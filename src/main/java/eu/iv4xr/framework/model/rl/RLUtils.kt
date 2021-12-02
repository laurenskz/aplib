package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassHashableState
import eu.iv4xr.framework.model.rl.valuefunctions.QFunction
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableValuefunction
import eu.iv4xr.framework.model.rl.valuefunctions.Valuefunction
import kotlin.math.abs

infix fun ClosedFloatingPointRange<Double>.sampleWithStepSize(step: Double) = Distributions.real(this.start, this.endInclusive, step)

data class StateWithGoalProgress<State : Identifiable>(val progress: List<Boolean>, val state: State) : DataClassHashableState()

/**
 * Compute the value of a state for an optimal policy
 */
fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.stateValue(state: State, discountFactor: Double, depth: Int): Double {
    if (isTerminal(state)) return 0.0
    return possibleActions(state).maxOf { action ->
        qValue(state, action, discountFactor, depth)
    }
}

/**
 * Compute the value of a state for the given policy
 */
fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.stateValue(state: State, policy: Policy<State, Action>, discountFactor: Double, depth: Int): Double {
    if (isTerminal(state)) return 0.0
    return policy.action(state).expectedValue { action ->
        qValue(state, policy, action, discountFactor, depth)

    }
}

/**
 * Compute the qValue of a state action pair for a specific policy
 */
fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.qValue(state: State, policy: Policy<State, Action>, action: Action, discountFactor: Double, depth: Int): Double {
    if (depth == 0) return expectedReward(state, action)
    return expectedReward(state, action) + discountFactor * transition(state, action).expectedValue { newState ->
        stateValue(newState, policy, discountFactor, depth - 1)
    }
}

/**
 * Compute the qValue for a state action pair for the optimal policy
 */
fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.qValue(state: State, action: Action, discountFactor: Double, depth: Int): Double {
    if (depth == 0) return expectedReward(state, action)
    return expectedReward(state, action) + discountFactor * transition(state, action).expectedValue { newState ->
        stateValue(newState, discountFactor, depth - 1)
    }
}

/**
 * Expected reward by performing a specific action in a specific state
 */
fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.expectedReward(state: State, action: Action): Double {
    return transition(state, action).expectedValue { newState ->
        reward(state, action, newState).expectedValue()
    }
}

fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.qValue(state: State, action: Action, discountFactor: Double, valueFunction: Valuefunction<State>): Double {
    return expectedReward(state, action) + discountFactor * transition(state, action).expectedValue { newState ->
        if (isTerminal(newState)) 0.0 else
            valueFunction.value(newState).toDouble()
    }
}

fun <State : Identifiable, Action : Identifiable> expectedUpdate(state: State, gamma: Float, mdp: MDP<State, Action>, valueFunction: Valuefunction<State>): Target<State> {
    return Target(state, mdp.possibleActions(state)
            .maxOf { a -> mdp.qValue(state, a, gamma.toDouble(), valueFunction).toFloat() })
}

fun <State : Identifiable, Action : Identifiable> valueIterationSweep(states: List<State>, valueFunction: TrainableValuefunction<State>, mdp: MDP<State, Action>, gamma: Float): Float {
    return states.maxOf {
        val current = valueFunction.value(it)
        val target = expectedUpdate(it, gamma, mdp, valueFunction)
        valueFunction.train(target)
        abs(current - target.target)
    }
}

fun <State : Identifiable, Action : Identifiable> expectedUpdate(state: State, gamma: Float, mdp: MDP<State, Action>, valueFunction: Valuefunction<State>, lookAhead: Int = 0): Target<State> {
    if (lookAhead == 0)
        return Target(state, mdp.possibleActions(state)
                .maxOf { a -> mdp.qValue(state, a, gamma.toDouble(), valueFunction).toFloat() })
    val max = mdp.possibleActions(state).maxOf { a ->
        mdp.transition(state, a).support().maxOf { sp ->
            var total = mdp.reward(state, a, sp).expectedValue().toFloat()
            if (!mdp.isTerminal(sp))
                total += gamma * expectedUpdate(sp, gamma, mdp, valueFunction, lookAhead - 1).target
            total
        }
    }
    return Target(state, max)
}

fun <State : Identifiable, Action : Identifiable> deepQValue(state: State, action: Action, gamma: Float, mdp: MDP<State, Action>, qFunction: QFunction<State, Action>, depth: Int): Double {
    return when (depth) {
        0 -> qFunction.qValue(state, action).toDouble()
        1 -> mdp.transition(state, action).expectedValue { sp -> mdp.reward(state, action, sp).expectedValue() + gamma * qFunction.stateValue(sp, mdp) }
        else -> mdp.transition(state, action).expectedValue { sp -> mdp.reward(state, action, sp).expectedValue() + gamma * mdp.possibleActions(sp).maxOf { deepQValue(sp, it, gamma, mdp, qFunction, depth - 1) } }
    }
}

fun <State : Identifiable, Action : Identifiable> bellmanResidual(states: List<State>, valueFunction: TrainableValuefunction<State>, mdp: MDP<State, Action>, gamma: Float): Double {
    return valueFunction.values(states).zip(states).map { (v, s) ->
        abs(expectedUpdate(s, gamma, mdp, valueFunction).target - v)
    }.sum().toDouble()
}


data class Reward(val reward: Double, val timeStep: Int)

data class Transition<S : Identifiable, A : Identifiable>(val s: S, val a: A)

data class SAS<S : Identifiable, A : Identifiable>(val state: S, val action: A, val sp: S) {
    fun transitionProb(model: ProbabilisticModel<S, A>) = model.transition(state, action).score(sp)
}

class TransitionLog<S : Identifiable, A : Identifiable>(val model: ProbabilisticModel<S, A>) {
    val steps = mutableListOf<Transition<S, A>>()
    var terminal: S? = null

    fun plausible() = transitionProbabilities().all { it > 0 }

    fun transitionProbabilities() = sas().map { model.transition(it.state, it.action).score(it.sp) }

    fun sas(): Sequence<SAS<S, A>> {
        val lastSas = terminal?.run { steps.last().let { SAS(it.s, it.a, this) } }
        val seq = steps.asSequence().windowed(2) {
            SAS(it[0].s, it[0].a, it[1].s)
        }
        if (lastSas == null) return seq
        return (seq + sequenceOf(lastSas))
    }
}

fun <S : Identifiable, A : Identifiable> analyzeFaulty(transitionLog: TransitionLog<S, A>, string: (S) -> String): String =
        transitionLog.sas().filter {
            it.transitionProb(transitionLog.model) <= 0
        }.joinToString("\n") {
            "Deviation from model, was in state:\n ${string(it.state)}\nTook action ${it.action}, ended up in:\n${string(it.sp)}"
        }

