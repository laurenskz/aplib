package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassHashableState
import eu.iv4xr.framework.model.rl.burlapadaptors.ReflectionBasedState

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


data class Reward(val reward: Double, val timeStep: Int)

data class Step<S : Identifiable, A : Identifiable>(val s: S, val a: A)

data class SAS<S : Identifiable, A : Identifiable>(val state: S, val action: A, val sp: S) {
    fun transitionProb(model: ProbabilisticModel<S, A>) = model.transition(state, action).score(sp)
}

class Progress<S : Identifiable, A : Identifiable>(val model: ProbabilisticModel<S, A>) {
    val steps = mutableListOf<Step<S, A>>()
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

fun <S : Identifiable, A : Identifiable> analyzeFaulty(progress: Progress<S, A>, string: (S) -> String): String =
        progress.sas().filter {
            it.transitionProb(progress.model) <= 0
        }.joinToString("\n") {
            "Deviation from model, was in state:\n ${string(it.state)}\nTook action ${it.action}, ended up in:\n${string(it.sp)}"
        }

