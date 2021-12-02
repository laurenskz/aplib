package eu.iv4xr.framework.model.rl.valuefunctions

import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP

interface Valuefunction<S> {
    fun value(state: S): Float
    fun values(states: List<S>): List<Float> = states.map { value(it) }
}

interface QFunction<S : Identifiable, A : Identifiable> {
    fun qValue(state: S, action: A): Float
    fun qForActions(state: S, possibilities: List<A>): List<Pair<A, Float>> = possibilities.map { it to qValue(state, it) }
    fun qValues(states: List<Pair<S, A>>): List<Float> = states.map { qValue(it.first, it.second) }
    fun qForActions(states: List<Pair<S, List<A>>>): List<List<Pair<A, Float>>> = states.map { qForActions(it.first, it.second) }

    fun stateValue(state: S, mdp: MDP<S, A>): Float {
        val maxOf = qValues(mdp.possibleActions(state).map { state to it }.toList()).maxOf { it }
        return maxOf
    }
}

data class Target<S>(val state: S, val target: Float)
data class QTarget<S : Identifiable, A : Identifiable>(val state: S, val action: A, val target: Float)

interface TrainableValuefunction<S> : Valuefunction<S> {
    fun train(target: Target<S>)
    fun train(targets: List<Target<S>>) = targets.forEach { train(it) }

}

interface TrainableQFunction<S : Identifiable, A : Identifiable> : QFunction<S, A> {
    fun train(target: QTarget<S, A>)
    fun train(targets: List<QTarget<S, A>>) = targets.forEach { train(it) }
}

class ValueFromQ<S : Identifiable, A : Identifiable>(val value: QFunction<S, A>, val mdp: MDP<S, A>) : Valuefunction<S> {
    override fun value(state: S): Float {
        return mdp.possibleActions(state).maxOf {
            value.qValue(state, it)
        }
    }
}

class QFromValue<S : Identifiable, A : Identifiable>(val value: TrainableValuefunction<S>, val mdp: MDP<S, A>, val gamma: Float) : QFunction<S, A>, TrainableValuefunction<S> by value {
    override fun qValue(state: S, action: A): Float {
        return mdp.transition(state, action).expectedValue {
            val reward = mdp.reward(state, action, it)
            val nextStateValue = value.value(it)
            reward.expectedValue() + gamma * nextStateValue
        }.toFloat()
    }

    override fun qForActions(state: S, possibilities: List<A>): List<Pair<A, Float>> {
        return super.qForActions(state, possibilities)
    }

    override fun qValues(states: List<Pair<S, A>>): List<Float> {
        return super.qValues(states)
    }

}