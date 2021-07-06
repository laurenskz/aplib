package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.rl.RandomWalkAction.*
import kotlin.math.exp
import kotlin.random.Random

data class RandomWalkState(val value: Double) : Identifiable

enum class RandomWalkAction : Identifiable {
    HOLD, DISCARD
}

class RandomWalk(private val offset: Distribution<Double>) : MDP<RandomWalkState, RandomWalkAction> {
    override fun possibleStates() = generateSequence(0.0) { it + 0.01 }.let { it + it.map { it * -1.0 } }.map { RandomWalkState(it) }

    override fun possibleActions(state: RandomWalkState) = values().asSequence()

    override fun isTerminal(state: RandomWalkState) = false

    override fun transition(current: RandomWalkState, action: RandomWalkAction) = when (action) {
        HOLD -> offset.map { RandomWalkState(current.value + it) }
        DISCARD -> always(RandomWalkState(0.0))
    }

    override fun reward(current: RandomWalkState, action: RandomWalkAction, newState: RandomWalkState) = when (action) {
        HOLD -> always(0.0)
        DISCARD -> always(current.value)
    }
}

class RandomWalkPolicy : Policy<RandomWalkState, RandomWalkAction> {

    fun sigmoid(x: Double) = 1.0 / (1.0 + exp(-x))


    override fun action(state: RandomWalkState): Distribution<RandomWalkAction> {
        val sellProb = if (state.value > 0) 0.3 else 0.5

        return Distributions.discrete(
                DISCARD to sellProb,
                HOLD to (1 - sellProb)
        )
    }
}

fun main() {
    val randomWalk = RandomWalk(Distributions.uniform(-0.01, 0.01))
    val policy = RandomWalkPolicy()
    println(randomWalk.stateValue(RandomWalkState(0.0), policy, 1.0, 10))
}