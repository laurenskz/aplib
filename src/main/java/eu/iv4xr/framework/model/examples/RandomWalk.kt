package eu.iv4xr.framework.model.examples

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.examples.RandomWalkAction.*
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import kotlin.math.exp

data class RandomWalkState(val value: Double) : Identifiable

/**
 * We can hold a stock or sell it
 */
enum class RandomWalkAction : Identifiable {
    HOLD, SELL
}

/**
 * Illustrates an example of a random walk. There is a hypothesis that stock prices follow a random walk
 * This MDP therefore can be seen as modelling a stock price. It is impossible to create a policy for this MDP
 * such that the expected reward is greater than 0
 */
class RandomWalk(private val offset: Distribution<Double>) : MDP<RandomWalkState, RandomWalkAction> {
    override fun possibleStates() = generateSequence(0.0) { it + 0.01 }.let { it + it.map { it * -1.0 } }.map { RandomWalkState(it) }

    override fun possibleActions(state: RandomWalkState) = values().asSequence()

    override fun isTerminal(state: RandomWalkState) = false

    override fun transition(current: RandomWalkState, action: RandomWalkAction) = when (action) {
        HOLD -> offset.map { RandomWalkState(current.value + it) }
        SELL -> always(RandomWalkState(0.0))
    }

    override fun reward(current: RandomWalkState, action: RandomWalkAction, newState: RandomWalkState) = when (action) {
        HOLD -> always(0.0)
        SELL -> always(current.value)
    }
}

/**
 * Try to create a policy such that the expected value of the initial state is greater than 0
 *
 * The idea is to sell it with greater probability if it has made a profit.
 */
class RandomWalkPolicy : Policy<RandomWalkState, RandomWalkAction> {

    fun sigmoid(x: Double) = 1.0 / (1.0 + exp(-x))


    override fun action(state: RandomWalkState): Distribution<RandomWalkAction> {
        val sellProb = sigmoid(state.value)
        return Distributions.discrete(
                SELL to sellProb,
                HOLD to (1 - sellProb)
        )
    }
}