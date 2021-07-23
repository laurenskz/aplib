package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.rl.*
import java.lang.IllegalArgumentException
import java.lang.IllegalStateException
import kotlin.random.Random

/**
 * Greedy algorithm based on qValues directly from the MDP, it chooses the action with the best Q value breaking ties arbitrarily
 */
class Greedy<ModelState : Identifiable, Action : Identifiable>(private val rlmdp: MDP<ModelState, Action>, private val discountFactor: Double, private val depth: Int) : Policy<ModelState, Action> {
    override fun action(state: ModelState): Distribution<Action> {
        val groups = rlmdp.possibleActions(state).groupBy {
            rlmdp.qValue(state, it, discountFactor, depth)
        }
        val maxKey = groups.keys.maxOf { it }
        val actions = groups[maxKey]
        if (actions.isNullOrEmpty()) throw IllegalArgumentException("No action for state")
        return Distributions.uniform(actions)
    }
}

/**
 * Implement RLAlgorithm interface
 */
class GreedyAlg<S : Identifiable, A : Identifiable>(private val discountFactor: Double, private val depth: Int) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        return Greedy(mdp, discountFactor, depth)
    }
}

