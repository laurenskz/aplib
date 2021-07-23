package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.Distributions.uniform
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.RLAlgorithm

class RandomPolicy<S : Identifiable, A : Identifiable>(private val mdp: MDP<S, A>) : Policy<S, A> {
    override fun action(state: S) = uniform(mdp.possibleActions(state).toList())
}

class RandomAlg<S : Identifiable, A : Identifiable>() : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>) = RandomPolicy(mdp)
}
