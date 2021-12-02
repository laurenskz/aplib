package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.RLAlgorithm
import kotlin.math.pow
import kotlin.random.Random

class MCPolicyGradient<S : Identifiable, A : Identifiable>(val policy: ModifiablePolicy<S, A>,
                                                           val episodes: Int,
                                                           val gamma: Double,
                                                           val random: Random) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        repeat(episodes) {
            val episode = mdp.sampleEpisode(policy, random)
            val rewards = episode.steps.map { it.r }
            var total = 0.0
            for (i in (episode.steps.indices.reversed())) {
                total = rewards[i] + gamma * total
                val sars = episode.steps[i]
                policy.update(PolicyGradientTarget(sars.s, sars.a, gamma.pow(i) * total))
            }
        }
        return policy
    }
}