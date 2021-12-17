package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.RLAlgorithm
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableValuefunction
import eu.iv4xr.framework.model.rl.valuefunctions.Valuefunction
import kotlin.math.pow
import kotlin.random.Random

class MCPolicyGradient<S : Identifiable, A : Identifiable>(val policy: ModifiablePolicy<S, A>,
                                                           val icm: ICMModule<S, A>,
                                                           val valueFunction: TrainableValuefunction<S>,
                                                           val visitFunction: TrainableValuefunction<S>,
                                                           val intrinsicWeight: Double,
                                                           val episodes: Int,
                                                           val gamma: Double,
                                                           val random: Random) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        repeat(episodes) {
            if (policy.action(mdp.initialState().sample(random)).supportWithDensities().values.any {
                        it > 0.999999
                    }) return policy
            val episode = mdp.sampleEpisode(policy, random)
            val rewards = episode.steps.map { it.r }
            val intrinsic = icm.train(episode.steps.map { it.toICM() })
            var total = 0.0
            for (i in (episode.steps.indices.reversed())) {
                total = (intrinsicWeight * intrinsic[i] + rewards[i]) + gamma * total
                val sars = episode.steps[i]
                visitFunction.train(Target(sars.s, visitFunction.value(sars.s) + 1))
                valueFunction.train(Target(sars.s, total.toFloat()))
                val current = valueFunction.value(sars.s)
                policy.update(PolicyGradientTarget(sars.s, sars.a, gamma.pow(i) * (total - current)))
            }
        }
        return policy
    }
}