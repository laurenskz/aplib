package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.RLAlgorithm
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms.totalReward
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableValuefunction
import eu.iv4xr.framework.model.rl.valuefunctions.Valuefunction
import kotlin.math.ceil
import kotlin.random.Random


data class A3CUpdate<S : Identifiable, A : Identifiable>(val target: Target<S>, val policyGradientTarget: PolicyGradientTarget<S, A>)
interface A3CTargetCreator<S : Identifiable, A : Identifiable> {
    fun createTargets(episodes: List<BurlapAlgorithms.Episode<S, A>>): List<List<A3CUpdate<S, A>>>
}

class BaseA3CTargetCreator<S : Identifiable, A : Identifiable>(val valueFunction: Valuefunction<S>, val gamma: Float, val mdp: MDP<S, A>) : A3CTargetCreator<S, A> {

    override fun createTargets(episodes: List<BurlapAlgorithms.Episode<S, A>>): List<List<A3CUpdate<S, A>>> {
        val finalStates = episodes.map { it.steps.last().sp }
        val finalValues = valueFunction.values(finalStates)
        val rs = finalValues.indices.map { if (mdp.isTerminal(finalStates[it])) 0f else finalValues[it] }
        return episodes.zip(rs).map { (episode, lastR) ->
            val steps = episode.steps
            val stateValues = valueFunction.values(steps.map { it.s })
            val discountedReturns = steps.indices.reversed().scan(lastR) { r, step ->
                (steps[step].r + gamma * r).toFloat()
            }.reversed()
            val returnMinusBaseline = discountedReturns.zip(stateValues).map { (r, base) -> r - base }
            returnMinusBaseline.indices.map {
                if (mdp.isTerminal(steps[it].s)) {
                    A3CUpdate(Target(steps[it].s, 0f), PolicyGradientTarget(steps[it].s, steps[it].a, 0.0))
                } else
                    A3CUpdate(Target(steps[it].s, discountedReturns[it]), PolicyGradientTarget(steps[it].s, steps[it].a, returnMinusBaseline[it].toDouble()))
            }
        }
    }
}

class A3C<S : Identifiable, A : Identifiable>(
        val policy: ModifiablePolicy<S, A>,
        val valueFunction: TrainableValuefunction<S>,
        val icm: ICMModule<S, A>,
        val random: Random,
        val gamma: Double,
        val episodes: Int,
        val rewardLogger: CuriosityRewardLogger,
        val batchSize: Int = 128,
        val eta: Double = 1.0
) : RLAlgorithm<S, A> {


    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        val ePolicy = EGreedyPolicy(0.0, mdp, policy)
        val targetCreator = BaseA3CTargetCreator(valueFunction, gamma.toFloat(), mdp)
        repeat(ceil(episodes / batchSize.toDouble()).toInt()) {
            val episodes = List(batchSize) { mutableListOf<BurlapAlgorithms.SARS<S, A>>() }
            var states = (0 until batchSize).map { mdp.initialState().sample(random) }
            println(policy.allActions(states).first().supportWithDensities())
            println(valueFunction.values(states))
            while (!states.all { mdp.isTerminal(it) }) {
                val allActions = ePolicy.allActions(states)
                val actions = allActions.map { it.sample(random) }
                val sars = actions.indices.map { mdp.executeAction(actions[it], states[it], random) }
                icm.train(sars.map { it.toICM() })
                episodes.indices.forEach { episodes[it].add(sars[it]) }
                states = sars.map { it.sp }
            }
            val intrinsicRewards = episodes.map { icm.intrinsicReward(it.map { it.toICM() }) }
            val extrinsicRewards = episodes.map { it.map { it.r } }
            val totalRewards = intrinsicRewards.zip(extrinsicRewards).map { (l, r) -> l.zip(r).map { (x, y) -> x + y } }
            val results = episodes.zip(intrinsicRewards).map { (episode, intrinsic) ->
                BurlapAlgorithms.Episode(episode.zip(intrinsic).map { (sars, r) -> sars.copy(r = sars.r + (eta / 2.0) * r) })
            }
            rewardLogger.intrinsicEpisodeReward(it * batchSize,
                    intrinsicRewards.map { totalReward(gamma, it.asSequence()) }.average().toFloat()
            )
            rewardLogger.extrinsicEpisodeReward(it * batchSize,
                    extrinsicRewards.map { totalReward(gamma, it.asSequence()) }.average().toFloat()
            )
            rewardLogger.episodeReward(it * batchSize,
                    totalRewards.map { totalReward(gamma, it.asSequence()) }.average().toFloat()
            )
            val targets = targetCreator.createTargets(results)
            for (step in (0 until targets.maxOf { it.size })) {
                val target = targets.map { it[step] }
                valueFunction.train(target.map { it.target })
                policy.updateAll(target.map { it.policyGradientTarget })
            }
        }
        return policy
    }
}