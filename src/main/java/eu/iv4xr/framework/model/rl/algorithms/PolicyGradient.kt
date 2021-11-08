package eu.iv4xr.framework.model.rl.algorithms

import cern.jet.stat.Gamma
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.RLAlgorithm
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableValuefunction
import java.io.OutputStream
import java.io.PrintWriter
import kotlin.math.absoluteValue
import kotlin.math.ceil
import kotlin.random.Random

data class PolicyGradientTarget<S : Identifiable, A : Identifiable>(val s: S, val a: A, val update: Double)
interface ModifiablePolicy<S : Identifiable, A : Identifiable> : Policy<S, A> {
    fun update(target: PolicyGradientTarget<S, A>)
    fun updateAll(targets: List<PolicyGradientTarget<S, A>>) = targets.forEach(::update)
}

interface RewardLogger {
    fun episodeReward(episode: Int, reward: Float)
}

interface CuriosityRewardLogger : RewardLogger {
    fun intrinsicEpisodeReward(episode: Int, reward: Float)
    fun extrinsicEpisodeReward(episode: Int, reward: Float)
}

class NoOpRewardLogger : RewardLogger {
    override fun episodeReward(episode: Int, reward: Float) {
    }

}

data class ICMSample<S : Identifiable, A : Identifiable>(val state: S, val action: A, val statePrime: S)

interface ICMModule<S : Identifiable, A : Identifiable> {
    fun intrinsicReward(sars: BurlapAlgorithms.SARS<S, A>): Double
    fun intrinsicReward(sars: List<BurlapAlgorithms.SARS<S, A>>): List<Double> = sars.map(::intrinsicReward)
    fun train(sars: List<BurlapAlgorithms.SARS<S, A>>): List<Double>
//    fun update(icmSample: ICMSample<S, A>)
//    fun updateAll(samples: List<ICMSample<S, A>>) = samples.forEach(::update)
}

class PrintRewardLogger(val outputStream: OutputStream) : RewardLogger {
    val writer = PrintWriter(outputStream)
    override fun episodeReward(episode: Int, reward: Float) {
        writer.println("Episode $episode, reward:$reward")
    }
}

class ActorCritic<S : Identifiable, A : Identifiable>(
        val policy: ModifiablePolicy<S, A>,
        val valueFunction: TrainableValuefunction<S>,
        val random: Random,
        val gamma: Double,
        val episodes: Int,
        val rewardLogger: RewardLogger = NoOpRewardLogger(),
        val batchSize: Int = 1
) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        repeat(ceil(episodes / batchSize.toDouble()).toInt()) {
            var i = 1.0
            var states = (0 until batchSize).map { mdp.initialState().sample(random) }
            val total = MutableList(batchSize) { 0.0 }
            while (!states.all { mdp.isTerminal(it) }) {
                val actions = policy.allActions(states).map { it.sample(random) }
                val sars = actions.indices.map { mdp.executeAction(actions[it], states[it], random) }
                val statesPrime = sars.map { it.sp }
                val values = valueFunction.values(statesPrime)
                val targets = actions.indices.map { sars[it].r + gamma * if (mdp.isTerminal(statesPrime[it])) 0f else values[it] }
                val baseLines = valueFunction.values(states)
                val updates = targets.indices.map { i * targets[it] - baseLines[it] }
                if (updates.any { it.absoluteValue > 0.001 }) {
                    valueFunction.train(targets.indices.map {
                        Target(states[it], targets[it].toFloat())
                    })
                    policy.updateAll(updates.indices.map { PolicyGradientTarget(states[it], actions[it], updates[it]) })
                }
                for (index in total.indices) {
                    total[index] += sars[index].r * i
                }
                i *= gamma
                states = statesPrime
            }
            rewardLogger.episodeReward(it * batchSize, total.average().toFloat())
        }
        return policy
    }
}


class CuriosityDriven<S : Identifiable, A : Identifiable>(
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
        repeat(ceil(episodes / batchSize.toDouble()).toInt()) {
            var i = 1.0
            val total = MutableList(batchSize) { 0.0 }
            val totalI = MutableList(batchSize) { 0.0 }
            val totalE = MutableList(batchSize) { 0.0 }

            var states = (0 until batchSize).map { mdp.initialState().sample(random) }
            println(policy.allActions(states).first().supportWithDensities())
            println(valueFunction.values(states))
            while (!states.all { mdp.isTerminal(it) }) {
                val actions = ePolicy.allActions(states).map { it.sample(random) }
                val sars = actions.indices.map { mdp.executeAction(actions[it], states[it], random) }
                val intrinsicRewards = icm.train(sars)
                val extrinsicRewards = sars.map { it.r }
                val totalRewards = intrinsicRewards.indices.map { (eta / 2.0) * intrinsicRewards[it] + extrinsicRewards[it] }
                val statesPrime = sars.map { it.sp }
                val values = valueFunction.values(statesPrime)
                val targets = totalRewards.indices.map { totalRewards[it] + gamma * if (mdp.isTerminal(statesPrime[it])) 0f else values[it] }
                val baseLines = valueFunction.values(states)
                valueFunction.train(targets.indices.map {
                    Target(states[it], targets[it].toFloat())
                })
                val updates = targets.indices.map { if (mdp.isTerminal(states[it])) 0.0 else targets[it] - baseLines[it] }
                policy.updateAll(updates.indices.map { PolicyGradientTarget(states[it], actions[it], updates[it]) })
                for (index in total.indices) {
                    total[index] += totalRewards[index] * i
                    totalI[index] += intrinsicRewards[index] * i
                    totalE[index] += extrinsicRewards[index] * i
                }
                i *= gamma
                states = statesPrime
            }
            rewardLogger.episodeReward(it * batchSize, total.average().toFloat())
            rewardLogger.intrinsicEpisodeReward(it * batchSize, totalI.average().toFloat())
            rewardLogger.extrinsicEpisodeReward(it * batchSize, totalE.average().toFloat())
        }
        return policy
    }
}