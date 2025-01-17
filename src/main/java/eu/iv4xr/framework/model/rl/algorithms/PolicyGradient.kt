package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.RLAlgorithm
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.valuefunctions.QTarget
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableQFunction
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

class NoOpRewardLogger : RewardLogger, CuriosityRewardLogger {
    override fun intrinsicEpisodeReward(episode: Int, reward: Float) {
    }

    override fun extrinsicEpisodeReward(episode: Int, reward: Float) {
    }

    override fun episodeReward(episode: Int, reward: Float) {
    }

}

data class ICMSample<S : Identifiable, A : Identifiable>(val state: S, val action: A, val statePrime: S)

fun <S : Identifiable, A : Identifiable> BurlapAlgorithms.SARS<S, A>.toICM() = ICMSample(s, a, sp)

interface ICMModule<S : Identifiable, A : Identifiable> {
    fun intrinsicReward(sars: ICMSample<S, A>): Double
    fun intrinsicReward(sars: List<ICMSample<S, A>>): List<Double> = sars.map(::intrinsicReward)
    fun train(sars: List<ICMSample<S, A>>): List<Double>
//    fun update(icmSample: ICMSample<S, A>)
//    fun updateAll(samples: List<ICMSample<S, A>>) = samples.forEach(::update)
}

class PrintRewardLogger(val outputStream: OutputStream) : RewardLogger {
    val writer = PrintWriter(outputStream)
    override fun episodeReward(episode: Int, reward: Float) {
        writer.println("Episode $episode, reward:$reward")
    }
}

class ICMActorCritic<S : Identifiable, A : Identifiable>(
        val policy: ModifiablePolicy<S, A>,
        val valueFunction: TrainableValuefunction<S>,
        val icm: ICMModule<S, A>,
        val random: Random,
        val gamma: Double,
        val episodes: Int,
        val eta: Double,
        val rewardLogger: RewardLogger = NoOpRewardLogger()
) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        repeat(episodes) {
            println("Episode $it")
            var i = 1.0
            var state = mdp.initialState().sample(random)
            while (!mdp.isTerminal(state)) {
                println()
                println("State:$state")
                val base = valueFunction.value(state)
                println("Value:$base")
                val action = policy.action(state)
                println("Action:" + action.supportWithDensities())
                val sars = mdp.executeAction(action.sample(random), state, random)
                println("Chose action ${sars.a}")
//                if (sars.r > 0) error("Reached first time in episode $it")
                val train = icm.train(listOf(sars.toICM()))
                val intrinsicReward = train.first()
                println("Intrinsic:$intrinsicReward")
                var sarsI = sars.copy(r = sars.r + (eta / 2.0) * intrinsicReward)
                val sp = if (mdp.isTerminal(sars.sp)) 0f else valueFunction.value(sars.sp)
                val target = sarsI.r + gamma * sp
                println("Value target:$target")
                valueFunction.train(Target(state, target.toFloat()))
                val newValue = valueFunction.value(state)
                println("Result:$newValue")
                val d = newValue - base
                println("Policy delta:$d")
                println()

                policy.update(PolicyGradientTarget(state, sars.a, d.toDouble()))
                i *= gamma
                state = sars.sp
            }
        }
        return policy
    }
}

class QActorCritic<S : Identifiable, A : Identifiable>(
        val policy: ModifiablePolicy<S, A>,
        val valueFunction: TrainableQFunction<S, A>,
        val icm: ICMModule<S, A>,
        val random: Random,
        val gamma: Double,
        val episodes: Int,
        val eta: Double,
        val rewardLogger: RewardLogger = NoOpRewardLogger()
) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        repeat(episodes) {
//            println("Episode $it")
            var i = 1.0
            var state = mdp.initialState().sample(random)
            var action = policy.action(state).sample(random)
            while (!mdp.isTerminal(state)) {
                val base = valueFunction.qValue(state, action)
//                println("Base:$base")
//                println("Policy:${policy.action(state).supportWithDensities()}")
//                println("State:$state")
//                println("Action:$action")
                val sars = mdp.executeAction(action, state, random)
                val train = icm.train(listOf(sars.toICM()))
                val intrinsicReward = train.first()
//                println("Intrinsic:$intrinsicReward")
                var sarsI = sars.copy(r = sars.r + (eta / 2.0) * intrinsicReward)
                val sp = if (mdp.isTerminal(sars.sp)) 0f else policy.action(sars.sp).expectedValue { valueFunction.qValue(sars.sp, it).toDouble() }.toFloat()
                val target = sarsI.r + gamma * sp
                valueFunction.train(QTarget(state, action, target.toFloat()))
                val newValue = valueFunction.qValue(state, action)
//                println("Result:$newValue")
                val d = newValue - base
//                println("Policy delta:$d")

                policy.update(PolicyGradientTarget(state, sars.a, d.toDouble()))
//                println("Result:" + policy.action(state).supportWithDensities())
//                println()

                i *= gamma
                state = sars.sp
                action = policy.action(state).sample(random)
            }
        }
        return policy
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
                val intrinsicRewards = icm.train(sars.map { it.toICM() })
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