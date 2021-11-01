package eu.iv4xr.framework.model.rl.algorithms

import cern.jet.stat.Gamma
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.RLAlgorithm
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableValuefunction
import java.io.OutputStream
import java.io.PrintWriter
import kotlin.random.Random

data class PolicyGradientTarget<S : Identifiable, A : Identifiable>(val s: S, val a: A, val update: Double)
interface ModifiablePolicy<S : Identifiable, A : Identifiable> : Policy<S, A> {
    fun update(target: PolicyGradientTarget<S, A>)
    fun updateAll(targets: List<PolicyGradientTarget<S, A>>) = targets.forEach(::update)
}

interface RewardLogger {
    fun episodeReward(episode: Int, reward: Float)
}

class NoOpRewardLogger : RewardLogger {
    override fun episodeReward(episode: Int, reward: Float) {
    }

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
        val rewardLogger: RewardLogger = NoOpRewardLogger()
) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        repeat(episodes) {
            var i = 1.0
            var state = mdp.initialState().sample(random)
            var total = 0.0
            while (!mdp.isTerminal(state)) {
                val sars = mdp.sampleSARS(policy, state, random)
                val target = sars.r + gamma * if (mdp.isTerminal(sars.sp)) 0f else valueFunction.value(sars.sp)
                if (target > 0.001) {
                    valueFunction.train(Target(state, target.toFloat()))
                    policy.update(PolicyGradientTarget(state, sars.a, i * target - valueFunction.value(state)))
                }
                total += sars.r * i
                i *= gamma
                state = sars.sp
            }
            rewardLogger.episodeReward(it, total.toFloat())
        }
        return policy
    }
}