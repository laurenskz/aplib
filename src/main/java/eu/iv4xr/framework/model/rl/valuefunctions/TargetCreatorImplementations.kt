package eu.iv4xr.framework.model.rl.valuefunctions

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

class TDTargetCreator<S : Identifiable, A : Identifiable>(val valuefunction: QFunction<S, A>, val gamma: Float) : TargetCreator<S, A> {

    override fun createTargets(episodes: List<BurlapAlgorithms.Episode<S, A>>, mdp: MDP<S, A>): List<QTarget<S, A>> {
        val samples = episodes.flatMap { it.steps }
        val inputs = samples.map { it.sp to mdp.possibleActions(it.sp).toList() }
        val nextStateValues = valuefunction.qForActions(inputs).mapIndexed { i, v -> if (mdp.isTerminal(samples[i].sp)) 0f else v.maxOf { it.second } }
        return List(samples.size) { QTarget(samples[it].s, samples[it].a, samples[it].r.toFloat() + gamma * nextStateValues[it]) }
    }
}

class NStepTDTargetCreator<S : Identifiable, A : Identifiable>(val valuefunction: QFunction<S, A>, val gamma: Float, val n: Int) : TargetCreator<S, A> {

    override fun createTargets(episodes: List<BurlapAlgorithms.Episode<S, A>>, mdp: MDP<S, A>): List<QTarget<S, A>> {
        return episodes.flatMap {
            it.steps.mapIndexed { i, step ->
                var reward = 0.0
                val maxOffset = min(n, it.steps.size - i)
                for (offset in (0 until maxOffset)) {
                    reward += gamma.pow(offset) * it.steps[i + offset].r
                }
                if ((i + maxOffset) < it.steps.size)
                    reward += gamma.pow(maxOffset) * valuefunction.qValue(it.steps[i + maxOffset].s, it.steps[i + maxOffset].a)
                QTarget(step.s, step.a, reward.toFloat())
            }
        }
    }
}