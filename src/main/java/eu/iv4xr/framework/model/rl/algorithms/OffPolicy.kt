package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.expectedUpdate
import eu.iv4xr.framework.model.rl.policies.GreedyPolicy
import eu.iv4xr.framework.model.rl.qValue
import eu.iv4xr.framework.model.rl.valuefunctions.QTarget
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableQFunction
import eu.iv4xr.framework.model.rl.valuefunctions.ValueFromQ
import kotlin.random.Random


class OffPolicyQLearning<S : Identifiable, A : Identifiable>(val qFunction: TrainableQFunction<S, A>, val gamma: Float, val mdp: MDP<S, A>, val random: Random) {

    val valueFunction = ValueFromQ(qFunction, mdp)

    fun train(episode: BurlapAlgorithms.Episode<S, A>) {
        episode.steps.reversed().forEach {
            train(it)
        }
    }

    fun train(it: BurlapAlgorithms.SARS<S, A>) {
        train(setOf(it), 1, 1)
    }

    fun improve(state: S, depth: Int = 2) {
        mdp.possibleActions(state).forEach {
            val target = expectedUpdate(state, gamma, mdp, valueFunction, depth)
            qFunction.train(QTarget(state, it, target.target))
        }
    }

    fun trainEPolicy(episodes: Int) {
        val epolicy = EGreedyPolicy(0.1, mdp, GreedyPolicy(qFunction, mdp))

        repeat(episodes) {
            var state = mdp.initialState().sample(random)
            while (!mdp.isTerminal(state)) {
                val sars = mdp.sampleSARS(epolicy, state, random)
                train(sars)
                state = sars.sp

            }
        }
    }

    fun train(experience: Set<BurlapAlgorithms.SARS<S, A>>, batchSize: Int, batches: Int) {
        repeat(batches) {
            val batch = experience.takeRandom(batchSize, random)
            val nextStates = batch.map { if (mdp.isTerminal(it.sp)) 0f else mdp.possibleActions(it.sp).maxOf { a -> qFunction.qValue(it.sp, a) } }
            val targets = batch.mapIndexed { i, t -> QTarget(t.s, t.a, (t.r + gamma * nextStates[i]).toFloat()) }
            qFunction.train(targets)
        }
    }
}