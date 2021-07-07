package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.behavior.policy.BoltzmannQPolicy
import burlap.behavior.policy.GreedyQPolicy
import burlap.behavior.singleagent.learning.tdmethods.QLearning
import burlap.mdp.singleagent.environment.SimulatedEnvironment
import burlap.statehashing.simple.SimpleHashableStateFactory
import eu.iv4xr.framework.model.rl.BurlapAction
import eu.iv4xr.framework.model.rl.BurlapState
import kotlin.random.Random

object BurlapAlgorithms {

    fun <S : BurlapState, A : BurlapAction> qLearning(discountFactor: Double, learningRate: Double, qInit: Double, numEpisodes: Int) = BurlapAlg<S, A>(Random(12)) {
        val qLearning = QLearning(domain, discountFactor, SimpleHashableStateFactory(), qInit, learningRate)
        repeat(numEpisodes) {
            qLearning.runLearningEpisode(SimulatedEnvironment(model, stateGenerator))
        }
        GreedyQPolicy(qLearning)
    }
}