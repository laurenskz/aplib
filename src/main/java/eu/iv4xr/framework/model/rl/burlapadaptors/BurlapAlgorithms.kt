package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.behavior.learningrate.ExponentialDecayLR
import burlap.behavior.learningrate.LearningRate
import burlap.behavior.policy.BoltzmannQPolicy
import burlap.behavior.policy.GreedyQPolicy
import burlap.behavior.singleagent.learning.tdmethods.QLearning
import burlap.mdp.singleagent.environment.SimulatedEnvironment
import burlap.statehashing.simple.SimpleHashableStateFactory
import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.model.rl.BurlapAction
import eu.iv4xr.framework.model.rl.BurlapState
import kotlin.random.Random

object BurlapAlgorithms {

    /**
     * Q learning, the most basic ALG
     */
    fun <S : BurlapState, A : BurlapAction> qLearning(discountFactor: Double, learningRate: Double, qInit: Double, numEpisodes: Int, random: Random) = BurlapAlg<S, A>(random) {
        val qLearning = QLearning(domain, discountFactor, SimpleHashableStateFactory(), qInit, learningRate)
        val greedyQPolicyWithQValues = GreedyQPolicyWithQValues(qLearning)
        repeat(numEpisodes) {
            qLearning.runLearningEpisode(SimulatedEnvironment(model, stateGenerator))
            val initialStateValue = mdp.initialState().expectedValue {
                greedyQPolicyWithQValues.qValue(it, greedyQPolicyWithQValues.action(it))
            }
//            println(initialStateValue)
        }
        greedyQPolicyWithQValues
    }
}