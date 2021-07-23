package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.behavior.functionapproximation.sparse.LinearVFA
import burlap.behavior.policy.EpsilonGreedy
import burlap.behavior.singleagent.learning.tdmethods.QLearning
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam
import burlap.mdp.singleagent.environment.SimulatedEnvironment
import burlap.statehashing.simple.SimpleHashableStateFactory
import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.model.rl.BurlapAction
import eu.iv4xr.framework.model.rl.BurlapState
import kotlin.random.Random
import kotlin.system.exitProcess

object BurlapAlgorithms {

    /**
     * Q learning, the most basic ALG
     */
    fun <S : BurlapState, A : BurlapAction> qLearning(discountFactor: Double, learningRate: Double, qInit: Double, numEpisodes: Int, random: Random) = BurlapAlg<S, A>(random) {
        val qLearning = QLearning(domain, discountFactor, DataClassStateFactory(), qInit, learningRate)
        val greedyQPolicyWithQValues = GreedyQPolicyWithQValues(qLearning)
        repeat(numEpisodes) {
            qLearning.runLearningEpisode(SimulatedEnvironment(model, stateGenerator))
        }
        greedyQPolicyWithQValues
    }

//    fun <S : BurlapState, A : BurlapAction> fittedVI(random: Random) = BurlapAlg<S, A>(random) {
//
//    }

    fun <S : BurlapState, A : BurlapAction> gradientSarsaLam(discountFactor: Double, learningRate: Double, lambda: Double, numEpisodes: Int, random: Random) = BurlapAlg<S, A>(random) {
        val features = mdp.features()
        val lam = GradientDescentSarsaLam(domain, discountFactor, LinearVFA(mdp.features()), learningRate, lambda)
        lam.setLearningPolicy(SafePolicy(mdp, EpsilonGreedy(lam, 0.1)))
        repeat(numEpisodes) {
            lam.runLearningEpisode(SimulatedEnvironment(model, stateGenerator))
        }
        GreedyQPolicyWithQValues(lam)
    }

}