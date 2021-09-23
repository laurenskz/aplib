package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.behavior.functionapproximation.dense.DenseLinearVFA
import burlap.behavior.functionapproximation.dense.DenseStateActionLinearVFA
import burlap.behavior.functionapproximation.sparse.LinearVFA
import burlap.behavior.policy.EpsilonGreedy
import burlap.behavior.singleagent.learning.lspi.LSPI
import burlap.behavior.singleagent.learning.lspi.SARSCollector
import burlap.behavior.singleagent.learning.lspi.SARSData
import burlap.behavior.singleagent.learning.tdmethods.QLearning
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam
import burlap.mdp.singleagent.environment.SimulatedEnvironment
import burlap.statehashing.simple.SimpleHashableStateFactory
import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.model.rl.BurlapAction
import eu.iv4xr.framework.model.rl.BurlapState
import eu.iv4xr.framework.model.rl.approximation.FeatureActionFactory
import eu.iv4xr.framework.model.rl.approximation.FeatureVectorFactory
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

    fun <S : BurlapState, A : BurlapAction> gradientSarsaLam(discountFactor: Double, learningRate: Double, lambda: Double, numEpisodes: Int, factory: FeatureActionFactory<S, A>, random: Random) = BurlapAlg<S, A>(random) {
        val lam = GradientDescentSarsaLam(domain, discountFactor, DenseStateActionLinearVFA(factory.stateActionFeatures(), 0.0), learningRate, lambda)

        repeat(numEpisodes) {
            lam.runLearningEpisode(SimulatedEnvironment(model, stateGenerator))
        }
        GreedyQPolicyWithQValues(lam)
    }

    fun <S : BurlapState, A : BurlapAction> lspi(discountFactor: Double, numEpisodes: Int, factory: FeatureActionFactory<S, A>, random: Random) = BurlapAlg<S, A>(random) {
        val collector = SARSCollector.UniformRandomSARSCollector(domain)
        val dataset = SARSData()
        collector.collectNInstances(SimulatedEnvironment(model, stateGenerator), 100, 5, dataset)
        val lspi = LSPI(domain, discountFactor, factory.stateActionFeatures(), dataset)
        lspi.runPolicyIteration(30, 1e-6)
//        repeat(numEpisodes) {
//            val simulatedEnvironment = FixedEnv(model, stateGenerator)
//            lspi.runLearningEpisode(simulatedEnvironment)
//        }
        GreedyQPolicyWithQValues(lspi)
    }
}