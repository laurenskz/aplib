package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.behavior.functionapproximation.dense.DenseStateActionLinearVFA
import burlap.behavior.singleagent.learning.lspi.LSPI
import burlap.behavior.singleagent.learning.lspi.SARSCollector
import burlap.behavior.singleagent.learning.lspi.SARSData
import burlap.behavior.singleagent.learning.tdmethods.QLearning
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam
import burlap.behavior.singleagent.planning.vfa.fittedvi.FittedVI
import burlap.mdp.singleagent.environment.SimulatedEnvironment
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.rl.BurlapAction
import eu.iv4xr.framework.model.rl.BurlapState
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.ai.KerasModel
import eu.iv4xr.framework.model.rl.ai.TensorflowModelTrainer
import eu.iv4xr.framework.model.rl.approximation.FeatureActionFactory
import eu.iv4xr.framework.model.rl.approximation.FeatureVectorFactory
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.Optimizer
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.OneHot
import kotlin.math.pow
import kotlin.random.Random

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

//    fun test(model: GraphTrainableModel, states: Array<FloatArray>, actions: IntArray, rewards: FloatArray, nextStates: Array<FloatArray>, gamma: Float, batchSize: Int) {
//        val nextRewards = model.predictSoftly(OnHeapDataset.create(states, FloatArray(states.size)), batchSize)
//        val next = nextRewards.mapNotNull { it.maxOrNull() }
//        val targets = FloatArray(rewards.size) { rewards[it] + gamma * next[it] }
//
//    }

    data class SARS<S, A>(val s: S, val a: A, val sp: S, val r: Double, val transitionProb: Double)
    data class Episode<S, A>(val steps: List<SARS<S, A>>) {
        fun totalReward(gamma: Double) = steps.mapIndexed { i, s -> gamma.pow(i) * s.r }.sum()
    }

    fun totalReward(gamma: Double, steps: Sequence<Double>) = steps.mapIndexed { i, s -> gamma.pow(i) * s }.sum()


    fun <S : BurlapState, A : BurlapAction> lspi(discountFactor: Double, numSamples: Int, maxEpisodeLength: Int, factory: FeatureActionFactory<S, A>, random: Random, numIterations: Int = 30) = BurlapAlg<S, A>(random) {
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

    fun <S : BurlapState, A : BurlapAction> fittedVI(discountFactor: Double, samples: Int, transitions: Int, maxDelta: Double, maxIterations: Int, model: KerasModel, factory: FeatureVectorFactory<S>, random: Random, completeStateDist: Distribution<S>) = BurlapAlg<S, A>(random) {
        val states = List(samples) { completeStateDist.sample(random) }
        val fittedVI = FittedVI(domain, discountFactor, TensorflowModelTrainer(model, factory), states, transitions, maxDelta, maxIterations)
        fittedVI.runVI()
        GreedyQPolicyWithQValues(fittedVI)
    }

}