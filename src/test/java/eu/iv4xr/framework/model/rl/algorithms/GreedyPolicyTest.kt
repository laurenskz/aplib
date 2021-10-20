package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.rl.StateWithGoalProgress
import eu.iv4xr.framework.model.rl.ai.BasicModel
import eu.iv4xr.framework.model.rl.approximation.*
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.policies.GreedyPolicy
import eu.iv4xr.framework.model.rl.sampleWithStepSize
import eu.iv4xr.framework.model.utils.DeterministicRandom
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.junit.Assert
import org.junit.Test
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.assertEquals

class TestModel : ModelDescription {
    override fun create(inputSize: Int, outputSize: Int): GraphTrainableModel {
        val model = Sequential.of(listOf(
                Input(inputSize.toLong()),
                Dense(1, kernelInitializer = Constant(1f), biasInitializer = Constant(0f))
        ))
        model.compile(Adam(), Losses.HUBER, Metrics.ACCURACY)
        model.init()
        return model
    }
}

internal class GreedyPolicyTest {
    val targets = listOf(100)
    val mdp = playgroundMDP(targets)

    val featureActionFactory = stateActionWithGoalProgressFactory(MergedFeatureFactory(PlaygroundState.factory, PlaygroundAction.factory), targets.size)

    //    val stateActionInputQModel = StateActionInputQModel(TestModel(), featureActionFactory)
//    val model = BaseDeepModel(stateActionInputQModel)
    val state = StateWithGoalProgress(listOf(false), PlaygroundState(20, 0.3, 0.3, 0))


    fun initModel(m: GraphTrainableModel) {
        m.compile(Adam(), Losses.HUBER, Metrics.ACCURACY)
        m.init()
    }


    fun <S, A> createTestModel(fac: FeatureActionFactory<S, A>) =
            BasicModel().description(fac.count()).also {
                it.compile(Adam(learningRate = 0.0001f), Losses.HUBER, Metrics.ACCURACY)
                it.init()
            }


//    @Test
//    fun testPlayground() {
//        val specialFactory = stateActionWithGoalProgressFactory(combinedPlaygroundFactory, targets.size)
//        val fac = featureActionFactory
//        val deepModel = createTestModel(fac)
//        val targetDeepModel = createTestModel(fac)
//        val dql = DeepQLearning(deepModel, targetDeepModel, fac, DeterministicRandom(), 1f, 128, 100000)
//        val policy = dql.train(mdp)
//        val doubles = ((0.0..1.0) sampleWithStepSize 0.3).support()
//        doubles.forEach { l ->
//            doubles.forEach { bl ->
//                println("State:$l,$bl")
//                val state1 = StateWithGoalProgress(listOf(false), PlaygroundState(1, l, bl, 0))
//                println(policy.qValues(state1))
//                println("\t" + policy.action(state1).supportWithDensities())
//                println("\t" + policy.action(state1).sample(DeterministicRandom()))
//            }
//        }
//    }
}

