package eu.iv4xr.framework.model.rl.approximation

import eu.iv4xr.framework.model.rl.StateWithGoalProgress
import eu.iv4xr.framework.model.rl.ai.BasicModel
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.sampleWithStepSize
import eu.iv4xr.framework.model.utils.DeterministicRandom
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.junit.Ignore
import org.junit.Test

internal class PlaygroundTest {

    @Test
    fun test() {
        val random = DeterministicRandom()
        repeat(3) {
            println(Playground().transition(PlaygroundState(100, 0.1, 0.0, 0), PlaygroundAction(100))
                    .sample(random))
        }
        repeat(3) {
            println(Playground().transition(PlaygroundState(100, 0.2, 0.3, 0), PlaygroundAction(100))
                    .sample(random))
        }
    }

    @Test
    fun testAlg() {
        val targets = listOf(100)
        val mdp = playgroundMDP(targets)
        val alg = BurlapAlgorithms.gradientSarsaLam(0.99, 0.1, 0.4, 1, stateActionWithGoalProgressFactory(combinedPlaygroundFactory, targets.size), DeterministicRandom())
        val policy = alg.train(mdp)
        val doubles = ((0.0..1.0) sampleWithStepSize 0.3).support()
        doubles.forEach { l ->
            doubles.forEach { bl ->
                println("State:$l,$bl")
                println("\t" + policy.action(StateWithGoalProgress(listOf(false), PlaygroundState(30, l, bl, 93))).sample(DeterministicRandom()))
            }
        }
    }

    @Test
    fun testFeatures() {
        val factory = MergedFeatureFactory(stateWithGoalProgressFactory(PlaygroundState.factory, 1), PlaygroundAction.factory)
        println(factory.features(StateWithGoalProgress(listOf(false), PlaygroundState(0, 0.9, 0.3, 2)) to PlaygroundAction(2)).toList())
    }

    @Test
    fun testLSPI() {
        val targets = listOf(100)
        val mdp = playgroundMDP(targets)
        val alg = BurlapAlgorithms.lspi(0.99, 5000, 100, stateActionWithGoalProgressFactory(combinedPlaygroundFactory, targets.size), DeterministicRandom())
        val policy = alg.train(mdp)
        val doubles = ((0.0..1.0) sampleWithStepSize 0.3).support()
        doubles.forEach { l ->
            doubles.forEach { bl ->
                println("State:$l,$bl")
                println("\t" + policy.action(StateWithGoalProgress(listOf(false), PlaygroundState(30, l, bl, 93))).sample(DeterministicRandom()))
            }
        }
    }

    @Test
    @Ignore
    fun testKerasModel() {
        val basicModel = BasicModel()
        basicModel.description(1).let {
            basicModel.init(it)
            val dataset = OnHeapDataset.Companion.create(arrayOf(FloatArray(1) { 1f }, FloatArray(1) { 0f }), floatArrayOf(0.65f, 0.15f))
            it.fit(dataset, epochs = 1000)
            println(it.evaluate(dataset))
            println(it.predictSoftly(FloatArray(1) { 1f }).toList())
            println(it.predictSoftly(FloatArray(1) { 0.01f }).toList())
        }
    }

    @Test
    @Ignore
    fun testFittedVI() {
        val targets = listOf(100)
        val mdp = playgroundMDP(targets)
        val factory = stateWithGoalProgressFactory(PlaygroundState.factory, targets.size)
        val alg = BurlapAlgorithms.fittedVI<StateWithGoalProgress<PlaygroundState>, PlaygroundAction>(
                0.99,
                20,
                1000000,
                5e-4,
                100,
                BasicModel(),
                factory,
                DeterministicRandom(),
                Playground.stateDist.map { StateWithGoalProgress(targets.map { false }, it) })
        val policy = alg.train(mdp)
        val doubles = ((0.0..1.0) sampleWithStepSize 0.3).support()
        doubles.forEach { l ->
            doubles.forEach { bl ->
                println("State:$l,$bl")
                println("\t" + policy.action(StateWithGoalProgress(listOf(false), PlaygroundState(30, l, bl, 93))).supportWithDensities())
                println("\t" + policy.action(StateWithGoalProgress(listOf(false), PlaygroundState(30, l, bl, 93))).sample(DeterministicRandom()))
            }
        }
    }
}