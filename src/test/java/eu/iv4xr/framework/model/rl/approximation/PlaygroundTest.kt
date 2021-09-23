package eu.iv4xr.framework.model.rl.approximation

import eu.iv4xr.framework.model.rl.StateWithGoalProgress
import eu.iv4xr.framework.model.rl.algorithms.GreedyAlg
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.sampleWithStepSize
import eu.iv4xr.framework.model.utils.DeterministicRandom
import nl.uu.cs.aplib.exampleUsages.fiveGame.Optimal
import org.junit.Test
import org.junit.jupiter.api.Assertions.*

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
        val alg = BurlapAlgorithms.gradientSarsaLam<StateWithGoalProgress<PlaygroundState>, PlaygroundAction>(0.99, 0.001, 0.4, 1, MergedFeatureFactory(stateWithGoalProgressFactory(PlaygroundState.factory, targets.size), PlaygroundAction.factory), DeterministicRandom())
        val policy = alg.train(mdp)
        val doubles = ((0.0..1.0) sampleWithStepSize 0.3).support()
        doubles.forEach { l ->
            doubles.forEach { bl ->
                println("State:$l,$bl")
                println(policy.action(StateWithGoalProgress(listOf(false), PlaygroundState(30, l, bl, 0))).support().first())
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
        val factory = MergedFeatureFactory(stateWithGoalProgressFactory(PlaygroundState.factory, targets.size), PlaygroundAction.factory)
        val alg = BurlapAlgorithms.lspi(0.99, 500, stateActionWithGoalProgressFactory(combinedPlaygroundFactory, targets.size), DeterministicRandom())
        val policy = alg.train(mdp)
        val doubles = ((0.0..1.0) sampleWithStepSize 0.3).support()
        doubles.forEach { l ->
            doubles.forEach { bl ->
                println("State:$l,$bl")
                println("\t"+policy.action(StateWithGoalProgress(listOf(false), PlaygroundState(30, l, bl, 93))).sample(DeterministicRandom()))
            }
        }
    }
}