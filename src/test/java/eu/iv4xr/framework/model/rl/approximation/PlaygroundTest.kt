package eu.iv4xr.framework.model.rl.approximation

import eu.iv4xr.framework.model.rl.StateWithGoalProgress
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.utils.DeterministicRandom
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
        val alg = BurlapAlgorithms.gradientSarsaLam<StateWithGoalProgress<PlaygroundState>, PlaygroundAction>(0.9, 0.01, 0.1, 1000, stateWithGoalProgressFactory(PlaygroundState.factory, targets.size), DeterministicRandom())
        alg.train(mdp)
    }

    @Test
    fun testLSPI() {
        val targets = listOf(100)
        val mdp = playgroundMDP(targets)
        val factory = MergedFeatureFactory(stateWithGoalProgressFactory(PlaygroundState.factory,targets.size), PlaygroundAction.factory)
        val alg = BurlapAlgorithms.lspi(0.9,1000,factory,DeterministicRandom())
        alg.train(mdp)
    }
}