package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.algorithms.*
import eu.iv4xr.framework.model.rl.valuefunctions.QTarget
import org.junit.Test
import org.junit.jupiter.api.Assertions.*
import kotlin.random.Random

class TFQFunctionTest {
    @Test
    fun test() {
        val size = 10
        val goal = Square(8, 8)
        val grid = (0..size).flatMap { x -> (0..size).map { y -> Square(x, y) } }.let { Grid(size, size, it) }
        val mdp = GridWorld(grid, goal, 1000)
        val factory = GridWorldState.factoryForGrid(grid)
        val qFunction = TFQFunction(factory, mdp) {
            QModelDefBuilder(Sequential(
            ), factory.shape, it, 1f)
        }
        val offPolicyQLearning = OffPolicyQLearning(qFunction, 0.9f, mdp, Random)
        offPolicyQLearning.trainEPolicy(100)
        val bab = mdp.sampleEpisode(GreedyPolicy(qFunction, mdp), Random)
        println(bab.steps.size)
    }

}