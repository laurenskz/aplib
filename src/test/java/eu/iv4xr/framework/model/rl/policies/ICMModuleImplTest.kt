package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.algorithms.*
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import org.junit.Test
import org.junit.jupiter.api.Assertions.*
import org.tensorflow.ndarray.Shape
import org.tensorflow.types.TFloat32
import kotlin.random.Random
import kotlin.system.exitProcess

internal class ICMModuleImplTest {

    @Test
    fun testEncoding() {
        val size = 50
        val grid = Grid(size, size, (0 until size).flatMap { x -> (0 until size).map { y -> Square(x, y) } })
        val factory = GridWorldState.tensorFactoryForGrid(grid)
        val actionFactory = GridWorldAction.factory
        val statePrimeSize = 32L
        val model = ICMModel(0.0, 0.01, factory.shape, actionFactory.shape,
                Sequential(
                        convLayer(4, 4, 32, "bab"),
                        maxPoolLayer(2, 2, 2),
                        convLayer(3, 3, 64, "bab2"),
                        maxPoolLayer(2, 2, 2),
                        flatten(),
                        dense(statePrimeSize)
                ),
                Sequential(dense(64), dense(statePrimeSize)),
                Sequential(dense(32), rawLayer(actionFactory.count().toLong()))
        )
        val icm = ICMModuleImpl(model, factory, actionFactory, "icmTest")
        val mdp = GridWorld(grid, Square(15, 15), 100)
        repeat(10000) {
            val episode = mdp.sampleEpisode(RandomPolicy(mdp), Random)
            icm.train(episode.steps.map { it.toICM() })
        }
    }
}