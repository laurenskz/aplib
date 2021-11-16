package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.algorithms.Grid
import eu.iv4xr.framework.model.rl.algorithms.GridWorldAction
import eu.iv4xr.framework.model.rl.algorithms.GridWorldState
import eu.iv4xr.framework.model.rl.algorithms.Square
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import org.junit.Test
import org.junit.jupiter.api.Assertions.*
import org.tensorflow.types.TFloat32
import kotlin.random.Random
import kotlin.system.exitProcess

internal class ICMModuleImplTest {

    @Test
    fun testICM() {
        val size = 4
        val goal = Square(20, 20)
        val actionFactory = GridWorldAction.factory
        val grid = Grid((0..size).flatMap { x -> (0..size).map { y -> Square(x, y) } })
        println(grid.squares.size)
        val factory = GridWorldState.factoryForGrid(grid)
        val model = ICMModel(0.2, 0.9, factory.count(), actionFactory.count(),
                Sequential(dense(factory.count().toLong())),
                Sequential(dense(factory.count().toLong())),
                Sequential(rawLayer(actionFactory.count().toLong()))
        )
        val icm = ICMModuleImpl(model, factory, actionFactory, "debug")
        repeat(1000) {
            println(it)
            println(icm.train(
                    listOf(
                            BurlapAlgorithms.SARS(GridWorldState(Square(0, 0), 0), GridWorldAction.DOWN, GridWorldState(Square(0, 0), 1), 0.0, 1.0),
                            BurlapAlgorithms.SARS(GridWorldState(Square(1, 1), 0), GridWorldAction.RIGHT, GridWorldState(Square(2, 1), 1), 0.0, 1.0)

                    )
            ))
            println(icm.train(
                    listOf(
                            BurlapAlgorithms.SARS(GridWorldState(Square(0, 0), 0), GridWorldAction.RIGHT, GridWorldState(Square(1, 0), 1), 0.0, 1.0)
                    )
            ))
            println(icm.train(
                    listOf(
                            BurlapAlgorithms.SARS(GridWorldState(Square(1, 1), 0), GridWorldAction.RIGHT, GridWorldState(Square(2, 1), 1), 0.0, 1.0)
                    )
            ))
        }
    }
}