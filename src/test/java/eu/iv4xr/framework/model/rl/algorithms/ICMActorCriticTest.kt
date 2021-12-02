package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.algorithms.GridWorldAction.*
import eu.iv4xr.framework.model.rl.approximation.*
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassAction
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassHashableState
import eu.iv4xr.framework.model.rl.policies.*
import org.junit.Test
import kotlin.random.Random
import kotlin.system.exitProcess


enum class GridWorldAction : DataClassAction {
    UP, DOWN, LEFT, RIGHT;

    companion object {
        val factory = OneHot(values().toList())
    }
}

data class Square(val x: Int, val y: Int)

data class Grid(val squares: List<Square>)

data class GridWorldState(val position: Square, val steps: Int) : DataClassHashableState() {
    companion object {
        fun factoryForGrid(grid: Grid): FeatureVectorFactory<GridWorldState> {
            return CompositeFeature(listOf(OneHot(grid.squares).from { it.position }))
        }
    }
}

class GridWorld(val grid: Grid, val goal: Square, val maxSteps: Int) : MDP<GridWorldState, GridWorldAction> {
    override fun possibleStates(): Sequence<GridWorldState> {
        return grid.squares.flatMap { (0..maxSteps).map { step -> GridWorldState(it, step) } }.asSequence()
    }

    override fun allPossibleActions(): Sequence<GridWorldAction> {
        return values().asSequence()
    }

    override fun possibleActions(state: GridWorldState): Sequence<GridWorldAction> {
        return allPossibleActions()
    }

    override fun isTerminal(state: GridWorldState): Boolean {
        return state.position == goal || state.steps == maxSteps
    }

    override fun transition(current: GridWorldState, action: GridWorldAction): Distribution<GridWorldState> {
        val offsets = when (action) {
            UP -> (0 to -1)
            DOWN -> (0 to 1)
            LEFT -> (-1 to 0)
            RIGHT -> (1 to 0)
        }
        val sp = current.position.copy(x = current.position.x + offsets.first, current.position.y + offsets.second)
        if (grid.squares.contains(sp)) {
            return always(GridWorldState(sp, current.steps + 1))
        }
        return always(current.copy(steps = current.steps + 1))
    }

    override fun reward(current: GridWorldState, action: GridWorldAction, newState: GridWorldState): Distribution<Double> {
        if (newState.position == goal) return always(10.0)
        return always(0.0)
    }

    override fun initialState(): Distribution<GridWorldState> {
        return always(GridWorldState(Square(0, 0), 0))
    }
}


internal class ICMActorCriticTest {


    @Test
    fun test() {

        val size = 100
        val goal = Square(80, 80)
        val grid = (0..size).flatMap { x -> (0..size).filter { flip(0.8).sample(Random) }.map { y -> Square(x, y) } }.let { Grid(it) }
        val factory = GridWorldState.factoryForGrid(grid)
        val mdp = GridWorld(grid, goal, 100000)
        val actionRepeatingFactory = ActionRepeatingFactory(factory, mdp.allPossibleActions().toList())
        val icm = CountBasedICMModule<GridWorldState, GridWorldAction>(LinearStateValueFunction(factory, 1.0)) { 1.0 / (it + 1) }
        val exploration = ExplorationPolicy(mdp, icm)

        val episode = mdp.sampleEpisode(exploration, Random)
        episode.steps.forEach {
            println(it)
        }
        exitProcess(0)
        val policy = SoftmaxPolicy(actionRepeatingFactory, mdp, 10.0)

//        println(policy.action(GridWorldState(Square(0, 0), 1)).supportWithDensities())
//        policy.update(PolicyGradientTarget(GridWorldState(Square(0, 0), 1), DOWN, 2.5))
//        println(policy.action(GridWorldState(Square(0, 0), 1)).supportWithDensities())
//        policy.update(PolicyGradientTarget(GridWorldState(Square(0, 0), 1), DOWN, -1.0))
//        println(policy.action(GridWorldState(Square(0, 0), 1)).supportWithDensities())
//        policy.update(PolicyGradientTarget(GridWorldState(Square(0, 0), 1), DOWN, -1.5))
//        println(policy.action(GridWorldState(Square(0, 0), 1)).supportWithDensities())
//        policy.update(PolicyGradientTarget(GridWorldState(Square(0, 0), 1), DOWN, -1.5))
//        println(policy.action(GridWorldState(Square(0, 0), 1)).supportWithDensities())
//        policy.update(PolicyGradientTarget(GridWorldState(Square(0, 0), 1), DOWN, -1.5))
//        println(policy.action(GridWorldState(Square(0, 0), 1)).supportWithDensities())
//        policy.update(PolicyGradientTarget(GridWorldState(Square(0, 0), 1), DOWN, -1.5))
//        exitProcess(0)

        val critic = QActorCritic(policy, QFromMerged(actionRepeatingFactory, 1.0, 1.0), icm, Random, 1.0, 1, 0.5)
//        critic.train(mdp)
        val p = critic.train(mdp)
        val steps = mdp.sampleEpisode(p, Random)
        println(steps.steps.size)
    }

    @Test
    fun testGenerator() {
        val size = 100
        val goal = Square(80, 80)
        val grid = (0..size).flatMap { x -> (0..size).map { y -> Square(x, y) } }.let { Grid(it) }
        val factory = GridWorldState.factoryForGrid(grid)
        val mdp = GridWorld(grid, goal, 100000)
        val actionRepeatingFactory = ActionRepeatingFactory(factory, mdp.allPossibleActions().toList())
        val icm = CountBasedICMModule<GridWorldState, GridWorldAction>(LinearStateValueFunction(factory, 1.0)) { 1.0 / (it + 1) }
        val qFunction = QFromMerged(actionRepeatingFactory, 1.0)
        val exploreFun = QFromMerged(actionRepeatingFactory, 1.0)
        val alg = ExploreAndConnect(icm, Random, qFunction, exploreFun, 0.999f, 0.999f, 3, 100, MCSampleGreedyPolicy(qFunction, mdp, 50, 0.99f, 10, Random))
        val policy = alg.train(mdp)
        println(mdp.sampleEpisode(policy, Random).steps.size)

    }
}