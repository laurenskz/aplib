package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.distribution.*
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.algorithms.GridWorldAction.*
import eu.iv4xr.framework.model.rl.approximation.*
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassHashableState
import eu.iv4xr.framework.model.rl.policies.*
import eu.iv4xr.framework.model.rl.valuefunctions.QTarget
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableQFunction
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableValuefunction
import org.junit.Test
import java.util.*
import kotlin.math.max
import kotlin.random.Random
import kotlin.system.exitProcess


enum class GridWorldAction : Identifiable {
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


class ExplorationPolicy<S : Identifiable, A : Identifiable>(val mdp: MDP<S, A>, val valueFunction: TrainableValuefunction<S>, val icmModule: ICMModule<S, A>, val gamma: Float) : Policy<S, A> {
    override fun action(state: S): Distribution<A> {
        println(state)
//        we would like to explore from the most promising state, this is the state from which we don't know what will happen
        val intrinsicRewards = mdp.possibleActions(state).map { a ->
            val transition = mdp.transition(state, a)
            a to transition.expectedValue { sp ->
                val reward = mdp.reward(state, a, sp).expectedValue()
                icmModule.train(listOf(BurlapAlgorithms.SARS(state, a, sp, reward, transition.score(sp)))).first()
            }
        }
        return always(intrinsicRewards.maxByOrNull { it.second }!!.first)
//        return Distributions.softmax(intrinsicRewards.toMap())
    }
}

class QLearning<S : Identifiable, A : Identifiable>(val qFunction: TrainableQFunction<S, A>, val gamma: Float, val mdp: MDP<S, A>, val random: Random) {

    fun train(episode: BurlapAlgorithms.Episode<S, A>) {
        episode.steps.reversed().forEach {
            train(it)
        }
    }

    private fun train(it: BurlapAlgorithms.SARS<S, A>) {
        train(setOf(it), 1, 1)
    }

    fun trainEPolicy(episodes: Int) {
        val epolicy = EGreedyPolicy(0.03, mdp, GreedyPolicy(qFunction, mdp))

        repeat(episodes) {
            println(it)
            var state = mdp.initialState().sample(random)
            while (!mdp.isTerminal(state)) {
                val sars = mdp.sampleSARS(epolicy, state, random)
                train(sars)
                state = sars.sp

            }
        }
    }

    fun train(experience: Set<BurlapAlgorithms.SARS<S, A>>, batchSize: Int, batches: Int) {
        repeat(batches) {
            val batch = experience.takeRandom(batchSize, random)
            val nextStates = batch.map { if (mdp.isTerminal(it.sp)) 0f else mdp.possibleActions(it.sp).maxOf { a -> qFunction.qValue(it.sp, a) } }
            val targets = batch.mapIndexed { i, t -> QTarget(t.s, t.a, (t.r + gamma * nextStates[i]).toFloat()) }
            qFunction.train(targets)
        }
    }
}

data class RunningAverage(var sum: Double, var count: Int) {
    val avg
        get() = sum / max(count.toDouble(), 0.001)

    fun add(delta: Double) {
        sum += delta
        count++
    }

    override fun toString(): String {
        return "RunningAverage(avg=$avg)"
    }

}

class ExplorationNode<S : Identifiable, A : Identifiable>(val state: S) : Comparable<ExplorationNode<S, A>> {
    var predecessor: ExplorationNode<S, A>? = null
    var sars: BurlapAlgorithms.SARS<S, A>? = null
    var sOut = 2.0
    val comparator = compareBy<ExplorationNode<S, A>> { it.priority }.reversed()

    val priority: Double
        get() = sOut

    override fun compareTo(other: ExplorationNode<S, A>): Int {
        return comparator.compare(this, other)
    }

    override fun toString(): String {
        return "ExplorationNode(state=$state, priority=$priority)"
    }


}

class ExplorationWithQueue<S : Identifiable, A : Identifiable>(
        val mdp: MDP<S, A>,
        val estimateUnfamiliarity: TrainableValuefunction<S>,
        val icmModule: ICMModule<S, A>,
        val random: Random) {
    val queue = PriorityQueue<ExplorationNode<S, A>>()
    val steps = mutableSetOf<BurlapAlgorithms.SARS<S, A>>()
    val ePolicy = RandomPolicy(mdp)
    var goalEpisode: BurlapAlgorithms.Episode<S, A>? = null

    init {

        mdp.initialState().support().forEach {
            queue.add(ExplorationNode(it))
        }
    }

    fun explore(): BurlapAlgorithms.Episode<S, A> {
        var i = 0
        while (!queue.isEmpty()) {
//            println(queue)
            val node = queue.remove()
            val state = node.state
            if (mdp.isTerminal(state)) continue
            val sars = mdp.sampleSARS(ePolicy, state, random)
            val new = ExplorationNode<S, A>(sars.sp)
            new.sars = sars
            val intrinsic = icmModule.train(listOf(sars, mdp.sampleSARS(ePolicy, sars.sp, random)))
            estimateUnfamiliarity.train(listOf(
                    Target(state, intrinsic.first().toFloat()),
                    Target(sars.sp, intrinsic.last().toFloat()))
            )
            val values = estimateUnfamiliarity.values(listOf(state, sars.sp))
            node.sOut = values.first().toDouble()
            new.sOut = values.last().toDouble()
            new.predecessor = node
            queue.add(new)
            queue.add(node)
            this.steps.add(sars)
            if (sars.r > 0) {
                return createEpisode(new)
            }
        }
        error("No goal found :(")
//        println("Succes!,${steps.size}")
    }

    fun createEpisode(node: ExplorationNode<S, A>): BurlapAlgorithms.Episode<S, A> {
        val steps = mutableListOf<BurlapAlgorithms.SARS<S, A>>()
        var current: ExplorationNode<S, A>? = node
        while (current != null) {
            current.sars?.also { steps.add(0, it) }
            current = current.predecessor
        }
        return BurlapAlgorithms.Episode(steps)
    }
}

internal class ICMActorCriticTest {


    @Test
    fun test() {

        val size = 100
        val goal = Square(80, 80)
        val grid = (0..size).flatMap { x -> (0..size).map { y -> Square(x, y) } }.let { Grid(it) }
        val factory = GridWorldState.factoryForGrid(grid)
        val mdp = GridWorld(grid, goal, 100000)
        val actionRepeatingFactory = ActionRepeatingFactory(factory, mdp.allPossibleActions().toList())
        val icm = CountBasedICMModule<GridWorldState, GridWorldAction>(LinearStateValueFunction(factory, 1.0)) { 1.0 / (it + 1) }
        val exploration = ExplorationPolicy(mdp, LinearStateValueFunction(factory, 1.0), icm, 0.99f)

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
        val lr = 1.0
        val explorer = ExplorationWithQueue(mdp, LinearStateValueFunction(factory, 1.0), icm, Random)
        val episode = explorer.explore()
        println(episode.steps.size)
        val qFunction = QFromMerged(actionRepeatingFactory, 1.0)
        val learning = QLearning(qFunction, 0.99f, mdp, Random)
        repeat(10) { learning.train(episode) }
        println(mdp.sampleEpisode(GreedyPolicy(qFunction, mdp), Random).steps.size)
        learning.trainEPolicy(100)
        println(mdp.sampleEpisode(GreedyPolicy(qFunction, mdp), Random).steps.size)

    }
}