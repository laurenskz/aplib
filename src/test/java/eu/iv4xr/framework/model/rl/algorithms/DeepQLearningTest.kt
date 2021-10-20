package eu.iv4xr.framework.model.rl.algorithms

import burlap.behavior.valuefunction.ValueFunction
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.algorithms.DeepQTestAction.*
import eu.iv4xr.framework.model.rl.approximation.*
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassHashableState
import eu.iv4xr.framework.model.rl.sampleWithStepSize
import eu.iv4xr.framework.model.rl.valuefunctions.StateValueFunction2
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableValuefunction
import eu.iv4xr.framework.model.rl.valuefunctions.ValueTable
import eu.iv4xr.framework.model.utils.DeterministicRandom
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.junit.Test
import java.time.Duration
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random


fun Float.clamp(range: ClosedFloatingPointRange<Float>) = max(min(this, range.endInclusive), range.start)
data class DeepQTestState(val x: Float, val y: Float) : DataClassHashableState() {
    companion object {
        val factory = CompositeFeature<DeepQTestState>(listOf(
                FloatFeature.from { it.x },
                FloatFeature.from { it.y },
        ))
    }

    fun clamped(xBounds: ClosedFloatingPointRange<Float>, yBounds: ClosedFloatingPointRange<Float>) = DeepQTestState(x.clamp(xBounds), y.clamp(yBounds))

    operator fun plus(other: DeepQTestState) = copy(x + other.x, y + other.y)
}

enum class DeepQTestAction : Identifiable {
    UP, DOWN, LEFT, RIGHT;

    companion object {
        val factory = OneHot(values().toList())
    }
}

class DeepQTestMDP(val targetX: Float, val targetY: Float, val delta: Float) : MDP<DeepQTestState, DeepQTestAction> {

    companion object {
        val bounds = (-10.0..10.0) sampleWithStepSize 0.1
        val stateDist = bounds.chain { x ->
            bounds.map { y ->
                DeepQTestState(x.toFloat(), y.toFloat())
            }
        }
    }

    override fun possibleStates(): Sequence<DeepQTestState> {
        return sequenceOf()
    }

    override fun allPossibleActions(): Sequence<DeepQTestAction> {
        return values().asSequence()
    }

    override fun possibleActions(state: DeepQTestState): Sequence<DeepQTestAction> {
        return allPossibleActions()
    }

    override fun isTerminal(state: DeepQTestState): Boolean {
        return abs(targetX - state.x) < delta && abs(targetY - state.y) < delta
    }

    private fun offset(action: DeepQTestAction) = when (action) {
        UP -> DeepQTestState(0f, -0.1f)
        DOWN -> DeepQTestState(0f, 0.1f)
        LEFT -> DeepQTestState(-0.1f, 0f)
        RIGHT -> DeepQTestState(0.1f, 0f)
    }

    override fun transition(current: DeepQTestState, action: DeepQTestAction): Distribution<DeepQTestState> {
        return always((current + offset(action)).clamped((-10f..10f), (-10f..10f)))
    }

    override fun reward(current: DeepQTestState, action: DeepQTestAction, newState: DeepQTestState): Distribution<Double> {
        if (isTerminal(newState)) return always(1.0)
        return always(0.0)
    }

    override fun initialState(): Distribution<DeepQTestState> {
        return always(DeepQTestState(0f, 0f))
    }
}

class DeepQTestModel : ModelDescription {
    override fun create(inputSize: Int, outputSize: Int): GraphTrainableModel {
        val model = Sequential.of(
                Input(inputSize.toLong()),
                Dense(12, activation = Activations.Relu,
                        kernelInitializer = HeNormal()),
                Dense(
                        outputSize,
                        activation = Activations.Linear,
                        kernelInitializer = HeNormal()
                ))
        model.compile(Adam(learningRate = 0.01f), Losses.HUBER, Metrics.MAE)
        model.init()
        return model
    }
}

internal class DeepQLearningTest {

    @Test
    fun test() {
        var valuefunction: TrainableValuefunction<DeepQTestState> = ValueTable(0.1f)
        valuefunction = StateValueFunction2(DeepQTestModel(), DeepQTestState.factory)
        val dql = Bab<DeepQTestState, DeepQTestAction>(valuefunction, 0.99f, DeterministicRandom(), 1000)
        val mdp = DeepQTestMDP(2f, 2f, 0.2f)
        val alg = dql.train(mdp)
        println(mdp.sampleEpisode(alg, Random, DeepQTestState(0.1f, 0.05f)).steps.size)
//        val policy = dql.train(mdp)
//        val randPolicy = RandomPolicy(mdp)
//        println("Done training the model")
//        val best = mdp.sampleEpisode(policy, DeterministicRandom())
//        val rand = mdp.sampleExecution(randPolicy, DeterministicRandom())
//        println(best.size)
//        println(rand.size)

    }

    @Test
    fun testPriorityBased() {
        var valuefunction: TrainableValuefunction<DeepQTestState> = ValueTable(0.1f)
        valuefunction = StateValueFunction2(DeepQTestModel(), DeepQTestState.factory)
        val experienceGenerator = PolicyBasedExperienceGenerator<DeepQTestState, DeepQTestAction>(100000, DeterministicRandom(), 0.99f, Duration.ofSeconds(1)) { RandomPolicy(it) }
        val dql = PriorityBased(valuefunction,
                0.99f, experienceGenerator, 0.0001, 40)
        val mdp = DeepQTestMDP(2f, 2f, 0.2f)
        val alg = dql.train(mdp)
        println(mdp.sampleEpisode(alg, Random, DeepQTestState(0.1f, 0.05f)).steps.size)
    }

    @Test
    fun testFrontier() {
        val f = FrontierBasedExperience<DeepQTestState, DeepQTestAction>()
        f.generate(DeepQTestMDP(2f,2f,0.2f))
    }
}