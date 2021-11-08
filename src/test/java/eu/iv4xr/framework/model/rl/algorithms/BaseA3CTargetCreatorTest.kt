package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.algorithms.TestState.*
import eu.iv4xr.framework.model.rl.approximation.MergedFeatureFactory
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms.Episode
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms.SARS
import eu.iv4xr.framework.model.rl.policies.LinearStateValueFunction
import eu.iv4xr.framework.model.rl.policies.SoftmaxPolicy
import eu.iv4xr.framework.model.rl.policies.TFPolicy
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import eu.iv4xr.framework.model.rl.valuefunctions.Valuefunction
import org.junit.Test
import org.junit.jupiter.api.Assertions.*
import org.tensorflow.op.core.OneHot
import kotlin.random.Random


private enum class TestState : Identifiable {
    ONE, TWO, THREE, TERMINAL;

    companion object {
        val factory = eu.iv4xr.framework.model.rl.approximation.OneHot(values().toList())
    }
}

private enum class TestAction : Identifiable {
    UP, DOWN, LEFT;

    companion object {
        val factory = eu.iv4xr.framework.model.rl.approximation.OneHot(values().toList())
    }
}

private class TestMDP : MDP<TestState, TestAction> {
    override fun possibleStates(): Sequence<TestState> {
        TODO("Not yet implemented")
    }

    override fun allPossibleActions(): Sequence<TestAction> {
        return TestAction.values().asSequence()
    }

    override fun possibleActions(state: TestState): Sequence<TestAction> {
        return allPossibleActions()
    }

    override fun isTerminal(state: TestState): Boolean {
        return state == TERMINAL
    }

    override fun transition(current: TestState, action: TestAction): Distribution<TestState> {
        TODO("Not yet implemented")
    }

    override fun reward(current: TestState, action: TestAction, newState: TestState): Distribution<Double> {
        TODO("Not yet implemented")
    }

    override fun initialState(): Distribution<TestState> {
        TODO("Not yet implemented")
    }
}

private class TestValueFunction : Valuefunction<TestState> {
    override fun value(state: TestState) = when (state) {
        ONE -> 1f
        TWO -> 2f
        THREE -> 3f
        TERMINAL -> -100000f
    }
}

internal class BaseA3CTargetCreatorTest {
    @Test
    fun test() {
        val episodes = listOf(
                Episode(steps = listOf(
                        SARS(ONE, TestAction.UP, TWO, 0.0, 1.0), // Value is 2.5 so advantage is 1.5
                        SARS(TWO, TestAction.UP, THREE, 0.0, 1.0), // value is 5 so advantage is 3
                        SARS(THREE, TestAction.UP, TERMINAL, 10.0, 1.0), //Value is 10 so advantage is 7
                        SARS(TERMINAL, TestAction.UP, TERMINAL, 0.0, 1.0), // For completeness
                )),
                Episode(steps = listOf(
                        SARS(ONE, TestAction.LEFT, ONE, 0.0, 1.0),// value is 0.25
                        SARS(ONE, TestAction.LEFT, ONE, 0.0, 1.0), // value is 0.5
                        SARS(ONE, TestAction.UP, TWO, 0.0, 1.0), //Resulting state is 2 so value is 1
                ))
        )
        val targetCreator = BaseA3CTargetCreator(TestValueFunction(), 0.5f, TestMDP())
        val targets = targetCreator.createTargets(episodes)
        assertEquals(targets.first()[0].target.target, 1.5f)
        assertEquals(targets.first()[1].target.target, 3.0f)
        assertEquals(targets.first()[2].target.target, 7.0f)
        assertEquals(targets.last()[0].target.target, -0.75f)
        assertEquals(targets.last()[1].target.target, -0.5f)
        assertEquals(targets.last()[2].target.target, 0.0f)
        targets.forEach {
            println("Episode:")
            it.forEach(::println)
        }
    }

    @Test
    fun testLoops() {
        val value = LinearStateValueFunction(TestState.factory, 0.1)
        val size = 80
        val policy = TFPolicy(TestState.factory, TestMDP(), 0.1f, batchSize = size)
//        val policy = SoftmaxPolicy(MergedFeatureFactory(TestState.factory, TestAction.factory), TestMDP(), 10.0)
        val ePolicy = EGreedyPolicy(0.1, TestMDP(), policy)
        val targetCreator = BaseA3CTargetCreator(value, 0.9f, TestMDP())
        val actions = mutableMapOf<TestAction, Int>(TestAction.UP to 0, TestAction.LEFT to 0, TestAction.DOWN to 0)
        (1 until 100000).forEach {
            val action = ePolicy.action(ONE)
            val episodes = listOf(
                    Episode(steps = List(size) {
                        val a = action.sample(Random)
                        SARS(ONE, a, ONE, 5.0 / (actions.computeIfPresent(a) { _, count -> count + 1 }?.toDouble()!!), 1.0)
                    } // Value is 2.5 so advantage is 1.5
                    ))
            println(it)
            println(policy.action(ONE).supportWithDensities())
            val targets = targetCreator.createTargets(episodes)
            value.train(targets.flatMap { it.map { it.target } })
            policy.updateAll(targets.flatMap { it.map { it.policyGradientTarget } })
            println(policy.action(ONE).supportWithDensities())


        }
    }
}