package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.algorithms.PolicyGradientTarget
import eu.iv4xr.framework.model.rl.approximation.*
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import nl.uu.cs.aplib.mainConcepts.Action
import org.junit.Test
import org.junit.jupiter.api.Assertions.*


data class StateTest(val x: Double, val y: Double, val z: Double) : Identifiable {
    companion object {
        val factory = CompositeFeature<StateTest>(listOf(
                DoubleFeature.from { it.x },
                DoubleFeature.from { it.y },
                DoubleFeature.from { it.z },
        ))
    }
}

enum class ActionDiscrete : Identifiable {
    ONE, TWO, THREE;

    companion object {
        val factory = OneHot(values().toList())
    }
}

data class ActionTest(val a: Double) : Identifiable {
    companion object {
        val factory: FeatureVectorFactory<ActionTest> = DoubleFeature.from { it.a }
    }
}

class ActionMDP<A : Identifiable>(val acs: List<A>) : MDP<StateTest, A> {
    override fun possibleStates(): Sequence<StateTest> {
        TODO("Not yet implemented")
    }

    override fun allPossibleActions(): Sequence<A> {
        return acs.asSequence()
    }

    override fun possibleActions(state: StateTest): Sequence<A> {
        return acs.asSequence()
    }

    override fun isTerminal(state: StateTest): Boolean {
        TODO("Not yet implemented")
    }

    override fun transition(current: StateTest, action: A): Distribution<StateTest> {
        TODO("Not yet implemented")
    }

    override fun reward(current: StateTest, action: A, newState: StateTest): Distribution<Double> {
        TODO("Not yet implemented")
    }

    override fun initialState(): Distribution<StateTest> {
        TODO("Not yet implemented")
    }
}


internal class LinearStateValueFunctionTest {


    @Test
    fun testSoftmaxPolicy() {
        val policy = SoftmaxPolicy(MergedFeatureFactory(StateTest.factory, ActionTest.factory), ActionMDP(listOf(
                ActionTest(0.0),
                ActionTest(1.0),
                ActionTest(2.0),
                ActionTest(3.0),
        )), 0.1)
        val state = StateTest(1.0, 1.0, 1.0)
        println(policy.action(state).supportWithDensities())
        policy.update(PolicyGradientTarget(state, ActionTest(0.0), 1000.0))
        println(policy.action(state).supportWithDensities())
    }

    @Test
    fun testTFPolicy() {
        val mdp = ActionMDP(listOf(
                ActionTest(0.0),
                ActionTest(1.0),
                ActionTest(2.0),
                ActionTest(3.0),
        ))
        val policy = TFPolicy(StateTest.factory, mdp, 0.1f)
        val state = StateTest(1.0, 1.0, 1.0)
        println(policy.action(state).supportWithDensities())
        println(policy.action(state).supportWithDensities())
        println(policy.action(state).supportWithDensities())
        policy.update(PolicyGradientTarget(state, ActionTest(0.0), 0.0))
        println(policy.action(state).supportWithDensities())
        policy.update(PolicyGradientTarget(state, ActionTest(1.0), 0.0))
        println(policy.action(state).supportWithDensities())
    }


    @Test
    fun testLinear() {
        val function = LinearStateValueFunction(StateTest.factory, 0.01)
        val state = StateTest(1.0, 1.0, 1.0)
        repeat(10000) {
            function.train(Target(state, -1f))
        }
        val x = function.value(state)
        println(x)
    }
}