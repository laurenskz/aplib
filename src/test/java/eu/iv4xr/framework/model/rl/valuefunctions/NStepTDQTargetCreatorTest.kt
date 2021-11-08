package eu.iv4xr.framework.model.rl.valuefunctions

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.valuefunctions.TestState.*
import org.junit.Test
import org.junit.jupiter.api.Assertions.*

private enum class TestState : Identifiable {
    ONE, TWO, THREE, FOUR, FIVE
}

private enum class TestAction : Identifiable {
    ADVANCE, REMAIN
}

private class TestQFunction : QFunction<TestState, TestAction> {
    override fun qValue(state: TestState, action: TestAction) = when (state) {
        ONE -> 0f
        TWO -> 0f
        THREE -> 0f
        FOUR -> 3f
        FIVE -> 3f
    }
}

private class TestMDP : MDP<TestState, TestAction> {
    override fun possibleStates(): Sequence<TestState> {
        return TestState.values().asSequence()
    }

    override fun allPossibleActions(): Sequence<TestAction> {
        return TestAction.values().asSequence()
    }

    override fun possibleActions(state: TestState): Sequence<TestAction> {
        return allPossibleActions()
    }

    override fun isTerminal(state: TestState): Boolean {
        TODO("Not yet implemented")
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

internal class NStepTDQTargetCreatorTest {
    @Test
    fun test() {
        val episodes = listOf(
                BurlapAlgorithms.Episode(
                        listOf(
                                // 0 + 0.9 * 1 + 0.81 * 0
                                BurlapAlgorithms.SARS(ONE, TestAction.ADVANCE, TWO, 0.0, 1.0),
                                // 1.0 + 0.9 * 1 + 0.81 * 3
                                BurlapAlgorithms.SARS(TWO, TestAction.ADVANCE, THREE, 1.0, 1.0),
                                BurlapAlgorithms.SARS(THREE, TestAction.ADVANCE, FOUR, 1.0, 1.0),
                                BurlapAlgorithms.SARS(FOUR, TestAction.REMAIN, FIVE, 1.0, 1.0),
                        )
                )
        )
        val targets = NStepTDQTargetCreator(TestQFunction(), 0.9f, 2).createTargets(
                episodes, TestMDP()
        )
        assertEquals(targets[0].target, 0.9f)
        assertEquals(targets[1].target, 4.33f)
        assertEquals(targets[2].target, 1.9f)
        assertEquals(targets[3].target, 1.0f)
        println(NStepTDQTargetCreator(TestQFunction(), 0.9f, Int.MAX_VALUE).createTargets(
                episodes, TestMDP()
        ))
    }
}