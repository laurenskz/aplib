package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.model.rl.RLAgentTest.TestAction.LEFT
import eu.iv4xr.framework.model.rl.RLAgentTest.TestModelState.ONE
import eu.iv4xr.framework.model.rl.RLAgentTest.TestModelState.PIT
import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import kotlin.math.PI

internal class RLMDPTest {

    val mdp: RLMDP<RLAgentTest.TestModelState, RLAgentTest.TestAction> = RLMDP(RLAgentTest.TestModel(), listOf(
            basicGoal(1.0) { it == 3 }
    ))

    @Test
    fun possibleStates() {
        assertEquals(8, mdp.possibleStates().count())
    }

    @Test
    fun possibleActions() {
        for (state in mdp.possibleStates()) {
            assertEquals(RLAgentTest.TestModel().possibleActions(state.state).toList(), mdp.possibleActions(state).toList())
        }
    }

    @Test
    fun isTerminal() {
        for (state in mdp.possibleStates()) {
            assertEquals(RLAgentTest.TestModel().isTerminal(state.state), mdp.isTerminal(state))
        }
    }

    @Test
    fun transition() {
        val newStates = mdp.transition(StateWithGoalProgress(listOf(false), ONE), LEFT)
//        Goal is complete if we end up in the pit
        assertEquals(0.1, newStates.score(StateWithGoalProgress(listOf(true), PIT)))
        assertEquals(0.9, newStates.score(StateWithGoalProgress(listOf(false), ONE)))
    }

    @Test
    fun reward() {
        val newStates = mdp.reward(StateWithGoalProgress(listOf(false), ONE), LEFT, StateWithGoalProgress(listOf(true), PIT))
        assertEquals(1.0, newStates.expectedValue())
        assertEquals(0.0, mdp.reward(StateWithGoalProgress(listOf(false), ONE), LEFT, StateWithGoalProgress(listOf(false), ONE)).expectedValue())
    }
}