package eu.iv4xr.framework.model.rl

import nl.uu.cs.aplib.AplibEDSL
import nl.uu.cs.aplib.AplibEDSL.goal
import nl.uu.cs.aplib.mainConcepts.GoalStructure
import nl.uu.cs.aplib.mainConcepts.GoalStructure.GoalsCombinator.SEQ
import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*

internal class RLMDPTest {

    @Test
    fun allPossibleGoalStates() {
        assertEquals(1, allPossibleGoalStates(0).count())
        assertEquals(2, allPossibleGoalStates(1).count())
        assertEquals(8, allPossibleGoalStates(3).count())
        assertEquals(16, allPossibleGoalStates(4).count())
        assertEquals(32, allPossibleGoalStates(5).count())
        assertTrue(allPossibleGoalStates(4).contains(listOf(false, true, false, false)))
    }

    @Test
    fun convert() {
        val twoNodes = GoalStructure(SEQ, goal("leaf1").lift(), goal("leaf2").lift())
        val topGoal = GoalStructure(SEQ, goal("leaf0").lift(), twoNodes)
        val names = convert(topGoal) {
            it.goal.name
        }
        assertEquals(3, names.size)
        assertTrue(names.contains("leaf0"))
        assertTrue(names.contains("leaf1"))
        assertTrue(names.contains("leaf2"))
    }

    @Test
    fun updateGoalStatus() {
        val goal = goal("leaf1")
        val goal1 = goal("leaf2")
        val goal2 = goal("leaf0")
        val twoNodes = GoalStructure(SEQ, goal.lift(), goal1.lift())
        val topGoal = GoalStructure(SEQ, goal2.lift(), twoNodes)
        updateGoalStatus(topGoal)
        assertFalse(topGoal.status.success())
        goal.status.setToSuccess()
        updateGoalStatus(topGoal)
        assertFalse(topGoal.status.success())
        goal1.status.setToSuccess()
        goal2.status.setToSuccess()
        updateGoalStatus(topGoal)
        assertTrue(topGoal.status.success())
    }
}