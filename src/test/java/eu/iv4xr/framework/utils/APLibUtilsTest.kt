package eu.iv4xr.framework.utils

import nl.uu.cs.aplib.AplibEDSL
import nl.uu.cs.aplib.mainConcepts.GoalStructure
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class APLibUtilsTest {
    @Test
    fun allPossibleGoalStatesTest() {
        assertEquals(1, allPossibleGoalStates(0).count())
        assertEquals(2, allPossibleGoalStates(1).count())
        assertEquals(8, allPossibleGoalStates(3).count())
        assertEquals(16, allPossibleGoalStates(4).count())
        assertEquals(32, allPossibleGoalStates(5).count())
        assertTrue(allPossibleGoalStates(4).contains(listOf(false, true, false, false)))
    }

    @Test
    fun convertTest() {
        val twoNodes = GoalStructure(GoalStructure.GoalsCombinator.SEQ, AplibEDSL.goal("leaf1").lift(), AplibEDSL.goal("leaf2").lift())
        val topGoal = GoalStructure(GoalStructure.GoalsCombinator.SEQ, AplibEDSL.goal("leaf0").lift(), twoNodes)
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
        val goal = AplibEDSL.goal("leaf1")
        val goal1 = AplibEDSL.goal("leaf2")
        val goal2 = AplibEDSL.goal("leaf0")
        val twoNodes = GoalStructure(GoalStructure.GoalsCombinator.SEQ, goal.lift(), goal1.lift())
        val topGoal = GoalStructure(GoalStructure.GoalsCombinator.SEQ, goal2.lift(), twoNodes)
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