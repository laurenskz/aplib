package eu.iv4xr.framework.model.rl.burlapadaptors

import eu.iv4xr.framework.model.rl.BurlapState
import eu.iv4xr.framework.model.rl.StateWithGoalProgress
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class BurlapUtilsKtTest {

    data class TestRefl(val x: String, val z: Int, val nested: Nested) : ImmutableReflectionBasedState

    data class Nested(val a: String, val b: String, val c: Nested2) : ImmutableReflectionBasedState

    data class Nested2(val greet: String) : ImmutableReflectionBasedState

    @Test
    fun testReflectionBasedState() {
        val test = TestRefl("hi!", 42, Nested("hi", "Bye", Nested2("Nested 2")))
        val values = test.variableKeys().map {
            test.get(it)
        }.toList()
        println(values)
        assertEquals("hi", values[0])
        assertEquals("Bye", values[1])
        assertEquals("Nested 2", values[2])
        assertEquals("hi!", values[3])
        assertEquals(42, values[4])
    }

    @Test
    fun testWithGoalProgress() {
        val stateWithGoalProgress = StateWithGoalProgress(listOf(true), Nested2("Hi!"))
        val values = stateWithGoalProgress.variableKeys().map {
            stateWithGoalProgress.get(it)
        }
        println(values)
    }
}