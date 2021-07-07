package eu.iv4xr.framework.model.rl.burlapadaptors

import eu.iv4xr.framework.model.rl.BurlapState
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class BurlapUtilsKtTest {

    data class TestRefl(val x: String, val z: Int, val nested: Nested) : ImmutableReflectionBasedState

    data class Nested(val a: String, val b: String, val c: Nested2) : ImmutableReflectionBasedState

    data class Nested2(val greet: String) : ImmutableReflectionBasedState

    @Test
    fun testReflectionBasedState() {
        val test = TestRefl("hi!", 42, Nested("hi", "Bye", Nested2("Nested 2")))
        test.variableKeys().forEach {
            println(test.get(it))
        }
    }
}