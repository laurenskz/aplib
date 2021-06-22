package eu.iv4xr.framework.model

import eu.iv4xr.framework.model.Test.greet
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class TestTest {
    @Test
    fun test() {
        assertEquals("Hi!", greet)
    }
}