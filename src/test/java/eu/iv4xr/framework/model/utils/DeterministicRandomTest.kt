package eu.iv4xr.framework.model.utils

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class DeterministicRandomTest {
    @Test
    fun testDouble() {
        val deterministicRandom = DeterministicRandom()
        assertTrue((0..1000).map {
            deterministicRandom.nextDouble()
        }.all {
            it in 0.0..1.0
        })
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
        println(deterministicRandom.nextDouble())
    }
}