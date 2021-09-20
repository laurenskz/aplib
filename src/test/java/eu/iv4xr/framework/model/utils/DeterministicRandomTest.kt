package eu.iv4xr.framework.model.utils

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.security.SecureRandom

internal class DeterministicRandomTest {
    @Test
    fun testDouble() {
        val deterministicRandom = DeterministicRandom()
        assertTrue((0..100000).map {
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

    @Test
    fun testInt() {
        val deterministicRandom = DeterministicRandom()
        println(deterministicRandom.nextInt(50))
        println(deterministicRandom.nextInt(50))
        println(deterministicRandom.nextInt(50))
        println(deterministicRandom.nextInt(50))
        println(deterministicRandom.nextInt(50))
        println(deterministicRandom.nextInt(50))
        println(deterministicRandom.nextInt(50))
        println(deterministicRandom.nextInt(50))
        println(deterministicRandom.nextInt(50))
        println(deterministicRandom.nextInt(50))
        println(deterministicRandom.nextInt(50))
    }
}