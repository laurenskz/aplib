package eu.iv4xr.framework.model

import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import kotlin.random.Random

internal class DiscreteDistributionTest {

    enum class Coin {
        HEADS, TAILS
    }

    enum class Color {
        RED, WHITE, GREEN, BLUE
    }

    val fairCoin = DiscreteDistribution(mapOf(Coin.HEADS to 0.5, Coin.TAILS to 0.5))
    val uniformColors = DiscreteDistribution(mapOf(Color.RED to 0.25, Color.WHITE to 0.25, Color.GREEN to 0.25, Color.BLUE to 0.25))

    @Test
    fun score() {
        assertEquals(0.5, fairCoin.score(Coin.TAILS))
    }

    @Test
    fun testScore() {
        assertEquals(0.5, uniformColors.score { listOf(Color.RED, Color.BLUE).contains(it) })
    }

    @Test
    fun filter() {
        val filtered = uniformColors.filter { it != Color.BLUE }
        assertEquals(0.333333, filtered.score(Color.WHITE), 0.001)
    }

    @Test
    fun sample() {
        val random = Random(123456)
        val counts = (0 until 1000).map {
            fairCoin.sample(random)
        }.groupBy { it }
                .mapValues { it.value.count() }
        assertEquals(511, counts[Coin.HEADS])
        assertEquals(489, counts[Coin.TAILS])
    }

    @Test
    fun support() {
    }

    @Test
    fun chain() {
    }

    @Test
    fun map() {
    }

    @Test
    fun getValues() {
    }
}