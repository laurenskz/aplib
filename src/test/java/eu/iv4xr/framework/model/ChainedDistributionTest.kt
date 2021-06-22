package eu.iv4xr.framework.model

import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import java.lang.IllegalArgumentException
import kotlin.random.Random

internal class ChainedDistributionTest {

    val fairCoin = DiscreteDistribution(mapOf(DiscreteDistributionTest.Coin.HEADS to 0.5, DiscreteDistributionTest.Coin.TAILS to 0.5))

    private val weather = ChainedDistribution(fairCoin) {
        // Flip a coin
        when (it) {
            DiscreteDistributionTest.Coin.HEADS -> DiscreteDistribution(mapOf("Cloudy" to 0.3, "Sun" to 0.7))
            DiscreteDistributionTest.Coin.TAILS -> DiscreteDistribution(mapOf("Rain" to 0.3, "Cloudy" to 0.7))
        }
    }

    private val temperature = weather.chain {
        when (it) {
            "Sun" -> DiscreteDistribution(mapOf(17.0 to 0.5, 18.0 to 0.5))
            "Cloudy" -> DiscreteDistribution(mapOf(15.0 to 0.5, 16.0 to 0.5))
            "Rain" -> DiscreteDistribution(mapOf(10.0 to 0.5, 11.0 to 0.5))
            else -> DiscreteDistribution(mapOf(-1.0 to 1.0))
        }
    }

    @Test
    fun score() {
        assertEquals(0.5, weather.score("Cloudy"))
        assertEquals(0.35, weather.score("Sun"))
        assertEquals(0.15, weather.score("Rain"))
    }

    @Test
    fun testScore() {
        assertEquals(1.0, temperature.score { it >= 10 })
        assertEquals(0.925, temperature.score { it >= 11 }, 0.01)
        assertEquals(0.85, temperature.score { it >= 12 }, 0.01)
    }

    @Test
    fun filter() {
        val newWeather = weather.filter { it != "Cloudy" }
        assertEquals(0.7, newWeather.score("Sun"))
        assertEquals(0.3, newWeather.score("Rain"))
    }

    @Test
    fun sample() {
        val counts = weather.sample(100000, Random(123456))
        assertEquals(34889, counts["Sun"])
        assertEquals(49992, counts["Cloudy"])
        assertEquals(15119, counts["Rain"])
    }

    @Test
    fun support() {
        val support = weather.support()
        assertEquals(3, support.count())
        assertEquals(setOf("Sun", "Cloudy", "Rain"), support.toSet())
    }

    @Test
    fun chain() {
        val season = weather.chain {
            when (it) {
                "Sun" -> DiscreteDistribution(mapOf("Winter" to 0.1, "Summer" to 0.9))
                "Cloudy" -> DiscreteDistribution(mapOf("Winter" to 0.5, "Summer" to 0.5))
                "Rain" -> DiscreteDistribution(mapOf("Winter" to 0.9, "Summer" to 0.1))
                else -> throw IllegalArgumentException()
            }
        }
        assertEquals(0.42, season.score("Winter"), 0.01)
        assertEquals(0.58, season.score("Summer"), 0.01)

    }

    @Test
    fun map() {
        val emotion = weather.map {
            when (it) {
                "Sun" -> ":)"
                "Cloudy" -> ":)"
                else -> ":("
            }
        }
        assertEquals(0.85, emotion.score(":)"))
        assertEquals(0.15, emotion.score(":("))
    }
}