package eu.iv4xr.framework.model.distribution

import eu.iv4xr.framework.utils.cons
import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import kotlin.random.Random

internal class DistributionDSLTest {

    @Test
    fun sample() {
        val weather = Distributions.discrete("Sun" to 0.3, "Rain" to 0.3, "Snow" to 0.4)
        val sample = weather.sample(10000, Random(123))
        assertEquals(2981, sample["Sun"])
        assertEquals(4030, sample["Snow"])
        assertEquals(2989, sample["Rain"])
    }

    @Test
    fun fold() {
        val dists = listOf(
            Distributions.uniform(1, 2),
            Distributions.uniform(3, 4),
            Distributions.uniform(5, 6),
        )
        val final = dists.foldD(always(listOf<Int>())) { l, i ->
            l + i
        }
        println(final.supportWithDensities())
    }

    @Test
    fun softmax() {
        val bab = Distributions.softmax(
            mapOf(
                "Sun" to 1.0,
                "Rain" to 2.0,
            )
        )
        println(bab.supportWithDensities())
    }

    @Test
    fun ifd() {
        val rain = Distributions.bernoulli(0.3)
        val temperature = ifd(rain, Distributions.deterministic(7), Distributions.deterministic(9))
        assertEquals(0.3, temperature.score(7))
    }

    @Test
    fun if_() {
        val rain = Distributions.bernoulli(0.3)
        val temperature = if_(rain, 7, 9)
        assertEquals(0.3, temperature.score(7))
    }

    @Test
    fun if_2() {
        val rain = flip(0.3)
        val temperature = if_(rain) {
            7
        }.else_ {
            19
        }
        assertEquals(0.3, temperature.score(7))
        assertEquals(0.7, temperature.score(19))
    }

    @Test
    fun ifd2() {
        val rain = flip(0.3)
        val state = ifd(rain) {
            Distributions.uniform("Sad", "Cold")
        }.elsed {
            Distributions.uniform("Sun", "Ice")
        }
        assertEquals(0.15, state.score("Sad"))
        assertEquals(0.15, state.score("Cold"))
        assertEquals(0.35, state.score("Sun"))
        assertEquals(0.35, state.score("Ice"))
    }

    @Test
    fun plus() {
        val grade1 = Distributions.discrete(5.0 to 0.1, 9.0 to 0.9)
        println(grade1.map { 0 }.supportWithDensities())
    }

}