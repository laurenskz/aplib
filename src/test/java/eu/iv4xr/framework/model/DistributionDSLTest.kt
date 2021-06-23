package eu.iv4xr.framework.model

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
    fun ifd() {
        val rain = Distributions.bernoulli(0.3)
        val temperature = ifd(rain, Distributions.constant(7), Distributions.constant(9))
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
        val grade1 = Distributions.uniform(5.0, 9.0)
        val grade2 = Distributions.uniform(12.0, 21.0)
        val sum = grade1 + grade2
        print(sum.supportWithDensities())
    }

}