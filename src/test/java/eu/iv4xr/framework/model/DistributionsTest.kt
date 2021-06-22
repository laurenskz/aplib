package eu.iv4xr.framework.model

import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import kotlin.random.Random

internal class DistributionsTest {

    @Test
    fun uniform() {
        val uniform = Distributions.uniform((0 until 4))
        assertEquals(0.25, uniform.score(0))
        assertEquals(0.25, uniform.score(1))
        assertEquals(0.25, uniform.score(2))
        assertEquals(0.25, uniform.score(3))
    }

    @Test
    fun discrete() {
        val weather = Distributions.discrete("Sun" to 0.3, "Rain" to 0.3, "Snow" to 0.4)
        val sample = weather.sample(10000, Random(123))
        assertEquals(2981, sample["Sun"])
        assertEquals(4030, sample["Snow"])
        assertEquals(2989, sample["Rain"])
    }

    @Test
    fun bernoulli() {
        val rain = Distributions.bernoulli(0.75) //75 percent chance of rain
        val sample = rain.sample(1000, Random(123))
        assertEquals(756, sample[true])
        assertEquals(244, sample[false])
    }
}