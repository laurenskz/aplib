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
    fun if_() {
        val rain = Distributions.bernoulli(0.3)
        val temperature = if_(rain, Distributions.constant(7), Distributions.constant(9))
        print(temperature.supportWithDensities())
    }
}