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

    fun <T> randList(dist: Distribution<T>, count: Int): Distribution<List<T>> {
        if (count == 0) return always(listOf<T>())
        return dist.chain { new ->
            randList(dist, count - 1).map { listOf(new) + it }
        }
    }


    /**
     * This distribution is very large, it contains 2 billion members, therefore methods like scoring won't work
     * But sampling is still efficient because we do it lazily
     */
    @Test
    fun sampleLargeDist() {
        randList(Distributions.uniform(0 until 2), 32).sample(10000, Random)

    }
}
