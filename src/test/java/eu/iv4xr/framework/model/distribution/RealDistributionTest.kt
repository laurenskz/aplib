package eu.iv4xr.framework.model.distribution

import eu.iv4xr.framework.model.rl.sampleWithStepSize
import eu.iv4xr.framework.model.utils.DeterministicRandom
import org.junit.Test
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import kotlin.test.assertFails

internal class RealDistributionTest {

    @Test
    fun testCount() {
        val distribution = RealDistribution(0.0, 1.45, 0.5)
        assertEquals(3, distribution.count)
        val dist = RealDistribution(0.0, 4.0, 0.9)
        assertEquals(5, dist.count)
        val dist2 = RealDistribution(0.0, 1.0, 0.5)
        assertEquals(3, dist2.count)

    }

    @Test
    fun testSample() {
        val dist = RealDistribution(0.0, 4.0, 0.9)
        val vals = dist.sample(150, DeterministicRandom())
        assertTrue(vals[0.0]!! > 10)
        assertTrue(vals[0.9]!! > 10)
        assertTrue(vals[1.8]!! > 10)
        assertTrue(vals[2.7]!! > 10)
        assertTrue(vals[3.6]!! > 10)
    }

    @Test
    fun testSupport() {
        val dist = RealDistribution(0.0, 4.0, 0.9)
        val support = dist.support()
        assertEquals(listOf(0.0, 0.9, 1.8, 2.7, 3.6), support.toList())
    }

    @Test
    fun testScore() {
        val dist = RealDistribution(0.0, 4.0, 0.9)
        assertEquals(5, dist.count)
        assertEquals(0.2, dist.score(0.6))
        assertEquals(0.0, dist.score(-1.0))
    }

    @Test
    fun testFilter() {
        val dist = RealDistribution(0.0, 4.0, 0.9)
        val newDist = dist.filter { it in (0.0..1.0) || it in (2.0..3.0) }
        assertEquals(listOf(0.0, 0.9, 2.7), newDist.support().toList())
        val sup = newDist.supportWithDensities()
        assertEquals(0.333, sup[0.0]!!, 0.01)
        assertEquals(0.333, sup[0.9]!!, 0.01)
        assertEquals(0.333, sup[2.7]!!, 0.01)
    }

    /**
     * Assert that map happens lazily as the distribution can be very big
     */
    @Test
    fun testMap() {
        val dist = RealDistribution(0.0, 4.0, 0.9)
        val random = DeterministicRandom()
        val next = dist.map { if (it > 2) error("") else "Hello $it" }
        assertEquals("Hello 1.8", next.sample(random))
        assertFails {
            next.sample(random)
        }
    }

    @Test
    fun testChain() {
        val x = (0.0..1.0) sampleWithStepSize 0.1
        val y = (0.0..1.0) sampleWithStepSize 0.1
        val sum = x + y
        assertEquals(21, sum.support().count())
    }
}