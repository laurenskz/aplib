package eu.iv4xr.framework.model.distribution

import org.junit.Assert

fun <T> assertProbability(t: T, prob: Double, distribution: Distribution<out T>, delta: Double = 0.01, message: String? = null) {
    if (message != null) {
        Assert.assertEquals(message, prob, distribution.score { it == t }, delta)
    } else {
        Assert.assertEquals(prob, distribution.score { it == t }, delta)
    }
}

fun <T> assertAlways(t: T, distribution: Distribution<out T>) = assertProbability(t, 1.0, distribution)