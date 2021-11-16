package eu.iv4xr.framework.model.distribution

import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange
import kotlin.math.exp

/**
 * Some basic probability distributions
 */
object Distributions {

    /**
     * Uniform distribution based on some elements
     */
    fun <T> uniform(vararg ts: T) = UniformDistribution(ts.asList())

    /**
     * Uniform distribution based on a collection of elements
     */
    fun <T> uniform(ts: Collection<T>) = UniformDistribution(ts)


    /**
     * Uniform distribution for all ints in the range
     */
    fun uniform(ints: IntRange) = UniformDistribution(ints.toList())

    /**
     * Uniform distribution for all longs in the range
     */
    fun uniform(longs: LongRange) = UniformDistribution(longs.toList())

    /**
     * Discrete distribution based on values with associated probability
     */
    fun <T> discrete(vararg elements: Pair<out T, Double>): Distribution<T> {
        return DiscreteDistribution(mapOf(*elements))
    }

    /**
     * Discrete distribution based on softmax of the associated elements
     */
    fun <T> softmax(elements: Map<T, Double>): Distribution<T> {
        val exponentiated = elements.mapValues { exp(it.value) }
        val sum = exponentiated.values.sum()
        return discrete(exponentiated.mapValues { it.value / sum })
    }

    /**
     * Discrete distribution based on values with associated probability
     */
    fun <T> discrete(elements: List<Pair<T, Double>>): Distribution<T> {
        return DiscreteDistribution(elements.toMap())
    }

    /**
     * Discrete distribution based on values with associated probability
     */
    fun <T> discrete(elements: Map<T, Double>): Distribution<T> {
        return DiscreteDistribution(elements)
    }

    /**
     * Bernoulli distribution with probability p
     */
    fun bernoulli(p: Double): Distribution<Boolean> {
        return discrete(true to p, false to (1 - p))
    }

    fun real(min: Double, max: Double, step: Double): Distribution<Double> {
        return RealDistribution(min, max, step)
    }

    /**
     * Distribution of 1 element with probability 1
     */
    fun <T> deterministic(t: T) = ConstantDistribution(t)
}


