package eu.iv4xr.framework.model.distribution

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
    fun <T> discrete(vararg elements: Pair<T, Double>): Distribution<T> {
        return DiscreteDistribution(mapOf(*elements))
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

    /**
     * Distribution of 1 element with probability 1
     */
    fun <T> deterministic(t: T) = ConstantDistribution(t)
}


