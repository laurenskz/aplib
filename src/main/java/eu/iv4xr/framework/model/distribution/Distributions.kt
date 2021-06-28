package eu.iv4xr.framework.model.distribution

/**
 * Some basic probability distributions
 */
object Distributions {

    /**
     * Uniform distribution based on some elements
     */
    fun <T> uniform(vararg ts: T): Distribution<T> {
        val prob = 1.0 / ts.size
        return DiscreteDistribution(ts.associateWith { prob })
    }

    /**
     * Uniform distribution based on a collection of elements
     */
    fun <T> uniform(ts: Collection<T>): Distribution<T> {
        val prob = 1.0 / ts.size
        return DiscreteDistribution(ts.associateWith { prob })
    }


    /**
     * Uniform distribution for all ints in the range
     */
    fun uniform(ints: IntRange): Distribution<Int> {
        return uniform(ints.toList())
    }

    /**
     * Uniform distribution for all longs in the range
     */
    fun uniform(longs: LongRange): Distribution<Long> {
        return uniform(longs.toList())
    }

    /**
     * Discrete distribution based on values with associated probability
     */
    fun <T> discrete(vararg elements: Pair<T, Double>): Distribution<T> {
        return DiscreteDistribution(mapOf(*elements))
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
    fun <T> deterministic(t: T): Distribution<T> {
        return discrete(t to 1.0)
    }
}


