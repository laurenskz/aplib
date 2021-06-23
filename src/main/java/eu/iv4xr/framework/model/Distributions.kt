package eu.iv4xr.framework.model

object Distributions {

    fun <T> uniform(vararg ts: T): Distribution<T> {
        val prob = 1.0 / ts.size
        return DiscreteDistribution(ts.associateWith { prob })
    }

    fun <T> uniform(ts: Collection<T>): Distribution<T> {
        val prob = 1.0 / ts.size
        return DiscreteDistribution(ts.associateWith { prob })
    }

    fun uniform(ints: IntRange): Distribution<Int> {
        return uniform(ints.toList())
    }

    fun uniform(longs: LongRange): Distribution<Long> {
        return uniform(longs.toList())
    }

    fun <T> discrete(vararg elements: Pair<T, Double>): Distribution<T> {
        return DiscreteDistribution(mapOf(*elements))
    }

    fun bernoulli(p: Double): Distribution<Boolean> {
        return discrete(true to p, false to (1 - p))
    }

    fun <T> constant(t: T): Distribution<T> {
        return discrete(t to 1.0)
    }

}


