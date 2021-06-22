package eu.iv4xr.framework.model

object Distributions {
    fun <T> uniform(vararg ts: T): Distribution<T> {
        val prob = 1.0 / ts.size
        return DiscreteDistribution(ts.associate { it to prob })
    }
}