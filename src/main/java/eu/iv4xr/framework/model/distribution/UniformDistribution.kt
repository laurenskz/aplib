package eu.iv4xr.framework.model.distribution

import kotlin.random.Random

class UniformDistribution<T>(private val values: Collection<T>) : Distribution<T> {

    private val prob = 1.0 / values.size;

    override fun score(t: T) = if (values.contains(t)) prob else 0.0

    override fun score(predicate: (T) -> Boolean) = values.count { predicate(it) } * prob

    override fun filter(predicate: (T) -> Boolean) = UniformDistribution(values.filter(predicate))

    override fun sample(random: Random) = values.random(random)

    override fun support() = values.distinct().asSequence()

    override fun <A> chain(continuation: (T) -> Distribution<A>) = ChainedDistribution(this, continuation)

    override fun <R> map(modifier: (T) -> R) = UniformDistribution(values.map(modifier))
}