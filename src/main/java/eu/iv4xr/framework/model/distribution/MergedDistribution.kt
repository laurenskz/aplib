package eu.iv4xr.framework.model.distribution

import kotlin.random.Random

class MergedDistribution<T>(val dists: List<Distribution<T>>) : Distribution<T> {
    override fun score(t: T): Double {
        return dists.sumByDouble { it.score(t) } / dists.size
    }

    override fun score(predicate: (T) -> Boolean): Double {
        return dists.sumByDouble { it.score(predicate) } / dists.size

    }

    override fun filter(predicate: (T) -> Boolean): Distribution<T> {
        return MergedDistribution(dists.map { it.filter(predicate) })
    }

    override fun sample(random: Random): T {
        val dist = dists.random()
        return dist.sample(random)
    }

    override fun support(): Sequence<T> {
        return dists.flatMap { it.support() }.asSequence()
    }

    override fun <A> chain(continuation: (T) -> Distribution<A>): Distribution<A> {
        return MergedDistribution(dists.map { it.chain(continuation) })
    }

    override fun <R> map(modifier: (T) -> R): Distribution<R> {
        return MergedDistribution(dists.map { it.map(modifier) })
    }
}