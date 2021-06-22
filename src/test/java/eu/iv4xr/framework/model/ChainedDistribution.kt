package eu.iv4xr.framework.model

import kotlin.random.Random

class ChainedDistribution<T, R>(private val dist: Distribution<T>, private val continuation: (T) -> Distribution<R>) : Distribution<R> {

    override fun score(t: R): Double {
        return dist.support()
                .sumByDouble { dist.score(it) * continuation(it).score(t) }
    }

    override fun score(predicate: (R) -> Boolean): Double {
        return dist.support()
                .sumByDouble { dist.score(it) * continuation(it).score(predicate) }
    }

    override fun filter(predicate: (R) -> Boolean): Distribution<R> {
        return ChainedDistribution(dist) {
            continuation(it).filter(predicate)
        }
    }

    override fun sample(random: Random): R {
        return continuation(dist.sample(random)).sample(random)
    }

    override fun support(): Sequence<R> {
        return dist.support()
                .flatMap { continuation(it).support() }
    }

    override fun <A> chain(continuation: (R) -> Distribution<A>): Distribution<A> {
        return ChainedDistribution(this, continuation)
    }

    override fun <A> map(modifier: (R) -> A): Distribution<A> {
        return ChainedDistribution(dist) {
            continuation(it).map(modifier)
        }
    }
}