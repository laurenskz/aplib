package eu.iv4xr.framework.model

import kotlin.random.Random

/**
 * Represents a distribution based on some distribution and a continuation.
 *
 * In this way we can create more complex distributions from simpler ones.
 */
class ChainedDistribution<T, R>(private val dist: Distribution<T>, private val continuation: (T) -> Distribution<R>) : Distribution<R> {

    /**
     * The probability of reaching the result via any path starting in the first distribution, then we sum because those are independent events
     */
    override fun score(t: R): Double {
        return dist.support()
                .sumByDouble { dist.score(it) * continuation(it).score(t) }
    }

    /**
     * Same as other score
     */
    override fun score(predicate: (R) -> Boolean): Double {
        return dist.support()
                .sumByDouble {
                    dist.score(it) * continuation(it).score(predicate)
                }
    }

    /**
     * We filter the distribution of the continuation lazily
     */
    override fun filter(predicate: (R) -> Boolean): Distribution<R> {
        return ChainedDistribution(dist) {
            continuation(it).filter(predicate)
        }
    }

    /**
     * Sample the base distribution of first and then sample again
     */
    override fun sample(random: Random): R {
        return continuation(dist.sample(random)).sample(random)
    }

    /**
     * We assess all possible outcomes and return the unique ones
     */
    override fun support(): Sequence<R> {
        return dist.support()
                .flatMap { continuation(it).support() }
                .distinct()
    }

    /**
     * Return a new chained distribution, we then thus have two continuations after each other
     */
    override fun <A> chain(continuation: (R) -> Distribution<A>): Distribution<A> {
        return ChainedDistribution(this, continuation)
    }

    /**
     * Map the distributions of the continuation
     */
    override fun <A> map(modifier: (R) -> A): Distribution<A> {
        return ChainedDistribution(dist) {
            continuation(it).map(modifier)
        }
    }
}