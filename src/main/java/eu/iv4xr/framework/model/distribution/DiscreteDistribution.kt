package eu.iv4xr.framework.model.distribution

import java.lang.IllegalStateException
import kotlin.random.Random

/**
 * The most basic distribution, it is discrete and based on a map of values with associated probabilities, it is required that
 * those probabilities sum to 1
 */
class DiscreteDistribution<T>(val values: Map<T, Double>, private val tolerance: Double = 0.0001) : Distribution<T> {

    init {
        if(values.any{ it.value.isNaN()})
            println("Nope!")
        assert((values.values.sum() - 1) < tolerance)
    }

    /**
     * Basic map lookup
     */
    override fun score(t: T): Double {
        return values[t] ?: 0.0;
    }

    /**
     * Possibly multiple values satisfy the predicate
     */
    override fun score(predicate: (T) -> Boolean): Double {
        return values.filterKeys(predicate).values.sum();
    }

    /**
     * We filter the map and then scale the resulting probabilities to sum to 1 again
     */
    override fun filter(predicate: (T) -> Boolean): Distribution<T> {
        val kept = values.filterKeys(predicate)
        if (kept.size == 1) {
            println("hi!")
        }
        val total = kept.values.sum()
        val scaled = kept.mapValues { it.value / total }
        return DiscreteDistribution(scaled)
    }

    /**
     * Sample a number between 0 and 1 and then go through the map to see which element corresponds to this sample
     */
    override fun sample(random: Random): T {
        val sampledNumber = random.nextDouble();
        var current = 0.0;
        for (value in values) {
            if (current + value.value > sampledNumber) {
                return value.key
            }
            current += value.value
        }
        throw IllegalStateException("No element could be sampled, support = ${supportWithDensities()}")
    }

    /**
     * Distinct values in the map
     */
    override fun support(): Sequence<T> {
        return values.keys.distinct().asSequence()
    }

    /**
     * Return a chained distribution so that the continuation is evaluated lazily
     */
    override fun <A> chain(continuation: (T) -> Distribution<A>): Distribution<A> {
        return ChainedDistribution(this, continuation)
    }

    /**
     * This is not evaluated lazily and directly applies the modifier to all elements and returns a new discrete distribution
     */
    override fun <R> map(modifier: (T) -> R): Distribution<R> {
        return ChainedDistribution(this) { always(modifier(it)) }
    }
}