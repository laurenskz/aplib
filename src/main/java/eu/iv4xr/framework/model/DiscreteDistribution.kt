package eu.iv4xr.framework.model

import java.lang.IllegalStateException
import kotlin.random.Random

class DiscreteDistribution<T>(val values: Map<T, Double>, private val tolerance: Double = 0.0001) : Distribution<T> {

    init {
        assert((values.values.sum() - 1) < tolerance)
    }

    override fun score(t: T): Double {
        return values[t] ?: 0.0;
    }

    override fun score(predicate: (T) -> Boolean): Double {
        return values.filterKeys(predicate).values.sum();
    }

    override fun filter(predicate: (T) -> Boolean): Distribution<T> {
        val kept = values.filterKeys(predicate)
        val total = kept.values.sum()
        val scaled = kept.mapValues { it.value / total }
        return DiscreteDistribution(scaled)
    }

    override fun sample(random: Random): T {
        val sampledNumber = random.nextDouble();
        var current = 0.0;
        for (value in values) {
            if (current + value.value > sampledNumber) {
                return value.key
            }
            current += value.value
        }
        throw IllegalStateException("No element could be sampled")
    }

    override fun support(): Sequence<T> {
        return values.keys.distinct().asSequence()
    }

    override fun <A> chain(continuation: (T) -> Distribution<A>): Distribution<A> {
        return ChainedDistribution(this, continuation)
    }

    override fun <R> map(modifier: (T) -> R): Distribution<R> {
        val result = mutableMapOf<R, Double>()
        for (value in supportWithDensities()) {
            result.compute(modifier(value.key)) { _, prob ->
                prob?.plus(value.value) ?: value.value
            }
        }
        return DiscreteDistribution(result)
    }
}