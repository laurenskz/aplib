package eu.iv4xr.framework.model.distribution

import kotlin.math.floor
import kotlin.math.roundToLong
import kotlin.random.Random


class RealDistribution(val min: Double, val max: Double, val step: Double) : Distribution<Double> {

    val count: Int = floor((max - min) / step).toInt() + 1
    val prob = 1.0 / count


    override fun score(t: Double): Double {
        if (t in min..max) return prob
        return 0.0
    }

    override fun score(predicate: (Double) -> Boolean): Double {
        return support().filter(predicate).count() * prob
    }

    override fun filter(predicate: (Double) -> Boolean): Distribution<Double> {
        return UniformDistribution(support().filter(predicate).toList())
    }

    override fun sample(random: Random): Double {
        val index = random.nextInt(count + 1)
        return min + index * step
    }

    override fun support() = generateSequence(min) { it + step }.takeWhile { it <= max }

    override fun <A> chain(continuation: (Double) -> Distribution<A>): Distribution<A> {
        return ChainedDistribution(this, continuation)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <R> map(modifier: (Double) -> R): Distribution<R> {
        val continuation = if (modifier(0.0) is Double)
            { it: Double -> always(((modifier(it) as Double) / step).roundToLong() * step) as Distribution<R> }
        else
            { it: Double -> always(modifier(it)) }
        return ChainedDistribution(this, continuation)
    }
}