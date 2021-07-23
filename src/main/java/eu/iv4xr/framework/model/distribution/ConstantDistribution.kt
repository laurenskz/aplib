package eu.iv4xr.framework.model.distribution

import java.lang.IllegalArgumentException
import kotlin.random.Random

class ConstantDistribution<T>(private val t: T) : Distribution<T> {
    override fun score(t: T) = if (this.t == t) 1.0 else 0.0

    override fun score(predicate: (T) -> Boolean) = if (predicate(t)) 1.0 else 0.0

    override fun filter(predicate: (T) -> Boolean) = if (!predicate(t)) throw IllegalArgumentException("Ends up with empty distribution") else this

    override fun sample(random: Random) = t

    override fun support() = sequenceOf(t)

    override fun <A> chain(continuation: (T) -> Distribution<A>) = continuation(t)

    override fun <R> map(modifier: (T) -> R) = ConstantDistribution(modifier(t))
}