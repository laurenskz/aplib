package eu.iv4xr.framework.model.distribution

import kotlin.random.Random

class SequenceUniform<T>(private val sequence: Sequence<T>, val count: Int) : Distribution<T> {

    private val prob = 1.0 / count;

    override fun score(t: T) = if (sequence.contains(t)) prob else 0.0

    override fun score(predicate: (T) -> Boolean) = sequence.count { predicate(it) } * prob

    override fun filter(predicate: (T) -> Boolean) = UniformDistribution(sequence.filter(predicate).toList())

    override fun sample(random: Random) = sequence.drop(random.nextInt(count)).first()

    override fun support() = sequence.distinct()

    override fun <A> chain(continuation: (T) -> Distribution<A>) = ChainedDistribution(this, continuation)

    override fun <R> map(modifier: (T) -> R) = ChainedDistribution(this) { always(modifier(it)) }
}