package eu.iv4xr.framework.model

import kotlin.random.Random

interface Distribution<T> {
    fun score(t: T): Double
    fun score(predicate: (T) -> Boolean): Double
    fun filter(predicate: (T) -> Boolean): Distribution<T>
    fun sample(random: Random): T
    fun support(): Sequence<T>
    fun <A> chain(continuation: (T) -> Distribution<A>): Distribution<A>
    fun <R> map(modifier: (T) -> R): Distribution<R>
    fun supportWithDensities(): Map<T, Double> = support().associateWith { score(it) }
}