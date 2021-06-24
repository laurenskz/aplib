package eu.iv4xr.framework.model

import kotlin.random.Random

/**
 * The fundamental building block of probabilistic modelling.
 *
 * A distribution represents a set of values in which each element has a corresponding probability.
 */
interface Distribution<T> {

    /**
     * Gives the probability of element t in this distribution
     */
    fun score(t: T): Double

    /**
     * Gives the sum of the probabilities of all elements that satisfy the predicate in this distribution
     */
    fun score(predicate: (T) -> Boolean): Double

    /**
     * Returns a new distribution in which all elements satisfy the predicate
     */
    fun filter(predicate: (T) -> Boolean): Distribution<T>

    /**
     * Draws a random element from this distribution based on their respective probabilities and the supplied random number generator
     */
    fun sample(random: Random): T

    /**
     * Returns all unique values in this distribution
     */
    fun support(): Sequence<T>

    /**
     * Creates a new distribution from the values in this distribution.
     *
     * Each continued distribution is weighted by the probability of the element that was passed to the continuation, in that way
     * the resulting probabilities sum to 1 again.
     */
    fun <A> chain(continuation: (T) -> Distribution<A>): Distribution<A>

    /**
     * Map the values in this distribution, i.e. transform each element.
     * If two values are mapped to the same result, then the probabilities are summed
     */
    fun <R> map(modifier: (T) -> R): Distribution<R>

    /**
     * Returns all unique elements, associated with their probability
     */
    fun supportWithDensities(): Map<T, Double> = support().associateWith { score(it) }
}