package eu.iv4xr.framework.model

import kotlin.random.Random

/**
 * Sample multiple times
 */
fun <T> Distribution<T>.sample(count: Int, random: Random): Map<T, Int> {
    return (0 until count).map {
        sample(random)
    }
            .groupBy { it }
            .mapValues { it.value.count() }
}

/**
 * Useful for printing all elements with their probability
 */
fun <T> Distribution<T>.densityString(): String {
    return supportWithDensities().map {
        it.key.toString() + " : " + it.value.toString() + "\n"
    }.joinToString("")
}

/**
 * Expected value of this distribution, given a function for associating a value with each element of this distribution
 */
fun <T> Distribution<T>.expectedValue(evaluator: (T) -> Double): Double {
    return supportWithDensities().map {
        evaluator(it.key) * it.value
    }.sum()
}

/**
 * All number distributions have an expected value
 */
fun <T : Number> Distribution<T>.expectedValue() = expectedValue { it.toDouble() }

/**
 * Make all numerical distributions have overloaded arithmetic operators
 */
@JvmName("plusDouble")
operator fun Distribution<Double>.plus(other: Distribution<Double>) = chain { other.plus(it) }
operator fun Distribution<Double>.plus(t: Double) = map { it.plus(t) }
operator fun Double.plus(other: Distribution<Double>) = other.map { this.plus(it) }

@JvmName("plusInt")
operator fun Distribution<Int>.plus(other: Distribution<Int>) = chain { other.plus(it) }
operator fun Distribution<Int>.plus(t: Int) = map { it.plus(t) }
operator fun Int.plus(other: Distribution<Int>) = other.map { this.plus(it) }

@JvmName("plusLong")
operator fun Distribution<Long>.plus(other: Distribution<Long>) = chain { other.plus(it) }
operator fun Distribution<Long>.plus(t: Long) = map { it.plus(t) }
operator fun Long.plus(other: Distribution<Long>) = other.map { this.plus(it) }

@JvmName("minusDouble")
operator fun Distribution<Double>.minus(other: Distribution<Double>) = chain { other.minus(it) }
operator fun Distribution<Double>.minus(t: Double) = map { it.minus(t) }
operator fun Double.minus(other: Distribution<Double>) = other.map { this.minus(it) }

@JvmName("minusInt")
operator fun Distribution<Int>.minus(other: Distribution<Int>) = chain { other.minus(it) }
operator fun Distribution<Int>.minus(t: Int) = map { it.minus(t) }
operator fun Int.minus(other: Distribution<Int>) = other.map { this.minus(it) }


@JvmName("minusLong")
operator fun Distribution<Long>.minus(other: Distribution<Long>) = chain { other.minus(it) }
operator fun Distribution<Long>.minus(t: Long) = map { it.minus(t) }
operator fun Long.minus(other: Distribution<Long>) = other.map { this.minus(it) }

@JvmName("timesDouble")
operator fun Distribution<Double>.times(other: Distribution<Double>) = chain { other.times(it) }
operator fun Distribution<Double>.times(t: Double) = map { it.times(t) }
operator fun Double.times(other: Distribution<Double>) = other.map { this.times(it) }

@JvmName("timesInt")
operator fun Distribution<Int>.times(other: Distribution<Int>) = chain { other.times(it) }
operator fun Distribution<Int>.times(t: Int) = map { it.times(t) }
operator fun Int.times(other: Distribution<Int>) = other.map { this.times(it) }


@JvmName("timesLong")
operator fun Distribution<Long>.times(other: Distribution<Long>) = chain { other.times(it) }
operator fun Distribution<Long>.times(t: Long) = map { it.times(t) }
operator fun Long.times(other: Distribution<Long>) = other.map { this.times(it) }

@JvmName("divDouble")
operator fun Distribution<Double>.div(other: Distribution<Double>) = chain { other.div(it) }
operator fun Distribution<Double>.div(t: Double) = map { it.div(t) }
operator fun Double.div(other: Distribution<Double>) = other.map { this.div(it) }


@JvmName("divInt")
operator fun Distribution<Int>.div(other: Distribution<Int>) = chain { other.div(it) }
operator fun Distribution<Int>.div(t: Int) = map { it.div(t) }
operator fun Int.div(other: Distribution<Int>) = other.map { this.div(it) }


@JvmName("divLong")
operator fun Distribution<Long>.div(other: Distribution<Long>) = chain { other.div(it) }
operator fun Distribution<Long>.div(t: Long) = map { it.div(t) }
operator fun Long.div(other: Distribution<Long>) = other.map { this.div(it) }


@JvmName("remDouble")
operator fun Distribution<Double>.rem(other: Distribution<Double>) = chain { other.rem(it) }
operator fun Distribution<Double>.rem(t: Double) = map { it.rem(t) }
operator fun Double.rem(other: Distribution<Double>) = other.map { this.rem(it) }


@JvmName("remInt")
operator fun Distribution<Int>.rem(other: Distribution<Int>) = chain { other.rem(it) }
operator fun Distribution<Int>.rem(t: Int) = map { it.rem(t) }
operator fun Int.rem(other: Distribution<Long>) = other.map { this.rem(it) }

@JvmName("remLong")
operator fun Distribution<Long>.rem(other: Distribution<Long>) = chain { other.rem(it) }
operator fun Distribution<Long>.rem(t: Long) = map { it.rem(t) }
operator fun Long.rem(other: Distribution<Long>) = other.map { this.rem(it) }

@JvmName("rangeToDouble")
operator fun Distribution<Double>.rangeTo(other: Distribution<Double>) = chain { other.rangeTo(it) }
operator fun Distribution<Double>.rangeTo(t: Double) = map { it.rangeTo(t) }
operator fun Double.rangeTo(other: Distribution<Double>) = other.map { this.rangeTo(it) }

@JvmName("rangeToInt")
operator fun Distribution<Int>.rangeTo(other: Distribution<Int>) = chain { other.rangeTo(it) }
operator fun Distribution<Int>.rangeTo(t: Int) = map { it.rangeTo(t) }
operator fun Int.rangeTo(other: Distribution<Int>) = other.map { this.rangeTo(it) }


@JvmName("rangeToLong")
operator fun Distribution<Long>.rangeTo(other: Distribution<Long>) = chain { other.rangeTo(it) }
operator fun Distribution<Long>.rangeTo(t: Long) = map { it.rangeTo(t) }
operator fun Long.rangeTo(other: Distribution<Long>) = other.map { this.rangeTo(it) }


/**
 * Flip a coin, it is true with probability p
 */
fun flip(p: Double) = Distributions.bernoulli(p)

/**
 * Make element t have probability 1. I.e. a deterministic distribution
 */
fun <T> always(t: T) = Distributions.deterministic(t)

/**
 * Computes the cartesian product of two distributions
 */
infix fun <T, R> Distribution<T>.times(b: Distribution<R>): Distribution<Pair<T, R>> {
    return chain { t -> b.map { r -> t to r } }
}

/**
 * Return a distribution based on the result of a boolean distribution
 */
fun <A> ifd(case: Distribution<Boolean>, t: Distribution<A>, f: Distribution<A>): Distribution<A> {
    return case.chain {
        if (it) t else f
    }
}

/**
 * Return a distribution (deterministic one) based on the result of a boolean
 */
fun <A> if_(case: Distribution<Boolean>, t: A, f: A): Distribution<A> {
    return case.chain {
        if (it) Distributions.deterministic(t) else Distributions.deterministic(f)
    }
}

/**
 * Create if else like syntax to use with a boolean distribution. Note that we do not require a distribution here, this is syntactic
 * sugar for wrapping the supplied value in a deterministic distribution
 */
fun <A> if_(case: Distribution<Boolean>, t: () -> A): IfBuilder<A> {
    return IfBuilder(case) {
        Distributions.deterministic(t())
    }
}

/**
 * Create if else like syntax, but based on distribution instead of a value
 */
fun <A> ifd(case: Distribution<Boolean>, t: () -> Distribution<A>): IfBuilder<A> {
    return IfBuilder(case, t)
}

/**
 * Wrapper for creating if else like syntax for boolean distributions
 */
class IfBuilder<A>(private val case: Distribution<Boolean>, private val t: () -> Distribution<A>) {

    /**
     * else_ option for when the user does not want to supply a distribution, we wrap it again (syntactic sugar)
     */
    fun else_(f: () -> A): Distribution<A> {
        return case.chain {
            if (it) t() else Distributions.deterministic(f())
        }
    }

    /**
     * else option for when the user wants to pass a distribution
     */
    fun elsed(f: () -> Distribution<A>): Distribution<A> {
        return case.chain {
            if (it) t() else f()
        }
    }
}
