package eu.iv4xr.framework.model

import kotlin.random.Random

fun <T> Distribution<T>.sample(count: Int, random: Random): Map<T, Int> {
    return (0 until count).map {
        sample(random)
    }
            .groupBy { it }
            .mapValues { it.value.count() }
}

fun flip(p: Double) = Distributions.bernoulli(p)

fun <A> ifd(case: Distribution<Boolean>, t: Distribution<A>, f: Distribution<A>): Distribution<A> {
    return case.chain {
        if (it) t else f
    }
}

fun <A> if_(case: Distribution<Boolean>, t: A, f: A): Distribution<A> {
    return case.chain {
        if (it) Distributions.constant(t) else Distributions.constant(f)
    }
}

fun <A> if_(case: Distribution<Boolean>, t: () -> A): IfBuilder<A> {
    return IfBuilder(case) {
        Distributions.constant(t())
    }
}

fun <A> ifd(case: Distribution<Boolean>, t: () -> Distribution<A>): IfBuilder<A> {
    return IfBuilder(case, t)
}


class IfBuilder<A>(private val case: Distribution<Boolean>, private val t: () -> Distribution<A>) {

    fun else_(f: () -> A): Distribution<A> {
        return case.chain {
            if (it) t() else Distributions.constant(f())
        }
    }

    fun elsed(f: () -> Distribution<A>): Distribution<A> {
        return case.chain {
            if (it) t() else f()
        }
    }
}
