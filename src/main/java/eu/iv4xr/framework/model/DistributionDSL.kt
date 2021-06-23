package eu.iv4xr.framework.model

import kotlin.random.Random

fun <T> Distribution<T>.sample(count: Int, random: Random): Map<T, Int> {
    return (0 until count).map {
        sample(random)
    }
            .groupBy { it }
            .mapValues { it.value.count() }
}

fun <A> if_(case: Distribution<Boolean>, t: Distribution<A>, f: Distribution<A>): Distribution<A> {
    return case.chain {
        if (it) t else f
    }
}
