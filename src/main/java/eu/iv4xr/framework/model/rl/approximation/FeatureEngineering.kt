package eu.iv4xr.framework.model.rl.approximation

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.spatial.Vec3
import kotlin.math.max
import kotlin.reflect.KClass
import kotlin.reflect.safeCast

interface FeatureVectorFactory<T> {
    fun features(t: T): DoubleArray = DoubleArray(count()).also { setFrom(t, it, 0) }
    fun setFrom(t: T, result: DoubleArray, start: Int)
    fun count(): Int
}

typealias FeatureActionFactory<S, A> = FeatureVectorFactory<Pair<S, A>>

class MergedFeatureFactory<T, A>(val first: FeatureVectorFactory<T>, val second: FeatureVectorFactory<A>) : FeatureVectorFactory<Pair<T, A>> {
    override fun setFrom(t: Pair<T, A>, result: DoubleArray, start: Int) {
        first.setFrom(t.first, result, start)
        second.setFrom(t.second, result, start + first.count())
    }

    override fun count(): Int {
        return first.count() + second.count()
    }
}

open class FeatureOwner<T>(val factory: FeatureVectorFactory<T>) : Identifiable

fun <T : FeatureOwner<T>> T.features() = factory.features(this)

class OneHot<T>(val ts: List<T>) : FeatureVectorFactory<T> {

    override fun setFrom(t: T, result: DoubleArray, start: Int) {
        ts.forEachIndexed { i, tp -> result[i + start] = if (t == tp) 1.0 else 0.0 }
    }

    override fun count(): Int {
        return ts.count()
    }
}

open class PrimitiveFeature<T>(val toDouble: (T) -> Double) : FeatureVectorFactory<T> {

    override fun setFrom(t: T, result: DoubleArray, start: Int) {
        result[start] = toDouble(t)
    }

    override fun count(): Int {
        return 1
    }
}

class ExtractFeature<T, V>(val factory: FeatureVectorFactory<V>, val lens: (T) -> V) : FeatureVectorFactory<T> {
    override fun features(t: T): DoubleArray {
        return factory.features(lens(t))
    }

    override fun setFrom(t: T, result: DoubleArray, start: Int) {
        return factory.setFrom(lens(t), result, start)
    }

    override fun count(): Int {
        return factory.count()
    }
}

class RepeatedFeature<T>(val repetitions: Int, val factory: FeatureVectorFactory<T>) : FeatureVectorFactory<List<T>> {

    override fun setFrom(t: List<T>, result: DoubleArray, start: Int) {
        for (i in (0 until max(repetitions, t.count()))) {
            factory.setFrom(t[i], result, start + i * factory.count())
        }
    }

    override fun count(): Int {
        return this.repetitions * factory.count()
    }
}

object Vec3Feature : FeatureVectorFactory<Vec3> {

    override fun setFrom(t: Vec3, result: DoubleArray, start: Int) {
        result[0] = t.x.toDouble()
        result[1] = t.y.toDouble()
        result[2] = t.z.toDouble()
    }

    override fun count() = 3
}

class SumTypeFeatureFactory<T : Any>(val factories: List<OptionalFeatureFactory<T, *>>) : FeatureVectorFactory<T> {

    val map: Map<KClass<*>, FeatureVectorFactory<T>> = factories.associate { it.clazz to it }
    val totalCount = factories.sumBy { it.count() }
    val counts = factories.scan(0) { acc, fac -> acc + fac.factory.count() }
    val offsets = factories.mapIndexed { idx, fac -> fac.clazz to counts[idx] }.toMap()


    override fun setFrom(t: T, result: DoubleArray, start: Int) {
        map[t::class]?.setFrom(t, result, start + (offsets[t::class] ?: error("Unrecognized class")))
    }

    override fun count() = totalCount
}

class OptionalFeatureFactory<T : Any, V : T>(val clazz: KClass<V>, val factory: FeatureVectorFactory<V>) : FeatureVectorFactory<T> {

    override fun setFrom(t: T, result: DoubleArray, start: Int) {
        clazz.safeCast(t)?.also {
            factory.setFrom(it, DoubleArray(0), 0)
        }
    }

    override fun count(): Int {
        return factory.count()
    }
}

/**
 * Encodes the type of the class as well as the representation given by the factory, note that
 */
class EncodedSumType<T : Any>(factories: List<OptionalFeatureFactory<T, *>>) : CompositeFeature<T>(listOf(
        OneHot(factories.map { it.clazz }).from { it::class },
        SumTypeFeatureFactory(factories)
))

infix fun <T : Any, V : T> KClass<V>.with(factory: FeatureVectorFactory<V>) = OptionalFeatureFactory<T, V>(this, factory)


open class CompositeFeature<T>(val featureVectorFactories: List<FeatureVectorFactory<T>>) : FeatureVectorFactory<T> {

    override fun setFrom(t: T, result: DoubleArray, start: Int) {
        var count = 0
        for (featureVectorFactory in featureVectorFactories) {
            featureVectorFactory.setFrom(t, result, start + count)
            count += featureVectorFactory.count()
        }
    }

    override fun count(): Int {
        return featureVectorFactories.sumBy { it.count() }
    }
}

fun <T, V> FeatureVectorFactory<V>.from(f: (T) -> V) = ExtractFeature(this, f)

object IntFeature : PrimitiveFeature<Int>({ it.toDouble() })
object DoubleFeature : PrimitiveFeature<Double>({ it })
object BoolFeature : PrimitiveFeature<Boolean>({ if (it) 1.0 else 0.0 })
