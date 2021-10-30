package eu.iv4xr.framework.model.rl.approximation

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.spatial.Vec3
import org.tensorflow.ndarray.FloatNdArray
import org.tensorflow.types.TFloat32
import kotlin.math.max
import kotlin.reflect.KClass
import kotlin.reflect.safeCast

interface DataBuffer {
    fun set(index: Int, value: Double)
}

class TensorBuffer(val tensor: TFloat32, val dims: LongArray) : DataBuffer {
    override fun set(index: Int, value: Double) {
        tensor.setFloat(value.toFloat(), *dims, index.toLong())
    }

}

class NDArrayBuffer(val buffer: FloatNdArray, val dims: LongArray) : DataBuffer {
    override fun set(index: Int, value: Double) {
        buffer.setFloat(value.toFloat(), *dims, index.toLong())
    }
}

class FloatBuffer(val array: FloatArray) : DataBuffer {
    override fun set(index: Int, value: Double) {
        array[index] = value.toFloat()
    }
}

class DoubleBuffer(val array: DoubleArray) : DataBuffer {
    override fun set(index: Int, value: Double) {
        array[index] = value
    }
}

interface FeatureVectorFactory<T> {
    fun features(t: T): DoubleArray = DoubleArray(count()).also { setFrom(t, DoubleBuffer(it), 0) }
    fun floatFeatures(t: T): FloatArray = FloatArray(count()).also { setFrom(t, FloatBuffer(it), 0) }
    fun setTensorFeatures(t: T, tensor: TFloat32, index: LongArray) = tensor.also { setFrom(t, TensorBuffer(it, index), 0) }
    fun setNdArrayFeatures(t: T, tensor: FloatNdArray, index: LongArray) = tensor.also { setFrom(t, NDArrayBuffer(it, index), 0) }
    fun setFrom(t: T, result: DataBuffer, start: Int)
    fun count(): Int
}

typealias FeatureActionFactory<S, A> = FeatureVectorFactory<Pair<S, A>>

class MergedFeatureFactory<T, A>(val first: FeatureVectorFactory<T>, val second: FeatureVectorFactory<A>) : FeatureVectorFactory<Pair<T, A>> {
    override fun setFrom(t: Pair<T, A>, result: DataBuffer, start: Int) {
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
    private val map = ts.mapIndexed { i, t -> t to i }.toMap()

    override fun setFrom(t: T, result: DataBuffer, start: Int) {
        val index = map[t] ?: return
        result.set(start + index, 1.0)
    }

    override fun count(): Int {
        return ts.count()
    }
}

open class PrimitiveFeature<T>(val toDouble: (T) -> Double) : FeatureVectorFactory<T> {

    override fun setFrom(t: T, result: DataBuffer, start: Int) {
        result.set(start, toDouble(t))
    }

    override fun count(): Int {
        return 1
    }
}

class ExtractFeature<T, V>(val factory: FeatureVectorFactory<V>, val lens: (T) -> V) : FeatureVectorFactory<T> {

    override fun setFrom(t: T, result: DataBuffer, start: Int) {
        return factory.setFrom(lens(t), result, start)
    }

    override fun count(): Int {
        return factory.count()
    }
}

class RepeatedFeature<T>(val repetitions: Int, val factory: FeatureVectorFactory<T>) : FeatureVectorFactory<List<T>> {

    override fun setFrom(t: List<T>, result: DataBuffer, start: Int) {
        for (i in t.indices.filter { it < repetitions }) {
            factory.setFrom(t[i], result, start + i * factory.count())
        }
    }

    override fun count(): Int {
        return this.repetitions * factory.count()
    }
}

object Vec3Feature : FeatureVectorFactory<Vec3> {

    override fun setFrom(t: Vec3, result: DataBuffer, start: Int) {
        result.set(0, t.x.toDouble())
        result.set(1, t.y.toDouble())
        result.set(2, t.z.toDouble())
    }

    override fun count() = 3
}

class SumTypeFeatureFactory<T : Any>(val factories: List<OptionalFeatureFactory<T, *>>) : FeatureVectorFactory<T> {

    val map: Map<KClass<*>, FeatureVectorFactory<T>> = factories.associate { it.clazz to it }
    val totalCount = factories.sumBy { it.count() }
    val counts = factories.scan(0) { acc, fac -> acc + fac.factory.count() }
    val offsets = factories.mapIndexed { idx, fac -> fac.clazz to counts[idx] }.toMap()


    override fun setFrom(t: T, result: DataBuffer, start: Int) {
        map[t::class]?.setFrom(t, result, start + (offsets[t::class] ?: error("Unrecognized class")))
    }

    override fun count() = totalCount
}

class OptionalFeatureFactory<T : Any, V : T>(val clazz: KClass<V>, val factory: FeatureVectorFactory<V>) : FeatureVectorFactory<T> {

    override fun setFrom(t: T, result: DataBuffer, start: Int) {
        clazz.safeCast(t)?.also {
            factory.setFrom(it, result, 0)
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
    private val count = featureVectorFactories.sumBy { it.count() }

    override fun setFrom(t: T, result: DataBuffer, start: Int) {
        var count = 0
        for (featureVectorFactory in featureVectorFactories) {
            featureVectorFactory.setFrom(t, result, start + count)
            count += featureVectorFactory.count()
        }
    }

    override fun count(): Int {
        return count
    }
}

fun <T, V> FeatureVectorFactory<V>.from(f: (T) -> V) = ExtractFeature(this, f)

object IntFeature : PrimitiveFeature<Int>({ it.toDouble() })
object DoubleFeature : PrimitiveFeature<Double>({ it })
object FloatFeature : PrimitiveFeature<Float>({ it.toDouble() })
object BoolFeature : PrimitiveFeature<Boolean>({ if (it) 1.0 else 0.0 })
