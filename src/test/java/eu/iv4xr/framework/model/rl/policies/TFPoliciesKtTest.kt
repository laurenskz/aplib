package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.approximation.CompositeFeature
import eu.iv4xr.framework.model.rl.approximation.DoubleFeature
import eu.iv4xr.framework.model.rl.approximation.from
import org.junit.Test
import org.junit.jupiter.api.Assertions.*
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.ndarray.NdArrays
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.types.TFloat32

data class TestState(val x: Double, val y: Double, val z: Double) : Identifiable {
    companion object {
        val factory = CompositeFeature<TestState>(listOf(
                DoubleFeature.from { it.x },
                DoubleFeature.from { it.y },
                DoubleFeature.from { it.z },
        ))
    }
}

internal class TFPoliciesKtTest {

    @Test
    fun testMSE() {
        val graph = Graph()
        var tf = Ops.create(graph)
        val x = tf.placeholder(TFloat32::class.java, Placeholder.shape(Shape.of(2, 3)))
        val y = tf.placeholder(TFloat32::class.java, Placeholder.shape(Shape.of(2, 3)))
        val model = Model(mutableMapOf(), tf, mutableListOf())
        val error = meanSquaredError(x).transform(model, y)
        val sess = Session(graph)
        val out = sess.runner()
                .feed(x, TestState.factory.createInputs(2, sequenceOf(
                        TestState(0.0, 0.0, 0.0),
                        TestState(0.0, 0.0, 0.0))))
                .feed(y, TestState.factory.createInputs(2, sequenceOf(
                        TestState(0.0, 0.0, 0.0),
                        TestState(3.0, 2.0, 1.0))))
                .fetch(error)
                .run()
                .first() as TFloat32
        out.print()
//        println(out.getFloat())
    }

    @Test
    fun testString() {
        val of = Shape.of(3,10,10)
        val ndArray = NdArrays.ofFloats(of)
        println(ndArray.prettyString())
    }
}