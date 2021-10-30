package eu.iv4xr.framework.model.rl.policies

import org.junit.Test
import org.tensorflow.ndarray.Shape
import org.tensorflow.types.TFloat32




internal class TFModelTest {

    @Test
    fun test() {
        val input = StateTest(1.0, 0.5, 1.3)
        val input2 = StateTest(1.7, 0.5, 1.3)
        val tensor = TFloat32.tensorOf(Shape.of(2, 3))
        val tensorFeatures = StateTest.factory.setTensorFeatures(input, tensor, longArrayOf(0))
//        StateTest.factory.setTensorFeatures(input2, tensor, longArrayOf(1))
//        val sessioned = Sessioned {
//            SoftmaxModel(it, 3, 3) { it }
//        }
//        val result = sessioned.session.runner()
//                .feed(sessioned.t.input, tensorFeatures)
//                .fetch(sessioned.t.softmaxed)
//                .run()
//                .first() as TFloat32
////        println(result.shape())
////        result.print()
//        tensorFeatures.print()

    }
}