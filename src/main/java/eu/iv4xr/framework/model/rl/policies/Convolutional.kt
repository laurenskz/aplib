package eu.iv4xr.framework.model.rl.policies

import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.Shape.UNKNOWN_SIZE
import org.tensorflow.op.core.Placeholder
import org.tensorflow.types.TFloat32

class Convolutional(val inputSize: LongArray) : ModelDefBuilder {
    override fun create(model: Model): ModelOperations {
        val tf = model.tf
        val inputNode = tf.placeholder(TFloat32::class.java, Placeholder.shape(Shape.of(UNKNOWN_SIZE, *inputSize)))
//        tf.nn.conv2d(inputNode, ,listOf(1L, 1L), "same")
        TODO()
    }
}