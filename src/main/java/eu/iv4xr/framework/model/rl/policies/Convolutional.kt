package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.approximation.TensorFactory
import eu.iv4xr.framework.model.rl.valuefunctions.QTarget
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableQFunction
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.Shape.UNKNOWN_SIZE
import org.tensorflow.ndarray.Shape.of
import org.tensorflow.op.core.Placeholder
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32


enum class TFQOperations {
    STATE_INPUT, ACTION_INDEX, OUTPUT, TRAIN, TARGETS
}

class TFQFunction<S : Identifiable, A : Identifiable>(val factory: TensorFactory<S>, val mdp: MDP<S, A>, val modelDefBuilder: (Int) -> ModelDefBuilder) : TrainableQFunction<S, A> {

    val actions = mdp.allPossibleActions().toList().mapIndexed { i, a -> a to i }.toMap()
    val sessioned = Sessioned(modelDefBuilder(actions.size), null)

    override fun qValue(state: S, action: A): Float {
        val index = actions[action]?.toLong() ?: error("Action not recognized")
        return (sessioned.session.runner()
                .feed(sessioned.input(TFQOperations.STATE_INPUT), factory.createFrom(state))
                .fetch(sessioned.output(TFQOperations.OUTPUT))
                .run()
                .first()
                as TFloat32)
                .getFloat(0, index)
    }

    override fun train(target: QTarget<S, A>) {
        val index = actions[target.action] ?: error("Action not recognized")
        sessioned.session.runner()
                .feed(sessioned.input(TFQOperations.STATE_INPUT), factory.createFrom(target.state))
                .feed(sessioned.input(TFQOperations.ACTION_INDEX), TInt32.vectorOf(index))
                .feed(sessioned.input(TFQOperations.TARGETS), TFloat32.vectorOf(target.target))
                .addTarget(sessioned.op(TFQOperations.TRAIN))
                .run()
    }
}

class QModelDefBuilder(val processing: Layer, val inShape: Shape, val outSize: Int, val lr: Float) : ModelDefBuilder {
    override fun create(model: Model): ModelOperations {
        val input = model.tf.placeholder(TFloat32::class.java, Placeholder.shape(inShape))
        val output = processing.transform(model, input)
        val actions = model.tf.placeholder(TInt32::class.java, Placeholder.shape(of(UNKNOWN_SIZE)))
        val targets = model.tf.placeholder(TFloat32::class.java, Placeholder.shape(of(UNKNOWN_SIZE)))
        val qs = rawLayer(outSize.toLong()).transform(model, output)
        val mask = model.tf.oneHot(actions, model.tf.constant(outSize), model.tf.constant(1f), model.tf.constant(0f))
        val maskedTargets = model.tf.math.mul(targets, mask)
        val loss = model.tf.math.mean(model.tf.math.square(model.tf.math.mul(model.tf.math.sub(maskedTargets, qs), mask)), model.tf.constant(1))
        val trainStep = optimize(model.withScope("Optimize"), loss, model.tf.constant(lr), add = false)
        return ModelOperations(
                mapOf(TFQOperations.TARGETS to targets, TFQOperations.ACTION_INDEX to actions, TFQOperations.STATE_INPUT to input),
                mapOf(TFQOperations.OUTPUT to qs),
                mapOf(TFQOperations.TRAIN to trainStep)
        )
    }

}

class Convolutional(val inputSize: LongArray) : ModelDefBuilder {
    override fun create(model: Model): ModelOperations {
        val tf = model.tf
        val inputNode = tf.placeholder(TFloat32::class.java, Placeholder.shape(Shape.of(UNKNOWN_SIZE, *inputSize)))
//        tf.nn.conv2d(inputNode, ,listOf(1L, 1L), "same")
        TODO()
    }
}