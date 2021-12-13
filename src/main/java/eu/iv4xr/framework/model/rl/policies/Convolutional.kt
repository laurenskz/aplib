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
        return qValues(listOf(state to action)).first()
    }

    override fun qValues(states: List<Pair<S, A>>): List<Float> {
        val indices = states.map { actions[it.second] ?: error("Unrecognized action:${it.second}") }
        val input = states.map { it.first }
        val output = sessioned.session.runner()
                .feed(sessioned.input(TFQOperations.STATE_INPUT), factory.createFrom(input))
                .fetch(sessioned.output(TFQOperations.OUTPUT))
                .run()
                .first()
                as TFloat32
        return input.indices.map { output.getFloat(it.toLong(), indices[it].toLong()) }
    }

    override fun train(targets: List<QTarget<S, A>>) {
        val indices = targets.map { actions[it.action] ?: error("Unrecognized action:${it.action}") }
        val inputs = targets.map { it.state }
        sessioned.session.runner()
                .feed(sessioned.input(TFQOperations.STATE_INPUT), factory.createFrom(inputs))
                .feed(sessioned.input(TFQOperations.ACTION_INDEX), TInt32.vectorOf(*indices.toIntArray()))
                .feed(sessioned.input(TFQOperations.TARGETS), TFloat32.vectorOf(*FloatArray(targets.size) { targets[it].target }))
                .addTarget(sessioned.op(TFQOperations.TRAIN))
                .run()

    }

    override fun train(target: QTarget<S, A>) {
        train(listOf(target))
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
        val maskedTargets = model.tf.math.mul(model.tf.expandDims(targets, model.tf.constant(-1)), mask)
        val square = model.tf.math.square(model.tf.math.sub(maskedTargets, qs))
        val maskedLoss = model.tf.math.mul(square, mask)
        println(maskedTargets.shape())
        val loss = model.tf.math.mean(model.tf.math.mean(maskedLoss, model.tf.constant(1)), model.tf.constant(0))
        println(loss.shape())
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