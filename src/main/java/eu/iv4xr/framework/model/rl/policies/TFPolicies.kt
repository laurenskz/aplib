package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions.discrete
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.algorithms.ModifiablePolicy
import eu.iv4xr.framework.model.rl.algorithms.PolicyGradientTarget
import eu.iv4xr.framework.model.rl.algorithms.RewardLogger
import eu.iv4xr.framework.model.rl.approximation.FeatureVectorFactory
import org.tensorflow.*
import org.tensorflow.framework.Summary
import org.tensorflow.framework.SummaryMetadata
import org.tensorflow.ndarray.FloatNdArray
import org.tensorflow.ndarray.NdArrays
import org.tensorflow.ndarray.Shape.*
import org.tensorflow.op.Op
import org.tensorflow.op.Ops
import org.tensorflow.op.RawOp
import org.tensorflow.op.core.*
import org.tensorflow.op.nn.BiasAdd
import org.tensorflow.op.nn.Softmax
import org.tensorflow.op.summary.*
import org.tensorflow.proto.framework.RunOptions
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.TInt64
import org.tensorflow.types.TString
import java.io.File
import java.io.FileOutputStream


fun FloatNdArray.print() {
    if (this.shape().isScalar) {
        print(this.getFloat())
        return
    }
    repeat(shape().asArray()[0].toInt()) {
        this[it.toLong()].print()
        println()
    }
}

fun TFloat32.print() {
    repeat(shape().asArray()[0].toInt()) {
        println(it)
        this[it.toLong()].print()
        println()
    }
}

class TFPolicy<State : Identifiable, A : Identifiable>(val factory: FeatureVectorFactory<State>, val mdp: MDP<State, A>, val lr: Float, val tensorboardSteps: Int = 100, val logDir: String? = null) : ModifiablePolicy<State, A> {

    val actions = mdp.allPossibleActions().toList()
    val indices = actions.mapIndexed { i, a -> a to i }.toMap()
    val sessioned = Sessioned(SoftmaxModel(factory.count().toLong(), actions.size.toLong(), Sequential()), logDir)
    var updateCounter = 0L

    override fun update(target: PolicyGradientTarget<State, A>) {
        updateAll(listOf(target))
    }

    override fun updateAll(targets: List<PolicyGradientTarget<State, A>>) {
        val input = createInputs(targets.size.toLong(), targets.asSequence().map { it.s })
        val lr = TFloat32.vectorOf(*FloatArray(targets.size) { targets[it].update.toFloat() * lr })
        val actions = TInt32.vectorOf(*IntArray(targets.size) {
            indices[targets[it].a] ?: error("Unrecognized action")
        })
        var runner = sessioned.session.runner()
                .fetch(sessioned.globalStep)
                .feed(sessioned.input(SoftmaxModel.Holders.INPUT_FEATURES), input)
                .feed(sessioned.input(SoftmaxModel.Holders.INPUT_LR), lr)
                .feed(sessioned.input(SoftmaxModel.Holders.INPUT_ACTIONS), actions)
                .addTarget(sessioned.op(SoftmaxModel.Holders.TRAIN_OP))
                .addTarget(sessioned.incGlobal)
        if (updateCounter % tensorboardSteps == 0L && sessioned.logDir != null) {
            runner.addTarget(sessioned.summary)
        }
        updateCounter = (runner.run().first() as TInt64).getLong()
    }

    private fun createInputs(count: Long, sequence: Sequence<State>): Tensor {
        val of = of(count, factory.count().toLong())
        val ndArray = NdArrays.ofFloats(of)
        sequence.forEachIndexed { i, t ->
            factory.setNdArrayFeatures(t, ndArray, longArrayOf(i.toLong()))
        }
        return TFloat32.tensorOf(ndArray)
    }

    override fun action(state: State): Distribution<A> {
        return allActions(listOf(state)).first()
    }

    override fun allActions(state: List<State>): List<Distribution<A>> {
        val input = createInputs(state.size.toLong(), state.asSequence())
        val softmaxed = sessioned.session.runner()
                .feed(sessioned.input(SoftmaxModel.Holders.INPUT_FEATURES), input)
                .fetch(sessioned.output(SoftmaxModel.Holders.SOFTMAX_OUT))
                .run().first() as TFloat32
        return (state.indices).map { stateIndex ->
            val possibleActions = mdp.possibleActions(state[stateIndex]).toSet()
            discrete(
                    actions.mapIndexed { actionIndex, a ->
                        a to softmaxed.getFloat(stateIndex.toLong(), actionIndex.toLong()).toDouble()
                    }.toMap()
            ).filter {
                it in possibleActions
            }
        }
    }
}

data class Model(val variables: MutableList<Variable<TFloat32>>, val tf: Ops, val logs: MutableList<Operand<TString>>)

interface Layer {
    fun transform(model: Model, input: Operand<TFloat32>): Operand<TFloat32>
}

class Sequential(vararg val layers: Layer) : Layer {

    override fun transform(model: Model, input: Operand<TFloat32>): Operand<TFloat32> {
        var output = input
        layers.forEach {
            output = it.transform(model, output)
        }
        return output
    }
}


fun graph(func: Model.(Operand<TFloat32>) -> Operand<TFloat32>): Layer = object : Layer {
    override fun transform(model: Model, input: Operand<TFloat32>): Operand<TFloat32> {
        return func(model, input)
    }
}

fun dense(outSize: Long) = graph {
    val raw = rawLayer(outSize).transform(this, it)
    tf.nn.relu(raw)
}

fun rawLayer(outSize: Long) = graph {
    val w = tf.variable(tf.random.randomUniform(tf.constant(of(it.shape().asArray().last(), outSize).asArray()), TFloat32::class.java))
    val b = tf.variable(tf.random.randomUniform(tf.constant(of(outSize).asArray()), TFloat32::class.java))
    variables.add(w)
    variables.add(b)
    val out = tf.linalg.matMul(it, w)
    val raw = tf.nn.biasAdd(out, b)
    raw
}

fun logGradientStep(model: Model, actionCount: Int, y: Operand<TFloat32>, lr: Operand<TFloat32>, actions: Operand<TInt32>): Merge<TFloat32> {
    val tf = model.tf
    val logSoftmax = tf.nn.logSoftmax(y)
    model.logs.add(tf.summary.histogramSummary(tf.constant("log"), logSoftmax))
    val mask = tf.oneHot(actions, tf.constant(actionCount), tf.constant(1.0f), tf.constant(0.0f))
    val toGrad = tf.math.mul(mask, logSoftmax)
    val gradients = tf.gradients(listOf(toGrad), model.variables)
    val trainSteps = model.variables.zip(gradients).map {
        val delta = tf.math.mul(lr, it.second as Operand<TFloat32>)
//        model.logs.add(tf.summary.histogramSummary(tf.constant("Gradient delta ${it.first.op().name()}"), delta))
//        model.logs.add(tf.summary.histogramSummary(tf.constant("Variable ${it.first.op().name()}"), it.first))
        tf.assignAdd(it.first, delta)
    }
    return tf.merge(trainSteps.toMutableList())
}

class TFRewardLogger(val sessioned: Sessioned, val writeEpisodes: Int = 100) : RewardLogger {
    val rewardIn = sessioned.tf.placeholder(TFloat32::class.java, Placeholder.shape(scalar()))
    val episodeIn = sessioned.tf.placeholder(TInt64::class.java, Placeholder.shape(scalar()))
    val writeSummary = WriteScalarSummary.create(sessioned.tf.scope(),
            sessioned.writer,
            episodeIn,
            sessioned.tf.constant("Reward"),
            rewardIn)

    override fun episodeReward(episode: Int, reward: Float) {
        if (episode % writeEpisodes == 0)
            sessioned.session.runner()
                    .feed(rewardIn, TFloat32.scalarOf(reward))
                    .feed(episodeIn, TInt64.scalarOf(episode.toLong()))
                    .addTarget(writeSummary)
                    .run()

    }
}

class Sessioned(builder: ModelDefBuilder, val logDir: String?) {
    val graph = Graph()
    var tf = Ops.create(graph)
    val session: Session
    val model = Model(mutableListOf(), tf, mutableListOf())
    private val signature = builder.create(model)
    val summary: Op
    val writer: SummaryWriter
    val globalStep = tf.variable(tf.constant(0L))
    val incGlobal = tf.assignAdd(globalStep, tf.constant(1L))

    init {
        session = Session(graph)
        writer = SummaryWriter.create(tf.scope())
        val create = CreateSummaryFileWriter.create(
                tf.scope(),
                writer,
                tf.constant(logDir ?: ""),
                tf.constant(0),
                tf.constant(0),
                tf.constant(".out")
        )
        graph.addInitializer(create)
        val merged = tf.summary.mergeSummary(model.logs)
        val writeSummary = WriteRawProtoSummary.create(tf.scope(),
                writer,
                globalStep,
                merged)
        summary = writeSummary
        session.run(tf.init())

    }

    fun input(any: Any): Operand<*> {
        return signature.inputs[any] ?: error("Input not found")
    }

    fun output(any: Any): Operand<*> {
        return signature.outputs[any] ?: error("Output not found")
    }

    fun op(any: Any): Op {
        return signature.methods[any] ?: error("Method not found")
    }


}

data class ModelOperations(val inputs: Map<Any, Operand<*>>, val outputs: Map<Any, Operand<*>>, val methods: Map<Any, Op>)
interface ModelDefBuilder {
    fun create(model: Model): ModelOperations
}

//class ICM()


class SoftmaxModel(val inputSize: Long, val finalOutput: Long, val layer: Layer) : ModelDefBuilder {

    enum class Holders {
        INPUT_FEATURES,
        INPUT_LR,
        INPUT_ACTIONS,
        TRAIN_OP,
        SOFTMAX_OUT
    }

    override fun create(model: Model): ModelOperations {
        val tf = model.tf
        val input = tf.placeholder(TFloat32::class.java, Placeholder.shape(of(UNKNOWN_SIZE, inputSize)))
        val lr = tf.placeholder(TFloat32::class.java, Placeholder.shape(of(UNKNOWN_SIZE)))
        val actions = tf.placeholder(TInt32::class.java, Placeholder.shape(of(UNKNOWN_SIZE)))
        val second = layer.transform(model, input)
        val raw = rawLayer(finalOutput).transform(model, second)
        model.logs.add(tf.summary.histogramSummary(tf.constant("input"), input))
        val softmaxed = tf.nn.softmax(raw)
        model.logs.add(tf.summary.histogramSummary(tf.constant("Action prob"), softmaxed))
        val trainStep = logGradientStep(model.copy(tf = tf.withSubScope("optimizer")), finalOutput.toInt(), raw, lr, actions)
        return ModelOperations(
                mapOf(Holders.INPUT_LR to lr, Holders.INPUT_FEATURES to input, Holders.INPUT_ACTIONS to actions),
                mapOf(Holders.SOFTMAX_OUT to softmaxed),
                mapOf(Holders.TRAIN_OP to trainStep)
        )
    }

}

