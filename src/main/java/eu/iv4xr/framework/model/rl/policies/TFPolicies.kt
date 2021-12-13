package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions.discrete
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.algorithms.*
import eu.iv4xr.framework.model.rl.approximation.FeatureVectorFactory
import eu.iv4xr.framework.model.rl.valuefunctions.QTarget
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableQFunction
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableValuefunction
import org.tensorflow.*
import org.tensorflow.ndarray.FloatNdArray
import org.tensorflow.ndarray.NdArrays
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.Shape.*
import org.tensorflow.op.Op
import org.tensorflow.op.Ops
import org.tensorflow.op.core.*
import org.tensorflow.op.random.RandomStandardNormal
import org.tensorflow.op.summary.*
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.TInt64
import org.tensorflow.types.TString
import java.io.FileOutputStream
import kotlin.math.sqrt


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

fun FloatNdArray.prettyString(): String {
    if (shape().asArray().isEmpty()) return this.getFloat().toString()
    if (shape().asArray().size == 1) {
        return (0 until shape().asArray().first()).map { this.getFloat(it) }.toString()
    }
    val smaller = (
            (0 until shape().asArray().first()).flatMap { this[it].prettyString().lines().map { "\t" + it } }
            ).joinToString("\n")
    return "[\n" + smaller + "\n]"
}

fun TFloat32.print() {

    repeat(shape().asArray()[0].toInt()) {
        this[it.toLong()].print()
        println()
    }
}

fun <S : Identifiable> FeatureVectorFactory<S>.createInputs(count: Long, sequence: Sequence<S>): Tensor {
    val of = of(count, count().toLong())
    val ndArray = NdArrays.ofFloats(of)
    sequence.forEachIndexed { i, t ->
        setNdArrayFeatures(t, ndArray, longArrayOf(i.toLong()))
    }
    return TFloat32.tensorOf(ndArray)
}

class CountBasedICMModule<S : Identifiable, A : Identifiable>(
        val valueFunction: TrainableValuefunction<S>,
        val countFunction: (Double) -> Double
) : ICMModule<S, A> {
    val counts = mutableMapOf<Pair<S, A>, Int>()

    override fun intrinsicReward(sars: ICMSample<S, A>): Double {
        return valueFunction.value(sars.statePrime).let { countFunction(it.toDouble()) }
    }

    private fun trainSample(sample: ICMSample<S, A>): Double {
        val current = valueFunction.value(sample.statePrime)
        valueFunction.train(Target(sample.statePrime, current + 1))
        return countFunction(current.toDouble())
    }

    override fun train(sars: List<ICMSample<S, A>>): List<Double> {
        return sars.map { trainSample(it) }
    }

}

class QCountBasedICMModule<S : Identifiable, A : Identifiable>(
        val valueFunction: TrainableQFunction<S, A>,
        val countFunction: (Double) -> Double
) : ICMModule<S, A> {
    val counts = mutableMapOf<Pair<S, A>, Int>()

    override fun intrinsicReward(sars: ICMSample<S, A>): Double {
        return valueFunction.qValue(sars.state, sars.action).let { countFunction(it.toDouble()) }
    }

    private fun trainSample(sample: ICMSample<S, A>): Double {
        val current = valueFunction.qValue(sample.state, sample.action)
        valueFunction.train(QTarget(sample.state, sample.action, current + 1))
        return countFunction(current.toDouble())
    }

    override fun train(sars: List<ICMSample<S, A>>): List<Double> {
        return sars.map { trainSample(it) }
    }

}

class ICMModuleImpl<S : Identifiable, A : Identifiable>(
        val model: ICMModel,
        val stateEncoder: FeatureVectorFactory<S>,
        val actionEncoder: FeatureVectorFactory<A>,
        val logDir: String? = null
) : ICMModule<S, A> {
    val sessioned = Sessioned(model, logDir)
    override fun intrinsicReward(sars: List<ICMSample<S, A>>): List<Double> {
        val output = feedInput(sars)
                .fetch(sessioned.output(ICMHolders.STATE_LOSS))
                .run()
                .first() as TFloat32
        return sars.indices.map { output.getFloat(it.toLong()).toDouble() }
    }

    override fun train(sars: List<ICMSample<S, A>>): List<Double> {
        val output = feedInput(sars)
                .addTarget(sessioned.op(ICMHolders.TRAIN_STEP))
                .fetch(sessioned.output(ICMHolders.STATE_LOSS))
                .addTarget(sessioned.summary)
                .addTarget(sessioned.incGlobal)
                .run()
                .first() as TFloat32
        return sars.indices.map { output.getFloat(it.toLong()).toDouble() }
    }

    private fun feedInput(sars: List<ICMSample<S, A>>): Session.Runner {
        return sessioned.session.runner()
                .feed(sessioned.input(ICMHolders.ACTION), actionEncoder.createInputs(sars.size.toLong(), sars.asSequence().map { it.action }))
                .feed(sessioned.input(ICMHolders.STATE_PRIME), stateEncoder.createInputs(sars.size.toLong(), sars.asSequence().map { it.statePrime }))
                .feed(sessioned.input(ICMHolders.STATE), stateEncoder.createInputs(sars.size.toLong(), sars.asSequence().map { it.state }))
    }

    override fun intrinsicReward(sars: ICMSample<S, A>): Double {
        return intrinsicReward(listOf(sars)).first()
    }
}

class TFPolicy<State : Identifiable, A : Identifiable>(val factory: FeatureVectorFactory<State>, val mdp: MDP<State, A>, val lr: Float, val tensorboardSteps: Int = 100, val logDir: String? = null, val batchSize: Int = 128) : ModifiablePolicy<State, A> {

    val actions = mdp.allPossibleActions().toList()
    val indices = actions.mapIndexed { i, a -> a to i }.toMap()
    val sessioned = Sessioned(SoftmaxModel(factory.count().toLong(), actions.size.toLong(), Sequential(), batchSize), logDir)
    var updateCounter = 0L

    override fun update(target: PolicyGradientTarget<State, A>) {
        updateAll(listOf(target))
    }

    override fun updateAll(targets: List<PolicyGradientTarget<State, A>>) {
        val input = factory.createInputs(targets.size.toLong(), targets.asSequence().map { it.s })
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
                .addTarget(sessioned.summary)
        updateCounter = (runner.run().first() as TInt64).getLong()
    }

    override fun action(state: State): Distribution<A> {
        return allActions(listOf(state)).first()
    }

    override fun allActions(state: List<State>): List<Distribution<A>> {
        val input = factory.createInputs(state.size.toLong(), state.asSequence())
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

fun TFloat32.toNdArray() = NdArrays.ofFloats(shape()).also { this.copyTo(it) }

data class Model(val variables: MutableMap<String, Variable<TFloat32>>, val tf: Ops, val logs: MutableList<Operand<TString>>)

interface Layer {
    fun transform(model: Model, input: Operand<TFloat32>): Operand<TFloat32>
    fun createConcrete(input: Shape): Sessioned {
        return Sessioned(object : ModelDefBuilder {

            override fun create(model: Model): ModelOperations {
                val placeholder = model.tf.placeholder(TFloat32::class.java, Placeholder.shape(input))
                val out = transform(model, placeholder)
                return ModelOperations(
                        mapOf("in" to placeholder), mapOf("out" to out), mapOf()
                )
            }
        }, logDir = null)
    }
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
    tf.math.sigmoid(raw)
}

fun dense(outSize: Long, name: String) = graph {
    val scope = tf.withSubScope("dense")
    val raw = rawLayer(outSize, name).transform(this.copy(tf = scope), it)
    scope.nn.relu(raw)
}

fun convLayer(width: Long, height: Long, filterCount: Long, name: String) = graph {
    val w = variables.getOrPut("${name}_filters") {
        val shape = longArrayOf(height, width, it.shape().size(-1), filterCount)
        val n = shape.fold(1, Long::times)
        val randomStandardNormal = tf.random.randomStandardNormal(tf.constant(shape), TFloat32::class.java)
        tf.variable(tf.math.mul(randomStandardNormal, tf.constant(sqrt(2f / n))))
    }
    val conv = tf.nn.conv2d(it, w, listOf(1L, 1L, 1L, 1L), "VALID")
    tf.nn.relu(conv)
}

fun maxPoolLayer(width: Int, height: Int, stride: Int) = graph {
    tf.nn.maxPool(it, tf.constant(intArrayOf(1, width, height, 1)), tf.constant(intArrayOf(1, stride, stride, 1)), "VALID")
}

fun flatten() = graph {
    val total = it.shape().tail().asArray().fold(1, Long::times)
    tf.reshape(it, tf.constant(longArrayOf(UNKNOWN_SIZE, total)))
}

fun rawLayer(outSize: Long) = graph {
    val keys = variables.keys
    val name = generateSequence(0) { it + 1 }.first { !keys.contains("layer:${it}_w") }
    rawLayer(outSize, "layer:$name").transform(this, it)

}

fun rawLayer(outSize: Long, name: String) = graph {
    val tfp = tf.withSubScope("variables")
    val w = variables.getOrPut("${name}_w") {
        tfp.variable(tfp.random.randomUniform(tfp.constant(of(it.shape().asArray().last(), outSize).asArray()), TFloat32::class.java))
    }
    val b = variables.getOrPut("${name}_b") {
        tfp.variable(tfp.random.randomUniform(tfp.constant(of(outSize).asArray()), TFloat32::class.java))
    }
    val out = tf.linalg.matMul(it, w)
    val raw = tf.nn.biasAdd(out, b)
    raw
}

fun meanSquaredError(x: Operand<TFloat32>) = graph {
    val squaredDiff = tf.math.square(tf.math.sub(x, it))
    tf.math.mean(squaredDiff, tf.constant(1))
}


fun optimize(model: Model, loss: Operand<TFloat32>, lr: Operand<TFloat32>, add: Boolean = true): Op {
    val tf = model.tf
    val variables = model.variables.values.toList()
    val gradients = tf.gradients(listOf(loss), variables)
    val trainSteps = variables.zip(gradients).map {
        val delta = tf.math.mul(lr, it.second as Operand<TFloat32>)
//        model.logs.add(tf.summary.histogramSummary(tf.constant("Gradient delta ${it.first.op().name()}"), delta))
//        model.logs.add(tf.summary.histogramSummary(tf.constant("Variable ${it.first.op().name()}"), it.first))
        if (add)
            tf.assignAdd(it.first, delta)
        else
            tf.assignSub(it.first, delta)
    }
//    return tf.noOp()
    return tf.merge(trainSteps.toMutableList())
}

fun logGradientStep(model: Model, actionCount: Int, logits: Operand<TFloat32>, lr: Operand<TFloat32>, actions: Operand<TInt32>, batchSize: Int): Merge<TFloat32> {
    val tf = model.tf
    val logSoftmax = tf.nn.logSoftmax(logits)
    val mask = tf.oneHot(actions, tf.constant(actionCount), tf.constant(1.0f), tf.constant(0.0f))
    val toGrad = tf.math.mul(mask, logSoftmax)
    val variables = model.variables.values.toList()
    val trainSteps = variables.map { variable ->
        val gradient: Operand<TFloat32> = tf.stack(tf.unstack(toGrad, batchSize.toLong()).map { sample ->
            tf.gradients(sample, listOf(variable)).first() as Operand<TFloat32>
        })
        var lrp = lr
        while (lrp.rank() < gradient.rank()) {
            lrp = tf.expandDims(lrp, tf.constant(-1))
        }
        val updates = tf.math.mul(lrp, gradient)
        val reduced = tf.reduceSum(updates, tf.constant(0))
        tf.assignAdd(variable, reduced)


    }
    return tf.merge(trainSteps.toMutableList())
}

class TFRewardLogger(val sessioned: Sessioned, val writeEpisodes: Int = 100) : CuriosityRewardLogger {
    val rewardIn = sessioned.tf.placeholder(TFloat32::class.java, Placeholder.shape(scalar()))
    val intrinsicRewardIn = sessioned.tf.placeholder(TFloat32::class.java, Placeholder.shape(scalar()))
    val extrinsicRewardIn = sessioned.tf.placeholder(TFloat32::class.java, Placeholder.shape(scalar()))
    val episodeIn = sessioned.tf.placeholder(TInt64::class.java, Placeholder.shape(scalar()))
    var lastWrite = 0
    var lastWriteI = 0
    var lastWriteE = 0
    val writeSummary = WriteScalarSummary.create(sessioned.tf.scope(),
            sessioned.writer,
            episodeIn,
            sessioned.tf.constant("Reward"),
            rewardIn)
    val intrinsicWriteSummary = WriteScalarSummary.create(sessioned.tf.scope(),
            sessioned.writer,
            episodeIn,
            sessioned.tf.constant("Intrinsic reward"),
            intrinsicRewardIn)
    val extrinsicWriteSummary = WriteScalarSummary.create(sessioned.tf.scope(),
            sessioned.writer,
            episodeIn,
            sessioned.tf.constant("Extrinsic reward"),
            extrinsicRewardIn)

    override fun episodeReward(episode: Int, reward: Float) {
        if (episode - lastWrite > writeEpisodes) {
            sessioned.session.runner()
                    .feed(rewardIn, TFloat32.scalarOf(reward))
                    .feed(episodeIn, TInt64.scalarOf(episode.toLong()))
                    .addTarget(writeSummary)
                    .run()
            lastWrite = episode
        }

    }

    override fun intrinsicEpisodeReward(episode: Int, reward: Float) {
        if (episode - lastWriteI > writeEpisodes) {
            sessioned.session.runner()
                    .feed(intrinsicRewardIn, TFloat32.scalarOf(reward))
                    .feed(episodeIn, TInt64.scalarOf(episode.toLong()))
                    .addTarget(intrinsicWriteSummary)
                    .run()
            lastWriteI = episode
        }
    }

    override fun extrinsicEpisodeReward(episode: Int, reward: Float) {
        if (episode - lastWriteE > writeEpisodes) {
            sessioned.session.runner()
                    .feed(extrinsicRewardIn, TFloat32.scalarOf(reward))
                    .feed(episodeIn, TInt64.scalarOf(episode.toLong()))
                    .addTarget(extrinsicWriteSummary)
                    .run()
            lastWriteE = episode
        }
    }
}


class Sessioned(builder: ModelDefBuilder, val logDir: String?) {
    val graph = Graph()
    var tf = Ops.create(graph)
    val session: Session
    val model = Model(mutableMapOf(), tf, mutableListOf())
    private val signature = builder.create(model)
    val summary: Op
    val writer: SummaryWriter
    val globalStep = tf.variable(tf.constant(0L))
    val incGlobal = tf.assignAdd(globalStep, tf.constant(1L))

    init {
        session = Session(graph)
        writer = SummaryWriter.create(tf.scope())
        if (model.logs.isNotEmpty() && logDir != null) {
            val create = CreateSummaryFileWriter.create(
                    tf.scope(),
                    writer,
                    tf.constant(logDir ?: ""),
                    tf.constant(0),
                    tf.constant(0),
                    tf.constant(".out")
            )
            graph.registerInitOp(create.op())
            val merged = tf.summary.mergeSummary(model.logs)
            val writeSummary = WriteRawProtoSummary.create(tf.scope(),
                    writer,
                    globalStep,
                    merged)
            summary = writeSummary
        } else {
            summary = tf.noOp()
        }
        val runner = session.runner()
        graph.initializers().forEach { runner.addTarget(it) }
        runner.run()
        logDir?.also { FileOutputStream("$it/graph.pbtxt").writer().write(graph.toGraphDef().toString()) }
    }

    fun run(input: TFloat32) = session.runner()
            .feed(signature.inputs.values.first(), input)
            .fetch(signature.outputs.values.first())
            .run().first()

    fun input(any: Any): Operand<*> {
        return signature.inputs[any] ?: error("Input not found:$any")
    }

    fun output(any: Any): Operand<*> {
        return signature.outputs[any] ?: error("Output not found:$any")
    }

    fun op(any: Any): Op {
        return signature.methods[any] ?: error("Method not found:$any")
    }


}

data class ModelOperations(val inputs: Map<Any, Operand<*>>, val outputs: Map<Any, Operand<*>>, val methods: Map<Any, Op>)


interface ModelDefBuilder {
    fun create(model: Model): ModelOperations
}

enum class ICMHolders {
    STATE, STATE_PRIME, ACTION, STATE_LOSS, TRAIN_STEP
}

fun Model.withScope(name: String) =
        this.copy(tf = tf.withSubScope(name))

class ICMModel(val beta: Double,
               val lr: Double,
               val stateSize: Int,
               val actionSize: Int,
               val stateEncodingNetwork: Layer,
               val statePrimeNetwork: Layer,
               val predictActionNetwork: Layer

) : ModelDefBuilder {

    override fun create(model: Model): ModelOperations {
        val tf = model.tf
        val state = tf.placeholder(TFloat32::class.java, Placeholder.shape(of(UNKNOWN_SIZE, stateSize.toLong())))
        val statePrime = tf.placeholder(TFloat32::class.java, Placeholder.shape(of(UNKNOWN_SIZE, stateSize.toLong())))
        val action = tf.placeholder(TFloat32::class.java, Placeholder.shape(of(UNKNOWN_SIZE, actionSize.toLong())))
        val stateFeatures = stateEncodingNetwork.transform(model.withScope("encodeState"), state)
        val statePrimeFeatures = stateEncodingNetwork.transform(model.withScope("encodeStatePrime"), statePrime)
        val forwardInput = tf.concat(listOf(stateFeatures, action), tf.constant(-1))
        val statePrimePrediction = statePrimeNetwork.transform(model.withScope("predictStatePrime"), forwardInput)
        val actionPredictionInput = tf.concat(listOf(stateFeatures, statePrimeFeatures), tf.constant(-1))
        model.logs.add(tf.summary.histogramSummary(tf.constant("Action input"), actionPredictionInput))
        val batchStateLoss = meanSquaredError(statePrimeFeatures).transform(model.withScope("stateLoss"), statePrimePrediction)
        val stateLoss = tf.math.mean(batchStateLoss, tf.constant(0))
        val predictedAction = predictActionNetwork.transform(model.withScope("predictAction"), actionPredictionInput)
        model.logs.add(tf.summary.histogramSummary(tf.constant("Action logits"), predictedAction))
        val batchedActionLoss = tf.withSubScope("actionLoss").nn.softmaxCrossEntropyWithLogits(predictedAction, action).loss()
        val actionLoss = tf.math.mean(batchedActionLoss, tf.constant(0))
        model.logs.add(tf.summary.scalarSummary(tf.constant("Action loss"), actionLoss))
        model.logs.add(tf.summary.scalarSummary(tf.constant("State loss"), stateLoss))
        val totalLoss = createLoss(model.tf.withSubScope("totalLoss"), actionLoss, stateLoss)
        val trainStep = optimize(model.withScope("Optimize"), totalLoss, tf.constant(lr.toFloat()), add = false)
        return ModelOperations(mapOf(
                ICMHolders.STATE to state, ICMHolders.STATE_PRIME to statePrime, ICMHolders.ACTION to action),
                mapOf(ICMHolders.STATE_LOSS to batchStateLoss),
                mapOf(ICMHolders.TRAIN_STEP to trainStep))
    }

    fun createLoss(tf: Ops, actionLoss: Operand<TFloat32>, stateLoss: Operand<TFloat32>): Operand<TFloat32> {
        val weightedAction = tf.math.mul(actionLoss, tf.constant(1f - beta.toFloat()))
        val weightedState = tf.math.mul(stateLoss, tf.constant(beta.toFloat()))
        return tf.math.add(weightedAction, weightedState)
    }

}


class SoftmaxModel(val inputSize: Long, val finalOutput: Long, val layer: Layer, val batchSize: Int) : ModelDefBuilder {

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
        model.logs.add(tf.summary.histogramSummary(tf.constant("Logits"), raw))
        model.logs.add(tf.summary.histogramSummary(tf.constant("input"), input))
        val softmaxed = tf.nn.softmax(raw)
        model.logs.add(tf.summary.histogramSummary(tf.constant("Action prob"), softmaxed))
        val trainStep = logGradientStep(model.copy(tf = tf.withSubScope("optimizer")), finalOutput.toInt(), raw, lr, actions, batchSize)
        return ModelOperations(
                mapOf(Holders.INPUT_LR to lr, Holders.INPUT_FEATURES to input, Holders.INPUT_ACTIONS to actions),
                mapOf(Holders.SOFTMAX_OUT to softmaxed),
                mapOf(Holders.TRAIN_OP to trainStep)
        )
    }

}


fun main() {
    val model = ICMModel(0.7, 0.1, 10, 2,
            Sequential(dense(5, "state1")),
            Sequential(dense(5)),
            Sequential(dense(2))
    )
    Sessioned(model, "icm")
}
