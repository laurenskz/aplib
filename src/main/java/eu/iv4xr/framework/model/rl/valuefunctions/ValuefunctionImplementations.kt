package eu.iv4xr.framework.model.rl.valuefunctions

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.algorithms.ModelDescription
import eu.iv4xr.framework.model.rl.approximation.FeatureActionFactory
import eu.iv4xr.framework.model.rl.approximation.FeatureVectorFactory
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassAction
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassHashableState
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset

class StateValueFunction2<S : Identifiable>(val model: ModelDescription, val features: FeatureVectorFactory<S>) : TrainableValuefunction<S> {


    val modelInstance = model.create(features.count(), 1)

    override fun value(state: S): Float {
        return modelInstance.predictSoftly(features.floatFeatures(state))[0]
    }

    override fun values(states: List<S>): List<Float> {
        val input = Array(states.size) { features.floatFeatures(states[it]) }
        return modelInstance.predictSoftly(OnHeapDataset.create(input, FloatArray(input.size)), states.size).map { it[0] }
    }

    override fun train(target: Target<S>) {
        modelInstance.fit(OnHeapDataset.Companion.create(arrayOf(features.floatFeatures(target.state)), floatArrayOf(target.target)))
    }

    override fun train(targets: List<Target<S>>) {
        val input = Array(targets.size) { features.floatFeatures(targets[it].state) }
        val labels = FloatArray(targets.size) { targets[it].target }
        val bab = modelInstance.fit(OnHeapDataset.Companion.create(input, labels), epochs = 1000)
        println(bab.epochHistory.last())
    }
}

class DownSampledValueFunction<S : Identifiable, R : Identifiable>(val valuefunction: TrainableValuefunction<R>, val downSampler: (S) -> R) : TrainableValuefunction<S> {
    override fun value(state: S): Float {
        return valuefunction.value(downSampler(state))
    }

    override fun train(target: Target<S>) {
        valuefunction.train(Target(downSampler(target.state), target.target))
    }
}

class QTable<S : DataClassHashableState, A : DataClassAction>(val learningRate: Float) : TrainableQFunction<S, A> {
    val qValues = mutableMapOf<Pair<S, A>, Float>()
    override fun qValue(state: S, action: A): Float {
        return qValues[state to action] ?: 0f
    }

    override fun train(target: QTarget<S, A>) {
        val current = qValue(target.state, target.action)
        qValues[target.state to target.action] = current + learningRate * (target.target - current)
    }
}

class ValueTable<S : DataClassHashableState>(val learningRate: Float, val initial: Float = 0f) : TrainableValuefunction<S> {
    private val values = mutableMapOf<S, Float>()
    val targets: List<Target<S>>
        get() = values.map { Target(it.key, it.value) }
    val states: List<S>
        get() = values.keys.toList()

    override fun value(state: S): Float {
        return values[state] ?: initial
    }

    override fun train(target: Target<S>) {
        val current = value(target.state)
        values[target.state] = current + learningRate * (target.target - current)
    }

}

class MergedValueFunction<S : DataClassHashableState>(table: ValueTable<S>)

class StateActionInputQModel<S : Identifiable, A : Identifiable>(val model: ModelDescription, val featureActionFactory: FeatureActionFactory<S, A>) : TrainableQFunction<S, A> {

    val modelInstance = model.create(featureActionFactory.count(), 1)

    fun create(): GraphTrainableModel {
        return model.create(featureActionFactory.count(), 1)
    }

    override fun qValue(state: S, action: A): Float {
        return modelInstance.predictSoftly(featureActionFactory.floatFeatures(state to action))[0]
    }

    override fun qValues(states: List<Pair<S, A>>): List<Float> {
        val input = Array(states.size) { featureActionFactory.floatFeatures(states[it]) }
        return modelInstance.predictSoftly(OnHeapDataset.create(input, FloatArray(input.size)), states.size).map { it[0] }
    }

    override fun train(target: QTarget<S, A>) {
        modelInstance.fit(OnHeapDataset.create(arrayOf(featureActionFactory.floatFeatures(target.state to target.action)), floatArrayOf(target.target)))
    }

    override fun train(targets: List<QTarget<S, A>>) {
        val input = Array(targets.size) { featureActionFactory.floatFeatures(targets[it].state to targets[it].action) }
        val labels = FloatArray(targets.size) { targets[it].target }
        modelInstance.fit(OnHeapDataset.Companion.create(input, labels))
    }
}