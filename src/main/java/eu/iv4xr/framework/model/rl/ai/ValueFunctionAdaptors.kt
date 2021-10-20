package eu.iv4xr.framework.model.rl.ai

import burlap.behavior.functionapproximation.DifferentiableStateActionValue
import burlap.behavior.functionapproximation.FunctionGradient
import burlap.behavior.functionapproximation.ParametricFunction
import burlap.behavior.functionapproximation.dense.DenseStateFeatures
import burlap.behavior.functionapproximation.supervised.SupervisedVFA
import burlap.behavior.valuefunction.ValueFunction
import burlap.mdp.core.action.Action
import burlap.mdp.core.state.State
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.approximation.FeatureActionFactory
import eu.iv4xr.framework.model.rl.approximation.FeatureVectorFactory
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.ClipGradientByValue
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.mnist
import org.nd4j.linalg.factory.Nd4j

class Neural(val features: (State, Action) -> DoubleArray, val network: MultiLayerNetwork) : DifferentiableStateActionValue {

    fun ba() {
        val build = NeuralNetConfiguration.Builder().build()
        val network = MultiLayerNetwork(MultiLayerConfiguration())
    }

    override fun numParameters(): Int {
        return network.numParams().toInt()
    }

    override fun getParameter(p0: Int): Double {
        return network.params().getDouble(p0);
    }

    override fun setParameter(p0: Int, p1: Double) {
        network.params().putScalar(p0.toLong(), p1)
    }

    override fun resetParameters() {
        TODO("Not yet implemented")
    }

    override fun copy() = Neural(features, network.clone())

    override fun evaluate(p0: State, p1: Action): Double {
        val input = features(p0, p1)
        val ndArray = Nd4j.create(input, intArrayOf(1, input.size))
        return network.output(ndArray).toDoubleVector()[0]
    }

    override fun gradient(p0: State?, p1: Action?): FunctionGradient {
        network.flattenedGradients
        TODO()
    }
}


interface KerasModel {
    fun description(inputSize: Int): GraphTrainableModel
    fun init(model: GraphTrainableModel)
}

@Suppress("UNCHECKED_CAST")
class TensorflowModelTrainer<S : Identifiable>(val model: KerasModel, val features: FeatureVectorFactory<S>) : SupervisedVFA {

    override fun train(p0: MutableList<SupervisedVFA.SupervisedVFAInstance>): ValueFunction {
        val input = p0.map { features.floatFeatures(it.s as S) }.toTypedArray()
        val labels = FloatArray(p0.size) { p0[it].v.toFloat() }
        val dataset = OnHeapDataset.create(input, labels)
        return model.description(features.count()).let {
            model.init(it)
            while (it.evaluate(dataset, Metrics.MAE) > 0.01) {
                it.fit(dataset, epochs = 100)
            }
            TensorflowValueFunction(it, features)
        }
    }
}

@Suppress("UNCHECKED_CAST")
class TensorflowValueFunction<S : Identifiable>(val model: InferenceModel, val features: FeatureVectorFactory<S>) : ValueFunction {
    override fun value(p0: State): Double {
        return model.predictSoftly(features.floatFeatures(p0 as S))[0].toDouble()
    }
}

class BasicModel : KerasModel {
    override fun description(inputSize: Int) = Sequential.of(
            Input(inputSize.toLong()),
            Dense(48, activation = Activations.Relu,
                    kernelInitializer = HeNormal()),
            Dense(24, activation = Activations.Relu,
                    kernelInitializer = HeNormal()),
            Dense(12, activation = Activations.Relu,
                    kernelInitializer = HeNormal()),
//            Dense(4, activation = Activations.Relu,
//                    kernelInitializer = GlorotNormal(),
//                    biasInitializer = GlorotNormal()),
//            Dense(4, activation = Activations.Relu,
//                    kernelInitializer = GlorotNormal(),
//                    biasInitializer = GlorotNormal()),
//            Dense(20, activation = Activations.Tanh,
//                    kernelInitializer = GlorotNormal(),
//                    biasInitializer = GlorotNormal()),
//            Dense(20, activation = Activations.Tanh,
//                    kernelInitializer = GlorotNormal(),
//                    biasInitializer = GlorotNormal()),
            Dense(
                    1,
                    activation = Activations.Linear,
                    kernelInitializer = HeNormal()
            )
    )

    override fun init(model: GraphTrainableModel) {
        model.compile(
                optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
                loss = Losses.HUBER,
                metric = Metrics.ACCURACY
        )
        model.init()
    }

}