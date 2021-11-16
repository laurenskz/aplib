package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.algorithms.ModifiablePolicy
import eu.iv4xr.framework.model.rl.algorithms.PolicyGradientTarget
import eu.iv4xr.framework.model.rl.approximation.FeatureActionFactory
import eu.iv4xr.framework.model.rl.approximation.FeatureVectorFactory
import eu.iv4xr.framework.model.rl.approximation.MergedFeatureFactory
import eu.iv4xr.framework.model.rl.valuefunctions.*
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import org.tensorflow.*
import kotlin.math.exp

class GreedyPolicy<S : Identifiable, A : Identifiable>(val qFunction: QFunction<S, A>, val mdp: MDP<S, A>) : Policy<S, A> {

    override fun action(state: S): Distribution<A> {
        val qForActions = qFunction.qForActions(state, mdp.possibleActions(state).toList())
        val action = qForActions.maxByOrNull { it.second }
                ?: error("No element found")
        return always(action.first)
    }
}

class EGreedyPolicy<S : Identifiable, A : Identifiable>(val epsilon: Double, val mdp: MDP<S, A>, val policy: Policy<S, A>) : Policy<S, A> {
    override fun action(state: S): Distribution<A> {
        return flip(epsilon).chain {
            val possibleActions = mdp.possibleActions(state).toList()
            if (it) {
                Distributions.uniform(possibleActions.toList())
            } else {
                policy.action(state)
            }
        }
    }
}


class QFromMerged<S : Identifiable, A : Identifiable>(val factory: FeatureActionFactory<S, A>, val learningRate: Double,
                                                      val initial: Double = 0.0
                                                      ) : TrainableQFunction<S, A> {
    val value = LinearStateValueFunction(factory, learningRate,initial)
    override fun qValue(state: S, action: A): Float {
        return value.value(state to action)
    }

    override fun train(target: QTarget<S, A>) {
        value.train(Target(target.state to target.action, target.target))
    }

}

class LinearStateValueFunction<S>(
        val factory: FeatureVectorFactory<S>,
        val learningRate: Double,
        val initial: Double = 0.0
) : TrainableValuefunction<S> {
    val weights = DoubleArray(factory.count()){initial}
    override fun value(state: S): Float {
        val features = factory.arrayFeatures(state)
        var sum = 0.0
        features.indices.forEach {
            sum += features[it] * weights[it]
        }
        return sum.toFloat()
    }

    override fun train(target: Target<S>) {
        val delta = target.target - value(target.state)
        val gradient = factory.arrayFeatures(target.state)
        gradient.indices.forEach {
            weights[it] += learningRate * delta * gradient[it]
        }
    }
}

class SoftmaxPolicy<S : Identifiable, A : Identifiable>(
        val factory: FeatureActionFactory<S, A>,
        val mdp: MDP<S, A>,
        val learningRate: Double
) : ModifiablePolicy<S, A> {

    val weights = DoubleArray(factory.count())

    private fun preference(state: S, action: A): Double {
        val features = factory.arrayFeatures(state to action)
        var sum = 0.0
        repeat(factory.count()) {
            sum += features[it] * weights[it]
        }
        return sum
    }

    inline private fun gradPreference(state: S, action: A) = factory.arrayFeatures(state to action)

    override fun update(target: PolicyGradientTarget<S, A>) {
        var gradLog = gradPreference(target.s, target.a)
        val possibleActions = mdp.possibleActions(target.s).toList()
        val prob = 1.0 / possibleActions.size
        possibleActions.forEach {
            val features = factory.arrayFeatures(target.s to it)
            features.indices.forEach {
                gradLog[it] -= prob * features[it]
            }
        }
        gradLog.indices.forEach {
            weights[it] += learningRate * gradLog[it] * target.update
        }
    }

    override fun action(state: S): Distribution<A> {
        val unnormalized = mdp.possibleActions(state).associate { it to exp(preference(state, it)) }
        val sum = unnormalized.values.sum()
        return Distributions.discrete(unnormalized.mapValues { it.value / sum })
    }
}

class Test() {
    fun bab() {
        val graph = Graph()

    }
}