package eu.iv4xr.framework.model.rl.policies

import burlap.behavior.learningrate.LearningRate
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.Indexable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.algorithms.ModifiablePolicy
import eu.iv4xr.framework.model.rl.algorithms.PolicyGradientTarget
import eu.iv4xr.framework.model.rl.approximation.FeatureActionFactory
import eu.iv4xr.framework.model.rl.approximation.FeatureVectorFactory
import eu.iv4xr.framework.model.rl.valuefunctions.QFunction
import eu.iv4xr.framework.model.rl.valuefunctions.Target
import eu.iv4xr.framework.model.rl.valuefunctions.TrainableValuefunction
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


class LinearStateValueFunction<S : Identifiable, A : Identifiable>(
        val factory: FeatureVectorFactory<S>,
        val learningRate: Double
) : TrainableValuefunction<S> {
    val weights = DoubleArray(factory.count())
    override fun value(state: S): Float {
        val features = factory.features(state)
        var sum = 0.0
        features.indices.forEach {
            sum += features[it] * weights[it]
        }
        return sum.toFloat()
    }

    override fun train(target: Target<S>) {
        val delta = target.target - value(target.state)
        val gradient = factory.features(target.state)
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
        val features = factory.features(state to action)
        var sum = 0.0
        repeat(factory.count()) {
            sum += features[it] * weights[it]
        }
        return sum
    }

    override fun update(target: PolicyGradientTarget<S, A>) {
        var gradLog = factory.features(target.s to target.a)
        val policy = action(target.s)
        policy.support().forEach {
            val score = policy.score(it)
            val features = factory.features(target.s to it)
            features.indices.forEach {
                gradLog[it] -= score * features[it]
            }
        }
        gradLog.indices.forEach {
            weights[it] += learningRate * gradLog[it]
        }
    }

    override fun action(state: S): Distribution<A> {
        val unnormalized = mdp.possibleActions(state).associate { it to exp(preference(state, it)) }
        val sum = unnormalized.values.sum()
        return Distributions.discrete(unnormalized.mapValues { it.value / sum })
    }
}