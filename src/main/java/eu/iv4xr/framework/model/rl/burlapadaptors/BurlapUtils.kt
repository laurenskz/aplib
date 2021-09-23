package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.behavior.functionapproximation.dense.DenseStateActionFeatures
import burlap.behavior.functionapproximation.dense.DenseStateFeatures
import burlap.behavior.functionapproximation.sparse.SparseStateFeatures
import burlap.behavior.functionapproximation.sparse.StateFeature
import burlap.behavior.policy.GreedyQPolicy
import burlap.behavior.policy.Policy
import burlap.behavior.valuefunction.QProvider
import burlap.mdp.auxiliary.StateGenerator
import burlap.mdp.core.action.Action
import burlap.mdp.core.action.ActionType
import burlap.mdp.core.state.State
import burlap.mdp.singleagent.SADomain
import burlap.mdp.singleagent.environment.EnvironmentOutcome
import burlap.mdp.singleagent.environment.SimulatedEnvironment
import burlap.mdp.singleagent.model.SampleModel
import burlap.statehashing.HashableState
import burlap.statehashing.HashableStateFactory
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.rl.*
import eu.iv4xr.framework.model.rl.approximation.FeatureActionFactory
import eu.iv4xr.framework.model.rl.approximation.FeatureVectorFactory
import java.lang.IllegalArgumentException
import kotlin.random.Random
import kotlin.reflect.KClass
import kotlin.reflect.full.memberProperties

/**
 * Turn any ENUM into a burlap action
 */
interface BurlapEnum<E : Enum<E>> : BurlapAction {
    fun get(): E
    override fun actionName(): String {
        return get().name
    }

    override fun copy(): BurlapEnum<E> {
        return this
    }
}

fun State.variables() = variableKeys().map { get(it) }

fun <S : BurlapState, A : Identifiable> MDP<S, A>.features() = compositeFeature(initialState().sample(Random(12)),
        listOf(BoolFeature(), EnumFeature(), ListFeatures(listOf(BoolFeature(), EnumFeature())))
)


data class CountedFeature(val count: Int, val features: List<StateFeature>)

interface FeatureExtractor {
    fun numFeatures(any: Any): Int
    fun features(any: Any): List<StateFeature>
    fun applicableTo(any: Any): Boolean
}

fun SparseStateFeatures.toString(state: State): String {
    val features = this.features(state)
    val toMutableList = (0 until this.numFeatures()).map { 0.0 }.toMutableList()
    features.forEach { toMutableList[it.id] = it.value }
    return toMutableList.toString()
}

fun compositeFeature(example: State, features: List<FeatureExtractor>): SparseStateFeatures {
    val result = extract(example, features)
    return object : SparseStateFeatures {
        override fun features(p0: State): MutableList<StateFeature> {
            return extract(p0, features).features.toMutableList()
        }

        override fun copy() = this

        override fun numFeatures(): Int {
            return result.count
        }
    }
}

private fun extract(state: State, features: List<FeatureExtractor>): CountedFeature {
    val objects = state.variableKeys().map { state.get(it) }
    return mergeFeatures(objects, features)
}

private fun mergeFeatures(objects: List<*>, features: List<FeatureExtractor>): CountedFeature {

    return objects.fold(CountedFeature(0, emptyList())) { (count, list), any ->
        val extractor = features.first { it.applicableTo(any as Any) }
        val newFeatures = extractor.features(any as Any).map { StateFeature(it.id + count, it.value) }
        CountedFeature(count + extractor.numFeatures(any), (list + newFeatures))
    }
}

class ListFeatures(val features: List<FeatureExtractor>) : FeatureExtractor {

    private fun countedFeatures(list: List<*>) = mergeFeatures(list, features)

    override fun numFeatures(any: Any) = countedFeatures(any as List<*>).count

    override fun features(any: Any) = countedFeatures(any as List<*>).features

    override fun applicableTo(any: Any) = any is List<*>
}

class BoolFeature : FeatureExtractor {
    override fun numFeatures(any: Any) = 1

    override fun features(any: Any) = if (any == true) listOf(StateFeature(0, 1.0)) else emptyList()

    override fun applicableTo(any: Any) = any is Boolean
}

class EnumFeature : FeatureExtractor {
    override fun numFeatures(any: Any) = any.javaClass.enumConstants.size

    override fun features(any: Any) = listOf(StateFeature((any as Enum<*>).ordinal, 1.0))

    override fun applicableTo(any: Any) = any is Enum<*>
}

/**
 * Turn any data class into an action
 */
interface DataClassAction : BurlapAction {
    override fun actionName(): String {
        return this.toString()
    }

    override fun copy(): DataClassAction {
        return this
    }
}

/**
 * Remain access to q values
 */
class GreedyQPolicyWithQValues(private val qProvider: QProvider) : GreedyQPolicy(qProvider), QProvider by qProvider {
    override fun action(s: State?): Action? {
        if (qProvider.qValues(s).size == 0) {
            println("He")
        }
        println("H")
        return super.action(s)
    }
}

/**
 * BurlapPolicy to internal policy
 */
class BurlapPolicy<S : BurlapState, A : BurlapAction>(val policy: Policy, val mdp: MDP<S, A>) : eu.iv4xr.framework.model.rl.Policy<S, A> {
    override fun action(state: S) = Distributions.discrete(
            mdp.possibleActions(state).associateWith { policy.actionProb(state, it) }
    )
}

/**
 * Create stategenerator  from MDP
 */
fun <S : BurlapState, A : BurlapAction> MDP<S, A>.stateGenerator(random: Random) = StateGenerator { initialState().sample(random) }

/**
 * Get all actionTypes of MDP
 */
fun <S : BurlapState, A : BurlapAction> MDP<S, A>.actionTypes() = object : ActionType {
    override fun typeName() = "Actions"

    override fun associatedAction(name: String) = allPossibleActions().first { it.actionName() == name }

    @Suppress("UNCHECKED_CAST")
    override fun allApplicableActions(p0: State) = possibleActions(p0 as S).toList()
}


@Suppress("UNCHECKED_CAST")
fun <T : BurlapState> FeatureVectorFactory<T>.stateFeatures() = object : DenseStateFeatures {
    override fun features(p0: State?): DoubleArray {
        return this@stateFeatures.features(p0 as T)
    }

    override fun copy(): DenseStateFeatures = this
}

fun <T : BurlapState, A : BurlapAction> FeatureActionFactory<T, A>.stateActionFeatures() = object : DenseStateActionFeatures {
    override fun features(p0: State?, p1: Action?): DoubleArray {
        return this@stateActionFeatures.features((p0 as T) to (p1 as A))
    }

    override fun copy(): DenseStateActionFeatures = this
}

class SafePolicy<S : BurlapState>(val mdp: MDP<S, *>, val policy: Policy) : Policy by policy {
    @Suppress("UNCHECKED_CAST")
    override fun action(p0: State): Action? {
        if (mdp.isTerminal(p0 as S)) {
            return null
        }
        return policy.action(p0)
    }

}

class FixedEnv(domain: SampleModel, initialState: StateGenerator) : SimulatedEnvironment(domain, initialState) {
    init {
        allowActionFromTerminalStates = false
    }

}

/**
 * Create sample model from MDP
 */
fun <S : BurlapState, A : BurlapAction> MDP<S, A>.toBurlapModel(random: Random): SampleModel {

    return object : SampleModel {
        @Suppress("UNCHECKED_CAST")
        override fun sample(state: State, action: Action): EnvironmentOutcome {
            val newState = transition(state as S, action as A).sample(random)
            val reward = reward(state, action, newState).sample(random)
            return EnvironmentOutcome(state, action, newState, reward, isTerminal(newState))

        }

        @Suppress("UNCHECKED_CAST")
        override fun terminal(p0: State): Boolean {
            return isTerminal(p0 as S)
        }
    }
}

/**
 * Utils class for use in lambda
 */
data class BurlapComponents<S : BurlapState, A : BurlapAction>(val domain: SADomain, val stateGenerator: StateGenerator, val model: SampleModel, val mdp: MDP<S, A>)

/**
 * Convenient way to integrate BurlapAlgorithms into our framework
 */
class BurlapAlg<S : BurlapState, A : BurlapAction>(private val random: Random, private val alg: BurlapComponents<S, A>.() -> Policy) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): eu.iv4xr.framework.model.rl.Policy<S, A> {
        val domain = SADomain()
        val model = mdp.toBurlapModel(random)
        domain.actionTypes = listOf(mdp.actionTypes())
        domain.model = model
        val policy = alg(BurlapComponents(domain, mdp.stateGenerator(random), model, mdp))
        return BurlapPolicy(policy, mdp)
    }
}

/**
 * This class can be used when
 */
open class DataClassHashableState : HashableState, BurlapState {
    override fun s() = this

    override fun variableKeys() = mutableListOf<Any>()

    override fun get(p0: Any?) = p0

    override fun copy() = this
}

class DataClassStateFactory : HashableStateFactory {
    override fun hashState(p0: State): HashableState {
        return p0 as DataClassHashableState ?: throw IllegalArgumentException("This state is not recognized")
    }
}

/**
 * Enumerates all memberproperties of a class
 */
open class ReflectionBasedState : BurlapState {

    companion object {
        val keys = mutableMapOf<KClass<*>, MutableList<Any>>()
        fun getVariableKeys(clazz: KClass<*>, any: Any) = keys.getOrPut(clazz) {
            clazz.memberProperties.flatMap { property ->
                val child = property.call(any)
                if (child is BurlapState) {
                    child.variableKeys().map {
                        { obj -> (BurlapState::get)(property.call(obj) as BurlapState, it) }
                    }
                } else listOf { obj: Any ->
                    property.call(obj)
                }
            }.toMutableList()
        }
    }

    override fun copy() = this

    override fun variableKeys() = getVariableKeys(this::class, this)

    @Suppress("UNCHECKED_CAST")
    override fun get(p0: Any?): Any? {
        return (p0 as (Any) -> Any)(this)
    }
}