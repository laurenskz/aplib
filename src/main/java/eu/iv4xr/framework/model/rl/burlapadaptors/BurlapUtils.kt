package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.behavior.policy.Policy
import burlap.mdp.auxiliary.StateGenerator
import burlap.mdp.core.action.Action
import burlap.mdp.core.action.ActionType
import burlap.mdp.core.state.State
import burlap.mdp.singleagent.SADomain
import burlap.mdp.singleagent.environment.EnvironmentOutcome
import burlap.mdp.singleagent.model.SampleModel
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.rl.BurlapAction
import eu.iv4xr.framework.model.rl.BurlapState
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.RLAlgorithm
import java.lang.reflect.Field
import kotlin.random.Random
import kotlin.reflect.KCallable
import kotlin.reflect.KClass
import kotlin.reflect.KProperty
import kotlin.reflect.cast
import kotlin.reflect.full.isSuperclassOf
import kotlin.reflect.full.memberProperties

interface BurlapEnum<E : Enum<E>> : BurlapAction {
    fun get(): E
    override fun actionName(): String {
        return get().name
    }

    override fun copy(): BurlapEnum<E> {
        return this
    }
}

class BurlapPolicy<S : BurlapState, A : BurlapAction>(val policy: Policy, val mdp: MDP<S, A>) : eu.iv4xr.framework.model.rl.Policy<S, A> {
    override fun action(state: S) = Distributions.discrete(
            mdp.possibleActions(state).associateWith { policy.actionProb(state, it) }
    )
}
fun <S : BurlapState, A : BurlapAction> MDP<S, A>.stateGenerator(random: Random) = object : StateGenerator {
    override fun generateState(): State {
        return initialState().sample(random)
    }
}

fun <S : BurlapState, A : BurlapAction> MDP<S, A>.actionTypes() = object : ActionType {
    override fun typeName() = "Actions"

    override fun associatedAction(name: String) = allPossibleActions().first { it.actionName() == name }

    override fun allApplicableActions(p0: State) = possibleActions(p0 as S).toList()
}


fun <S : BurlapState, A : BurlapAction> MDP<S, A>.toBurlapModel(random: Random): SampleModel {

    return object : SampleModel {
        override fun sample(state: State, action: Action): EnvironmentOutcome {
            val newState = transition(state as S, action as A).sample(random)
            val reward = reward(state as S, action as A, newState).sample(random)
            return EnvironmentOutcome(state, action, newState, reward, isTerminal(newState))

        }

        override fun terminal(p0: State): Boolean {
            return isTerminal(p0 as S)
        }
    }
}

data class BurlapComponents<S : BurlapState, A : BurlapAction>(val domain: SADomain, val stateGenerator: StateGenerator, val model: SampleModel, val mdp: MDP<S, A>)

class BurlapAlg<S : BurlapState, A : BurlapAction>(private val random: Random, private val alg: BurlapComponents<S, A>.() -> Policy) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): eu.iv4xr.framework.model.rl.Policy<S, A> {
        val domain = SADomain()
        val model = mdp.toBurlapModel(Random)
        domain.actionTypes = listOf(mdp.actionTypes())
        domain.model = model
        val policy = alg(BurlapComponents(domain, mdp.stateGenerator(random), model, mdp))
        return BurlapPolicy(policy, mdp)
    }
}

interface ReflectionBasedState : BurlapState {
    override fun variableKeys(): MutableList<Any> {
        return this::class.memberProperties.flatMap { property ->
            val child = property.call(this)
            if (child is BurlapState) {
                child.variableKeys().map {
                    { bab -> (BurlapState::get)(property.call(bab) as BurlapState, it) }
                }
            } else listOf { bab: Any ->
                property.call(bab)
            }
        }.toMutableList()
    }

    override fun get(p0: Any?): Any? {
        return (p0 as (Any) -> Any)(this)
    }
}

interface ImmutableReflectionBasedState : ReflectionBasedState {
    override fun copy(): State {
        return this
    }
}