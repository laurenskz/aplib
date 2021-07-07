package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.behavior.policy.GreedyQPolicy
import burlap.behavior.policy.Policy
import burlap.behavior.singleagent.learning.tdmethods.QLearning
import burlap.mdp.auxiliary.StateGenerator
import burlap.mdp.core.action.Action
import burlap.mdp.core.action.ActionType
import burlap.mdp.core.state.State
import burlap.mdp.singleagent.SADomain
import burlap.mdp.singleagent.environment.EnvironmentOutcome
import burlap.mdp.singleagent.environment.SimulatedEnvironment
import burlap.mdp.singleagent.model.SampleModel
import burlap.statehashing.simple.SimpleHashableStateFactory
import eu.iv4xr.framework.model.distribution.Distributions.discrete
import eu.iv4xr.framework.model.distribution.Distributions.uniform
import eu.iv4xr.framework.model.examples.RandomWalk
import eu.iv4xr.framework.model.examples.RandomWalkAction
import eu.iv4xr.framework.model.examples.RandomWalkState
import eu.iv4xr.framework.model.rl.*
import kotlin.random.Random


class BurlapPolicy<S : BurlapState, A : BurlapAction>(val policy: Policy, val mdp: MDP<S, A>) : eu.iv4xr.framework.model.rl.Policy<S, A> {
    override fun action(state: S) = discrete(
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
    override fun train(mdp: MDP<S, A>, timeout: Long): eu.iv4xr.framework.model.rl.Policy<S, A> {
        val domain = SADomain()
        val model = mdp.toBurlapModel(Random)
        domain.actionTypes = listOf(mdp.actionTypes())
        domain.model = model
        val policy = alg(BurlapComponents(domain, mdp.stateGenerator(random), model, mdp))
        return BurlapPolicy(policy, mdp)
    }
}

fun main() {
    val randomWalk = RandomWalk(uniform(-0.1, 0.1))
    BurlapAlg<RandomWalkState, RandomWalkAction>(Random(12)) {
        val qLearning = QLearning(domain, 0.9, SimpleHashableStateFactory(), 0.0, 0.1)
        (0 until 100).forEach {
            qLearning.runLearningEpisode(SimulatedEnvironment(model, stateGenerator))
        }
        GreedyQPolicy(qLearning)
    }.train(randomWalk, 0)


}