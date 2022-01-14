package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import kotlin.random.Random

interface MDP<State : Identifiable, Action : Identifiable> {
    fun possibleStates(): Sequence<State>

    fun allPossibleActions(): Sequence<Action>

    fun possibleActions(state: State): Sequence<Action>

    fun isTerminal(state: State): Boolean

    fun transition(current: State, action: Action): Distribution<State>

    fun reward(current: State, action: Action, newState: State): Distribution<Double>

    fun initialState(): Distribution<State>
    fun sampleEpisode(policy: Policy<State, Action>, random: Random, maxSteps: Int = Int.MAX_VALUE): BurlapAlgorithms.Episode<State, Action> {
        return sampleEpisode(policy, random, initialState().sample(random), maxSteps)
    }

    fun sampleEpisode(policy: Policy<State, Action>, random: Random, initialState: State, maxSteps: Int = Int.MAX_VALUE): BurlapAlgorithms.Episode<State, Action> {
        var state = initialState().sample(random)
        val sars = mutableListOf<BurlapAlgorithms.SARS<State, Action>>()
        var stepCount = 0
        while (!isTerminal(state) && stepCount < maxSteps) {
            val newSars = sampleSARS(policy, state, random)
            sars.add(newSars)
            state = newSars.sp
            stepCount++
        }
        return BurlapAlgorithms.Episode(sars)
    }

    fun sampleSARS(policy: Policy<State, Action>, state: State, random: Random): BurlapAlgorithms.SARS<State, Action> {
        val action = policy.action(state).sample(random)
        return executeAction(action, state, random)
    }

    fun executeAction(action: Action, state: State, random: Random): BurlapAlgorithms.SARS<State, Action> {
        val transition = transition(state, action)
        val sp = transition.sample(random)
        val r = reward(state, action, sp).sample(random)
        return BurlapAlgorithms.SARS(state, action, sp, r, 0.0)
    }
}