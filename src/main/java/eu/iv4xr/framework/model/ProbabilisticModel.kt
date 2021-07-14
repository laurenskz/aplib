package eu.iv4xr.framework.model

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.rl.Identifiable
import nl.uu.cs.aplib.mainConcepts.SimpleState
import java.lang.IllegalArgumentException


/**
 * This is the main model the user has to define to define the logic of their game
 */
interface ProbabilisticModel<ModelState : Identifiable, ModelAction : Identifiable> {
    /**
     * Gives all valid states for the model
     */
    fun possibleStates(): Sequence<ModelState>

    /**
     * Gives all valid actions for a specific state
     */
    fun possibleActions(state: ModelState): Sequence<ModelAction>

    /**
     * Returns the result of executing the action in the environment. The result will be proposed to all goals
     */
    fun executeAction(action: ModelAction, state: SimpleState): Any

    /**
     * Maps Aplib agent state to a state in the model
     */
    fun convertState(state: SimpleState): ModelState

    /**
     * Returns true if the state is terminal, i.e. interaction with the environment should stop
     */
    fun isTerminal(state: ModelState): Boolean

    /**
     * Transition function of the MDP. describes the distribution of next states given an action
     */
    fun transition(current: ModelState, action: ModelAction): Distribution<ModelState>

    /**
     * The proposal that is returned after executing a specific action
     */
    fun proposal(current: ModelState, action: ModelAction, result: ModelState): Distribution<out Any>

    /**
     * Returns all possible actions
     */
    fun possibleActions(): Sequence<ModelAction>

    /**
     * Distribution of possible initial states
     */
    fun initialState(): Distribution<ModelState>
}

fun terminal(): Nothing = throw IllegalArgumentException("Terminal states cannot transition")

fun <S : Identifiable, A : Identifiable> ProbabilisticModel<S, A>.distinctStatesFrom(visited: MutableSet<S>, state: S) {
    if (state in visited) return
    visited.add(state)
    if (isTerminal(state)) return
    val newStates = possibleActions(state).flatMap { transition(state, it).support() }.distinct()
    for (newState in newStates) {
        distinctStatesFrom(visited, newState)
    }
}

fun <S : Identifiable, A : Identifiable> ProbabilisticModel<S, A>.tree() {
    initialState().chain { s ->
        Distributions.uniform(possibleActions(s)).map { s to it }
    }
}

fun <S : Identifiable, A : Identifiable> ProbabilisticModel<S, A>.distinctStates() = mutableSetOf<S>().also { visited ->
    initialState().support().forEach { distinctStatesFrom(visited, it) }
}
