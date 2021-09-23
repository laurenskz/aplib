package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.distribution.Distribution

interface MDP<State : Identifiable, Action : Identifiable> {
    fun possibleStates(): Sequence<State>

    fun allPossibleActions(): Sequence<Action>

    fun possibleActions(state: State): Sequence<Action>

    fun isTerminal(state: State): Boolean

    fun transition(current: State, action: Action): Distribution<State>

    fun reward(current: State, action: Action, newState: State): Distribution<Double>

    fun initialState(): Distribution<State>

}