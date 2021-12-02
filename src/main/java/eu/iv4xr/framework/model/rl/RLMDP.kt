package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.ConstantDistribution
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.utils.allPossibleGoalStates


/**
 * We extend the state of the model with the states of all the goals given, all goals are initially false
 *
 * Goals drive the reward function, but the reward is only given the first time the goal is achieved
 */
open class RLMDP<State : Identifiable, Action : Identifiable>(private val model: ProbabilisticModel<State, Action>, private val goals: List<MDPGoal>) : MDP<StateWithGoalProgress<State>, Action> {
    override fun possibleStates() = model.possibleStates().flatMap { modelState -> allPossibleGoalStates(goals.size).map { StateWithGoalProgress(it, modelState) } }


    override fun possibleActions(state: StateWithGoalProgress<State>) = if (isTerminal(state)) emptySequence() else model.possibleActions(state.state)

    override fun isTerminal(state: StateWithGoalProgress<State>) = state.progress.all { it } || model.isTerminal(state.state)

    override fun transition(current: StateWithGoalProgress<State>, action: Action): Distribution<StateWithGoalProgress<State>> {
        if (isTerminal(current)) throw IllegalArgumentException("Can't perform action in terminal state")
        return model.transition(current.state, action).chain { newState ->
            model.proposal(current.state, action, newState).map { proposal ->
                StateWithGoalProgress(
                        current.progress.mapIndexed { idx, status ->
                            status || goals[idx].completed(proposal)
                        },
                        newState)
            }
        }
    }

    /**
     * The reward distribution is actually deterministic because it is conditioned on two specific states, which contain the goal statuses
     * We are only interested in completing goals, therefore we can completely determine rewards from the goals that have been completed
     * by taking this transition. The two states are actually determined by distributions.
     */
    override fun reward(current: StateWithGoalProgress<State>, action: Action, newState: StateWithGoalProgress<State>) = always(
            goals.mapIndexedNotNull { index, goal ->
                goal.reward().takeIf { !current.progress[index] && newState.progress[index] }
            }.sum()
    )

    override fun allPossibleActions() = model.possibleActions()
    override fun initialState(): Distribution<StateWithGoalProgress<State>> {
        return model.initialState().map { StateWithGoalProgress(goals.map { false }, it) }
    }

}

interface Heuristic<State : Identifiable, Action : Identifiable> {
    fun reward(state: State, action: Action, statePrime: State): Double
}

class HeuristicMDP<State : Identifiable, Action : Identifiable>(private val mdp: MDP<State, Action>) : MDP<State, Action> by mdp {
    private val heuristics = mutableListOf<Heuristic<State, Action>>()

    fun addHeuristic(heuristic: Heuristic<State, Action>) = heuristics.add(heuristic)

    override fun reward(current: State, action: Action, newState: State): Distribution<Double> {
        return mdp.reward(current, action, newState).map {
            it + heuristics.sumByDouble { it.reward(current, action, newState) }
        }
    }
}


class NonTerminalRLMDP<State : Identifiable, Action : Identifiable>(private val model: ProbabilisticModel<State, Action>, private val goals: List<MDPGoal>) : RLMDP<State, Action>(model, goals) {

    val heuristics = mutableListOf<Heuristic<State, Action>>()

    override fun reward(current: StateWithGoalProgress<State>, action: Action, newState: StateWithGoalProgress<State>): ConstantDistribution<Double> {
        if (isTerminal(current)) return always(0.0)
        return super.reward(current, action, newState).map {
            it + heuristics.sumByDouble { it.reward(current.state, action, newState.state) }
        }
    }

    override fun transition(current: StateWithGoalProgress<State>, action: Action): Distribution<StateWithGoalProgress<State>> {
        if (isTerminal(current)) return always(current)
        return super.transition(current, action)
    }

    override fun possibleActions(state: StateWithGoalProgress<State>): Sequence<Action> {
        if (isTerminal(state)) return allPossibleActions()
        return super.possibleActions(state)
    }
}



