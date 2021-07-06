package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.utils.allPossibleGoalStates


class RLMDP<State : Identifiable, Action : Identifiable>(private val model: ProbabilisticModel<State, Action>, private val goals: List<MDPGoal>) : MDP<StateWithGoalProgress<State>, Action> {
    override fun possibleStates() = model.possibleStates().flatMap { modelState -> allPossibleGoalStates(goals.size).map { StateWithGoalProgress(it, modelState) } }

    override fun possibleActions(state: StateWithGoalProgress<State>) = if (isTerminal(state)) emptySequence() else model.possibleActions(state.state)

    override fun isTerminal(state: StateWithGoalProgress<State>) = model.isTerminal(state.state)

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
}


