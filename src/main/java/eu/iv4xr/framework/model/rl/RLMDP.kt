package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.utils.cons
import nl.uu.cs.aplib.AplibEDSL
import nl.uu.cs.aplib.mainConcepts.BasicAgent
import nl.uu.cs.aplib.mainConcepts.GoalStructure
import nl.uu.cs.aplib.mainConcepts.SimpleState
import kotlin.random.Random

data class StateWithGoalProgress<State : Identifiable>(val progress: List<Boolean>, val state: State) : Identifiable


fun allPossibleGoalStates(count: Int): Sequence<List<Boolean>> = if (count == 0) sequenceOf(listOf()) else
    allPossibleGoalStates(count - 1)
            .flatMap { listOf(true cons it, false cons it) }


class RLMDP<State : Identifiable, Action : Identifiable>(private val model: ProbabilisticModel<State, Action>, private val goals: List<MDPGoal>) : MDP<StateWithGoalProgress<State>, Action> {
    override fun possibleStates() = model.possibleStates().flatMap { modelState -> allPossibleGoalStates(goals.size).map { StateWithGoalProgress(it, modelState) } }

    override fun possibleActions(state: StateWithGoalProgress<State>) = model.possibleActions(state.state)

    override fun isTerminal(state: StateWithGoalProgress<State>) = model.isTerminal(state.state)

    override fun transition(current: StateWithGoalProgress<State>, action: Action): Distribution<StateWithGoalProgress<State>> {
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

fun <T> convert(goal: GoalStructure, onItem: (GoalStructure.PrimitiveGoal) -> T): List<T> {
    if (goal is GoalStructure.PrimitiveGoal) {
        return listOf(onItem(goal))
    }
    return goal.subgoals.flatMap { convert(it, onItem) }
}

fun updateGoalStatus(goal: GoalStructure) {
    goal.subgoals.forEach { updateGoalStatus(it) }
    if (goal is GoalStructure.PrimitiveGoal) {
        if (goal.goal.status.success()) {
            goal.status.setToSuccess()
        }
    } else if (goal.subgoals.all { it.status.success() }) {
        goal.status.setToSuccess()
    }
}

class RLAgent<ModelState : Identifiable, Action : Identifiable>(private val model: ProbabilisticModel<ModelState, Action>) : BasicAgent() {

    lateinit var policy: Policy<StateWithGoalProgress<ModelState>, Action>

    fun trainWith(alorithm: RLAlgorithm, timeout: Long): RLAgent<ModelState, Action> {
        policy = alorithm.train(model, goal, timeout)
        return this
    }

    fun withPolicy(policy: Policy<StateWithGoalProgress<ModelState>, Action>): RLAgent<ModelState, Action> {
        this.policy = policy
        return this
    }

    override fun attachState(state: SimpleState?): RLAgent<ModelState, Action> {
        super.attachState(state)
        return this
    }

    override fun setGoal(g: GoalStructure?): RLAgent<ModelState, Action> {
        super.setGoal(g)
        return this
    }


    override fun update() {
        lockEnvironment()
        try {
            state.updateState()
            val modelState = model.convertState(state)
            val goalProgress = convert(goal) { it.status.success() }
            val action = policy.action(StateWithGoalProgress(goalProgress, modelState))
            val proposal = model.executeAction(action.sample(Random), state)
            convert(goal) { it.goal.propose(proposal) }
            updateGoalStatus(goal)
        } finally {
            unlockEnvironment()
        }
    }
}