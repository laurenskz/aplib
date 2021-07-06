package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.utils.cons
import nl.uu.cs.aplib.mainConcepts.BasicAgent
import nl.uu.cs.aplib.mainConcepts.GoalStructure
import nl.uu.cs.aplib.mainConcepts.SimpleState
import java.lang.IllegalArgumentException
import kotlin.random.Random

data class StateWithGoalProgress<State : Identifiable>(val progress: List<Boolean>, val state: State) : Identifiable


fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.stateValue(state: State, discountFactor: Double, depth: Int): Double {
    if (isTerminal(state)) return 0.0
    return possibleActions(state).maxOf { action ->
        qValue(state, action, discountFactor, depth)
    }
}

fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.stateValue(state: State, policy: Policy<State, Action>, discountFactor: Double, depth: Int): Double {
    if (isTerminal(state)) return 0.0
    return policy.action(state).expectedValue { action ->
        qValue(state, policy, action, discountFactor, depth)

    }
}

fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.qValue(state: State, policy: Policy<State, Action>, action: Action, discountFactor: Double, depth: Int): Double {
    if (depth == 0) return expectedReward(state, action)
    return expectedReward(state, action) + discountFactor * transition(state, action).expectedValue { newState ->
        stateValue(newState, policy, discountFactor, depth - 1)
    }
}


fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.qValue(state: State, action: Action, discountFactor: Double, depth: Int): Double {
    if (depth == 0) return expectedReward(state, action)
    return expectedReward(state, action) + discountFactor * transition(state, action).expectedValue { newState ->
        stateValue(newState, discountFactor, depth - 1)
    }
}

fun <State : Identifiable, Action : Identifiable> MDP<State, Action>.expectedReward(state: State, action: Action): Double {
    return transition(state, action).expectedValue { newState ->
        reward(state, action, newState).expectedValue()
    }
}

fun allPossibleGoalStates(count: Int): Sequence<List<Boolean>> = if (count == 0) sequenceOf(listOf()) else
    allPossibleGoalStates(count - 1)
            .flatMap { listOf(true cons it, false cons it) }


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
        if (goal.goal.status.failed()) {
            goal.status.setToFail(goal.goal.status.info)
        }
    } else if (goal.subgoals.all { it.status.success() }) {
        goal.status.setToSuccess()
    } else if (goal.subgoals.any { it.status.failed() }) {
        goal.status.setToFail("Some subgoal failed")
    }
}

class RLAgent<ModelState : Identifiable, Action : Identifiable>(private val model: ProbabilisticModel<ModelState, Action>, private val random: Random) : BasicAgent() {

    lateinit var policy: Policy<StateWithGoalProgress<ModelState>, Action>

    lateinit var mdp: MDP<StateWithGoalProgress<ModelState>, Action>

    fun trainWith(alorithm: RLAlgorithm, timeout: Long): RLAgent<ModelState, Action> {
        mdp = createRlMDP(model, goal)
        policy = alorithm.train(mdp, timeout)
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
        goal = g
        return this
    }


    override fun update() {
        lockEnvironment()
        try {
            state.updateState()
            val modelState = model.convertState(state)
            if (model.isTerminal(modelState)) {
                handleTerminalState()
                return
            }
            val goalProgress = convert(goal) { it.status.success() }
            val action = policy.action(StateWithGoalProgress(goalProgress, modelState))
            val proposal = model.executeAction(action.sample(random), state)
            convert(goal) { it.goal.propose(proposal) }
            updateGoalStatus(goal)
        } finally {
            unlockEnvironment()
        }
    }

    private fun handleTerminalState() {
        convert(goal) {
            if (!it.status.success()) {
                it.status.setToFail("In terminal state according to model")
            }
        }
        updateGoalStatus(goal)
    }
}