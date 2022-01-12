package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.utils.convert
import eu.iv4xr.framework.utils.updateGoalStatus
import nl.uu.cs.aplib.mainConcepts.BasicAgent
import nl.uu.cs.aplib.mainConcepts.GoalStructure
import nl.uu.cs.aplib.mainConcepts.SimpleState
import kotlin.math.pow
import kotlin.random.Random

/**
 * Integration of trained algorithms with MDPs
 * the model is a model of the environment the agent is meant to be executed in
 * the random is used to sample actions
 */
class RLAgent<ModelState : Identifiable, Action : Identifiable>(private val model: ProbabilisticModel<ModelState, Action>, private val random: Random) : BasicAgent() {

    lateinit var policy: Policy<StateWithGoalProgress<ModelState>, Action>

    lateinit var mdp: NonTerminalRLMDP<ModelState, Action>

    val rewards = mutableListOf<Reward>()

    var transitions = TransitionLog(model)
        private set

    private var timeStep = 0

    fun trainWith(alorithm: RLAlgorithm<StateWithGoalProgress<ModelState>, Action>): RLAgent<ModelState, Action> {
        policy = alorithm.train(mdp)
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
        mdp = createRlMDP(model, goal)
        return this
    }


    /**
     * Updates the environment
     * Then the state
     * Then chooses an action to execute in the environment based on the model of the environment
     * Executes the action and updates the status of all goals
     *
     * This behaviour is different from other agents as it does not follow the BDI philosophy
     */
    override fun update() {
        lockEnvironment()
        try {
            val modelState = currentState()
            if (model.isTerminal(modelState)) {
                handleTerminalState(modelState)
                return
            }
            val goalProgress = convert(goal) { it.status.success() }
            val stateWithProgress = StateWithGoalProgress(goalProgress, modelState)
            val action = policy.action(stateWithProgress)
            val sampledAction = action.sample(random)
            transitions.steps.add(Transition(modelState, sampledAction))
            val proposal = model.executeAction(sampledAction, state)
            convert(goal) {
                if (!it.goal.status.success()) {
                    it.goal.propose(proposal)
                    if (it.goal.status.success()) {
                        rewards.add(Reward(it.maxBudgetAllowed, timeStep))
                    }
                }

            }
            updateGoalStatus(goal)
            timeStep++
        } finally {
            unlockEnvironment()
        }
    }

    fun currentState(): ModelState {
        state.updateState(id)
        return model.convertState(state)
    }

    override fun restart() {
        timeStep = 0
        rewards.clear()
        resetGoal()
        state.env().resetWorker()
        state.updateState(id)
        transitions = TransitionLog(model)
    }

    fun resetGoal() {
        convert(goal) { it.goal.status.resetToInProgress() }
        updateGoalStatus(goal)
    }

    fun totalReward(discountFactor: Double) =
            rewards.sumByDouble { it.reward * discountFactor.pow(it.timeStep) }

    /**
     * If we are in a terminal state according to the model we cannot perform an action any more
     * Therefore all goals that have not been completed have failed, and we update them accordingly
     */
    private fun handleTerminalState(state: ModelState) {
        transitions.terminal = state
        convert(goal) {
            if (!it.status.success()) {
                val message = "In terminal state according to model"
                it.goal.status.setToFail(message)
                it.status.setToFail(message)
            }
        }
        updateGoalStatus(goal)
    }
}