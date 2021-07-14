package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.utils.convert
import eu.iv4xr.framework.utils.updateGoalStatus
import nl.uu.cs.aplib.mainConcepts.BasicAgent
import nl.uu.cs.aplib.mainConcepts.GoalStructure
import nl.uu.cs.aplib.mainConcepts.SimpleState
import kotlin.random.Random

/**
 * Integration of trained algorithms with MDPs
 * the model is a model of the environment the agent is meant to be executed in
 * the random is used to sample actions
 */
class RLAgent<ModelState : Identifiable, Action : Identifiable>(private val model: ProbabilisticModel<ModelState, Action>, private val random: Random) : BasicAgent() {

    lateinit var policy: Policy<StateWithGoalProgress<ModelState>, Action>

    lateinit var mdp: MDP<StateWithGoalProgress<ModelState>, Action>

    fun trainWith(alorithm: RLAlgorithm<StateWithGoalProgress<ModelState>, Action>): RLAgent<ModelState, Action> {
        mdp = createRlMDP(model, goal)
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

    fun resetGoal() {
        convert(goal) { it.goal.status.resetToInProgress() }
        updateGoalStatus(goal)
    }

    /**
     * If we are in a terminal state according to the model we cannot perform an action any more
     * Therefore all goals that have not been completed have failed, and we update them accordingly
     */
    private fun handleTerminalState() {
        convert(goal) {
            if (!it.status.success()) {
                it.status.setToFail("In terminal state according to model")
            }
        }
        updateGoalStatus(goal)
    }
}