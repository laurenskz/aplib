package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.utils.convert
import eu.iv4xr.framework.utils.updateGoalStatus
import nl.uu.cs.aplib.mainConcepts.BasicAgent
import nl.uu.cs.aplib.mainConcepts.GoalStructure
import nl.uu.cs.aplib.mainConcepts.SimpleState
import kotlin.random.Random

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