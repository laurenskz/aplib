package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.rl.policies.GreedyPolicy
import eu.iv4xr.framework.model.rl.valuefunctions.BeliefQFunction
import eu.iv4xr.framework.model.rl.valuefunctions.QFunction
import eu.iv4xr.framework.utils.convert
import eu.iv4xr.framework.utils.updateGoalStatus
import nl.uu.cs.aplib.mainConcepts.BasicAgent
import nl.uu.cs.aplib.mainConcepts.GoalStructure
import nl.uu.cs.aplib.mainConcepts.SimpleState
import kotlin.math.pow
import kotlin.random.Random


interface RLExecutionEngine<ModelState : Identifiable, Action : Identifiable> {
    var policy: Policy<StateWithGoalProgress<ModelState>, Action>

    var mdp: NonTerminalRLMDP<ModelState, Action>

    val rewards: MutableList<Reward>

    var transitions: TransitionLog<ModelState, Action>

    fun executeAction(state: SimpleState)
    fun restart() {}
    fun setGoal(g: GoalStructure) {

    }
}

class BeliefExecutionEngine<ModelState : Identifiable, BeliefState : Identifiable, Action : Identifiable>(
    val beliefParse: (SimpleState) -> BeliefState,
    val beliefDist: (BeliefState) -> Distribution<ModelState>,
    val model: ProbabilisticModel<ModelState, Action>,
    val random: Random

) : RLExecutionEngine<ModelState, Action> {
    override lateinit var policy: Policy<StateWithGoalProgress<ModelState>, Action>
    override lateinit var mdp: NonTerminalRLMDP<ModelState, Action>
    override val rewards: MutableList<Reward>
        get() = TODO("Not yet implemented")
    override var transitions: TransitionLog<ModelState, Action>
        get() = TODO("Not yet implemented")
        set(value) {}
    private lateinit var beliefPolicy: Policy<StateWithGoalProgress<BeliefState>, Action>

    override fun executeAction(state: SimpleState) {
        val beliefState = beliefParse(state)
        val real = beliefDist(beliefState).sample(random)
        val withGoal = StateWithGoalProgress(listOf(false), real)
        val action = policy.action(withGoal).sample(random)
        model.executeAction(action, state)
    }

    override fun setGoal(g: GoalStructure) {
        mdp = createRlMDP(model, g)
    }

    private fun progress(beliefState: StateWithGoalProgress<BeliefState>): Distribution<StateWithGoalProgress<ModelState>> {
        return beliefDist(beliefState.state).map {
            StateWithGoalProgress(beliefState.progress, it)
        }
    }

    fun setQFunction(qFunction: QFunction<StateWithGoalProgress<ModelState>, Action>) {
        beliefPolicy = GreedyPolicy(
            BeliefQFunction(
                qFunction, this::progress
            )
        ) {
            mdp.possibleActions(progress(it).sample(Random(9))).toList()
        }
    }

}

class RLMDPExecutionEngine<ModelState : Identifiable, Action : Identifiable>(
    private val model: ProbabilisticModel<ModelState, Action>,
    private val random: Random
) : RLExecutionEngine<ModelState, Action> {

    private lateinit var goal: GoalStructure
    override lateinit var policy: Policy<StateWithGoalProgress<ModelState>, Action>

    override lateinit var mdp: NonTerminalRLMDP<ModelState, Action>

    override val rewards = mutableListOf<Reward>()

    override var transitions = TransitionLog(model)

    private var timeStep = 0

    fun resetGoal() {
        convert(goal) { it.goal.status.resetToInProgress() }
        updateGoalStatus(goal)
    }

    override fun restart() {
        timeStep = 0
        rewards.clear()
        resetGoal()
        transitions = TransitionLog(model)
    }

    fun trainWith(alorithm: RLAlgorithm<StateWithGoalProgress<ModelState>, Action>) {
        policy = alorithm.train(mdp)
    }

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

    override fun setGoal(g: GoalStructure) {
        goal = g
        mdp = createRlMDP(model, goal)
    }


    override fun executeAction(state: SimpleState) {
        val modelState = model.convertState(state)
        if (model.isTerminal(modelState)) {
            handleTerminalState(modelState)
            return
        }
        val goalProgress = convert(goal) { it.status.success() }
        val stateWithProgress = StateWithGoalProgress(goalProgress, modelState)
        val action = policy.action(stateWithProgress)
        println("Action:${action.supportWithDensities()}")
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
    }

}


/**
 * Integration of trained algorithms with MDPs
 * the model is a model of the environment the agent is meant to be executed in
 * the random is used to sample actions
 */
class RLAgent<ModelState : Identifiable, Action : Identifiable>(
    private val executionEngine: RLExecutionEngine<ModelState, Action>,
    id: String? = null
) : BasicAgent(id, "") {

    constructor(
        model: ProbabilisticModel<ModelState, Action>,
        random: Random
    ) : this(RLMDPExecutionEngine(model, random))

    var policy: Policy<StateWithGoalProgress<ModelState>, Action> by executionEngine::policy

    var mdp: NonTerminalRLMDP<ModelState, Action> by executionEngine::mdp

    val rewards by executionEngine::rewards


    var transitions: TransitionLog<ModelState, Action> by executionEngine::transitions

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

    override fun setGoal(g: GoalStructure): RLAgent<ModelState, Action> {
        goal = g
        executionEngine.setGoal(g)
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
            state.updateState(id)
            executionEngine.executeAction(state)
        } finally {
            unlockEnvironment()
        }
    }

    override fun restart() {
        state.env().resetWorker()
        state.updateState(id)
        executionEngine.restart()
    }
}
