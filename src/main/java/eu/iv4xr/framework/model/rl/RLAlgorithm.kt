package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import nl.uu.cs.aplib.agents.State
import nl.uu.cs.aplib.mainConcepts.GoalStructure

interface RLAlgorithm {
    fun <S : Identifiable, A : Identifiable> train(mdp: MDP<S, A>, timeout: Long): Policy<S, A>
}

interface MDPGoal {
    fun completed(proposal: Any): Boolean
    fun reward(): Double
}

class WrappedMDPGoal(private val goal: GoalStructure.PrimitiveGoal) : MDPGoal {
    override fun completed(proposal: Any): Boolean {
        return goal.goal.wouldBeSolvedBy(proposal)
    }

    override fun reward() = goal.maxBudgetAllowed

}

fun <ModelState : Identifiable, Action : Identifiable> RLAlgorithm.train(model: ProbabilisticModel<ModelState, Action>, goal: GoalStructure, timeout: Long): Policy<StateWithGoalProgress<ModelState>, Action> {
    return train(RLMDP(model, convert(goal) { WrappedMDPGoal(it) }), timeout)
}