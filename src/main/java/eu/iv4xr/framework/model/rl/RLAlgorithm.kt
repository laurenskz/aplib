package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.utils.convert
import nl.uu.cs.aplib.mainConcepts.GoalStructure

interface RLAlgorithm<S : Identifiable, A : Identifiable> {
    fun train(mdp: MDP<S, A>): Policy<S, A>
}

interface MDPGoal {
    fun completed(proposal: Any): Boolean
    fun reward(): Double
}

fun basicGoal(reward: Double, predicate: (Any) -> Boolean): MDPGoal {
    return object : MDPGoal {
        override fun completed(proposal: Any) = predicate(proposal)

        override fun reward() = reward
    }
}

class WrappedMDPGoal(private val goal: GoalStructure.PrimitiveGoal) : MDPGoal {
    override fun completed(proposal: Any): Boolean {
        return goal.goal.wouldBeSolvedBy(proposal)
    }

    override fun reward() = goal.maxBudgetAllowed

}


fun <Action : Identifiable, ModelState : Identifiable> createRlMDP(model: ProbabilisticModel<ModelState, Action>, goal: GoalStructure) =
        RLMDP(model, convert(goal) { WrappedMDPGoal(it) })