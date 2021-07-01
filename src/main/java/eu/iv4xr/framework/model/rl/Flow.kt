package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel

interface MDPGoal {
    fun completed(proposal: Any): Boolean
    fun reward(): Double
}


fun <State : Identifiable, Action : Identifiable> container(init: Container<State, Action>.() -> Unit) = Container<State, Action>().apply(init)

class Container<State : Identifiable, Action : Identifiable> {
    private lateinit var description: String
    private lateinit var model: ProbabilisticModel<State, Action>
    private lateinit var algorithm: RLAlgorithm
    private lateinit var goals: List<MDPGoal>
}