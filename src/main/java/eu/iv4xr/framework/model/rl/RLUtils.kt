package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.distribution.expectedValue

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



