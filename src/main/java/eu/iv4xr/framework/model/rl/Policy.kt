package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.distribution.Distribution

interface Policy<State, Action> {
    fun action(state: State): Distribution<Action>
}