package eu.iv4xr.framework.model

import nl.uu.cs.aplib.mainConcepts.Environment
import nl.uu.cs.aplib.mainConcepts.SimpleState


interface ProbabilisticModel<ModelState, ModelAction> {
    fun possibleStates(): Sequence<ModelState>
    fun possibleActions(state: ModelState): Sequence<ModelAction>

    /**
     * Returns the result of executing the action in the environment.
     */
    fun executeAction(action: ModelAction, environment: Environment?): Any
    fun convertState(state: SimpleState): ModelState
    fun transition(current: ModelState, action: ModelAction): Distribution<ModelState>
}