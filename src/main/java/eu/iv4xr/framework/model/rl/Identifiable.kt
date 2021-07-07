package eu.iv4xr.framework.model.rl

import burlap.mdp.core.action.Action
import burlap.mdp.core.state.State

interface Identifiable {
}

interface Indexable : Identifiable {
    /**
     * The total number of elements that are part of this index family
     */
    fun count(): Long

    /**
     * The index of this instance in the index family
     */
    fun index(): Int
}


interface BurlapState : State,Identifiable
interface BurlapAction : Action,Identifiable

interface FeatureVector : Identifiable {
    /**
     * Represent this state as a list of features
     */
    fun features(): FloatArray
}
