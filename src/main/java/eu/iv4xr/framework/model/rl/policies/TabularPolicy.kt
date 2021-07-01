package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.rl.Indexable
import eu.iv4xr.framework.model.rl.Policy

class TabularPolicy<S : Indexable, A : Indexable>(val stateCount: Int, val actionCount: Int) : Policy<S, A> {


    override fun action(state: S): Distribution<A> {
        TODO("Not yet implemented")
    }
}