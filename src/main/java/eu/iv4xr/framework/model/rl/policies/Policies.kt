package eu.iv4xr.framework.model.rl.policies

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.Indexable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.valuefunctions.QFunction

class GreedyPolicy<S : Identifiable, A : Identifiable>(val qFunction: QFunction<S, A>, val mdp: MDP<S, A>) : Policy<S, A> {

    override fun action(state: S): Distribution<A> {
        val qForActions = qFunction.qForActions(state, mdp.possibleActions(state).toList())
        val action = qForActions.maxByOrNull { it.second }
                ?: error("No element found")
        return always(action.first)
    }
}

class EGreedyPolicy<S : Identifiable, A : Identifiable>(val epsilon: Double, val mdp: MDP<S, A>, val policy: Policy<S, A>) : Policy<S, A> {
    override fun action(state: S): Distribution<A> {
        return flip(epsilon).chain {
            val possibleActions = mdp.possibleActions(state).toList()
            if (it) {
                Distributions.uniform(possibleActions.toList())
            } else {
                policy.action(state)
            }
        }
    }
}