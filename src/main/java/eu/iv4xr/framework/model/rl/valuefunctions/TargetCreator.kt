package eu.iv4xr.framework.model.rl.valuefunctions

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms

interface TargetCreator<S : Identifiable, A : Identifiable> {
    fun createTargets(episodes: List<BurlapAlgorithms.Episode<S, A>>, mdp: MDP<S, A>): List<QTarget<S, A>>
}
