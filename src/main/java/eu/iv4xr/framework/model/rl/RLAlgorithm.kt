package eu.iv4xr.framework.model.rl

interface RLAlgorithm {
    fun <S : Identifiable, A : Identifiable> train(mdp: MDP<S, A>, timeout: Long): Policy<S, A>
}