package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.behavior.policy.GreedyQPolicy
import burlap.behavior.policy.Policy
import burlap.behavior.singleagent.learning.tdmethods.QLearning
import burlap.mdp.auxiliary.StateGenerator
import burlap.mdp.core.action.Action
import burlap.mdp.core.action.ActionType
import burlap.mdp.core.state.State
import burlap.mdp.singleagent.SADomain
import burlap.mdp.singleagent.environment.EnvironmentOutcome
import burlap.mdp.singleagent.environment.SimulatedEnvironment
import burlap.mdp.singleagent.model.SampleModel
import burlap.statehashing.simple.SimpleHashableStateFactory
import eu.iv4xr.framework.model.distribution.Distributions.discrete
import eu.iv4xr.framework.model.distribution.Distributions.uniform
import eu.iv4xr.framework.model.examples.RandomWalk
import eu.iv4xr.framework.model.examples.RandomWalkAction
import eu.iv4xr.framework.model.examples.RandomWalkState
import eu.iv4xr.framework.model.rl.*
import kotlin.random.Random


fun main() {
    val randomWalk = RandomWalk(uniform(-0.1, 0.1))
    BurlapAlg<RandomWalkState, RandomWalkAction>(Random(12)) {
        val qLearning = QLearning(domain, 0.9, SimpleHashableStateFactory(), 0.0, 0.1)
        (0 until 100).forEach {
            qLearning.runLearningEpisode(SimulatedEnvironment(model, stateGenerator))
        }
        GreedyQPolicy(qLearning)
    }.train(randomWalk)


}