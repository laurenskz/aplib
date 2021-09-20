package eu.iv4xr.framework.model.rl.approximation

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.times
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.sampleWithStepSize
import nl.uu.cs.aplib.mainConcepts.SimpleState

data class PlaygroundState(val luck: Int, val badLuck: Int, val money: Int) : Identifiable

data class PlaygroundAction(val betAmount: Int) : Identifiable

class Playground : ProbabilisticModel<PlaygroundState, PlaygroundAction> {
    override fun possibleStates(): Sequence<PlaygroundState> {
        return (-1000..1000).flatMap { luck ->
            (-10000..10000).flatMap { money ->
                (-1000..1000).map { badLuck ->
                    PlaygroundState(luck, badLuck, money)
                }
            }
        }.asSequence()
    }

    override fun possibleActions(state: PlaygroundState): Sequence<PlaygroundAction> {
        return sequenceOf(0, 100, 400, 200, 50, 30).map { PlaygroundAction(it) }
    }

    override fun executeAction(action: PlaygroundAction, state: SimpleState): Any {
        TODO("Not yet implemented")
    }

    override fun convertState(state: SimpleState): PlaygroundState {
        TODO("Not yet implemented")
    }

    override fun isTerminal(state: PlaygroundState): Boolean {
        return false
    }

    override fun transition(current: PlaygroundState, action: PlaygroundAction): Distribution<PlaygroundState> {
        val betAmount = action.betAmount
        val distribution = (0.0..1.0) sampleWithStepSize 0.01
        val newLuck = Distributions.uniform(-1000..10000)
        val newBadLuck = Distributions.uniform(-1000..10000)
//        (newLuck times newBadLuck).map {
//        }
//        return distribution.map { it * current.luck }
        TODO()
//        Distributions.un
//        current.luck
    }

    override fun proposal(current: PlaygroundState, action: PlaygroundAction, result: PlaygroundState): Distribution<out Any> {
        TODO("Not yet implemented")
    }

    override fun possibleActions(): Sequence<PlaygroundAction> {
        TODO("Not yet implemented")
    }

    override fun initialState(): Distribution<PlaygroundState> {
        TODO("Not yet implemented")
    }
}