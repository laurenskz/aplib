package eu.iv4xr.framework.model.rl.approximation

import burlap.behavior.functionapproximation.dense.DenseLinearVFA
import burlap.behavior.functionapproximation.dense.DenseStateFeatures
import burlap.behavior.valuefunction.ValueFunction
import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.distribution.times
import eu.iv4xr.framework.model.rl.*
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassAction
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassHashableState
import eu.iv4xr.framework.model.rl.burlapadaptors.stateFeatures
import nl.uu.cs.aplib.mainConcepts.SimpleState

fun <T : Identifiable> stateWithGoalProgressFactory(wrapped: FeatureVectorFactory<T>, count: Int): FeatureVectorFactory<StateWithGoalProgress<T>> = CompositeFeature(listOf(
        wrapped.from { it.state },
        RepeatedFeature(count, BoolFeature).from { it.progress }
))

data class PlaygroundState(val tries: Int, val luck: Double, val badLuck: Double, val money: Int) : DataClassHashableState() {
    companion object {
        val factory: FeatureVectorFactory<PlaygroundState> = CompositeFeature(listOf(
                IntFeature.from { it.tries },
                DoubleFeature.from { it.luck },
                DoubleFeature.from { it.badLuck },
                IntFeature.from { it.money }
        ))
    }
}

data class PlaygroundAction(val betAmount: Int) : DataClassAction {
    companion object {
        val factory: FeatureVectorFactory<PlaygroundAction> = IntFeature.from { it.betAmount }
    }
}


fun playgroundMDP(targets: List<Int>) = RLMDP(Playground(), targets.map { t -> basicGoal<Int>(1.0) { it > t } })


class Playground : ProbabilisticModel<PlaygroundState, PlaygroundAction> {
    private val actions = sequenceOf(0, 100, 400, 200, 50, 30).map { PlaygroundAction(it) }

    override fun possibleStates(): Sequence<PlaygroundState> {
        return emptySequence()
    }

    override fun possibleActions(state: PlaygroundState): Sequence<PlaygroundAction> {
        return actions
    }

    override fun executeAction(action: PlaygroundAction, state: SimpleState): Any {
        TODO("Not yet implemented")
    }

    override fun convertState(state: SimpleState): PlaygroundState {
        TODO("Not yet implemented")
    }

    override fun isTerminal(state: PlaygroundState): Boolean {
        if (state.tries == 0) {
            println("What!!!?")
        }
        return state.tries <= 0
    }

    override fun transition(current: PlaygroundState, action: PlaygroundAction): Distribution<PlaygroundState> {
        println("We now have $current")
        val betAmount = action.betAmount
        return ((0.0..1.0) sampleWithStepSize 0.01).chain { newLuck ->
            ((0.0..1.0) sampleWithStepSize 0.01).chain { newBadLuck ->
                ((0.0..1.0) sampleWithStepSize 0.01).chain { rand ->
                    val addition = (rand * betAmount * (current.luck - current.badLuck)).toInt()
                    always(PlaygroundState(current.tries - 1, newLuck, newBadLuck, current.money + addition))
                }
            }
        }
    }

    override fun proposal(current: PlaygroundState, action: PlaygroundAction, result: PlaygroundState): Distribution<out Any> {
        return always(result.money)
    }

    override fun possibleActions() = actions

    override fun initialState() = always(
            PlaygroundState(100, 0.0, 0.0, 0),
    )
}