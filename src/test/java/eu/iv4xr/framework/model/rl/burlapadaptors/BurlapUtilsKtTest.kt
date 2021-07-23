package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.behavior.functionapproximation.sparse.SparseStateFeatures
import burlap.behavior.functionapproximation.sparse.StateFeature
import burlap.mdp.core.state.State
import eu.iv4xr.framework.model.rl.BurlapState
import eu.iv4xr.framework.model.rl.StateWithGoalProgress
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGameModelState
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGameSquare
import org.junit.Ignore
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import kotlin.random.Random

internal class BurlapUtilsKtTest {

    data class TestRefl(val x: String, val z: Int, val nested: Nested) : ReflectionBasedState()

    data class Nested(val a: String, val b: String, val c: Nested2) : ReflectionBasedState()

    data class Nested2(val greet: String) : ReflectionBasedState()

    @Test
    fun testReflectionBasedState() {
        val test = TestRefl("hi!", 42, Nested("hi", "Bye", Nested2("Nested 2")))
        val values = test.variableKeys().map {
            test.get(it)
        }.toList()
        println(values)
        assertEquals("hi", values[0])
        assertEquals("Bye", values[1])
        assertEquals("Nested 2", values[2])
        assertEquals("hi!", values[3])
        assertEquals(42, values[4])
    }

    fun SparseStateFeatures.toString(state: State): String {
        val features = this.features(state)
        val toMutableList = (0 until this.numFeatures()).map { 0.0 }.toMutableList()
        features.forEach { toMutableList[it.id] = it.value }
        return toMutableList.toString()
    }

    @Test
    @Disabled
    fun testWithGoalProgress() {
        val stateWithGoalProgress = StateWithGoalProgress(listOf(true), Nested2("Hi!"))
        val values = stateWithGoalProgress.variableKeys().map {
            stateWithGoalProgress.get(it)
        }
        assertEquals(values[0], listOf(true))
        assertEquals(values[1], "Hi!")
    }

    @Test
    fun testStateFeatures() {
        val state = StateWithGoalProgress(listOf(false), FiveGameModelState(listOf(FiveGameSquare.EMPTY, FiveGameSquare.CROSS)))
        val featureList = compositeFeature(state, listOf(BoolFeature(), EnumFeature(), ListFeatures(listOf(BoolFeature(), EnumFeature()))))
        println(featureList.numFeatures())
        println(featureList.features(state).map { "${it.id}:${it.value}" })
        println(featureList.toString(state))
        println(featureList.toString(state.copy(progress = listOf(true))))
    }
}