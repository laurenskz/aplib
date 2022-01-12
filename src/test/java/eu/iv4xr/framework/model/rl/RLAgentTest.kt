package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.RLAgentTest.TestAction.LEFT
import eu.iv4xr.framework.model.rl.RLAgentTest.TestAction.RIGHT
import eu.iv4xr.framework.model.rl.RLAgentTest.TestModelState.*
import eu.iv4xr.framework.model.rl.algorithms.GreedyAlg
import eu.iv4xr.framework.model.terminal
import eu.iv4xr.framework.model.utils.DeterministicRandom
import nl.uu.cs.aplib.AplibEDSL.goal
import nl.uu.cs.aplib.mainConcepts.Environment
import nl.uu.cs.aplib.mainConcepts.GoalStructure
import nl.uu.cs.aplib.mainConcepts.GoalStructure.GoalsCombinator.SEQ
import nl.uu.cs.aplib.mainConcepts.SimpleState
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.lang.IllegalArgumentException
import kotlin.random.Random

internal class RLAgentTest {

    class TestState : SimpleState() {
        var state: String = ""

        override fun env() = super.env() as TestEnv

        override fun updateState(agentID:String) {
            super.updateState(agentID)
            state = env().state
        }
    }

    class TestEnv(private val random: Random) : Environment() {

        var state = "One"
        fun left(): Int {
            if (flip(0.1).sample(random)) {
                state = "Pit"
                return 3
            }
            if (state == "Two") {
                state = "One"
            }
            return 0
        }

        fun right(): Int {
            if (flip(0.1).sample(random)) {
                state = "Pit"
                return 3
            }

            if (state == "One") {
                state = "Two"
                return 0
            } else if (state == "Two") {
                state = "Terminal"
                return Distributions.discrete(3 to 0.5, 0 to 0.5).sample(random)
            }
            return 0
        }
    }

    enum class TestModelState : Identifiable {
        ONE, TWO, TERMINAL, PIT, EXCLUDED
    }

    enum class TestAction : Identifiable {
        LEFT, RIGHT
    }

    class TestModel : ProbabilisticModel<TestModelState, TestAction> {
        override fun possibleStates() = TestModelState.values().asSequence()

        override fun possibleActions(state: TestModelState) = if (isTerminal(state)) emptySequence() else TestAction.values().asSequence()

        override fun executeAction(action: TestAction, state: SimpleState): Any {
            val env = (state as TestState).env()
            return when (action) {
                LEFT -> env.left()
                RIGHT -> env.right()
            }
        }

        override fun convertState(state: SimpleState) = when ((state as TestState).state) {
            "One" -> ONE
            "Two" -> TWO
            else -> TERMINAL
        }

        override fun isTerminal(state: TestModelState) = state in listOf(TERMINAL, PIT)

        override fun transition(current: TestModelState, action: TestAction) = when (action) {
            LEFT -> when (current) {
                ONE -> ONE
                TWO -> ONE
                else -> terminal()
            }
            RIGHT -> when (current) {
                ONE -> TWO
                TWO -> TERMINAL
                else -> terminal()
            }
        }.let {
            Distributions.discrete(it to 0.9, PIT to 0.1)
        }

        override fun proposal(current: TestModelState, action: TestAction, result: TestModelState) = when (result) {
            ONE -> always(0)
            TWO -> always(0)
            TERMINAL -> Distributions.discrete(3 to 0.5, 0 to 0.5)
            PIT -> always(3)
            else -> throw IllegalArgumentException("State is ignored")
        }

        override fun possibleActions() = TestAction.values().asSequence()
        override fun initialState() = Distributions.uniform(ONE, TWO)
    }

    @Test
    fun testAgent() {
        val testEnv = TestEnv(DeterministicRandom())
        val state1 = TestState()
        state1.setEnvironment(testEnv)
        val get3 = goal("Get value 3").toSolve<Int> { it == 3 }.lift().maxbudget(3.0)
        val get9 = goal("Get value 9").toSolve<Int> { it == 9 }.lift()
        val g = GoalStructure(SEQ, get3, get9)
        val agent = RLAgent(TestModel(), Random(134))
                .setGoal(g)
                .attachState(state1)
                .trainWith(GreedyAlg(0.97, 6))
        assertEquals("", state1.state)
        state1.updateState("")

        while (g.status.inProgress()) {
            assertTrue(listOf("One", "Two").contains(state1.state))
            agent.update()
        }

        assertEquals("Pit", state1.state)
        assertFalse(g.status.inProgress())

        assertTrue(get3.status.success())
        assertTrue(get9.status.failed())
        assertEquals("In terminal state according to model", get9.status.info)
        assertTrue(g.status.failed())
        assertEquals("Some subgoal failed", g.status.info)
    }


}