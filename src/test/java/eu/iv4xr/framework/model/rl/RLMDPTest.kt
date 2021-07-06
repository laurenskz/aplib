package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.RLMDPTest.SlotMachine.*
import eu.iv4xr.framework.model.rl.RLMDPTest.TestAction.LEFT
import eu.iv4xr.framework.model.rl.RLMDPTest.TestAction.RIGHT
import eu.iv4xr.framework.model.rl.RLMDPTest.TestModelState.*
import eu.iv4xr.framework.model.rl.RLMDPTest.SlotMachineAction.*
import eu.iv4xr.framework.model.rl.algorithms.GreedyAlg
import eu.iv4xr.framework.model.terminal
import nl.uu.cs.aplib.AplibEDSL.goal
import nl.uu.cs.aplib.mainConcepts.Environment
import nl.uu.cs.aplib.mainConcepts.GoalStructure
import nl.uu.cs.aplib.mainConcepts.GoalStructure.GoalsCombinator.SEQ
import nl.uu.cs.aplib.mainConcepts.SimpleState
import org.junit.jupiter.api.Test

import org.junit.jupiter.api.Assertions.*
import kotlin.random.Random

internal class RLMDPTest {

    class TestState : SimpleState() {
        var state: String = ""

        override fun env() = super.env() as TestEnv

        override fun updateState() {
            super.updateState()
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
        ONE, TWO, TERMINAL, PIT
    }

    enum class TestAction : Identifiable {
        LEFT, RIGHT
    }

    class TestModel : ProbabilisticModel<TestModelState, TestAction> {
        override fun possibleStates() = TestModelState.values().asSequence()

        override fun possibleActions(state: TestModelState) = TestAction.values().asSequence()

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
        }

    }

    @Test
    fun allPossibleGoalStates() {
        assertEquals(1, allPossibleGoalStates(0).count())
        assertEquals(2, allPossibleGoalStates(1).count())
        assertEquals(8, allPossibleGoalStates(3).count())
        assertEquals(16, allPossibleGoalStates(4).count())
        assertEquals(32, allPossibleGoalStates(5).count())
        assertTrue(allPossibleGoalStates(4).contains(listOf(false, true, false, false)))
    }

    @Test
    fun convert() {
        val twoNodes = GoalStructure(SEQ, goal("leaf1").lift(), goal("leaf2").lift())
        val topGoal = GoalStructure(SEQ, goal("leaf0").lift(), twoNodes)
        val names = convert(topGoal) {
            it.goal.name
        }
        assertEquals(3, names.size)
        assertTrue(names.contains("leaf0"))
        assertTrue(names.contains("leaf1"))
        assertTrue(names.contains("leaf2"))
    }

    @Test
    fun updateGoalStatus() {
        val goal = goal("leaf1")
        val goal1 = goal("leaf2")
        val goal2 = goal("leaf0")
        val twoNodes = GoalStructure(SEQ, goal.lift(), goal1.lift())
        val topGoal = GoalStructure(SEQ, goal2.lift(), twoNodes)
        updateGoalStatus(topGoal)
        assertFalse(topGoal.status.success())
        goal.status.setToSuccess()
        updateGoalStatus(topGoal)
        assertFalse(topGoal.status.success())
        goal1.status.setToSuccess()
        goal2.status.setToSuccess()
        updateGoalStatus(topGoal)
        assertTrue(topGoal.status.success())
    }

    @Test
    fun testRLMDP() {
        val testEnv = TestEnv(Random(3020))
        val state1 = TestState()
        state1.setEnvironment(testEnv)
        val get3 = goal("Get value 3").toSolve<Int> { it == 3 }.lift().maxbudget(3.0)
        val get9 = goal("Get value 9").toSolve<Int> { it == 9 }.lift()
        val g = GoalStructure(SEQ, get3, get9)
        val agent = RLAgent(TestModel(), Random(134))
                .setGoal(g)
                .attachState(state1)
                .trainWith(GreedyAlg(0.97, 6), 10)
        assertEquals("", state1.state)
        state1.updateState()

        assertEquals("One", state1.state)
        agent.update()
        assertTrue(g.status.inProgress())

        assertEquals("One", state1.state)
        agent.update()
        assertTrue(g.status.inProgress())

        assertEquals("Two", state1.state)
        agent.update()
        assertTrue(g.status.inProgress())

        assertEquals("One", state1.state)
        agent.update()
        assertTrue(g.status.inProgress())

        assertEquals("Two", state1.state)
        agent.update()
        assertTrue(g.status.inProgress())

        assertEquals("One", state1.state)
        agent.update()
        assertTrue(g.status.inProgress())

        assertEquals("Two", state1.state)
        agent.update()
        assertTrue(g.status.inProgress())

        assertEquals("One", state1.state)
        agent.update()
        assertTrue(g.status.inProgress())

        assertEquals("One", state1.state)
        agent.update()
        assertTrue(g.status.inProgress())

        assertEquals("One", state1.state)
        agent.update()
        assertTrue(g.status.inProgress())

        assertEquals("Two", state1.state)
        agent.update()

        assertEquals("Pit", state1.state)
        assertFalse(g.status.inProgress())

        assertTrue(get3.status.success())
        assertTrue(get9.status.failed())
        assertEquals("In terminal state according to model", get9.status.info)
        assertTrue(g.status.failed())
        assertEquals("Some subgoal failed", g.status.info)
    }

    enum class SlotMachine : Identifiable {
        BAD, MEDIUM, GOOD, DONE
    }

    enum class SlotMachineAction : Identifiable {
        DRAW, LEFT, RIGHT
    }

    class SlotMachineMDP : MDP<SlotMachine, SlotMachineAction> {
        override fun possibleStates() = SlotMachine.values().asSequence()

        override fun possibleActions(state: SlotMachine) = SlotMachineAction.values().asSequence()

        override fun isTerminal(state: SlotMachine) = state == DONE

        private fun left(current: SlotMachine) = when (current) {
            BAD -> BAD
            MEDIUM -> BAD
            GOOD -> MEDIUM
            DONE -> terminal()
        }

        private fun right(current: SlotMachine) = when (current) {
            BAD -> MEDIUM
            MEDIUM -> GOOD
            GOOD -> GOOD
            DONE -> terminal()
        }

        override fun transition(current: SlotMachine, action: SlotMachineAction) = when (action) {
            DRAW -> always(DONE)
            SlotMachineAction.LEFT -> always(left(current))
            SlotMachineAction.RIGHT -> always(right(current))
        }

        override fun reward(current: SlotMachine, action: SlotMachineAction, newState: SlotMachine) = when (current) {
            BAD -> Distributions.uniform(-1.0, -0.5)
            MEDIUM -> Distributions.uniform(0.0, 0.5)
            GOOD -> Distributions.uniform(1.0, 0.5)
            DONE -> terminal()
        }.takeIf { newState == DONE } ?: always(0.0)
    }

    @Test
    fun stateValue() {
        val mdp = SlotMachineMDP()
//        Starting in state BAD and following optimal policy will of course go to machine good
        assertEquals(0.75, mdp.stateValue(BAD, 1.0, 10))
//        Same applies here
        assertEquals(0.75, mdp.stateValue(MEDIUM, 1.0, 10))
//        Discounting takes some time to move
        assertEquals(0.1875, mdp.stateValue(BAD, 0.5, 10))
        assertEquals(0.375, mdp.stateValue(MEDIUM, 0.5, 10))
        assertEquals(0.75, mdp.stateValue(GOOD, 0.5, 10))

    }

    @Test
    fun expectedReward() {
        val mdp = SlotMachineMDP()
        assertEquals(0.25, mdp.expectedReward(MEDIUM, DRAW))
        assertEquals(0.0, mdp.expectedReward(BAD, SlotMachineAction.RIGHT))
    }


    @Test
    fun qValue() {
        val mdp = SlotMachineMDP()
//        Values of all slot machines and then in terminal state
        assertEquals(0.75, mdp.qValue(GOOD, DRAW, 1.0, 10))
        assertEquals(0.25, mdp.qValue(MEDIUM, DRAW, 1.0, 10))
        assertEquals(-0.75, mdp.qValue(BAD, DRAW, 1.0, 10))
        assertEquals(0.75, mdp.qValue(BAD, SlotMachineAction.LEFT, 1.0, 10))
    }

    class SlotMachinePolicy : Policy<SlotMachine, SlotMachineAction> {
        override fun action(state: SlotMachine) = when (state) {
            BAD -> always(SlotMachineAction.RIGHT)
            MEDIUM -> Distributions.uniform(DRAW, SlotMachineAction.RIGHT)
            GOOD -> always(DRAW)
            DONE -> terminal()
        }
    }

    @Test
    fun stateValuePolicy() {
        val mdp = SlotMachineMDP()
        val policy = SlotMachinePolicy()
        assertEquals(0.5, mdp.stateValue(BAD, policy, 1.0, 10))
        assertEquals(0.15625, mdp.stateValue(BAD, policy, 0.5, 10))
    }

    @Test
    fun qValuePolicy() {
        val mdp = SlotMachineMDP()
        val policy = SlotMachinePolicy()
        assertEquals(0.5, mdp.qValue(BAD, policy, SlotMachineAction.LEFT, 1.0, 10))
        assertEquals(0.5, mdp.qValue(BAD, policy, SlotMachineAction.RIGHT, 1.0, 10))
    }
}