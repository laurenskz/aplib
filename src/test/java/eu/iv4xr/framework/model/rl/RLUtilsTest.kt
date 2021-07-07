package eu.iv4xr.framework.model.rl

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.rl.RLUtilsTest.SlotMachine.*
import eu.iv4xr.framework.model.rl.RLUtilsTest.SlotMachineAction.*
import eu.iv4xr.framework.model.terminal
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test


internal class RLUtilsTest {
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

        override fun allPossibleActions() = SlotMachineAction.values().asSequence()
        override fun initialState() = Distributions.uniform(BAD, MEDIUM, GOOD)
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