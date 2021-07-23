package nl.uu.cs.aplib.exampleUsages.fiveGame

import burlap.statehashing.simple.IISimpleHashableState
import burlap.statehashing.simple.SimpleHashableStateFactory
import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distinctStates
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.assertAlways
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassStateFactory
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.RandomPlayer
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.SQUARE
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGameEnv.FiveGameConf
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame_withAgent.FiveGameState
import org.junit.Ignore
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

internal class FiveGameModelTest {

    @Test
    fun testBasic() {
        val conf = FiveGameConf(5) { false }
        val fiveGameModel = FiveGameModel(conf, 5, SQUARE.CIRCLE)
        assertEquals(12, fiveGameModel.lines.size)
    }

    class Execution<S : Identifiable, A : Identifiable>(val random: Random, val model: ProbabilisticModel<S, A>) {
        var prevState: S? = null
        var state = model.initialState().sample(random)
        var lastAction: A? = null

        fun execute(action: A): Distribution<out Any> {
            prevState = state
            val last = state
            state = model.transition(state, action).sample(random)
            lastAction = action
            return model.proposal(last, action, state)
        }
    }

    fun <S : Identifiable, A : Identifiable> ProbabilisticModel<S, A>.sampleExecution(random: Random, lambda: Execution<S, A>.() -> Unit) {
        Execution(random, this).also(lambda)
    }

    @Test
    fun ticTacToe() {
        val conf = FiveGameConf(3) { false }
        val fiveGameModel = FiveGameModel(conf, 3, SQUARE.CIRCLE)
        fiveGameModel.sampleExecution(Random(2)) {
            assertEquals(8, model.possibleActions(state).count())
            assertAlways(FiveGame.GAMESTATUS.UNFINISHED, execute(FiveGameAction(1, 0)))
            assertEquals(6, model.possibleActions(state).count())
            assertAlways(FiveGame.GAMESTATUS.UNFINISHED, execute(FiveGameAction(0, 0)))
            assertEquals(4, model.possibleActions(state).count())
            assertAlways(FiveGame.GAMESTATUS.UNFINISHED, execute(FiveGameAction(0, 1)))
            assertEquals(2, model.possibleActions(state).count())
            assertAlways(FiveGame.GAMESTATUS.CIRCLEWON, execute(FiveGameAction(0, 2)))
        }
    }

    @Test
    fun testInitialState() {
        val conf = FiveGameConf(3) { false }
        val fiveGameModel = FiveGameModel(conf, 3, SQUARE.CIRCLE)
        assertEquals(9, fiveGameModel.initialState().support().count())
    }

    @Test
    fun testConvertState() {

        val thegame = FiveGame(5, 0)
        val state = FiveGameState().setEnvironment(FiveGameEnv().attachGame(thegame))
        val fiveGameModel = FiveGameModel(state.env().conf, 5, SQUARE.CIRCLE)
        val opponent = RandomPlayer(SQUARE.CROSS, thegame)
        opponent.rnd = java.util.Random(1234)
        thegame.attachOpponent(opponent)
        thegame.move(SQUARE.CIRCLE, 0, 0)
        state.updateState()
        assertEquals(fiveGameModel.stateString(fiveGameModel.convertState(state)).filter { it in "?OX#" },
                thegame.toString().filter { it in "?OX#" })

        thegame.move(SQUARE.CIRCLE, 1, 0)
        state.updateState()
        assertEquals(fiveGameModel.stateString(fiveGameModel.convertState(state)).filter { it in "?OX#" },
                thegame.toString().filter { it in "?OX#" })
    }

    @Test
    fun testExecuteAction() {
        val thegame = FiveGame(5, 0)
        val randomPlayer = RandomPlayer(SQUARE.CROSS)
        randomPlayer.rnd = java.util.Random(1234)
        thegame.attachOpponent(randomPlayer)
        val state = FiveGameState().setEnvironment(FiveGameEnv().attachGame(thegame))
        val fiveGameModel = FiveGameModel(state.env().conf, 5, SQUARE.CIRCLE)
        fiveGameModel.executeAction(FiveGameAction(4, 3), state)
        assertEquals(SQUARE.CIRCLE, thegame.board[4][3])
    }

    @Disabled
    @Test
    fun testStatistics() {
        // creating an instance of the FiveGame
        val thegame = FiveGame(5, 15, 2, java.util.Random(22))
        val games = listOf(
                FiveGame(5, 15, 5, java.util.Random(22)),
                FiveGame(5, 14, 5, java.util.Random(22)),
                FiveGame(5, 13, 5, java.util.Random(22)),
                FiveGame(5, 12, 5, java.util.Random(22)),
                FiveGame(3, 0, 3, java.util.Random(22)),
                FiveGame(4, 0, 4, java.util.Random(22)),
        )
        for (game in games) {
            val state = FiveGameState().setEnvironment(FiveGameEnv().attachGame(game))
            val model = FiveGameModel(state.env().conf, game.winSize, SQUARE.CIRCLE)
            println("${game.boardsize} by ${game.boardsize} & ${game.numOfBlocked} & ${model.distinctStates().count()} \\\\")
        }
        // create an agent state and an environment, attached to the game:
        println("Measuring states")
        // creatint the agent:
    }

    @Test
    fun testState() {
        val fact = DataClassStateFactory()
        assertEquals(
                fact.hashState(FiveGameModelState(listOf(FiveGameSquare.CIRCLE, FiveGameSquare.EMPTY, FiveGameSquare.EMPTY))),
                fact.hashState(FiveGameModelState(listOf(FiveGameSquare.CIRCLE, FiveGameSquare.EMPTY, FiveGameSquare.EMPTY)))
        )
        assertEquals(
                fact.hashState(FiveGameModelState(listOf(FiveGameSquare.CIRCLE, FiveGameSquare.EMPTY, FiveGameSquare.EMPTY))).hashCode(),
                fact.hashState(FiveGameModelState(listOf(FiveGameSquare.CIRCLE, FiveGameSquare.EMPTY, FiveGameSquare.EMPTY))).hashCode()
        )
        assertNotEquals(
                fact.hashState(FiveGameModelState(listOf(FiveGameSquare.CIRCLE, FiveGameSquare.CROSS, FiveGameSquare.EMPTY))),
                fact.hashState(FiveGameModelState(listOf(FiveGameSquare.CIRCLE, FiveGameSquare.EMPTY, FiveGameSquare.EMPTY)))
        )
    }
}