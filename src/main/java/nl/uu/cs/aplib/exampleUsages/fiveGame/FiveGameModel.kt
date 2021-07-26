package nl.uu.cs.aplib.exampleUsages.fiveGame

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions.uniform
import eu.iv4xr.framework.model.distribution.SequenceUniform
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassAction
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassHashableState
import eu.iv4xr.framework.model.rl.burlapadaptors.ReflectionBasedState
import eu.iv4xr.framework.utils.allPossible
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.SQUARE
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGameEnv.FiveGameConf
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGameSquare.*
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame_withAgent.FiveGameState
import nl.uu.cs.aplib.mainConcepts.SimpleState
import java.lang.IllegalArgumentException
import java.lang.IllegalStateException

enum class FiveGameSquare {
    CIRCLE, CROSS, EMPTY
}

data class FiveGameSpot(val x: Int, val y: Int) {
    fun valid(boardSize: Int) = (x in 0 until boardSize) && (y in 0 until boardSize)
}

data class FiveGameLine(val spots: List<FiveGameSpot>, val desired: Int) {
    val complete = spots.size == desired
}

fun extend(line: FiveGameLine, next: (FiveGameSpot) -> FiveGameSpot) = FiveGameLine(line.spots + listOf(next(line.spots.last())), line.desired)

fun horizontal(line: FiveGameLine) = extend(line) { FiveGameSpot(it.x + 1, it.y) }
fun vertical(line: FiveGameLine) = extend(line) { FiveGameSpot(it.x, it.y + 1) }
fun diagonal1(line: FiveGameLine) = extend(line) { FiveGameSpot(it.x + 1, it.y + 1) }
fun diagonal2(line: FiveGameLine) = extend(line) { FiveGameSpot(it.x - 1, it.y + 1) }

fun lineFrom(spot: FiveGameSpot, desired: Int, extender: (FiveGameLine) -> FiveGameLine) =
        generateSequence(FiveGameLine(listOf(spot), desired), extender).first { it.complete }


data class FiveGameAction(val x: Int, val y: Int) : DataClassAction

data class FiveGameModelState(val spots: List<FiveGameSquare>) : DataClassHashableState()

open class FiveGameModel(private val conf: FiveGameConf, private val desired: Int, private val symbol: SQUARE) : ProbabilisticModel<FiveGameModelState, FiveGameAction> {

    private val opponent = if (symbol == SQUARE.CROSS) SQUARE.CIRCLE else SQUARE.CROSS

    val lines = allLines()

    override fun possibleStates() = allPossible(conf.availableSpots, FiveGameSquare.values().toList())
            .map(::FiveGameModelState)


    override fun possibleActions(state: FiveGameModelState): Sequence<FiveGameAction> {
        return if (isTerminal(state)) emptySequence() else
            state.spots.asSequence().mapIndexedNotNull { index, fiveGameSquare ->
                conf.spots[index].takeIf { fiveGameSquare == EMPTY }
            }.map { FiveGameAction(it.x, it.y) }
    }

    override fun executeAction(action: FiveGameAction, state: SimpleState): Any {
        if (state !is FiveGameState) throw IllegalStateException("Wrong state")
        return state.env().move(symbol, action.x, action.y)
    }

    override fun convertState(state: SimpleState): FiveGameModelState {
        if (state !is FiveGameState) throw IllegalStateException("Wrong state")
        return conf.spots.map {
            convertSquare(state.board[it.x][it.y])
        }.let(::FiveGameModelState)

    }

    private fun convertSquare(square: SQUARE) = when (square) {
        SQUARE.EMPTY -> EMPTY
        SQUARE.CIRCLE -> CIRCLE
        SQUARE.CROSS -> CROSS
        else -> throw IllegalArgumentException("Illegal state")
    }

    private fun allLines(): List<List<Int>> {
        return (0 until conf.boardSize).flatMap { x ->
            (0 until conf.boardSize).flatMap { y ->
                val spot = FiveGameSpot(x, y)
                listOf(
                        lineFrom(spot, desired, ::horizontal),
                        lineFrom(spot, desired, ::vertical),
                        lineFrom(spot, desired, ::diagonal1),
                        lineFrom(spot, desired, ::diagonal2)
                )
            }
        }.filter {
            it.spots.all { it.valid(conf.boardSize) && !conf.blocked[it.x][it.y] }
        }.map {
            it.spots.map { conf.spotPos(it.x, it.y) }
        }
    }

    fun getSymbol(state: FiveGameModelState, x: Int, y: Int): String {
        if (conf.spots.any { it.x == x && it.y == y }) {
            return when (state.spots[conf.spotPos(x, y)]) {
                EMPTY -> "?"
                CIRCLE -> "O"
                CROSS -> "X"
            }
        }
        return if (conf.blocked[x][y]) "#" else "?"
    }

    fun stateString(state: FiveGameModelState): String = (0 until conf.boardSize).joinToString("\n") { row ->
        (0 until conf.boardSize).joinToString("") { column -> getSymbol(state, column, row) }
    }

    override fun isTerminal(state: FiveGameModelState) = gameStatus(state) != FiveGame.GAMESTATUS.UNFINISHED

    private fun markSpot(state: FiveGameModelState, index: Int, piece: SQUARE) = FiveGameModelState(
            state.spots.let {
                val mutable = it.toMutableList()
                mutable[index] = convertSquare(piece)
                mutable.toList()
            }
    )

    override fun transition(current: FiveGameModelState, action: FiveGameAction): Distribution<FiveGameModelState> {
        if (isTerminal(current)) {
            throw IllegalArgumentException("Attempting to perform move in terminal state")
        }
        val spotIndex = conf.spots.indexOfFirst { it.x == action.x && it.y == action.y }
        if (current.spots[spotIndex] != EMPTY) throw IllegalArgumentException("Square is occupied")
        val state = markSpot(current, spotIndex, symbol)
        if (gameStatus(state) != FiveGame.GAMESTATUS.UNFINISHED) {
            return always(state)
        }
        return uniform(
                state.spots.mapIndexedNotNull { index, fiveGameSquare ->
                    markSpot(state, index, opponent).takeIf { fiveGameSquare == EMPTY }
                }
        )
    }

    override fun proposal(current: FiveGameModelState, action: FiveGameAction, result: FiveGameModelState) =
            always(gameStatus(result))

    override fun possibleActions() = conf.spots.map {
        FiveGameAction(it.x, it.y)
    }.asSequence()

    open protected fun gameStatus(state: FiveGameModelState): FiveGame.GAMESTATUS {
        val winner = lines.mapNotNull {
            when {
                it.all { state.spots[it] == CROSS } -> FiveGame.GAMESTATUS.CROSSWON
                it.all { state.spots[it] == CIRCLE } -> FiveGame.GAMESTATUS.CIRCLEWON
                else -> null
            }
        }.firstOrNull()
        if (winner != null) return winner
        if (state.spots.all { it != EMPTY }) return FiveGame.GAMESTATUS.TIE
        return FiveGame.GAMESTATUS.UNFINISHED
    }

    override fun initialState(): Distribution<FiveGameModelState> {
        val empty = (0 until conf.availableSpots).map { EMPTY }.let(::FiveGameModelState)
        val seq = sequence {
            for (index in 0 until conf.availableSpots) {
                yield(markSpot(empty, index, opponent))
            }
        }
        return SequenceUniform(seq, conf.availableSpots)
    }
}