package nl.uu.cs.aplib.exampleUsages.fiveGame

import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.GAMESTATUS.*
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.SQUARE.*
import java.util.*

@Suppress("WHEN_ENUM_CAN_BE_NULL_IN_JAVA")
class FiveGameWithChange(size: Int, numOfBlocked: Int, winSize: Int, random: Random) : FiveGame(size, numOfBlocked, winSize, random) {
    override fun getGameStatus(): GAMESTATUS {
        val square = board[0][0]

        return when (super.getGameStatus()) {
            UNFINISHED -> UNFINISHED
            TIE -> TIE
            CIRCLEWON -> when (square) {
                EMPTY -> UNFINISHED
                CIRCLE -> CIRCLEWON
                CROSS -> TIE
                BLOCKED -> TIE
            }
            CROSSWON -> when (square) {
                EMPTY -> UNFINISHED
                CIRCLE -> TIE
                CROSS -> CROSSWON
                BLOCKED -> TIE
            }
        }
    }
}

class FiveGameModelWithChange(val conf: FiveGameEnv.FiveGameConf, desired: Int, symbol: FiveGame.SQUARE) : FiveGameModel(conf, desired, symbol) {
    override fun gameStatus(state: FiveGameModelState): FiveGame.GAMESTATUS {
        val spotIndex = conf.spots.indexOfFirst { it.x == 0 && it.y == 0 }.takeIf { it != -1 }
        val square = spotIndex?.let { state.spots[it] }
        return when (super.gameStatus(state)) {
            UNFINISHED -> UNFINISHED
            TIE -> TIE
            CIRCLEWON -> when (square) {
                FiveGameSquare.CIRCLE -> CIRCLEWON
                FiveGameSquare.CROSS -> TIE
                FiveGameSquare.EMPTY -> UNFINISHED
                null -> TIE
            }
            CROSSWON -> when (square) {
                FiveGameSquare.CIRCLE -> TIE
                FiveGameSquare.CROSS -> CROSSWON
                FiveGameSquare.EMPTY -> UNFINISHED
                null -> TIE
            }
        }
    }
}