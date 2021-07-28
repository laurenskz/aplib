package nl.uu.cs.aplib.exampleUsages.fiveGame

import eu.iv4xr.framework.model.rl.RLAgent
import eu.iv4xr.framework.model.rl.RLAlgorithm
import eu.iv4xr.framework.model.rl.StateWithGoalProgress
import eu.iv4xr.framework.model.rl.algorithms.GreedyAlg
import eu.iv4xr.framework.model.rl.algorithms.RandomAlg
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import nl.uu.cs.aplib.AplibEDSL
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.*
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGameSetup.*
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame_withAgent.FiveGameState
import java.io.File
import kotlin.random.Random
import kotlin.system.exitProcess
import kotlin.time.DurationUnit
import kotlin.time.ExperimentalTime
import kotlin.time.measureTimedValue


sealed class Algorithm

class Optimal(val discountFactor: Double, val steps: Int) : Algorithm()

object Rand : Algorithm()

data class QLearning(val episodes: Int, val discountFactor: Double, val learningRate: Double, val qInit: Double) : Algorithm()

enum class FiveGameSetup {
    NORMAL, WITH_CHANGE, WITH_CHANGE_AND_UPDATED_MODEL
}

data class AlgorithmSetup(
        val desiredState: GAMESTATUS,
        val fiveGameSetup: FiveGameSetup,
        val algorithm: Algorithm,
        val trials: Int
)

data class AlgorithmResult(
        val setup: AlgorithmSetup,
        val totalTimeMS: Long,
        val totalRequiredEpisodes: Int
)

fun csvHeader() =
        "totalEpisodes,totalTime,desiredState,repetitions,setup,algorithm,qEpisodes,qLR"


fun csvRow(result: AlgorithmResult): String {
    val setup = result.setup
    val qEpisodes = if (setup.algorithm is QLearning) setup.algorithm.episodes else ""
    val qLR = if (setup.algorithm is QLearning) setup.algorithm.learningRate else ""
    return "${result.totalRequiredEpisodes},${result.totalTimeMS},${setup.desiredState},${setup.trials},${setup.fiveGameSetup},${setup.algorithm.javaClass.simpleName},$qEpisodes,$qLR"
}

fun gameFor(setup: FiveGameSetup): FiveGame = when (setup) {
    NORMAL -> FiveGame(3, 0, 3, java.util.Random()).attachOpponent(RandomPlayer(SQUARE.CROSS))
    WITH_CHANGE -> FiveGameWithChange(3, 0, 3, java.util.Random()).attachOpponent(RandomPlayer(SQUARE.CROSS))
    WITH_CHANGE_AND_UPDATED_MODEL -> FiveGameWithChange(3, 0, 3, java.util.Random()).attachOpponent(RandomPlayer(SQUARE.CROSS))
}


fun modelFor(setup: FiveGameSetup, game: FiveGame): FiveGameModel {
    val predicate: (Pair<Int, Int>) -> Boolean = { pair -> game.board[pair.first][pair.second] == SQUARE.BLOCKED }
    return when (setup) {
        NORMAL -> FiveGameModel(FiveGameEnv.FiveGameConf(game.boardsize, predicate), game.winSize, SQUARE.CIRCLE)
        WITH_CHANGE -> FiveGameModel(FiveGameEnv.FiveGameConf(game.boardsize, predicate), game.winSize, SQUARE.CIRCLE)
        WITH_CHANGE_AND_UPDATED_MODEL -> FiveGameModelWithChange(FiveGameEnv.FiveGameConf(game.boardsize, predicate), game.winSize, SQUARE.CIRCLE)
    }
}

fun algFor(algorithm: Algorithm): RLAlgorithm<StateWithGoalProgress<FiveGameModelState>, FiveGameAction> = when (algorithm) {
    is Optimal -> GreedyAlg(algorithm.discountFactor, algorithm.steps)
    is Rand -> RandomAlg()
    is QLearning -> BurlapAlgorithms.qLearning(algorithm.discountFactor, algorithm.learningRate, algorithm.qInit, algorithm.episodes, Random(1234))
}


@ExperimentalTime
fun main(args: Array<String>) {
    val out = File(args[0]).printWriter()
    val algs = listOf(
            Rand,
            Optimal(0.95, 4),
            QLearning(1000, 0.95, 0.4, 0.0),
            QLearning(3000, 0.95, 0.4, 0.0),
            QLearning(5000, 0.95, 0.4, 0.0),
            QLearning(10000, 0.95, 0.4, 0.0),
            QLearning(15000, 0.95, 0.4, 0.0),
            QLearning(30000, 0.95, 0.4, 0.0),
    )
    val statuses = listOf(GAMESTATUS.CIRCLEWON, GAMESTATUS.CROSSWON, GAMESTATUS.TIE)
    val gameSetups = listOf(WITH_CHANGE, WITH_CHANGE_AND_UPDATED_MODEL, NORMAL)
    val trials = 100
    val setups = algs.flatMap { alg -> statuses.flatMap { status -> gameSetups.map { gameSetup -> AlgorithmSetup(status, gameSetup, alg, trials) } } }
    val results = setups.map {
        val game = gameFor(it.fiveGameSetup)
        val model = modelFor(it.fiveGameSetup, game)
        achieveStatus(game, model, it)
    }
    out.println(csvHeader())
    out.println(results.joinToString("\n") { csvRow(it) })
    out.flush()

}

@ExperimentalTime
private fun achieveStatus(thegame: FiveGame, model: FiveGameModel, setup: AlgorithmSetup): AlgorithmResult {
    val state = FiveGameState().setEnvironment(FiveGameEnv().attachGame(thegame))
    val agent = RLAgent(model, Random(123))
            .attachState(state)
    val g = AplibEDSL.goal("goal").toSolve { st: GAMESTATUS -> st == setup.desiredState }.lift()
    g.maxbudget(10.0)
    agent.setGoal(g)
    val time = measureTimedValue {
        agent.trainWith(algFor(setup.algorithm))
        val nrOfSuccesses = 0 until setup.trials
        nrOfSuccesses.map {
            repeatUntilSuccess(agent)
        }.sum()
    }
    return AlgorithmResult(setup, time.duration.toLong(DurationUnit.MILLISECONDS), time.value)
}


private fun repeatUntilSuccess(agent: RLAgent<FiveGameModelState, FiveGameAction>): Int {
    return generateSequence(1) { it + 1 }.first {
        agent.restart()
        while (agent.goal.status.inProgress()) {
            agent.update()
        }
        agent.goal.status.success()
    }
}