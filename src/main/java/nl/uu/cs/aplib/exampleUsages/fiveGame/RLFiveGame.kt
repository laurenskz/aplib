package nl.uu.cs.aplib.exampleUsages.fiveGame

import eu.iv4xr.framework.model.rl.RLAgent
import eu.iv4xr.framework.model.rl.RLAlgorithm
import eu.iv4xr.framework.model.rl.StateWithGoalProgress
import eu.iv4xr.framework.model.rl.algorithms.GreedyAlg
import eu.iv4xr.framework.model.rl.algorithms.RandomAlg
import eu.iv4xr.framework.model.rl.analyzeFaulty
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import nl.uu.cs.aplib.AplibEDSL
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.*
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame_withAgent.FiveGameState
import kotlin.random.Random

fun main() {

    val status = GAMESTATUS.CIRCLEWON

    achieveStatus(GAMESTATUS.CIRCLEWON)
    achieveStatus(GAMESTATUS.TIE)
    achieveStatus(GAMESTATUS.CROSSWON)
    // now we let the agent play against an automated random player:
    // now we let the agent play against an automated random player:
//    while (thegame.getGameStatus() == GAMESTATUS.UNFINISHED) {
//        opponent.move()
//        if (thegame.getGameStatus() != GAMESTATUS.UNFINISHED) {
//            thegame.print()
//            thegame.printStatus()
//            break
//        }
//        agent.update()
//        thegame.print()
//        thegame.printStatus()
//        println("(press a ENTER to continue)")
//        consoleInput.nextLine()
//    }

}

private fun achieveStatus(status: GAMESTATUS) {
    val winSize = 3
    // creating an instance of the FiveGame
    val thegame = FiveGame(3, 0, winSize, java.util.Random()).attachOpponent(RandomPlayer(SQUARE.CROSS))
    // create an agent state and an environment, attached to the game:
    val state = FiveGameState().setEnvironment(FiveGameEnv().attachGame(thegame))
    // creatint the agent:
    val model = FiveGameModel(state.env().conf, winSize, SQUARE.CIRCLE)
    val agent = RLAgent(model, Random(123))
            .attachState(state)
    // define a goal and specify a tactic:
    val g = AplibEDSL.goal("goal").toSolve { st: GAMESTATUS -> st == status }.lift()
    g.maxbudget(10.0)
    agent.setGoal(g)
    val outcome = listOf<RLAlgorithm<StateWithGoalProgress<FiveGameModelState>, FiveGameAction>>(
            BurlapAlgorithms.qLearning(0.95, 0.4, 0.0, 100000, Random(1234)),
//            BurlapAlgorithms.gradientSarsaLam(0.7, 0.4, 0.7, 10000, Random(1234)),
//            GreedyAlg(0.95, 4),
            RandomAlg()).map { it ->
        agent.trainWith(it)
        val nrOfSuccesses = 0..4
        nrOfSuccesses.map {
            neededEpisodes(agent, model)
        }.average()
    }.joinToString(" & ")
    println("$status & $outcome \\\\")

}


private fun neededEpisodes(agent: RLAgent<FiveGameModelState, FiveGameAction>, model: FiveGameModel): Int {
    return generateSequence(1) { it + 1 }.first {
        agent.restart()
        while (agent.goal.status.inProgress()) {
            agent.update()
        }
        if (!agent.progress.plausible()) {
            println(analyzeFaulty(agent.progress) {
                model.stateString(it)
            })
        }
        agent.goal.status.success()
    }
}