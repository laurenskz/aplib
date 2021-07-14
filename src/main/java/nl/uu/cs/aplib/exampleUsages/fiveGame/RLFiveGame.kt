package nl.uu.cs.aplib.exampleUsages.fiveGame

import eu.iv4xr.framework.model.distinctStates
import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.model.rl.RLAgent
import eu.iv4xr.framework.model.rl.algorithms.Greedy
import eu.iv4xr.framework.model.rl.algorithms.GreedyAlg
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.stateValue
import nl.uu.cs.aplib.AplibEDSL
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.*
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame_withAgent.FiveGameState
import nl.uu.cs.aplib.mainConcepts.BasicAgent
import org.junit.Test
import java.util.*
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
    println("Trying to achieve status:$status")
    val winSize = 3
    // creating an instance of the FiveGame
    val thegame = { FiveGame(3, 0, winSize, java.util.Random()) }
    // create an agent state and an environment, attached to the game:
    val state = FiveGameState().setEnvironment(FiveGameEnv().attachGame(thegame()))
    // creatint the agent:
    val model = FiveGameModel(state.env().conf, winSize, SQUARE.CIRCLE)
    val agent = RLAgent(model, Random(123))
            .attachState(state)
    // define a goal and specify a tactic:
    val g = AplibEDSL.goal("goal").toSolve { st: GAMESTATUS -> st == status }.lift()
    g.maxbudget(10.0)
    agent.setGoal(g)
//    agent.trainWith(BurlapAlgorithms.qLearning(0.95, 0.4, 0.0, 100000, Random(1234)))
    agent.trainWith(GreedyAlg(0.95, 4))
    println(agent.mdp.initialState().expectedValue {
        agent.mdp.stateValue(it, 1.0, 4)
    })
    return

    val outComes = (0..100).map {
        state.env().attachGame(thegame())
        val opponent = RandomPlayer(SQUARE.CROSS, state.env().thegame)
        agent.resetGoal()
        while (state.env().thegame.gameStatus == GAMESTATUS.UNFINISHED) {
            opponent.move()
            agent.update()
        }
        state.env().thegame.gameStatus
    }.groupBy { it }
            .mapValues { it.component2().size }
    println(outComes)
}