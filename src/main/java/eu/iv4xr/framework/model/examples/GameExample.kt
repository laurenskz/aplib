package eu.iv4xr.framework.model.examples

import eu.iv4xr.framework.model.distribution.*
import eu.iv4xr.framework.model.distribution.Distributions.deterministic

/**
 * A location on an infinte grid
 */
data class Location(private val x: Int, private val y: Int) {
    operator fun plus(offset: Location) = Location(x + offset.x, y + offset.y)
}

/**
 * The state of the game
 */
enum class State {
    ALIVE, DEAD, WON
}

/**
 * A game has an enemy, player and goal, if the enemy touches the player he loses
 */
data class Game(val goal: Location, val player: Location, val enemy: Location, val state: State)

/**
 * Enemies move randomly, sometimes (10 percent chance) they take big steps
 */
fun updateEnemy(game: Game): Distribution<Game> {
    val basicOffset = Distributions.uniform(-1, 0, 1)
    val basicOffset2 = Distributions.discrete(-1 to 0.25, 0 to 0.5, 1 to 0.25)
    val xyOffset = ifd(flip(0.1), basicOffset * 2, basicOffset)
    val locationDelta = (xyOffset times xyOffset).map { Location(it.first, it.second) }
    return locationDelta.map { game.copy(enemy = game.enemy + it) }
}

/**
 * Returns the state of the game
 */
fun gameState(game: Game): State {
    if (game.enemy == game.player) return State.DEAD
    if (game.goal == game.player) return State.WON
    return State.ALIVE
}

/**
 * Makes sure the game state is actual
 */
fun refreshGameState(game: Game) = game.copy(state = gameState(game))

/**
 * Describes the transition of a game
 */
fun performMove(game: Game, move: Location): Distribution<Game> {
    val playerMoved = refreshGameState(game.copy(player = game.player + move))
    if (playerMoved.state == State.DEAD) return deterministic(playerMoved)
    val enemyMoved = updateEnemy(playerMoved)
    return enemyMoved.map(::refreshGameState)
}