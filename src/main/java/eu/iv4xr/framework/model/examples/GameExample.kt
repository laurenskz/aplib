package eu.iv4xr.framework.model.examples

import eu.iv4xr.framework.model.*
import eu.iv4xr.framework.model.Distributions.deterministic

data class Location(private val x: Int, private val y: Int) {
    operator fun plus(offset: Location) = Location(x + offset.x, y + offset.y)
}

enum class State {
    ALIVE, DEAD, WON
}

data class Game(val goal: Location, val player: Location, val enemy: Location, val state: State)

fun updateEnemy(game: Game): Distribution<Game> {
    val basicOffset = Distributions.uniform(-1, 0, 1)
    val xyOffset = ifd(flip(0.1), basicOffset * 2, basicOffset)
    val locationDelta = (xyOffset times xyOffset).map { Location(it.first, it.second) }
    return locationDelta.map { game.copy(enemy = game.enemy + it) }
}

fun gameState(game: Game): State {
    if (game.enemy == game.player) return State.DEAD
    if (game.goal == game.player) return State.WON
    return State.ALIVE
}

fun refreshGameState(game: Game) = game.copy(state = gameState(game))

fun performMove(game: Game, move: Location): Distribution<Game> {
    val playerMoved = refreshGameState(game.copy(player = game.player + move))
    if (playerMoved.state == State.DEAD) return deterministic(playerMoved)
    val enemyMoved = updateEnemy(playerMoved)
    return enemyMoved.map(::refreshGameState)
}