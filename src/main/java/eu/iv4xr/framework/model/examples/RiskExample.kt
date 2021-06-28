package eu.iv4xr.framework.model.examples

import eu.iv4xr.framework.model.distribution.*
import java.util.ArrayList

data class Roll(val high: Int, val low: Int?)

/**
 * Simulate a 6 sided dice roll
 */
fun rollDie() = Distributions.uniform(1..6)

/**
 * Roll n dice and put the result in a sorted list
 */
fun roll(count: Int): Distribution<List<Int>> {
    if (count == 1) return rollDie().map { listOf(it) }
    return rollDie().chain { new ->
        roll(count - 1).map { (listOf(new) + it).sorted() }
    }
}

/**
 * Attackers use 3 dice and the 2 highest count
 */
fun attackRoll() = roll(3).map { Roll(it[2], it[1]) }

/**
 * Defenders use 2 or 1 dice
 */
fun defenseRoll(count: Int) = when (count) {
    2 -> roll(2).map { Roll(it[1], it[0]) }
    1 -> roll(1).map { Roll(it[0], null) }
    else -> throw IllegalArgumentException("Defender can use 1 or 2 dice")
}

/**
 * Defend with a tactic based on the value of the attack roll
 */
fun defend(attack: Roll) = defenseRoll(tactic(attack))

/**
 * When to use 1 dice and when to use 2 as defender
 */
fun tactic(attack: Roll) = when (attack) {
    Roll(6, 6) -> 1
    Roll(5, 5) -> 1
    Roll(5, 4) -> 1
    Roll(6, 5) -> 1
    Roll(6, 4) -> 1
    else -> 2
}

/**
 * Simulate a battle of 1 dice
 */
fun battle(attack: Int, defense: Int): Int = if (attack > defense) 1 else -1

/**
 * Simulate the battle of 2 rolls
 */
fun battle(attack: Roll, defense: Roll): Int {
    var score = 0
    score += battle(attack.high, defense.high)
    if (defense.low != null && attack.low != null) score += battle(attack.low, defense.low)
    return score
}

/**
 * Simulate all outcomes
 */
fun simulation() = attackRoll().chain { attack ->
    defend(attack).map { defense ->
        battle(attack, defense)
    }
}

/**
 * Compute interesting statistics
 */
fun main() {
    val maybeTrue = flip(0.1)
    val list = ArrayList<Int>();
    list.map {  }

    val allPossibleAttacks = (1..6).flatMap { x -> (1..6).map { y -> Roll(x, y) } }.filter { it.low != null && it.high >= it.low }
    println("Throws where defender should use 1 die")
    for (attack in allPossibleAttacks) {
        val expectedValue1 = defenseRoll(1).map { battle(attack, it) }.expectedValue()
        val expectedValue2 = defenseRoll(2).map { battle(attack, it) }.expectedValue()
        if (expectedValue1 < expectedValue2) println(attack)
    }
    val expectedAttackAdvantage = simulation()
    println("\nProbability distribution of armies delta in advantage of attacker:")
    print(expectedAttackAdvantage.densityString())
    println("\nExpected value of attacker advantage:")
    print(expectedAttackAdvantage.expectedValue { it.toDouble() })
}