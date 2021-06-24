package eu.iv4xr.framework.model.examples

import eu.iv4xr.framework.model.Distribution
import eu.iv4xr.framework.model.Distributions
import eu.iv4xr.framework.model.densityString
import eu.iv4xr.framework.model.expectedValue

data class Roll(val high: Int, val low: Int?)

fun rollDie() = Distributions.uniform(1..6)

fun roll(count: Int): Distribution<List<Int>> {
    if (count == 1) return rollDie().map { listOf(it) }
    return rollDie().chain { new ->
        roll(count - 1).map { (listOf(new) + it).sorted() }
    }
}

fun attackRoll() = roll(3).map { Roll(it[2], it[1]) }

fun defenseRoll(count: Int) = when (count) {
    2 -> roll(2).map { Roll(it[1], it[0]) }
    1 -> roll(1).map { Roll(it[0], null) }
    else -> throw IllegalArgumentException("Defender can use 1 or 2 dice")
}

fun defend(attack: Roll) = defenseRoll(tactic(attack))

fun tactic(attack: Roll) = when (attack) {
    Roll(6, 6) -> 1
    Roll(5, 5) -> 1
    Roll(5, 4) -> 1
    Roll(6, 5) -> 1
    Roll(6, 4) -> 1
    else -> 2
}

fun battle(attack: Int, defense: Int): Int = if (attack > defense) 1 else -1

fun battle(attack: Roll, defense: Roll): Int {
    var score = 0
    score += battle(attack.high, defense.high)
    if (defense.low != null && attack.low != null) score += battle(attack.low, defense.low)
    return score
}

fun simulation() = attackRoll().chain { attack ->
    defend(attack).map { defense ->
        battle(attack, defense)
    }
}

fun main() {
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