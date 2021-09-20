package eu.iv4xr.framework.model.utils

import kotlin.random.Random


class DeterministicRandom : Random() {


    var seed: Long = 123456789
    val m = (1L shl 48) - 1
    val a = 1103515245
    val c = 12345

    override fun nextBits(bitCount: Int): Int {
        seed = (a * seed + c) and m
        return (seed ushr (48 - bitCount)).toInt()
    }
}