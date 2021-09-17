package eu.iv4xr.framework.model.utils

import burlap.debugtools.DPrint.c
import cern.clhep.Units.m
import kotlin.random.Random


class DeterministicRandom : Random() {


    var seed: Long = 123456789
    val m = 2147483648
    val a = 1103515245
    val c = 12345

    override fun nextBits(bitCount: Int): Int {
        seed = (a * seed + c) % m
        return (seed.toInt().ushr(31 - bitCount))
    }
}