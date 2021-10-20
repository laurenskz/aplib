package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassHashableState
import org.junit.Test
import org.junit.jupiter.api.Assertions.*

internal class StateInformationTest {

    data class Bab(val double: Double) : DataClassHashableState()

    @Test
    fun test() {
        val info = StateInformation<Bab>(100)
        info.add(Bab(0.4), 10.0)
        info.add(Bab(0.5), 20.0)
        info.add(Bab(0.6), 8.0)
        info.add(Bab(0.5), 30.0)
        info.add(Bab(0.5), 2.0)
        assertEquals(3, info.size)
        assertEquals(Bab(0.5), info.take())
        assertEquals(2, info.size)
    }
}