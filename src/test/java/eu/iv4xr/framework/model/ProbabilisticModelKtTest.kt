package eu.iv4xr.framework.model

import eu.iv4xr.framework.model.rl.RLAgentTest
import nl.uu.cs.aplib.exampleUsages.DumbDoctorModel
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class ProbabilisticModelKtTest {

    @Test
    fun distinct() {
        assertEquals(6, DumbDoctorModel().distinctStates().size)
        assertEquals(4, RLAgentTest.TestModel().distinctStates().size)
    }
}