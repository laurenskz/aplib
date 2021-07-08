package eu.iv4xr.framework.model.rl.burlapadaptors

import eu.iv4xr.framework.model.rl.RLMDP
import eu.iv4xr.framework.model.rl.StateWithGoalProgress
import eu.iv4xr.framework.model.rl.basicGoal
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms.qLearning
import eu.iv4xr.framework.model.rl.qValue
import nl.uu.cs.aplib.exampleUsages.DumbDoctorAction
import nl.uu.cs.aplib.exampleUsages.DumbDoctorModel
import nl.uu.cs.aplib.exampleUsages.DumbDoctorState
import org.junit.Assert
import org.junit.jupiter.api.Test
import kotlin.random.Random

internal class BurlapAlgorithmsTest {

    @Test
    fun testMDP() {
        val mdp = RLMDP(DumbDoctorModel(), listOf(basicGoal(10.0) { it: Int -> it >= 5 }))
        val alg = qLearning<StateWithGoalProgress<DumbDoctorState>, DumbDoctorAction>(0.9, 1.0, 0.0, 300, Random(1234))
        val policy = alg.train(mdp)
        if (policy !is BurlapPolicy) {
            kotlin.test.fail("We expect greedy qpolicy")
        }
        val qs = policy.policy
        if (qs !is GreedyQPolicyWithQValues) {
            kotlin.test.fail("We expect greedy qpolicy")
        }
        mdp.possibleStates().forEach { s ->
            mdp.possibleActions(s).forEach {
                Assert.assertEquals(mdp.qValue(s, it, 0.9, 5), qs.qValue(s, it), 0.01)
            }
        }
    }
}