package nl.uu.cs.aplib.exampleUsages

import burlap.behavior.policy.GreedyQPolicy
import burlap.behavior.singleagent.learning.tdmethods.QLearning
import burlap.mdp.singleagent.environment.SimulatedEnvironment
import burlap.statehashing.simple.SimpleHashableStateFactory
import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.RLAgent
import eu.iv4xr.framework.model.rl.StateWithGoalProgress
import eu.iv4xr.framework.model.rl.algorithms.GreedyAlg
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlg
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms.qLearning
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapEnum
import eu.iv4xr.framework.model.rl.burlapadaptors.ImmutableReflectionBasedState
import eu.iv4xr.framework.model.rl.qValue
import nl.uu.cs.aplib.AplibEDSL.goal
import nl.uu.cs.aplib.environments.ConsoleEnvironment
import nl.uu.cs.aplib.exampleUsages.DumbDoctor.DoctorBelief
import nl.uu.cs.aplib.exampleUsages.DumbDoctorAction.*
import nl.uu.cs.aplib.mainConcepts.SimpleState
import kotlin.random.Random

data class DumbDoctorState(val happiness: Int) : Identifiable, ImmutableReflectionBasedState

enum class DumbDoctorAction : Identifiable, BurlapEnum<DumbDoctorAction> {
    OPENING, QUESTION, SMART_QUESTION;

    override fun get() = this
}

class DumbDoctorModel : ProbabilisticModel<DumbDoctorState, DumbDoctorAction> {
    override fun possibleStates() = (0..5).map(::DumbDoctorState).asSequence()


    override fun possibleActions(state: DumbDoctorState) = values().asSequence()

    override fun executeAction(action: DumbDoctorAction, state: SimpleState): Int {
        if (state !is DoctorBelief) throw IllegalArgumentException()
        val question = when (action) {
            OPENING -> "How do you feel today?"
            QUESTION -> "Please explain a bit more..."
            SMART_QUESTION -> "I see... And why is that?"
        }
        state.env().ask(question)
        return ++state.patientHappiness

    }

    override fun convertState(state: SimpleState): DumbDoctorState {
        if (state !is DoctorBelief) throw IllegalArgumentException()
        return DumbDoctorState(state.patientHappiness)
    }

    override fun isTerminal(state: DumbDoctorState) = state.happiness >= 5

    override fun transition(current: DumbDoctorState, action: DumbDoctorAction) = when (action) {
        OPENING -> if (current.happiness == 0) DumbDoctorState(1) else current
        QUESTION -> if (current.happiness > 0) DumbDoctorState(current.happiness + 1) else current
        SMART_QUESTION -> if (current.happiness > 0) DumbDoctorState(current.happiness + 1) else current
    }.let { always(it) }

    override fun proposal(current: DumbDoctorState, action: DumbDoctorAction, result: DumbDoctorState): Distribution<out Any> {
        return always(result.happiness)
    }

    override fun possibleActions() = DumbDoctorAction.values().asSequence()
    override fun initialState() = always(DumbDoctorState(0))
}

fun main() {
    val g = goal("the-goal").toSolve { happiness: Int -> happiness >= 5 }
    val topgoal = g.lift()
    topgoal.maxbudget(10.0)
    val belief = DoctorBelief()
    belief.setEnvironment(ConsoleEnvironment())
    val dumbDoctorModel = DumbDoctorModel()
    val doctorAgent = RLAgent(dumbDoctorModel, Random(123))
            .attachState(belief)
            .setGoal(topgoal)
            .trainWith(qLearning(0.9, 0.1, 0.0, 10000, Random(1234)))
//            .trainWith(GreedyAlg(0.9, 5), 10)
    println("Printing from model")
    doctorAgent.mdp.possibleStates().forEach { s ->
        doctorAgent.mdp.possibleActions(s).forEach {
            println("Model: $s,$it: ${doctorAgent.mdp.qValue(s, it, 0.9, 10)}")
        }
    }

    println(doctorAgent.mdp.qValue(StateWithGoalProgress(listOf(false), DumbDoctorState(0)), OPENING, 0.9, 10))

    // run the doctor-agent until it solves its goal:
    while (topgoal.status.inProgress()) {
        doctorAgent.update()
    }
}