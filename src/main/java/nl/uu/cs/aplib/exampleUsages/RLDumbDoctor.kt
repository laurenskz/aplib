package nl.uu.cs.aplib.exampleUsages

import eu.iv4xr.framework.model.ProbabilisticModel
import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.always
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.RLAgent
import nl.uu.cs.aplib.AplibEDSL.goal
import nl.uu.cs.aplib.environments.ConsoleEnvironment
import nl.uu.cs.aplib.exampleUsages.DumbDoctor.DoctorBelief
import nl.uu.cs.aplib.exampleUsages.DumbDoctorAction.*
import nl.uu.cs.aplib.mainConcepts.SimpleState

data class DumbDoctorState(val happines: Int) : Identifiable

enum class DumbDoctorAction : Identifiable {
    OPENING, QUESTION, SMART_QUESTION
}

class DumbDoctorModel : ProbabilisticModel<DumbDoctorState, DumbDoctorAction> {
    override fun possibleStates() = generateSequence(0) { it + 1 }.map(::DumbDoctorState)


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

    override fun isTerminal(state: DumbDoctorState) = false

    override fun transition(current: DumbDoctorState, action: DumbDoctorAction) = when (action) {
        OPENING -> if (current.happines == 0) DumbDoctorState(1) else current
        QUESTION -> if (current.happines > 0) DumbDoctorState(current.happines + 1) else current
        SMART_QUESTION -> if (current.happines > 0) DumbDoctorState(current.happines + 1) else current
    }.let { always(it) }

    override fun proposal(current: DumbDoctorState, action: DumbDoctorAction, result: DumbDoctorState): Distribution<out Any> {
        return always(result.happines)
    }
}

fun main() {
    val g = goal("the-goal").toSolve { happiness: Int -> happiness >= 5 }
    val topgoal = g.lift()
    topgoal.maxbudget(10.0)
    val belief = DoctorBelief()
    belief.setEnvironment(ConsoleEnvironment())
    val doctorAgent = RLAgent(DumbDoctorModel())
            .attachState(belief)
            .setGoal(topgoal)
//            .trainWith()

    // run the doctor-agent until it solves its goal:
    while (topgoal.status.inProgress()) {
        doctorAgent.update()
    }
}