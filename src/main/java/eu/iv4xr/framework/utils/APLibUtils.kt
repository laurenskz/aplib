package eu.iv4xr.framework.utils

import nl.uu.cs.aplib.mainConcepts.GoalStructure

fun <T> convert(goal: GoalStructure, onItem: (GoalStructure.PrimitiveGoal) -> T): List<T> {
    if (goal is GoalStructure.PrimitiveGoal) {
        return listOf(onItem(goal))
    }
    return goal.subgoals.flatMap { convert(it, onItem) }
}

fun updateGoalStatus(goal: GoalStructure) {
    goal.subgoals.forEach { updateGoalStatus(it) }
    if (goal is GoalStructure.PrimitiveGoal) {
        if (goal.goal.status.success()) {
            goal.status.setToSuccess()
        }
        if (goal.goal.status.failed()) {
            goal.status.setToFail(goal.goal.status.info)
        }
    } else if (goal.subgoals.all { it.status.success() }) {
        goal.status.setToSuccess()
    } else if (goal.subgoals.any { it.status.failed() }) {
        goal.status.setToFail("Some subgoal failed")
    }
}

fun allPossibleGoalStates(count: Int): Sequence<List<Boolean>> = if (count == 0) sequenceOf(listOf()) else
    allPossibleGoalStates(count - 1)
            .flatMap { listOf(true cons it, false cons it) }
