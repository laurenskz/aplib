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
        } else if (goal.goal.status.failed()) {
            goal.status.setToFail(goal.goal.status.info)
        }else if (goal.goal.status.inProgress()) {
            goal.status.resetToInProgress()
        }
    } else if (goal.subgoals.all { it.status.success() }) {
        goal.status.setToSuccess()
    } else if (goal.subgoals.any { it.status.failed() }) {
        goal.status.setToFail("Some subgoal failed")
    } else if (goal.subgoals.any() { it.status.inProgress() }) {
        goal.status.resetToInProgress()
    }
}

fun allPossibleGoalStates(count: Int) = allPossible(count, listOf(true, false))

fun <R> allPossible(count: Int, element: List<R>): Sequence<List<R>> = if (count == 0) sequenceOf(listOf()) else
    allPossible(count - 1, element)
            .flatMap { big -> element.map { small -> small cons big } }