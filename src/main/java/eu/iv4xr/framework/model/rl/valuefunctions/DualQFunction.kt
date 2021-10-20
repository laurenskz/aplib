package eu.iv4xr.framework.model.rl.valuefunctions

import eu.iv4xr.framework.model.rl.Identifiable

interface CopyableTrainableQFunction<S : Identifiable, A : Identifiable> : TrainableQFunction<S, A> {
    fun copyFrom(other: QFunction<S, A>) {}
}


class DualQFunction<S : Identifiable, A : Identifiable>(val model: CopyableTrainableQFunction<S, A>, val targetModel: CopyableTrainableQFunction<S, A>) {
}