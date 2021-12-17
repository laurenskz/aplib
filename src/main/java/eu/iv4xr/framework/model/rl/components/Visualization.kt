package eu.iv4xr.framework.model.rl.components

import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.valuefunctions.Valuefunction

interface Image {
    fun writeTo(path: String)
    fun display(label: String)
}

interface Visualizer<S : Identifiable> {
    fun visualize(valueFunction: Valuefunction<S>): Image
}