package eu.iv4xr.framework.model.rl.burlapadaptors

import burlap.mdp.core.state.State
import eu.iv4xr.framework.model.rl.BurlapAction
import eu.iv4xr.framework.model.rl.BurlapState
import java.lang.reflect.Field
import kotlin.reflect.KCallable
import kotlin.reflect.KClass
import kotlin.reflect.KProperty
import kotlin.reflect.cast
import kotlin.reflect.full.isSuperclassOf
import kotlin.reflect.full.memberProperties

interface BurlapEnum<E : Enum<E>> : BurlapAction {
    fun get(): E
    override fun actionName(): String {
        return get().name
    }

    override fun copy(): BurlapEnum<E> {
        return this
    }
}

interface ReflectionBasedState : BurlapState {
    override fun variableKeys(): MutableList<Any> {
        return this::class.memberProperties.flatMap { property ->
            val child = property.call(this)
            if (child is BurlapState) {
                child.variableKeys().map {
                    { bab -> (BurlapState::get)(property.call(bab) as BurlapState, it) }
                }
            } else listOf { bab: Any ->
                property.call(bab)
            }
        }.toMutableList()
    }

    override fun get(p0: Any?): Any? {
        return (p0 as (Any) -> Any)(this)
    }
}

interface ImmutableReflectionBasedState : ReflectionBasedState {
    override fun copy(): State {
        return this
    }
}