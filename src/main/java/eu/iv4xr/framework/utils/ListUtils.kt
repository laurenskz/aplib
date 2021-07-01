package eu.iv4xr.framework.utils

infix fun <T> T.cons(tail: List<T>) = listOf(this) + tail