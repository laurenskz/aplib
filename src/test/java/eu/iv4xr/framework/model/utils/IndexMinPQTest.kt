package eu.iv4xr.framework.model.utils

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

internal class IndexMinPQTest {

    @Test
    fun test() {
        val queue = IndexMinPQ<Double>(10000)
        queue.insert(0, 9.0)
        queue.insert(1, 10.0)
        assertFalse { queue.isEmpty }
        assertEquals(queue.delMin(), 0)
        assertEquals(queue.delMin(), 1)
        assertTrue { queue.isEmpty }
        queue.insert(0, 9.0)
        queue.insert(1, 10.0)
        queue.changeKey(1, 5.0)
        assertEquals(1, queue.delMin())
        assertFalse { queue.isEmpty }
    }
}