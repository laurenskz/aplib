package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.distribution.*
import eu.iv4xr.framework.model.rl.Identifiable
import eu.iv4xr.framework.model.rl.MDP
import eu.iv4xr.framework.model.rl.Policy
import eu.iv4xr.framework.model.rl.RLAlgorithm
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.policies.GreedyPolicy
import eu.iv4xr.framework.model.rl.valuefunctions.*
import java.util.*
import kotlin.random.Random

class ExplorationPolicy<S : Identifiable, A : Identifiable>(val mdp: MDP<S, A>, val icmModule: ICMModule<S, A>) : Policy<S, A> {
    override fun action(state: S): Distribution<A> {
//        we would like to explore from the most promising state, this is the state from which we don't know what will happen
        val intrinsicRewards = mdp.possibleActions(state).map { a ->
            val transition = mdp.transition(state, a)
            a to transition.expectedValue { sp ->
                val reward = mdp.reward(state, a, sp).expectedValue()
                icmModule.intrinsicReward(listOf(BurlapAlgorithms.SARS(state, a, sp, reward, transition.score(sp)).toICM())).first()
            }
        }.toList()
        return always(intrinsicRewards.maxByOrNull { it.second }!!.first)
//        return Distributions.softmax(intrinsicRewards.toMap())
    }
}

class ICMEGreedyPolicy<S : Identifiable, A : Identifiable>(val mdp: MDP<S, A>, val random: Random, val icmModule: ICMModule<S, A>, val greedy: Policy<S, A>) : Policy<S, A> {
    override fun action(state: S): Distribution<A> {
        val sars = mdp.executeAction(mdp.possibleActions(state).toList().random(random), state, random)
        val epsilon = icmModule.intrinsicReward(sars.toICM())
        return flip(epsilon).chain {
            if (it) Distributions.uniform(mdp.possibleActions(state).toList())
            else greedy.action(state)
        }
    }

}

class ExplorationNode<S : Identifiable, A : Identifiable>(val state: S) : Comparable<ExplorationNode<S, A>> {
    var predecessor: ExplorationNode<S, A>? = null
    var sars: BurlapAlgorithms.SARS<S, A>? = null
    var sOut = 2.0
    val comparator = compareBy<ExplorationNode<S, A>> { it.priority }.reversed()

    val priority: Double
        get() = sOut

    override fun compareTo(other: ExplorationNode<S, A>): Int {
        return comparator.compare(this, other)
    }

    override fun toString(): String {
        return "ExplorationNode(state=$state, priority=$priority)"
    }


}

class ICMMDP<S : Identifiable, A : Identifiable>(val icmModule: ICMModule<S, A>, val mdp: MDP<S, A>) : MDP<S, A> by mdp {
    override fun reward(current: S, action: A, newState: S): Distribution<Double> {
        return always(icmModule.train(listOf(ICMSample(current, action, newState))).first())
    }
}

interface GoalDiscoverer<S : Identifiable, A : Identifiable> {
    fun explore(): BurlapAlgorithms.Episode<S, A>
}

class RandomStartICM<S : Identifiable, A : Identifiable>(
        val mdp: MDP<S, A>,
        val icmModule: ICMModule<S, A>,
        private val exploreFun: TrainableQFunction<S, A>,
        val gamma: Float,
        val random: Random,
        val maxSteps: Int = Int.MAX_VALUE,
        val epsilon: Double = 0.2,
) : GoalDiscoverer<S, A> {
    private val ePolicy = EGreedyPolicy(epsilon, mdp, GreedyPolicy(exploreFun, mdp))

    override fun explore(): BurlapAlgorithms.Episode<S, A> {
        val starts = mutableMapOf<S, List<BurlapAlgorithms.SARS<S, A>>>()
        mdp.initialState().support().forEach {
            starts[it] = emptyList()
        }
        val qLearning = OffPolicyQLearning(exploreFun, gamma, mdp, random)
        var i = 0
        while (i++ < maxSteps) {
            val start = starts.keys.random(random)
            var state = start
            val steps = mutableListOf<BurlapAlgorithms.SARS<S, A>>()
            steps.addAll(starts[start] ?: error("No prefix for episode"))
            while (!mdp.isTerminal(state)) {
                val sars = mdp.sampleSARS(ePolicy, state, random)
                val reward = icmModule.train(listOf(sars.toICM())).first()
                qLearning.train(sars.copy(r = reward))
                steps.add(sars)
                state = sars.sp
            }
            val episode = BurlapAlgorithms.Episode(steps)
            if (episode.steps.any { it.r > 0 }) {
                return episode
            }
            episode.steps.forEachIndexed { index, s -> starts[s.s] = episode.steps.subList(0, index) }
        }
        return BurlapAlgorithms.Episode(emptyList())
    }

}

class ExplorationWithQueue<S : Identifiable, A : Identifiable>(
        val mdp: MDP<S, A>,
        val icmModule: ICMModule<S, A>,
        val exploreFun: TrainableQFunction<S, A>,
        val random: Random,
        val gamma: Double,
        val maxSteps: Int = Int.MAX_VALUE,
        val epsilon: Double = 0.2,
) {
    val queue = PriorityQueue<ExplorationNode<S, A>>()
    val steps = mutableSetOf<BurlapAlgorithms.SARS<S, A>>()
    val ePolicy = EGreedyPolicy(epsilon, mdp, GreedyPolicy(exploreFun, mdp))

    init {

        mdp.initialState().support().forEach {
            queue.add(ExplorationNode(it))
        }
    }

    fun explore(): BurlapAlgorithms.Episode<S, A> {
        val qLearning = OffPolicyQLearning(exploreFun, gamma.toFloat(), mdp, random)
        var i = 0
        while (i < maxSteps) {
            println(i++)
            var state = mdp.initialState().sample(random)
            val steps = mutableListOf<BurlapAlgorithms.SARS<S, A>>()
            var done = false
            while (!mdp.isTerminal(state)) {
                val sars = mdp.sampleSARS(ePolicy, state, random)
                val reward = icmModule.train(listOf(sars.toICM())).first()
                qLearning.train(sars.copy(r = reward))
                steps.add(sars)
                done = done || sars.r > 0
                state = sars.sp
            }
            if (done) return BurlapAlgorithms.Episode(steps)
        }
        return BurlapAlgorithms.Episode(listOf())
    }

    fun exploreMC(): BurlapAlgorithms.Episode<S, A> {
        val policy = RandomPolicy(mdp)
        var i = 0
        while (true) {
            println(i++)
            val episode = mdp.sampleEpisode(policy, random)
            if (episode.steps.last().r > 0) {
                return episode
            }
        }
    }

    fun exploreWithQueue(): BurlapAlgorithms.Episode<S, A> {
        while (!queue.isEmpty()) {
            val node = queue.remove()
            println(node)
            val state = node.state
            if (node.priority < 1.0) {
                val message = queue.maxByOrNull { it.priority }
                println(message)
            }
            if (mdp.isTerminal(state)) continue
            val sars = mdp.sampleSARS(ePolicy, state, random)
            icmModule.train(listOf(sars.toICM()))
            val new = ExplorationNode<S, A>(sars.sp)
            new.sars = sars
            node.sOut = icmModule.intrinsicReward(mdp.possibleActions(sars.s).map { mdp.executeAction(it, sars.s, random).toICM() }.toList()).maxOf { it }
            new.sOut = icmModule.intrinsicReward(mdp.possibleActions(sars.sp).map { mdp.executeAction(it, sars.sp, random).toICM() }.toList()).maxOf { it }
            new.predecessor = node
            queue.add(node)
            queue.add(new)
            if (sars.r > 0) {
                return createEpisode(new)
            }
        }
        error("No goal found :(")
    }

    fun createEpisode(node: ExplorationNode<S, A>): BurlapAlgorithms.Episode<S, A> {
        val steps = mutableListOf<BurlapAlgorithms.SARS<S, A>>()
        var current: ExplorationNode<S, A>? = node
        while (current != null) {
            current.sars?.also { steps.add(0, it) }
            current = current.predecessor
        }
        return BurlapAlgorithms.Episode(steps)
    }
}

class ExploreAndLearn<S : Identifiable, A : Identifiable>(
        val random: Random,
        val qFunction: TrainableQFunction<S, A>,
        val goalDiscoverer: GoalDiscoverer<S, A>,
        val gamma: Float,
        val trainRepititions: Int,
) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        val episode = goalDiscoverer.explore()
        val learning = OffPolicyQLearning(qFunction, gamma, mdp, random)
        repeat(trainRepititions) { learning.train(episode) }
        return GreedyPolicy(qFunction, mdp)
    }

}

class ExploreAndConnect<S : Identifiable, A : Identifiable>(
        val icmModule: ICMModule<S, A>,
        val random: Random,
        val qFunction: TrainableQFunction<S, A>,
        val exploreFun: TrainableQFunction<S, A>,
        val exploreGamma: Float,
        val gamma: Float,
        val trainRepititions: Int,
        val connectEpisodes: Int,
        val connectPolicy: Policy<S, A>,
        val maxSteps: Int = Int.MAX_VALUE,
        val epsilon: Double = 0.2

) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        val explorer = ExplorationWithQueue(mdp, icmModule, exploreFun, random, exploreGamma.toDouble(), maxSteps, epsilon)
        val episode = explorer.explore()
        val learning = OffPolicyQLearning(qFunction, gamma, mdp, random)
        repeat(trainRepititions) { learning.train(episode) }
        repeat(connectEpisodes) {
            val sampled = mdp.sampleEpisode(connectPolicy, Random)
            learning.train(sampled)
        }
        return GreedyPolicy(qFunction, mdp)
    }
}
