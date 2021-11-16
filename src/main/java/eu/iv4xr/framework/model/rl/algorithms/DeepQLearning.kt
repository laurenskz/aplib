package eu.iv4xr.framework.model.rl.algorithms

import eu.iv4xr.framework.model.distribution.Distribution
import eu.iv4xr.framework.model.distribution.Distributions.uniform
import eu.iv4xr.framework.model.distribution.expectedValue
import eu.iv4xr.framework.model.distribution.flip
import eu.iv4xr.framework.model.rl.*
import eu.iv4xr.framework.model.rl.approximation.FeatureVectorFactory
import eu.iv4xr.framework.model.rl.burlapadaptors.BurlapAlgorithms
import eu.iv4xr.framework.model.rl.burlapadaptors.DataClassHashableState
import eu.iv4xr.framework.model.rl.policies.GreedyPolicy
import eu.iv4xr.framework.model.rl.valuefunctions.*
import eu.iv4xr.framework.model.utils.IndexMinPQ
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import java.time.Duration
import kotlin.math.abs
import kotlin.random.Random


fun <E> Collection<E>.takeRandom(n: Int, random: Random): List<E> {
    return List(n) { this.random(random) }
}

interface DeepQModel<S : Identifiable, A : Identifiable> {
    fun fit(samples: List<BurlapAlgorithms.SARS<S, A>>, gamma: Float, mdp: MDP<S, A>)
    fun qValues(state: S, gamma: Float, mdp: MDP<S, A>): List<Pair<A, Float>>
}

interface ModelDescription {
    fun create(inputSize: Int, outputSize: Int): GraphTrainableModel
}

interface BattlingQModelInstance<S : Identifiable, A : Identifiable, Model> {
    fun create(): Model
    fun fit(model: Model, targetModel: Model, samples: List<BurlapAlgorithms.SARS<S, A>>, gamma: Float, mdp: MDP<S, A>)
    fun qValues(state: S, gamma: Float, mdp: MDP<S, A>, model: Model): List<Pair<A, Float>>
    fun updateTargetModel(model: Model, targetModel: Model)
}

abstract class BattlingQModelKeras<S : Identifiable, A : Identifiable> : BattlingQModelInstance<S, A, GraphTrainableModel> {
    override fun updateTargetModel(model: GraphTrainableModel, targetModel: GraphTrainableModel) {
        targetModel.layers.zip(model.layers).forEach { (t, m) ->
            t.weights = m.weights
        }
    }
}

class BaseDeepModel<S : Identifiable, A : Identifiable, T>(val instance: BattlingQModelInstance<S, A, T>) : DeepQModel<S, A>, TrainableQFunction<S, A> {
    val model = instance.create()
    override fun fit(samples: List<BurlapAlgorithms.SARS<S, A>>, gamma: Float, mdp: MDP<S, A>) {
        instance.fit(model, model, samples, gamma, mdp)
    }

    override fun qValues(state: S, gamma: Float, mdp: MDP<S, A>): List<Pair<A, Float>> {
        return instance.qValues(state, gamma, mdp, model)
    }

    override fun qValue(state: S, action: A): Float {
        TODO()
    }

    override fun train(target: QTarget<S, A>) {
        TODO("Not yet implemented")
    }
}

class BattlingQModel<S : Identifiable, A : Identifiable, M>(val instance: BattlingQModelInstance<S, A, M>) : DeepQModel<S, A> {
    val model = instance.create()
    val targetModel = instance.create()
    var fitCount = 0


    fun updateTargetModel() {
        instance.updateTargetModel(model, targetModel)
    }

    override fun fit(samples: List<BurlapAlgorithms.SARS<S, A>>, gamma: Float, mdp: MDP<S, A>) {
        instance.fit(model, targetModel, samples, gamma, mdp)
        if ((fitCount + 1) % 100 == 0) {
            fitCount = 0
            updateTargetModel()
        }
        fitCount++
    }

    override fun qValues(state: S, gamma: Float, mdp: MDP<S, A>): List<Pair<A, Float>> {
        return instance.qValues(state, gamma, mdp, model)
    }
}

class StateValueFunction<S : Identifiable, A : Identifiable>(val model: ModelDescription, val features: FeatureVectorFactory<S>) : BattlingQModelKeras<S, A>() {

    var sampleCount = 0
    override fun create(): GraphTrainableModel {
        return model.create(features.count(), 1)
    }

    override fun fit(model: GraphTrainableModel, targetModel: GraphTrainableModel, samples: List<BurlapAlgorithms.SARS<S, A>>, gamma: Float, mdp: MDP<S, A>) {
        val maxState = FloatArray(samples.size) { samples[it].r.toFloat() + if (mdp.isTerminal(samples[it].sp)) 0f else gamma * targetModel.predictSoftly(features.floatFeatures(samples[it].sp))[0] }
        val features = samples.map { features.floatFeatures(it.s) }.toTypedArray()
        model.fit(OnHeapDataset.create(features, maxState))
        sampleCount++
    }

    override fun qValues(state: S, gamma: Float, mdp: MDP<S, A>, model: GraphTrainableModel): List<Pair<A, Float>> {
        if (sampleCount < 100) {
            return mdp.possibleActions(state).map { a ->
                a to mdp.transition(state, a).expectedValue { sp ->
                    mdp.reward(state, a, sp).expectedValue()
                }.toFloat()
            }.toList()
        }
        val sps = mdp.possibleActions(state).flatMap {
            mdp.transition(state, it).support()
        }.toSet().toList()
        val actions = sps.map { features.floatFeatures(it) }.toTypedArray()
        val predictSoftly = model.predictSoftly(OnHeapDataset.create(actions, FloatArray(actions.size)), actions.size)
        val predictions = predictSoftly.map { it[0] }
        val spMap = sps.zip(predictions).toMap()
        return mdp.possibleActions(state).map { a ->
            a to mdp.transition(state, a).expectedValue { sp ->
                mdp.reward(state, a, sp).expectedValue() + gamma * (spMap[sp] ?: error("sp not found"))
            }.toFloat()
        }.toList()
    }
}


class AllActionOutputQModel<S : Identifiable, A : Identifiable>(val model: ModelDescription, val features: FeatureVectorFactory<S>, val allActions: List<A>) : BattlingQModelKeras<S, A>() {
    override fun create(): GraphTrainableModel {
        return model.create(features.count(), allActions.size)
    }

    override fun fit(model: GraphTrainableModel, targetModel: GraphTrainableModel, samples: List<BurlapAlgorithms.SARS<S, A>>, gamma: Float, mdp: MDP<S, A>) {
        val features = samples.map { features.floatFeatures(it.s) }.toTypedArray()
        val targets = samples.map { it.r.toFloat() + if (mdp.isTerminal(it.sp)) 0f else gamma * qValues(it.sp, gamma, mdp, targetModel).maxOf { it.second } }
        val currentQs = model.predictSoftly(OnHeapDataset.create(features, FloatArray(features.size)), features.size)
        val broadTargets = currentQs.mapIndexed { i, qs ->
            FloatArray(qs.size) { qI ->
                if (qI == allActions.indexOf(samples[i].a)) {
                    targets[i]
                } else qs[qI]
            }
        }
        TODO()
    }

    override fun qValues(state: S, gamma: Float, mdp: MDP<S, A>, model: GraphTrainableModel): List<Pair<A, Float>> {

        val possibleActions = mdp.possibleActions(state)
        return allActions.zip(model.predictSoftly(features.floatFeatures(state)).toList())
                .filter { it.first in possibleActions }
    }
}


//class SampleLearning<S : Identifiable, A : Identifiable>(val stateDist: Distribution<S>, val model: DeepQModel<S, A>, val random: Random, val states: Int, val gamma: Float, val batchSize: Int) : RLAlgorithm<S, A> {
//    override fun train(mdp: MDP<S, A>): Policy<S, A> {
//        val memory = mutableListOf<BurlapAlgorithms.SARS<S, A>>()
//        val policy = GreedyPolicy(model, mdp, gamma)
//        val ePolicy = EGreedyPolicy(0.1, mdp, policy)
//        var i = 0
//        while (i < states) {
//            val s = stateDist.sample(random)
//            val expectedReward = ePolicy.action(s).expectedValue { a ->
//                mdp.transition(s, a).expectedValue { sp ->
//                    mdp.reward(s, a, sp).expectedValue()
//                }
//            }
//            val takeSample = expectedReward != 0.0
//            if (expectedReward == 0.0 && i < batchSize) {
//                continue
//            }
//            val a = ePolicy.action(s).sample(random)
//            val sp = mdp.transition(s, a).sample(random)
//            val r = mdp.reward(s, a, sp).expectedValue()
//            memory.add(BurlapAlgorithms.SARS(s, a, sp, r))
//            if (i > batchSize && (i % 4 == 0))
//                model.fit(memory.takeRandom(batchSize, random), gamma, mdp)
//            i++
//        }
//        return ePolicy
//    }
//}

class DeepQLearning<S : Identifiable, A : Identifiable>(val qFunction: TrainableQFunction<S, A>, val random: Random, val gamma: Float, val batchSize: Int, val trainIterations: Int, val epsilon: Double, val QTargetCreator: QTargetCreator<S, A> = TDQTargetCreator(qFunction, gamma)) : RLAlgorithm<S, A> {

    override fun train(mdp: MDP<S, A>): GreedyPolicy<S, A> {
        val memory = mutableListOf<BurlapAlgorithms.Episode<S, A>>()
        val greedy = GreedyPolicy(qFunction, mdp)
        val policy = EGreedyPolicy(0.2, mdp, greedy)

        for (i in (0 until trainIterations)) {
            val sars = mdp.sampleEpisode(policy, random)
            memory.add(sars)
            performTraining(i, memory, mdp)
        }
        return greedy
    }

    private fun performTraining(i: Int, memory: List<BurlapAlgorithms.Episode<S, A>>, mdp: MDP<S, A>) {
        if (i > batchSize && (i % 4) == 0) {
            val samples = memory.takeRandom(batchSize, random)
            val targets = QTargetCreator.createTargets(samples, mdp)
            qFunction.train(targets)
        }
    }
}

class DeepSARSA<S : Identifiable, A : Identifiable>(val qFunction: TrainableQFunction<S, A>, val random: Random, val gamma: Float, val episodes: Int, val n: Int, val QTargetCreator: QTargetCreator<S, A> = NStepTDQTargetCreator(qFunction, gamma, n)) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        val greedy = GreedyPolicy(qFunction, mdp)
        val policy = EGreedyPolicy(1.0, mdp, greedy)
        (0..episodes).forEach {
            println(it)
            qFunction.train(QTargetCreator.createTargets(listOf(mdp.sampleEpisode(policy, random)), mdp))
            policy.epsilon = 1.0 - (it / episodes.toDouble())
        }
        return greedy
    }

}

class DeepBaba<S : Identifiable, A : Identifiable>(val qFunction: TrainableQFunction<S, A>, val random: Random, val gamma: Float, val episodes: Int, val QTargetCreator: QTargetCreator<S, A> = NStepTDQTargetCreator(qFunction, gamma, Int.MAX_VALUE), val maxSteps: Int = 1000) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        val greedy = GreedyPolicy(qFunction, mdp)
        val policy = EGreedyPolicy(1.0, mdp, greedy)
        val step = 0.9 / episodes
        for (i in (0..episodes)) {
            val samples = (0..1000).map {
                mdp.sampleEpisode(policy, Random, maxSteps)
            }
            policy.epsilon -= step
            val sample = samples.maxByOrNull { it.totalReward(gamma.toDouble()) } ?: continue
            println(policy.epsilon)
            println(sample.totalReward(gamma.toDouble()))
            qFunction.train(QTargetCreator.createTargets(listOf(sample), mdp))
        }
        return policy
    }
}

class Bab<S : DataClassHashableState, A : Identifiable>(val valuefunction: TrainableValuefunction<S>,
                                                        val gamma: Float,
                                                        val random: Random,
                                                        val episodes: Int) : RLAlgorithm<S, A> {
    override fun train(mdp: MDP<S, A>): Policy<S, A> {
//        val value = ValueTable<S>(1.0f)
//        val policy = GreedyPolicy(QFromValue(value, mdp, gamma), mdp)
        repeat(episodes) {
            val sampleEpisode = mdp.sampleEpisode(RandomPolicy(mdp), random, 100000)
            if (sampleEpisode.totalReward(gamma.toDouble()) > 0.01)
                sampleEpisode.steps.reversed().forEach { sars ->
                    valuefunction.train(expectedUpdate(sars.s, gamma, mdp, valuefunction))
                }

        }
//        println(bellmanResidual(value.states, value, mdp, gamma))
//        do {
//            val delta = valueIterationSweep(value.states, value, mdp, gamma)
//            println(delta)
//        } while (delta > 0.001)
//        println(bellmanResidual(value.states, value, mdp, gamma))
//        println(value.targets.size)
//        valuefunction.train(value.targets)
        return GreedyPolicy(QFromValue(valuefunction, mdp, gamma), mdp)
    }
}

class StateInformation<S : DataClassHashableState>(n: Int) {
    private val priorityQue = IndexMinPQ<Double>(n)
    private val mappings = mutableMapOf<S, Int>()
    private val mappingsReversed = mutableMapOf<Int, S>()
    private val available = (0 until n).toMutableList()
    val size: Int
        get() = mappings.size
    val isEmpty: Boolean
        get() = size == 0

    fun take(): S {
        val index = priorityQue.delMin()
        val s = mappingsReversed[index]!!
        mappingsReversed.remove(index)
        mappings.remove(s)
        available.add(index)
        return s
    }

    fun delete(state: S) {
        mappings[state]?.also(priorityQue::delete)
    }

    fun add(state: S, priority: Double) {
        mappings[state]?.also { if (-priority < priorityQue.keyOf(it)) priorityQue.decreaseKey(it, -priority) }
                ?: available.removeFirst().also {
                    priorityQue.insert(it, -priority)
                    mappings[state] = it
                    mappingsReversed[it] = state
                }
    }
}


class PredecessorCache<S : DataClassHashableState> {
    val predecessors = mutableMapOf<S, MutableSet<Pair<S, Float>>>()

    fun addPredecessor(state: S, predecessor: S, prob: Float) {
        val pair = predecessor to prob
        val set = predecessors[state]
        if (set != null) {
            if (!set.contains(pair)) {
                set.add(pair)
            }
        } else {
            predecessors[state] = mutableSetOf(pair)
        }
    }

    fun getPredecessors(state: S) = predecessors[state]?.toList() ?: emptyList<Pair<S, Float>>()
}

data class Sample<S : Identifiable>(val result: S, val predecessor: S, val reward: Double, val prob: Double)

interface ExperienceGenerator<S : Identifiable, A : Identifiable> {
    fun generate(mdp: MDP<S, A>): List<Sample<S>>
}

class PolicyBasedExperienceGenerator<S : Identifiable, A : Identifiable>(
        val maxSteps: Int,
        val random: Random,
        val gamma: Float,
        val totalTime: Duration,
        val ePolicy: (MDP<S, A>) -> (Policy<S, A>) = ::RandomPolicy) : ExperienceGenerator<S, A> {
    override fun generate(mdp: MDP<S, A>): List<Sample<S>> {
        val time = System.currentTimeMillis()
        val episode = sequence<BurlapAlgorithms.Episode<S, A>> {
            while (true) {
                yield(mdp.sampleEpisode(ePolicy(mdp), random, maxSteps))
            }
        }
                .takeWhile { System.currentTimeMillis() < time + totalTime.toMillis() }
                .maxByOrNull { it.totalReward(gamma.toDouble()) }
        println(episode?.totalReward(gamma.toDouble()))
        println((episode?.steps?.map { it.transitionProb }))
        return episode?.steps?.map { Sample(it.sp, it.s, it.transitionProb, it.r) } ?: emptyList()
    }
}

class FrontierBasedExperience<S : DataClassHashableState, A : Identifiable> : ExperienceGenerator<S, A> {
    override fun generate(mdp: MDP<S, A>): List<Sample<S>> {
        val frontier = mdp.initialState().support().toMutableList()
        val visited = mutableSetOf<S>()
        while (frontier.isNotEmpty()) {
            val state = frontier.first()
            visited.add(state)
            frontier.addAll(
                    mdp.possibleActions(state)
                            .flatMap { mdp.transition(state, it).support() }
                            .filter { !visited.contains(it) }
            )
        }
        println(visited.size)
        return emptyList()
    }
}

class PriorityBased<S : DataClassHashableState, A : Identifiable>(
        val valueFunction: TrainableValuefunction<S>,
        val gamma: Float,
        val experienceGenerator: ExperienceGenerator<S, A>,
        val epsilon: Double,
        val episodes: Int,
) : RLAlgorithm<S, A> {

    override fun train(mdp: MDP<S, A>): Policy<S, A> {
        val stateInformation = StateInformation<S>(10000)
        val predecessorCache = PredecessorCache<S>()
        repeat(episodes) {
            println(it)
            val states = experienceGenerator.generate(mdp)
            createGraph(states, predecessorCache)
            incorporateRewards(states, stateInformation, predecessorCache, mdp, valueFunction)
            updateStates(stateInformation, predecessorCache, mdp, valueFunction)

        }
//        println(table.targets.size)
//        println("Training the value function now!")
//        valueFunction.train(table.targets)
        return GreedyPolicy(QFromValue(valueFunction, mdp, gamma), mdp)
    }

    private fun updateStates(stateInformation: StateInformation<S>, predecessorCache: PredecessorCache<S>, mdp: MDP<S, A>, table: TrainableValuefunction<S>) {
        while (!stateInformation.isEmpty) {
            process(stateInformation.take(), stateInformation, predecessorCache, mdp, table)
        }
    }

    private fun incorporateRewards(states: List<Sample<S>>, stateInformation: StateInformation<S>, predecessorCache: PredecessorCache<S>, mdp: MDP<S, A>, table: TrainableValuefunction<S>) {
        states.filter { it.reward > 0 }.forEach {
            process(it.predecessor, stateInformation, predecessorCache, mdp, table)
        }
    }

    private fun createGraph(states: List<Sample<S>>, predecessorCache: PredecessorCache<S>) {
        states.forEach {
            predecessorCache.addPredecessor(it.result, it.predecessor, it.prob.toFloat())
        }
    }

    fun process(state: S, stateInformation: StateInformation<S>, predecessorCache: PredecessorCache<S>, mdp: MDP<S, A>, valueFunction: TrainableValuefunction<S>) {
        val current = valueFunction.value(state)
        val target = expectedUpdate(state, gamma, mdp, valueFunction)
        valueFunction.train(target)
        val change = abs(current - target.target)
        stateInformation.delete(state)
        predecessorCache.getPredecessors(state).forEach {
            val priority = change * it.second.toDouble()
            if (priority > epsilon) {
                stateInformation.add(it.first, priority)
            }
        }
    }
}


class EDecayPolicy<S : Identifiable, A : Identifiable>(var initialEpsilon: Double, var finalEpsilon: Double, val mdp: MDP<S, A>, val policy: Policy<S, A>) : Policy<S, A> {
    val ePolicy = EGreedyPolicy(initialEpsilon, mdp, policy)
    override fun action(state: S): Distribution<A> {
        return ePolicy.action(state)
    }

    fun setEpsilon(progress: Double) {
        ePolicy.epsilon = initialEpsilon + (finalEpsilon - initialEpsilon) * progress
    }

}

class EGreedyPolicy<S : Identifiable, A : Identifiable>(var epsilon: Double, val mdp: MDP<S, A>, val policy: Policy<S, A>) : Policy<S, A> {
    override fun action(state: S): Distribution<A> {
        return flip(epsilon).chain {
            val possibleActions = mdp.possibleActions(state).toList()
            if (it) {
                uniform(possibleActions.toList())
            } else {
                policy.action(state)
            }
        }
    }

    override fun allActions(state: List<S>): List<Distribution<A>> {
        val dists = policy.allActions(state)
        return dists.mapIndexed { i, d ->
            flip(epsilon).chain {
                if (it) uniform(mdp.possibleActions(state[i]).toList())
                else d
            }
        }
    }
}