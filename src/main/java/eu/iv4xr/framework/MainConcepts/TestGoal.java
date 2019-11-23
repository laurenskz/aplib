package eu.iv4xr.framework.MainConcepts;

import java.util.function.* ;

import static eu.iv4xr.framework.MainConcepts.ObservationEvent.* ;
import nl.uu.cs.aplib.MainConcepts.*;

/**
 * A TestGoal is a Goal, but you can additionally specify a test oracle. When a TestGoal
 * receives a proposal from a test-agent (through the method {@link #propose_(Object)}),
 * and if this proposal is judged as solving the goal, the method {@link #propose_(Object)}
 * will automatically pass the proposal to be inspected by the test-oracle.
 * 
 * <p>A test-oracle expresses a test on the correctness of the proposal (not to be confused with
 * the goal-predicate itself). E.g. the goal may only require that a certain object becomes
 * accessible to the agent, whereas the oracle would test whether this object has the correct
 * properties.
 * 
 * A test-oracle
 * is essentially a function f that will inspect a given proposal-object and returns
 * a verdict. 
 * 
 * @author Wish
 *
 */
public class TestGoal extends Goal {

	/**
	 * Create a blank test-goal with the specified name.
	 */
	public TestGoal(String name) {
		super(name);
	}
	
	TestAgent owningTestAgent ;
	Function<Object,VerdictEvent> oracle ;
	
	/**
	 * Specify the test-oracle associated with for this TestGoal. Note that every TestGoal
	 * must have an oracle.
	 * 
	 * @param testagent  The test-agent to which this TestGoal will be associated to (the one that will work on this goal).
	 * @param oracle     The oracle predicate/function.
	 */
	public TestGoal oracle_(TestAgent testagent, Function<Object,VerdictEvent> oracle) {
		this.oracle = oracle ;
		owningTestAgent = testagent ;
		return this ;
	}
	
	/**
	 * Specify the test-oracle associated with for this TestGoal. Note that every TestGoal
	 * must have an oracle.
	 * 
	 * @param testagent  The test-agent to which this TestGoal will be associated to (the one that will work on this goal).
	 * @param oracle     The oracle predicate/function.
	 */
	public <Proposal> TestGoal oracle(TestAgent testagent, Function<Proposal,VerdictEvent> oracle) {
		return oracle_(testagent, o -> oracle.apply((Proposal) o)) ;
	}	
	
	/**
	 * We override {@link Goal#propose_(Object)} so that it now automatically check the
	 * oracle when the given proposal solves this Goal.
	 */
	@Override
	protected void propose_(Object proposal) {
		if (oracle == null) throw new IllegalArgumentException("The field oracle is null.") ;
		if (owningTestAgent == null) throw new IllegalArgumentException("The field owningTestAgent is null.") ;
		super.propose_(proposal);
		if (getStatus().success()) {
			var verdict = oracle.apply(proposal) ;
			owningTestAgent.registerVerdict(verdict);
		}
	}
	
	/**
	 * Set the given function as a goal function. A proposal o is a solution if abs(goalfunction(o))
	 * is a value less than epsilon (default is 0.005).
	 * The more general
	 * typing of the method's signature is for convenience, to allow you to explicitly
	 * specify the type of the goal's proposals domain at the point where this method is
	 * called, e.g. as in:
	 * 
	 * <pre>
	 *   Goal g = new Goal() . ftoSolve((Integer x) -> x - 9999) ;
	 * </pre>
	 * 
	 * The method returns this Goal itself so that it can be used in the Fluent Interface style.
	 */
	@Override
	public <Proposal> TestGoal ftoSolve(Function<Proposal,Double> predicateToSolve) {
		return (TestGoal) super.ftoSolve(predicateToSolve);
	}

	/**
	 * Set the predicate which would serve as the predicate to solve. The more general
	 * typing of the method's signature is for convenience, to allow you to explicitly
	 * specify the type of the goal's proposals domain at the point where this method is
	 * called, e.g. as in:
	 * 
	 * <pre>
	 *   Goal g = new Goal() . toSolve((Integer x) -> x==9999) ;
	 * </pre>
	 * 
	 * The method returns this Goal itself so that it can be used in the Fluent Interface style.
	 */
	@Override
	public <Proposal> TestGoal toSolve(Predicate<Proposal> predicateToSolve) {
		return (TestGoal) super.toSolve(predicateToSolve) ;
	}

	/**
	 * Set the value of eplison.
	 * The method returns this Goal itself so that it can be used in the Fluent Interface style.
	 */
	@Override
	public TestGoal withEpsilon(Double e) {
		return (TestGoal) super.withEpsilon(e) ;
	}
	
	/**
	 * Set the strategy to that a solving agent can use to solve this goal.
	 * The method returns this Goal itself so that it can be used in the Fluent Interface style.
	 */
	@Override
	public TestGoal withTactic(Tactic S) {
		return (TestGoal) super.withTactic(S) ;
	}
	
}
