package nl.uu.cs.aplib.MainConcepts;

import java.util.* ;
import java.util.stream.Collectors;

/**
 * A GoalStructure is a generalization of a {@link Goal}. It is a tree-shaped structure
 * that conceptually represents a more complex goal. The simplest GoalStructure is an object
 * of type {@link PrimitiveGoal}. A {@link PrimitiveGoal} itself is a subclass
 * of GoalTree. Such a GoalStructure represents a single leaf, containing a single instance of {@link Goal},
 * which is the concrete goal represented by this leaf.
 * 
 * <p>
 * More complex GoalStructure can be constructed by combining subgoals. There are two types
 * of nodes available to combine sub-GoalStructure: the <b>SEQ</b> and <b>FIRSTOF</b> nodes:
 * 
 * <ol>
 *    <li> SEQ g1,g2,... represents a series of goals that all have to be solved,
 *       and solved in the order as they are listed.
 *    <li> FIRSTof g1,g2,... represents a series of alternative goals. They will be
 *        tried one at a time, starting from g1. If one is solved, the entire ALT
 *        is solved. If all subgoals fail, the ALT fails.
 * </ol>       
 *        
 * @author wish
 *
 */
public class GoalStructure {
	
	/**
	 * Represent the available types of {@link GoalStructure}. There are three types: SEQ, FIRSTOF, and
	 * PRIMITIVE. If a GoalStructure is marked as PRIMITIVE, then it is a leaf (in other words, it is
	 * a {@link PrimitiveGoal}). If a GoalStructure h is marked as SEQ, it represents a tree of the form
	 * SEQ g1,g2,... where g1,g2,... are h' subgoals. If h is marked as FIRSTOF, it represents a
	 * tree of the form FIRSTof g1,g2,....
	 */
	static public enum GoalsCombinator { SEQ, FIRSTOF, PRIMITIVE }
	
	GoalStructure parent = null ;
	List<GoalStructure> subgoals ;
	GoalsCombinator combinator ;
	ProgressStatus status = new ProgressStatus() ;
	
	/**
	 * Maximum limit on the budget that can be allocated to this goal structure.
	 */
	double bmax = Double.POSITIVE_INFINITY ;
	
	/**
	 * Total budget that is spent so far on this goal structure.
	 */
	double consumedBudget = 0 ;
	
	long consumedTime = 0 ;
	
	/**
	 * The budget that remains for this goal structure.
	 */
	double budget = Double.POSITIVE_INFINITY ;
	
	/**
	 * Construct a new GoalStructure with the specified type of node (SEQ, FIRSTOFF, or PRIMITIVE)
	 * and the given subgoals.
	 */
	public GoalStructure(GoalsCombinator type, GoalStructure ... subgoals) {
		combinator = type ;
		this.subgoals = new LinkedList<GoalStructure>() ;
		for (GoalStructure g : subgoals) {
			this.subgoals.add(g) ;
			g.parent = this ;
		}
	}
	
	/**
	 * Return the type of this GoalStructure (SEQ, FIRSTOF, or PRIMITIVE).
	 */
	public GoalsCombinator getCombinatorType() { return combinator ; }
	
	public List<GoalStructure> getSubgoals() { return subgoals ; }
	
	/**
	 * Return the parent of this GoalStructure. It returns null if it has no parent.
	 */
	public GoalStructure getParent() { return parent ; }
	
	/**
	 * True is this goal has no parent.
	 */
	public boolean isTopGoal() { return parent == null ; }
	
	/**
	 * Set the status of this goal to success, and propagating this accordingly
	 * to its ancestors.
	 */
	void setStatusToSuccess(String info) {
		status.setToSuccess(info) ;
		if (! isTopGoal()) {
			switch(parent.combinator) {
			   case FIRSTOF : parent.setStatusToSuccess(info); break ;
			   case SEQ : 
				    int i = parent.subgoals.indexOf(this) ;
				    if (i == parent.subgoals.size()-1)
					  	 parent.setStatusToSuccess(info); 
				    break ;
			}
		}
	}
	
	/** 
	 * Set the status of this goal to fail, and propagating this accordingly
	 * to its ancestors.
	 */
	void setStatusToFail(String reason) {
		status.setToFail(reason);
		if (! isTopGoal()) {
			switch(parent.combinator) {
			   case SEQ : parent.setStatusToFail(reason); break;
			   case FIRSTOF :
				    int i = parent.subgoals.indexOf(this) ;
					if (i == parent.subgoals.size()-1)
						parent.setStatusToFail(reason);
					break;
			}
		}
	}
	
	
	void setStatusToFailBecauseBudgetExhausted() {
	    String reason = "The budget is exhausted" ;
		status.setToFail(reason);
		if (! isTopGoal()) {
			if (parent.budget <= 0d) {
				parent.setStatusToFailBecauseBudgetExhausted(); 
				return ;
			}
			switch(parent.combinator) {
			   case SEQ : parent.setStatusToFail(reason); break;
			   case FIRSTOF :
				    int i = parent.subgoals.indexOf(this) ;
					if (i == parent.subgoals.size()-1)
						parent.setStatusToFail(reason);
					break;
			}
		}
	}
	
	/**
	 * To abort the entire goal tree; this is done by marking this goal, all
	 * the way to the root, as fail.
	 */
	void abort() {
		status.setToFail("abort() is invoked.") ; 
		if (! isTopGoal()) parent.abort() ;
 	}
	
	/**
	 * Get the status of this GoalStructure. The status is INPROGRESS if the GoalStructure is
	 * not solved or failed yet. It is SUCCESS if the GoalStructure was solved, and
	 * FAILED if the GoalStructure has been marked as such.
	 */
	public ProgressStatus getStatus() { return status ; }
	
	/**
	 * Assuming this goal is closed (that is, it has been solved or failed), this
	 * method will return the next {@link PrimitiveGoal} to solve. The method will traverse
	 * up through the parent of this GoalStructure to look for this next goal. If none
	 * is found, null is returned.
	 * 
	 * <p>
	 * If a new {@link PrimitiveGoal} can be found, it will be adopted. So, budget will
	 * also be allocated for it. Recursively, all its ancestors that just become current
	 * will also get freshly allocated budget.
	 */
	PrimitiveGoal getNextPrimitiveGoal_andAllocateBudget() {
		
		if (status.inProgress())
            // this method should not be called on a goal-structure that is still in progress
			throw new IllegalArgumentException() ;
		
		if (isTopGoal()) return null ;
		
		// So... this goal structure is either solved or failed, and is not the top-goal
		
		if (parent.status.success() || parent.status.failed()) 
			return parent.getNextPrimitiveGoal_andAllocateBudget() ;
		
		// this case implies that the parent doesn't fail. So, it must have some budget left!
		
		switch(parent.combinator) {
		  case SEQ :
			   if(status.failed())
				  // this case should have caught by the if-parent above; as it implies that the
				  // patent failed
				  return parent.getNextPrimitiveGoal_andAllocateBudget() ;
			   // else: so, this goal is solved:
			   int k = parent.subgoals.indexOf(this) ;
			   if (k == parent.subgoals.size() - 1 ) 
				  // this case should have been caught by the if-parent case above; as it implies
				  // that the parent succeeded
				  return parent.getNextPrimitiveGoal_andAllocateBudget() ;
			   else
				  return parent.subgoals.get(k+1).getDeepestFirstPrimGoal_andAllocateBudget() ;
		  case FIRSTOF :
			   if(status.success())
				  // this case should have been caught by the if-parent case above; as it implies
				  // that the parent succeeded
				  return parent.getNextPrimitiveGoal_andAllocateBudget() ;
			   // else: so, this goal failed:
			   k = parent.subgoals.indexOf(this) ;
			   if (k == parent.subgoals.size() - 1 ) 
					// this case should have caught by the if-parent above; as it implies that the
					// patent failed
					return parent.getNextPrimitiveGoal_andAllocateBudget() ;
			   else
					return parent.subgoals.get(k+1).getDeepestFirstPrimGoal_andAllocateBudget() ;
		}
		// this case should not happen
		return null ;
	}
	
	
	PrimitiveGoal getDeepestFirstPrimGoal_andAllocateBudget() {
		// allocate budget:
		if (isTopGoal())
			budget = Math.min(budget,bmax) ;
		else
			budget = Math.min(bmax,parent.budget) ; 
		// find the first deepest primitive subgoal:
		if (this instanceof PrimitiveGoal) 
			 return (PrimitiveGoal) this ;
		else return subgoals.get(0).getDeepestFirstPrimGoal_andAllocateBudget() ;
	}
	


	public GoalStructure maxbudget(double b) {
		if (b <= 0 || ! Double.isFinite(b)) throw new IllegalArgumentException() ;
		bmax = b ;
		return this ;
	}
	
	/**
	 * Register that the agent has consumed the given amount of budget. Delta will be subtracted
	 * from this goal-structure's budget, as well as that of its ancestors.
	 */
	void registerConsumedBudget(double delta) {
		consumedBudget += delta ;
		budget -= delta ;
		if (! isTopGoal()) parent.registerConsumedBudget(delta);
	}
	
	void registerUsedTime(long duration) {
		consumedTime += duration ;
		if (! isTopGoal()) parent.registerUsedTime(duration);
	}
	
	
	/**
	 * Return the remaining budget for this goal structure.
	 */
	public double getBudget() { return budget ; }
	
	private String space(int k) { String s = "" ; for(int i=0; i<k; i++) s += " " ; return s ; }
	
	String showGoalStructureStatusWorker(int level) {
		String indent =  space(3*(level+1)) ;
		String s = "" ;
		if (this instanceof PrimitiveGoal) {
			s += indent + ((PrimitiveGoal) this).goal.getName() + ": " + status ;
		}
		else 
			s += indent + combinator + ": " + status ; 
		if (bmax < Double.POSITIVE_INFINITY) s += "\n" + indent + "Max. budget:" + bmax ;
		s += "\n" + indent + "Consumed budget:" + consumedBudget + "\n" ;
		for (GoalStructure gt : subgoals) s += gt.showGoalStructureStatusWorker(level+1) + "\n" ;
		return s ;
	}
	
	private String indent(int indentation, String s) {
		String[] lines = s.split("\n") ;
		String z = "" ;
		for (int k=0; k<lines.length; k++) {
			z += space(k) ;
			z += lines[k] ;
			if (k>0) z += "\n" ;
		}
		return z ;
	}
	
	/**
	 * Format a summary of the state of this GoalStructure to a readable string.
	 */
	public String showGoalStructureStatus() { return showGoalStructureStatusWorker(0) ; }
	
	/**
	 * Print a summary of the state of this GoalStructure.
	 */
	public void printGoalStructureStatus() { 
		System.out.println("\n** Goal status:") ;
		System.out.println(showGoalStructureStatus()) ; 
	}
	
	/**
	 * A special subclass of {@link GoalStructure} to represent a leaf, wrapping around
	 * an instance of {@link Goal}.
	 */
	static public class PrimitiveGoal extends GoalStructure {
		Goal goal ;
		
		/**
		 * Create an instance of PrimitiveGoal, wrapping around the given {@link Goal}.
		 */
		public PrimitiveGoal(Goal g) { 
			super(GoalsCombinator.PRIMITIVE) ;
			goal = g ; 
		}
		
		PrimitiveGoal getDeepestFirstPrimGoal_andAllocateBudget() {
			return this ;
		}

	}

}
