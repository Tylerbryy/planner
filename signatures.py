"""
DSPy Signatures for LLM-Assisted Planning

This module defines the signatures (interfaces) for the LLM-based planning system.
Each signature specifies the input/output structure for different planning tasks.
"""

import dspy


class DecomposeProblem(dspy.Signature):
    """
    Decompose a large-scale planning problem into simpler sub-tasks.

    Given a complex planning problem with initial state, goal state, and available actions,
    break it down into a sequence of intermediate sub-goals that are easier to solve.
    """

    initial_state = dspy.InputField(desc="The initial state of the planning problem")
    goal_state = dspy.InputField(desc="The desired goal state to achieve")
    available_actions = dspy.InputField(desc="List of available actions in the domain")
    domain_description = dspy.InputField(desc="Description of the planning domain")

    sub_tasks = dspy.OutputField(desc="List of intermediate sub-goals/tasks in sequential order")
    reasoning = dspy.OutputField(desc="Explanation of the decomposition strategy")


class InspireHeuristic(dspy.Signature):
    """
    LLM4Inspire: Provide heuristic guidance using general knowledge.

    Uses the LLM's general knowledge to suggest promising actions or states
    without relying on domain-specific information.
    """

    current_state = dspy.InputField(desc="Current state in the planning process")
    goal_state = dspy.InputField(desc="Target goal state")
    available_actions = dspy.InputField(desc="Actions available from current state")

    suggested_actions = dspy.OutputField(desc="Ranked list of promising actions to explore")
    heuristic_value = dspy.OutputField(desc="Estimated distance/cost to goal")
    reasoning = dspy.OutputField(desc="General knowledge-based reasoning for suggestions")


class PredictIntermediateConditions(dspy.Signature):
    """
    LLM4Predict: Infer intermediate conditions using domain-specific knowledge.

    Uses domain-specific patterns and constraints to predict necessary intermediate
    states or conditions required to reach the goal from a sub-task.
    """

    sub_task = dspy.InputField(desc="Current sub-task to solve")
    previous_state = dspy.InputField(desc="State achieved in previous sub-task")
    next_sub_task = dspy.InputField(desc="Next sub-task to achieve")
    domain_constraints = dspy.InputField(desc="Domain-specific rules and constraints")

    intermediate_conditions = dspy.OutputField(desc="Predicted intermediate state conditions")
    required_actions = dspy.OutputField(desc="Actions likely needed to achieve conditions")
    confidence = dspy.OutputField(desc="Confidence level in predictions")


class ValidatePlan(dspy.Signature):
    """
    Validate a generated plan against domain constraints and requirements.

    Checks if a proposed plan is valid, achieves the goal, and follows domain rules.
    """

    plan = dspy.InputField(desc="Sequence of actions in the proposed plan")
    initial_state = dspy.InputField(desc="Starting state")
    goal_state = dspy.InputField(desc="Desired goal state")
    domain_constraints = dspy.InputField(desc="Domain rules and constraints")

    is_valid = dspy.OutputField(desc="Whether the plan is valid (true/false)")
    validation_errors = dspy.OutputField(desc="List of validation errors if invalid")
    plan_quality_score = dspy.OutputField(desc="Quality score (efficiency, optimality)")


class GenerateSearchHeuristic(dspy.Signature):
    """
    Generate a search heuristic value for state prioritization.

    Estimates how promising a state is for reaching the goal, used to guide search.
    """

    state = dspy.InputField(desc="State to evaluate")
    goal_state = dspy.InputField(desc="Target goal state")
    domain_info = dspy.InputField(desc="Domain information for context")

    heuristic_value = dspy.OutputField(desc="Numeric heuristic estimate (lower is better)")
    key_factors = dspy.OutputField(desc="Key factors influencing the estimate")


class RefineSubTask(dspy.Signature):
    """
    Refine a sub-task based on search feedback and exploration results.

    Adjusts sub-task goals or constraints based on what was learned during search.
    """

    original_sub_task = dspy.InputField(desc="Original sub-task definition")
    search_results = dspy.InputField(desc="Results from attempting the sub-task")
    failures = dspy.InputField(desc="Failed attempts and reasons")

    refined_sub_task = dspy.OutputField(desc="Improved sub-task definition")
    adjustments = dspy.OutputField(desc="Explanation of refinements made")
