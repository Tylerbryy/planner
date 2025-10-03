"""
DSPy Modules for LLM-Assisted Planning

This module implements the core planning components using DSPy modules.
Each module encapsulates specific planning functionality using defined signatures.
"""

import dspy
from typing import List, Dict, Any, Tuple
from signatures import (
    DecomposeProblem,
    InspireHeuristic,
    PredictIntermediateConditions,
    ValidatePlan,
    GenerateSearchHeuristic,
    RefineSubTask
)


class ProblemDecomposer(dspy.Module):
    """
    Decomposes large-scale planning problems into manageable sub-tasks.

    This module breaks down complex planning problems by identifying
    intermediate goals that partition the search space.
    """

    def __init__(self):
        super().__init__()
        self.decompose = dspy.ChainOfThought(DecomposeProblem)

    def forward(self, initial_state: str, goal_state: str,
                available_actions: str, domain_description: str) -> dspy.Prediction:
        """
        Decompose a planning problem into sub-tasks.

        Args:
            initial_state: Starting state of the problem
            goal_state: Desired final state
            available_actions: Available actions in the domain
            domain_description: Description of the planning domain

        Returns:
            Prediction with sub_tasks and reasoning
        """
        result = self.decompose(
            initial_state=initial_state,
            goal_state=goal_state,
            available_actions=available_actions,
            domain_description=domain_description
        )
        return result


class LLM4InspireModule(dspy.Module):
    """
    LLM4Inspire: Uses general LLM knowledge for heuristic guidance.

    Provides search guidance based on the LLM's general world knowledge
    without requiring domain-specific training.
    """

    def __init__(self):
        super().__init__()
        self.inspire = dspy.ChainOfThought(InspireHeuristic)

    def forward(self, current_state: str, goal_state: str,
                available_actions: str) -> dspy.Prediction:
        """
        Generate heuristic guidance using general knowledge.

        Args:
            current_state: Current state in planning
            goal_state: Target goal state
            available_actions: Available actions from current state

        Returns:
            Prediction with suggested_actions, heuristic_value, and reasoning
        """
        result = self.inspire(
            current_state=current_state,
            goal_state=goal_state,
            available_actions=available_actions
        )
        return result


class LLM4PredictModule(dspy.Module):
    """
    LLM4Predict: Uses domain-specific knowledge for intermediate condition inference.

    Predicts necessary intermediate states and conditions by leveraging
    domain-specific patterns and constraints.
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(PredictIntermediateConditions)

    def forward(self, sub_task: str, previous_state: str,
                next_sub_task: str, domain_constraints: str) -> dspy.Prediction:
        """
        Predict intermediate conditions for sub-task transition.

        Args:
            sub_task: Current sub-task
            previous_state: State from previous sub-task
            next_sub_task: Next sub-task to achieve
            domain_constraints: Domain-specific rules

        Returns:
            Prediction with intermediate_conditions, required_actions, and confidence
        """
        result = self.predict(
            sub_task=sub_task,
            previous_state=previous_state,
            next_sub_task=next_sub_task,
            domain_constraints=domain_constraints
        )
        return result


class PlanValidator(dspy.Module):
    """
    Validates generated plans against domain constraints.

    Checks plan validity, goal achievement, and constraint satisfaction.
    """

    def __init__(self):
        super().__init__()
        self.validate = dspy.ChainOfThought(ValidatePlan)

    def forward(self, plan: str, initial_state: str,
                goal_state: str, domain_constraints: str) -> dspy.Prediction:
        """
        Validate a proposed plan.

        Args:
            plan: Sequence of actions
            initial_state: Starting state
            goal_state: Desired goal state
            domain_constraints: Domain rules

        Returns:
            Prediction with is_valid, validation_errors, and plan_quality_score
        """
        result = self.validate(
            plan=plan,
            initial_state=initial_state,
            goal_state=goal_state,
            domain_constraints=domain_constraints
        )
        return result


class HeuristicGenerator(dspy.Module):
    """
    Generates search heuristic values for state evaluation.

    Used to prioritize states during search based on estimated goal distance.
    """

    def __init__(self):
        super().__init__()
        self.generate_heuristic = dspy.Predict(GenerateSearchHeuristic)

    def forward(self, state: str, goal_state: str,
                domain_info: str) -> dspy.Prediction:
        """
        Generate heuristic value for a state.

        Args:
            state: State to evaluate
            goal_state: Target goal
            domain_info: Domain information

        Returns:
            Prediction with heuristic_value and key_factors
        """
        result = self.generate_heuristic(
            state=state,
            goal_state=goal_state,
            domain_info=domain_info
        )
        return result


class SubTaskRefiner(dspy.Module):
    """
    Refines sub-tasks based on search feedback.

    Adjusts sub-task definitions when initial attempts fail or struggle.
    """

    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(RefineSubTask)

    def forward(self, original_sub_task: str, search_results: str,
                failures: str) -> dspy.Prediction:
        """
        Refine a sub-task based on feedback.

        Args:
            original_sub_task: Original sub-task definition
            search_results: Results from search attempts
            failures: Failed attempts and reasons

        Returns:
            Prediction with refined_sub_task and adjustments
        """
        result = self.refine(
            original_sub_task=original_sub_task,
            search_results=search_results,
            failures=failures
        )
        return result


class SearchSpacePruner(dspy.Module):
    """
    Prunes the search space using LLM guidance.

    Combines multiple modules to intelligently reduce search space
    by eliminating unlikely states and actions.
    """

    def __init__(self, mode: str = "inspire"):
        """
        Initialize the search space pruner.

        Args:
            mode: Either "inspire" (LLM4Inspire) or "predict" (LLM4Predict)
        """
        super().__init__()
        self.mode = mode

        if mode == "inspire":
            self.guidance_module = LLM4InspireModule()
        elif mode == "predict":
            self.guidance_module = LLM4PredictModule()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'inspire' or 'predict'")

        self.heuristic_gen = HeuristicGenerator()

    def forward(self, current_state: str, goal_state: str,
                available_actions: str, **kwargs) -> dspy.Prediction:
        """
        Prune search space and suggest promising directions.

        Args:
            current_state: Current planning state
            goal_state: Target goal state
            available_actions: Available actions
            **kwargs: Additional args for specific modes (e.g., domain_constraints)

        Returns:
            Prediction with pruning guidance
        """
        if self.mode == "inspire":
            # Use general knowledge for guidance
            guidance = self.guidance_module(
                current_state=current_state,
                goal_state=goal_state,
                available_actions=available_actions
            )
            return guidance

        elif self.mode == "predict":
            # Use domain-specific predictions
            sub_task = kwargs.get('sub_task', '')
            previous_state = kwargs.get('previous_state', current_state)
            next_sub_task = kwargs.get('next_sub_task', goal_state)
            domain_constraints = kwargs.get('domain_constraints', '')

            guidance = self.guidance_module(
                sub_task=sub_task,
                previous_state=previous_state,
                next_sub_task=next_sub_task,
                domain_constraints=domain_constraints
            )
            return guidance


class HybridGuidanceModule(dspy.Module):
    """
    Combines both LLM4Inspire and LLM4Predict for enhanced guidance.

    Uses both general knowledge and domain-specific predictions
    to provide robust search guidance.
    """

    def __init__(self):
        super().__init__()
        self.inspire_module = LLM4InspireModule()
        self.predict_module = LLM4PredictModule()

    def forward(self, current_state: str, goal_state: str,
                available_actions: str, sub_task: str = "",
                previous_state: str = "", next_sub_task: str = "",
                domain_constraints: str = "") -> Dict[str, Any]:
        """
        Get guidance from both inspire and predict modules.

        Returns:
            Dictionary containing both guidance results
        """
        # Get general knowledge guidance
        inspire_result = self.inspire_module(
            current_state=current_state,
            goal_state=goal_state,
            available_actions=available_actions
        )

        # Get domain-specific predictions if sub-task info provided
        predict_result = None
        if sub_task and domain_constraints:
            predict_result = self.predict_module(
                sub_task=sub_task or goal_state,
                previous_state=previous_state or current_state,
                next_sub_task=next_sub_task or goal_state,
                domain_constraints=domain_constraints
            )

        return {
            "inspire": inspire_result,
            "predict": predict_result
        }
