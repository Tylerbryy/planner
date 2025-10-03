"""
LLM-Assisted Planner

Main planner implementation that integrates LLM guidance with traditional planning.
Implements the approach from the paper with problem decomposition and two paradigms:
- LLM4Inspire: General knowledge-based guidance
- LLM4Predict: Domain-specific knowledge-based predictions
"""

import dspy
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from modules import (
    ProblemDecomposer,
    LLM4InspireModule,
    LLM4PredictModule,
    PlanValidator,
    HeuristicGenerator,
    SubTaskRefiner,
    SearchSpacePruner,
    HybridGuidanceModule
)


class GuidanceMode(Enum):
    """Enumeration of guidance modes for LLM-assisted planning."""
    INSPIRE = "inspire"  # LLM4Inspire: General knowledge
    PREDICT = "predict"  # LLM4Predict: Domain-specific knowledge
    HYBRID = "hybrid"    # Both inspire and predict


@dataclass
class PlanningState:
    """Represents a state in the planning process."""
    state_description: str
    actions_taken: List[str]
    heuristic_value: float = float('inf')
    parent: Optional['PlanningState'] = None


@dataclass
class SubTask:
    """Represents a sub-task in the decomposed planning problem."""
    description: str
    initial_state: str
    goal_state: str
    priority: int = 0
    completed: bool = False


class LLMAssistedPlanner(dspy.Module):
    """
    Main LLM-Assisted Planner that orchestrates the planning process.

    This planner uses problem decomposition and LLM guidance to solve
    large-scale planning problems by partitioning the search space.
    """

    def __init__(self,
                 guidance_mode: GuidanceMode = GuidanceMode.PREDICT,
                 max_iterations: int = 100,
                 max_sub_tasks: int = 10,
                 enable_refinement: bool = True):
        """
        Initialize the LLM-Assisted Planner.

        Args:
            guidance_mode: Which LLM guidance paradigm to use
            max_iterations: Maximum planning iterations
            max_sub_tasks: Maximum number of sub-tasks to decompose into
            enable_refinement: Whether to refine sub-tasks based on feedback
        """
        super().__init__()

        self.guidance_mode = guidance_mode
        self.max_iterations = max_iterations
        self.max_sub_tasks = max_sub_tasks
        self.enable_refinement = enable_refinement

        # Initialize modules
        self.decomposer = ProblemDecomposer()
        self.validator = PlanValidator()
        self.heuristic_gen = HeuristicGenerator()

        # Initialize guidance modules based on mode
        if guidance_mode == GuidanceMode.INSPIRE:
            self.pruner = SearchSpacePruner(mode="inspire")
            self.inspire_module = LLM4InspireModule()
        elif guidance_mode == GuidanceMode.PREDICT:
            self.pruner = SearchSpacePruner(mode="predict")
            self.predict_module = LLM4PredictModule()
        elif guidance_mode == GuidanceMode.HYBRID:
            self.hybrid_module = HybridGuidanceModule()
            self.inspire_module = LLM4InspireModule()
            self.predict_module = LLM4PredictModule()

        if enable_refinement:
            self.refiner = SubTaskRefiner()

    def forward(self,
                initial_state: str,
                goal_state: str,
                available_actions: str,
                domain_description: str = "",
                domain_constraints: str = "") -> dspy.Prediction:
        """
        Plan a solution from initial state to goal state.

        Args:
            initial_state: Starting state description
            goal_state: Desired goal state description
            available_actions: Description of available actions
            domain_description: Description of the planning domain
            domain_constraints: Domain-specific constraints and rules

        Returns:
            Prediction containing the plan, sub_tasks, and planning_trace
        """
        # Step 1: Decompose the problem into sub-tasks
        decomposition = self.decomposer(
            initial_state=initial_state,
            goal_state=goal_state,
            available_actions=available_actions,
            domain_description=domain_description
        )

        # Parse sub-tasks
        sub_tasks = self._parse_sub_tasks(decomposition.sub_tasks, initial_state, goal_state)

        # Step 2: Solve each sub-task with LLM guidance
        overall_plan = []
        planning_trace = []
        current_state = initial_state

        for i, sub_task in enumerate(sub_tasks):
            trace_entry = {
                "sub_task_index": i,
                "sub_task": sub_task.description,
                "attempts": []
            }

            # Solve the sub-task
            sub_plan, final_state, attempt_trace = self._solve_sub_task(
                sub_task=sub_task,
                current_state=current_state,
                available_actions=available_actions,
                domain_constraints=domain_constraints,
                next_sub_task=sub_tasks[i + 1] if i + 1 < len(sub_tasks) else None
            )

            trace_entry["attempts"] = attempt_trace
            planning_trace.append(trace_entry)

            if sub_plan:
                overall_plan.extend(sub_plan)
                current_state = final_state
                sub_task.completed = True
            else:
                # Failed to find solution for sub-task
                trace_entry["status"] = "failed"
                break

        # Step 3: Validate the overall plan
        plan_str = " -> ".join(overall_plan) if overall_plan else "No plan found"

        validation = self.validator(
            plan=plan_str,
            initial_state=initial_state,
            goal_state=goal_state,
            domain_constraints=domain_constraints
        )

        return dspy.Prediction(
            plan=overall_plan,
            plan_string=plan_str,
            sub_tasks=[st.description for st in sub_tasks],
            decomposition_reasoning=decomposition.reasoning,
            is_valid=validation.is_valid,
            validation_errors=validation.validation_errors,
            plan_quality_score=validation.plan_quality_score,
            planning_trace=planning_trace,
            guidance_mode=self.guidance_mode.value
        )

    def _parse_sub_tasks(self,
                        sub_tasks_str: str,
                        initial_state: str,
                        goal_state: str) -> List[SubTask]:
        """
        Parse sub-tasks from LLM output into structured objects.

        Args:
            sub_tasks_str: String description of sub-tasks
            initial_state: Overall initial state
            goal_state: Overall goal state

        Returns:
            List of SubTask objects
        """
        # Simple parsing - split by newlines or numbered items
        lines = [line.strip() for line in sub_tasks_str.split('\n') if line.strip()]

        sub_tasks = []
        for i, line in enumerate(lines[:self.max_sub_tasks]):
            # Remove numbering if present (e.g., "1.", "1)", "Task 1:")
            clean_line = line
            for prefix in [".", ")", ":"]:
                if prefix in clean_line:
                    parts = clean_line.split(prefix, 1)
                    if len(parts) > 1 and parts[0].strip().isdigit():
                        clean_line = parts[1].strip()
                        break

            sub_task = SubTask(
                description=clean_line,
                initial_state=initial_state if i == 0 else f"state_after_subtask_{i-1}",
                goal_state=goal_state if i == len(lines) - 1 else f"state_after_subtask_{i}",
                priority=i
            )
            sub_tasks.append(sub_task)

        return sub_tasks

    def _solve_sub_task(self,
                       sub_task: SubTask,
                       current_state: str,
                       available_actions: str,
                       domain_constraints: str,
                       next_sub_task: Optional[SubTask] = None) -> Tuple[List[str], str, List[Dict]]:
        """
        Solve a single sub-task using LLM-guided search.

        Args:
            sub_task: The sub-task to solve
            current_state: Current state before sub-task
            available_actions: Available actions
            domain_constraints: Domain constraints
            next_sub_task: Next sub-task (if any)

        Returns:
            Tuple of (plan_actions, final_state, attempt_trace)
        """
        attempt_trace = []
        max_attempts = 3

        for attempt in range(max_attempts):
            attempt_info = {
                "attempt_number": attempt + 1,
                "mode": self.guidance_mode.value
            }

            # Get LLM guidance based on mode
            if self.guidance_mode == GuidanceMode.INSPIRE:
                guidance = self.inspire_module(
                    current_state=current_state,
                    goal_state=sub_task.goal_state,
                    available_actions=available_actions
                )
                attempt_info["inspire_guidance"] = {
                    "suggested_actions": guidance.suggested_actions,
                    "heuristic": guidance.heuristic_value,
                    "reasoning": guidance.reasoning
                }

                # Use suggested actions as plan
                suggested = self._parse_actions(guidance.suggested_actions)
                plan = suggested[:5]  # Take top 5 actions

            elif self.guidance_mode == GuidanceMode.PREDICT:
                guidance = self.predict_module(
                    sub_task=sub_task.description,
                    previous_state=current_state,
                    next_sub_task=next_sub_task.description if next_sub_task else sub_task.goal_state,
                    domain_constraints=domain_constraints
                )
                attempt_info["predict_guidance"] = {
                    "intermediate_conditions": guidance.intermediate_conditions,
                    "required_actions": guidance.required_actions,
                    "confidence": guidance.confidence
                }

                # Use required actions as plan
                plan = self._parse_actions(guidance.required_actions)

            elif self.guidance_mode == GuidanceMode.HYBRID:
                hybrid_guidance = self.hybrid_module(
                    current_state=current_state,
                    goal_state=sub_task.goal_state,
                    available_actions=available_actions,
                    sub_task=sub_task.description,
                    previous_state=current_state,
                    next_sub_task=next_sub_task.description if next_sub_task else sub_task.goal_state,
                    domain_constraints=domain_constraints
                )

                inspire_result = hybrid_guidance["inspire"]
                predict_result = hybrid_guidance["predict"]

                attempt_info["hybrid_guidance"] = {
                    "inspire": {
                        "suggested_actions": inspire_result.suggested_actions,
                        "reasoning": inspire_result.reasoning
                    },
                    "predict": {
                        "required_actions": predict_result.required_actions if predict_result else None,
                        "confidence": predict_result.confidence if predict_result else None
                    }
                }

                # Combine both guidances
                inspire_actions = self._parse_actions(inspire_result.suggested_actions)
                predict_actions = self._parse_actions(
                    predict_result.required_actions if predict_result else ""
                )
                plan = self._merge_action_lists(inspire_actions, predict_actions)

            attempt_info["generated_plan"] = plan
            attempt_trace.append(attempt_info)

            # Simulate plan execution to get final state
            # In a real planner, this would execute actions and check validity
            if plan:
                final_state = f"{current_state}_after_{sub_task.description}"
                return plan, final_state, attempt_trace

            # If plan failed and refinement is enabled, refine the sub-task
            if self.enable_refinement and attempt < max_attempts - 1:
                refinement = self.refiner(
                    original_sub_task=sub_task.description,
                    search_results=str(attempt_info),
                    failures=f"Attempt {attempt + 1} failed to generate valid plan"
                )
                sub_task.description = refinement.refined_sub_task
                attempt_info["refinement"] = refinement.adjustments

        # Failed after all attempts
        return [], current_state, attempt_trace

    def _parse_actions(self, actions_str: str) -> List[str]:
        """
        Parse action string into list of actions.

        Args:
            actions_str: String containing actions

        Returns:
            List of action strings
        """
        if not actions_str:
            return []

        # Split by common delimiters
        actions = []
        for line in actions_str.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Remove numbering or bullets
            for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()

            # Split by commas if present
            if ',' in line:
                actions.extend([a.strip() for a in line.split(',') if a.strip()])
            else:
                actions.append(line)

        return actions

    def _merge_action_lists(self, list1: List[str], list2: List[str]) -> List[str]:
        """
        Merge two action lists, prioritizing actions that appear in both.

        Args:
            list1: First action list (from inspire)
            list2: Second action list (from predict)

        Returns:
            Merged action list
        """
        # Prioritize actions in both lists
        common = [a for a in list1 if a in list2]
        unique_1 = [a for a in list1 if a not in list2]
        unique_2 = [a for a in list2 if a not in list1]

        return common + unique_1 + unique_2


class IncrementalPlanner(LLMAssistedPlanner):
    """
    Incremental planner that solves problems incrementally with continuous LLM feedback.

    Extends the base planner with iterative refinement and progressive search.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_history = []

    def forward(self, *args, **kwargs) -> dspy.Prediction:
        """
        Plan with incremental search and continuous refinement.
        """
        # Call parent forward method
        result = super().forward(*args, **kwargs)

        # Add search history to result
        result.search_history = self.search_history

        return result

    def _solve_sub_task(self, *args, **kwargs):
        """
        Solve sub-task with search history tracking.
        """
        plan, final_state, trace = super()._solve_sub_task(*args, **kwargs)

        # Record in search history
        self.search_history.append({
            "sub_task": args[0].description if args else "unknown",
            "plan": plan,
            "trace": trace
        })

        return plan, final_state, trace
