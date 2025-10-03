"""
DSPy Optimizer for LLM-Assisted Planner

This module provides optimization capabilities to tune the planner's performance
using DSPy's built-in optimizers like BootstrapFewShot and MIPROv2.
"""

import dspy
from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass
import json

from planner import LLMAssistedPlanner, GuidanceMode


@dataclass
class PlanningExample:
    """
    Training example for planner optimization.

    Contains a planning problem and its expected solution.
    """
    initial_state: str
    goal_state: str
    available_actions: str
    domain_description: str
    domain_constraints: str
    expected_plan: List[str]
    expected_sub_tasks: Optional[List[str]] = None


class PlannerOptimizer:
    """
    Optimizer for the LLM-Assisted Planner.

    Uses DSPy's optimization framework to improve planner performance
    on specific planning domains or problem types.
    """

    def __init__(self,
                 planner: LLMAssistedPlanner,
                 optimizer_type: str = "bootstrap",
                 num_threads: int = 4):
        """
        Initialize the planner optimizer.

        Args:
            planner: The planner instance to optimize
            optimizer_type: Type of optimizer ("bootstrap", "mipro", "random_search")
            num_threads: Number of threads for parallel optimization
        """
        self.planner = planner
        self.optimizer_type = optimizer_type
        self.num_threads = num_threads

    def plan_quality_metric(self,
                           example: PlanningExample,
                           prediction: dspy.Prediction,
                           trace: Optional[Any] = None) -> float:
        """
        Metric to evaluate plan quality.

        Combines multiple factors:
        - Plan validity
        - Plan length (shorter is better)
        - Goal achievement
        - Constraint satisfaction

        Args:
            example: The planning example with ground truth
            prediction: The planner's prediction
            trace: Optional execution trace

        Returns:
            Quality score (0.0 to 1.0, higher is better)
        """
        score = 0.0
        weights = {
            "validity": 0.4,
            "length": 0.2,
            "goal_achievement": 0.3,
            "decomposition": 0.1
        }

        # Validity check
        if prediction.is_valid == "true" or prediction.is_valid == True:
            score += weights["validity"]

        # Length check (compare to expected plan length)
        if example.expected_plan:
            expected_len = len(example.expected_plan)
            actual_len = len(prediction.plan)

            if actual_len == 0:
                length_score = 0.0
            elif actual_len <= expected_len:
                length_score = 1.0
            else:
                # Penalize longer plans
                length_score = max(0.0, 1.0 - (actual_len - expected_len) / expected_len)

            score += weights["length"] * length_score

        # Goal achievement (heuristic based on validation)
        if hasattr(prediction, 'plan_quality_score'):
            try:
                quality_str = str(prediction.plan_quality_score)
                # Try to extract numeric score
                if '/' in quality_str:
                    num, denom = quality_str.split('/')
                    goal_score = float(num) / float(denom)
                elif quality_str.replace('.', '').isdigit():
                    goal_score = float(quality_str)
                else:
                    goal_score = 0.5  # Default if can't parse
            except:
                goal_score = 0.5
        else:
            goal_score = 0.5

        score += weights["goal_achievement"] * goal_score

        # Decomposition quality (number of sub-tasks)
        if example.expected_sub_tasks:
            expected_subtasks = len(example.expected_sub_tasks)
            actual_subtasks = len(prediction.sub_tasks)

            if actual_subtasks == expected_subtasks:
                decomp_score = 1.0
            else:
                decomp_score = max(0.0, 1.0 - abs(actual_subtasks - expected_subtasks) / expected_subtasks)

            score += weights["decomposition"] * decomp_score
        else:
            # Reward reasonable decomposition (3-7 sub-tasks)
            actual_subtasks = len(prediction.sub_tasks)
            if 3 <= actual_subtasks <= 7:
                decomp_score = 1.0
            elif actual_subtasks < 3:
                decomp_score = actual_subtasks / 3.0
            else:
                decomp_score = max(0.0, 1.0 - (actual_subtasks - 7) / 7.0)

            score += weights["decomposition"] * decomp_score

        return score

    def optimize(self,
                trainset: List[PlanningExample],
                valset: Optional[List[PlanningExample]] = None,
                metric: Optional[Callable] = None,
                **optimizer_kwargs) -> LLMAssistedPlanner:
        """
        Optimize the planner on a training set.

        Args:
            trainset: Training examples
            valset: Validation examples (optional)
            metric: Custom metric function (optional)
            **optimizer_kwargs: Additional arguments for the optimizer

        Returns:
            Optimized planner
        """
        # Use default metric if none provided
        if metric is None:
            metric = self.plan_quality_metric

        # Convert examples to DSPy format
        train_examples = [self._example_to_dspy(ex) for ex in trainset]
        val_examples = [self._example_to_dspy(ex) for ex in valset] if valset else None

        # Choose optimizer
        if self.optimizer_type == "bootstrap":
            optimizer = dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
                num_threads=self.num_threads,
                **optimizer_kwargs
            )

        elif self.optimizer_type == "mipro":
            # MIPROv2 for more advanced optimization
            optimizer = dspy.MIPROv2(
                metric=metric,
                auto="medium",
                num_threads=self.num_threads,
                **optimizer_kwargs
            )

        elif self.optimizer_type == "random_search":
            # Random search with bootstrapping
            optimizer = dspy.BootstrapFewShotWithRandomSearch(
                metric=metric,
                max_bootstrapped_demos=8,
                num_candidate_programs=10,
                num_threads=self.num_threads,
                **optimizer_kwargs
            )

        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")

        # Run optimization
        print(f"\nOptimizing planner with {self.optimizer_type}...")
        print(f"Training examples: {len(train_examples)}")
        if val_examples:
            print(f"Validation examples: {len(val_examples)}")

        compile_kwargs = {"trainset": train_examples}
        if val_examples:
            compile_kwargs["valset"] = val_examples

        optimized_planner = optimizer.compile(self.planner, **compile_kwargs)

        print("Optimization complete!")

        return optimized_planner

    def _example_to_dspy(self, example: PlanningExample) -> dspy.Example:
        """
        Convert PlanningExample to DSPy Example format.

        Args:
            example: Planning example

        Returns:
            DSPy Example object
        """
        dspy_example = dspy.Example(
            initial_state=example.initial_state,
            goal_state=example.goal_state,
            available_actions=example.available_actions,
            domain_description=example.domain_description,
            domain_constraints=example.domain_constraints,
            expected_plan=example.expected_plan,
            expected_sub_tasks=example.expected_sub_tasks or []
        ).with_inputs(
            "initial_state",
            "goal_state",
            "available_actions",
            "domain_description",
            "domain_constraints"
        )

        return dspy_example

    def evaluate(self,
                testset: List[PlanningExample],
                metric: Optional[Callable] = None) -> Dict[str, float]:
        """
        Evaluate planner performance on a test set.

        Args:
            testset: Test examples
            metric: Metric function to use

        Returns:
            Dictionary with evaluation metrics
        """
        if metric is None:
            metric = self.plan_quality_metric

        test_examples = [self._example_to_dspy(ex) for ex in testset]

        scores = []
        valid_plans = 0
        total_plan_length = 0

        print(f"\nEvaluating planner on {len(test_examples)} examples...")

        for i, example in enumerate(test_examples):
            # Run planner
            prediction = self.planner(
                initial_state=example.initial_state,
                goal_state=example.goal_state,
                available_actions=example.available_actions,
                domain_description=example.domain_description,
                domain_constraints=example.domain_constraints
            )

            # Calculate score
            score = metric(testset[i], prediction)
            scores.append(score)

            # Track statistics
            if prediction.is_valid:
                valid_plans += 1
            total_plan_length += len(prediction.plan)

            print(f"  Example {i+1}/{len(test_examples)}: Score = {score:.3f}")

        # Calculate aggregate metrics
        avg_score = sum(scores) / len(scores) if scores else 0.0
        validity_rate = valid_plans / len(test_examples) if test_examples else 0.0
        avg_plan_length = total_plan_length / len(test_examples) if test_examples else 0.0

        results = {
            "average_score": avg_score,
            "validity_rate": validity_rate,
            "average_plan_length": avg_plan_length,
            "num_examples": len(test_examples),
            "individual_scores": scores
        }

        print(f"\nEvaluation Results:")
        print(f"  Average Score: {avg_score:.3f}")
        print(f"  Validity Rate: {validity_rate:.1%}")
        print(f"  Avg Plan Length: {avg_plan_length:.1f}")

        return results


def create_training_dataset(domain: str = "blocks_world") -> List[PlanningExample]:
    """
    Create a training dataset for a specific domain.

    Args:
        domain: Domain name ("blocks_world", "logistics", "navigation")

    Returns:
        List of training examples
    """
    if domain == "blocks_world":
        return [
            PlanningExample(
                initial_state="A on table, B on A, C on table",
                goal_state="A on B, B on C, C on table",
                available_actions="pickup, putdown, stack, unstack",
                domain_description="Blocks world with 3 blocks",
                domain_constraints="Can only hold one block, cannot pickup block with something on top",
                expected_plan=["unstack B from A", "putdown B", "pickup A", "stack A on B",
                             "pickup B", "stack B on C"],
                expected_sub_tasks=["Clear block A", "Move A to B", "Move B to C"]
            ),
            PlanningExample(
                initial_state="A on table, B on table, C on B",
                goal_state="A on C, C on B, B on table",
                available_actions="pickup, putdown, stack, unstack",
                domain_description="Blocks world with 3 blocks",
                domain_constraints="Can only hold one block, cannot pickup block with something on top",
                expected_plan=["pickup A", "stack A on C"],
                expected_sub_tasks=["Move A to C"]
            ),
        ]

    elif domain == "logistics":
        return [
            PlanningExample(
                initial_state="Package1 at CityA, Truck1 at CityA",
                goal_state="Package1 at CityB",
                available_actions="load, unload, drive",
                domain_description="Logistics delivery",
                domain_constraints="Package must be in truck to transport",
                expected_plan=["load Package1 Truck1 CityA", "drive Truck1 CityA CityB",
                             "unload Package1 Truck1 CityB"],
                expected_sub_tasks=["Load package", "Transport package", "Unload package"]
            ),
        ]

    else:
        raise ValueError(f"Unknown domain: {domain}")


def example_optimization_workflow():
    """
    Example workflow showing how to optimize a planner.
    """
    print("\n" + "="*70)
    print("PLANNER OPTIMIZATION WORKFLOW")
    print("="*70)

    # Setup DSPy with GPT-5
    lm = dspy.LM('openai/gpt-5', max_tokens=50000, temperature=1.0)
    dspy.settings.configure(lm=lm)

    # Create planner
    planner = LLMAssistedPlanner(
        guidance_mode=GuidanceMode.PREDICT,
        max_iterations=50,
        enable_refinement=True
    )

    # Create training data
    trainset = create_training_dataset("blocks_world")
    print(f"\nCreated training set with {len(trainset)} examples")

    # Create optimizer
    optimizer = PlannerOptimizer(
        planner=planner,
        optimizer_type="bootstrap",
        num_threads=2
    )

    # Evaluate before optimization
    print("\n" + "-"*70)
    print("BEFORE OPTIMIZATION")
    print("-"*70)
    before_results = optimizer.evaluate(trainset)

    # Optimize
    print("\n" + "-"*70)
    print("OPTIMIZING...")
    print("-"*70)
    optimized_planner = optimizer.optimize(trainset[:1], trainset[1:])  # Use first as train, rest as val

    # Evaluate after optimization
    print("\n" + "-"*70)
    print("AFTER OPTIMIZATION")
    print("-"*70)
    optimizer.planner = optimized_planner
    after_results = optimizer.evaluate(trainset)

    # Compare
    print("\n" + "="*70)
    print("OPTIMIZATION COMPARISON")
    print("="*70)
    print(f"\nAverage Score:")
    print(f"  Before: {before_results['average_score']:.3f}")
    print(f"  After:  {after_results['average_score']:.3f}")
    print(f"  Improvement: {(after_results['average_score'] - before_results['average_score']):.3f}")

    print(f"\nValidity Rate:")
    print(f"  Before: {before_results['validity_rate']:.1%}")
    print(f"  After:  {after_results['validity_rate']:.1%}")

    return optimized_planner


if __name__ == "__main__":
    # Run example optimization workflow
    example_optimization_workflow()
