"""
Example usage of the LLM-Assisted Planner

Demonstrates how to use the planner with different guidance modes on
sample planning problems including blocks world and logistics domains.
"""

import dspy
from planner import LLMAssistedPlanner, GuidanceMode, IncrementalPlanner
import json


def setup_dspy(model_name: str = "gpt-4o-mini"):
    """
    Configure DSPy with a language model.

    Args:
        model_name: Name of the model to use
    """
    # Configure the language model with custom parameters
    lm = dspy.LM(
        f'openai/{model_name}',
        max_tokens=50000,
        temperature=1.0
    )
    dspy.settings.configure(lm=lm)
    print(f"Configured DSPy with {model_name} (max_tokens=50000, temperature=1.0)")


def blocks_world_example():
    """
    Example: Classic Blocks World planning problem.

    Initial state: A on table, B on table, C on A
    Goal state: A on B, B on C, C on table
    """
    print("\n" + "="*70)
    print("BLOCKS WORLD EXAMPLE")
    print("="*70)

    initial_state = """
    Blocks: A, B, C
    Configuration:
    - Block C is on Block A
    - Block A is on the table
    - Block B is on the table
    - Robot arm is empty
    """

    goal_state = """
    Target configuration:
    - Block A is on Block B
    - Block B is on Block C
    - Block C is on the table
    """

    available_actions = """
    Available actions:
    1. pickup(block): Pick up a block from the table if arm is empty and block has nothing on top
    2. putdown(block): Put down the currently held block on the table
    3. stack(block1, block2): Place block1 on top of block2 if holding block1 and block2 has nothing on top
    4. unstack(block1, block2): Pick up block1 from block2 if arm is empty and block1 is on block2
    """

    domain_description = """
    Blocks World domain where blocks can be stacked on each other or placed on a table.
    Only one block can be moved at a time using a robot arm.
    A block can only be picked up if nothing is on top of it.
    """

    domain_constraints = """
    Constraints:
    - Can only hold one block at a time
    - Cannot pick up a block that has another block on top
    - Cannot stack a block on another block that already has something on top
    - All blocks must end up in valid positions
    """

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "available_actions": available_actions,
        "domain_description": domain_description,
        "domain_constraints": domain_constraints
    }


def logistics_example():
    """
    Example: Logistics planning problem.

    Transport packages between locations using trucks.
    """
    print("\n" + "="*70)
    print("LOGISTICS EXAMPLE")
    print("="*70)

    initial_state = """
    Locations: CityA, CityB, CityC
    Packages: Package1 at CityA, Package2 at CityA
    Trucks: Truck1 at CityA, Truck2 at CityB
    """

    goal_state = """
    Target state:
    - Package1 should be at CityC
    - Package2 should be at CityB
    """

    available_actions = """
    Available actions:
    1. load_truck(package, truck, location): Load package onto truck at location
    2. unload_truck(package, truck, location): Unload package from truck at location
    3. drive_truck(truck, from_location, to_location): Drive truck from one location to another
    """

    domain_description = """
    Logistics domain where packages need to be transported between cities using trucks.
    Trucks can carry multiple packages and drive between locations.
    """

    domain_constraints = """
    Constraints:
    - Packages can only be loaded/unloaded when truck is at the same location
    - Trucks can only carry packages while driving
    - Each package must reach its destination
    - Minimize total driving distance
    """

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "available_actions": available_actions,
        "domain_description": domain_description,
        "domain_constraints": domain_constraints
    }


def robot_navigation_example():
    """
    Example: Robot navigation in a grid world.
    """
    print("\n" + "="*70)
    print("ROBOT NAVIGATION EXAMPLE")
    print("="*70)

    initial_state = """
    Grid: 5x5 grid world
    Robot position: (0, 0) - bottom left
    Obstacles: (1,1), (2,2), (3,1)
    Items to collect: Key at (4,0), Door at (4,4)
    """

    goal_state = """
    Target state:
    - Robot at position (4, 4)
    - Key collected
    - Door opened (requires key)
    """

    available_actions = """
    Available actions:
    1. move_up(): Move robot one cell up (increase y)
    2. move_down(): Move robot one cell down (decrease y)
    3. move_left(): Move robot one cell left (decrease x)
    4. move_right(): Move robot one cell right (increase x)
    5. pick_item(): Pick up item at current location
    6. use_item(item): Use an item from inventory
    """

    domain_description = """
    Grid world where a robot navigates to collect items and reach a goal.
    The robot must avoid obstacles and collect necessary items to complete the task.
    """

    domain_constraints = """
    Constraints:
    - Cannot move into obstacle cells
    - Cannot move outside grid boundaries (0-4 for both x and y)
    - Must collect key before opening door
    - Must minimize number of moves
    """

    return {
        "initial_state": initial_state,
        "goal_state": goal_state,
        "available_actions": available_actions,
        "domain_description": domain_description,
        "domain_constraints": domain_constraints
    }


def run_planner_example(problem_data: dict, guidance_mode: GuidanceMode, problem_name: str):
    """
    Run the planner on a problem with specified guidance mode.

    Args:
        problem_data: Dictionary with problem specification
        guidance_mode: Which guidance mode to use
        problem_name: Name of the problem for display
    """
    print(f"\n{'='*70}")
    print(f"Running {problem_name} with {guidance_mode.value.upper()} mode")
    print(f"{'='*70}\n")

    # Initialize planner
    planner = LLMAssistedPlanner(
        guidance_mode=guidance_mode,
        max_iterations=50,
        max_sub_tasks=8,
        enable_refinement=True
    )

    # Run planner
    result = planner(
        initial_state=problem_data["initial_state"],
        goal_state=problem_data["goal_state"],
        available_actions=problem_data["available_actions"],
        domain_description=problem_data["domain_description"],
        domain_constraints=problem_data["domain_constraints"]
    )

    # Display results
    print(f"\n{'─'*70}")
    print("PLANNING RESULTS")
    print(f"{'─'*70}")

    print(f"\n📋 Sub-tasks identified ({len(result.sub_tasks)}):")
    for i, task in enumerate(result.sub_tasks, 1):
        print(f"  {i}. {task}")

    print(f"\n💡 Decomposition reasoning:")
    print(f"  {result.decomposition_reasoning}")

    print(f"\n🎯 Generated plan ({len(result.plan)} actions):")
    for i, action in enumerate(result.plan, 1):
        print(f"  {i}. {action}")

    print(f"\n✓ Plan validation:")
    print(f"  Valid: {result.is_valid}")
    if result.validation_errors:
        print(f"  Errors: {result.validation_errors}")
    print(f"  Quality score: {result.plan_quality_score}")

    print(f"\n🔍 Planning trace summary:")
    for trace in result.planning_trace:
        status = "✓" if trace.get("status") != "failed" else "✗"
        print(f"  {status} Sub-task {trace['sub_task_index']}: {trace['sub_task']}")
        print(f"     Attempts: {len(trace['attempts'])}")

    print(f"\n{'─'*70}\n")

    return result


def compare_guidance_modes(problem_data: dict, problem_name: str):
    """
    Compare all three guidance modes on the same problem.

    Args:
        problem_data: Problem specification
        problem_name: Name of the problem
    """
    print(f"\n{'='*70}")
    print(f"COMPARING GUIDANCE MODES: {problem_name}")
    print(f"{'='*70}\n")

    modes = [GuidanceMode.INSPIRE, GuidanceMode.PREDICT, GuidanceMode.HYBRID]
    results = {}

    for mode in modes:
        result = run_planner_example(problem_data, mode, problem_name)
        results[mode.value] = {
            "plan_length": len(result.plan),
            "is_valid": result.is_valid,
            "num_sub_tasks": len(result.sub_tasks),
            "quality_score": result.plan_quality_score
        }

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Mode':<15} {'Plan Length':<15} {'Valid':<10} {'Sub-tasks':<12} {'Quality':<10}")
    print(f"{'-'*70}")

    for mode_name, metrics in results.items():
        print(f"{mode_name:<15} {metrics['plan_length']:<15} "
              f"{str(metrics['is_valid']):<10} {metrics['num_sub_tasks']:<12} "
              f"{metrics['quality_score']:<10}")

    print(f"\n{'='*70}\n")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("LLM-ASSISTED PLANNER - EXAMPLE DEMONSTRATIONS")
    print("="*70)

    # Setup DSPy with GPT-5
    setup_dspy("gpt-5")

    # Choose examples to run
    print("\nAvailable examples:")
    print("1. Blocks World (classic planning problem)")
    print("2. Logistics (package delivery)")
    print("3. Robot Navigation (grid world)")
    print("4. Compare all modes on Blocks World")

    # For demonstration, run Blocks World with PREDICT mode
    print("\n" + "="*70)
    print("Running Blocks World example with LLM4Predict mode...")
    print("="*70)

    blocks_problem = blocks_world_example()
    result = run_planner_example(blocks_problem, GuidanceMode.PREDICT, "Blocks World")

    # Optionally, run comparison
    print("\nWould you like to compare all guidance modes? (This will make multiple LLM calls)")
    print("Uncomment the line below to run full comparison:\n")
    print("# compare_guidance_modes(blocks_problem, 'Blocks World')")

    # Show how to run other examples
    print("\n" + "="*70)
    print("To run other examples, uncomment the relevant lines:")
    print("="*70)
    print("""
    # Logistics example
    # logistics_problem = logistics_example()
    # run_planner_example(logistics_problem, GuidanceMode.HYBRID, "Logistics")

    # Robot navigation example
    # robot_problem = robot_navigation_example()
    # run_planner_example(robot_problem, GuidanceMode.INSPIRE, "Robot Navigation")

    # Full comparison
    # compare_guidance_modes(blocks_problem, "Blocks World")
    """)


if __name__ == "__main__":
    main()
