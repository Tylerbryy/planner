# LLM-Assisted Planner with DSPy

A sophisticated planning system that leverages Large Language Models (LLMs) through DSPy to solve large-scale planning problems via problem decomposition and intelligent search space pruning.

**Based on**: *"Inspire or Predict? Exploring New Paradigms in Assisting Classical Planners with Large Language Models"*
Yu et al., arXiv:2508.11524 [cs.AI], August 2025

## Overview

This implementation is based on the research paper that addresses large-scale planning challenges by:

1. **Problem Decomposition**: Breaking complex planning problems into simpler sub-tasks
2. **LLM-Guided Search**: Using LLMs to prune the search space and provide heuristic guidance
3. **Two Paradigms**:
   - **LLM4Inspire**: Uses general LLM knowledge for heuristic guidance
   - **LLM4Predict**: Uses domain-specific knowledge to predict intermediate conditions

## Features

- ✅ **Problem Decomposition**: Automatically breaks down large planning problems into manageable sub-tasks
- ✅ **Multiple Guidance Modes**: LLM4Inspire, LLM4Predict, and Hybrid approaches
- ✅ **Plan Validation**: Validates generated plans against domain constraints
- ✅ **Search Space Pruning**: Intelligently reduces search space using LLM guidance
- ✅ **Optimization Support**: Built-in DSPy optimizers (BootstrapFewShot, MIPROv2, Random Search)
- ✅ **Multiple Domains**: Academic examples (Blocks World, Logistics, Robot Navigation) + Real-world business examples (Supply Chain, Manufacturing, Marketing, etc.)

## Architecture

### Core Components

```
planner/
├── signatures.py          # DSPy signatures defining LLM interfaces
├── modules.py             # DSPy modules implementing planning components
├── planner.py             # Main planner orchestrator
├── optimizer.py           # Optimization and training utilities
├── example.py             # Academic examples (Blocks World, Logistics, etc.)
├── business_examples.py   # Real-world business scenarios
├── requirements.txt       # Python dependencies
├── CITATION.bib          # BibTeX citation for the research paper
└── README.md             # This file
```

### Key Modules

1. **Signatures** (`signatures.py`):
   - `DecomposeProblem`: Decompose planning problems into sub-tasks
   - `InspireHeuristic`: Provide general knowledge-based guidance
   - `PredictIntermediateConditions`: Predict intermediate states with domain knowledge
   - `ValidatePlan`: Validate plan correctness
   - `GenerateSearchHeuristic`: Generate heuristic values for search
   - `RefineSubTask`: Refine sub-tasks based on feedback

2. **Modules** (`modules.py`):
   - `ProblemDecomposer`: Decomposes problems into sub-tasks
   - `LLM4InspireModule`: General knowledge guidance
   - `LLM4PredictModule`: Domain-specific predictions
   - `PlanValidator`: Plan validation
   - `SearchSpacePruner`: Search space reduction
   - `HybridGuidanceModule`: Combines both paradigms

3. **Planners** (`planner.py`):
   - `LLMAssistedPlanner`: Main planner with configurable guidance modes
   - `IncrementalPlanner`: Extends base planner with incremental search

## Installation

```bash
# Install required dependencies
pip install dspy-ai openai

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

## Quick Start

### Basic Usage

```python
import dspy
from planner import LLMAssistedPlanner, GuidanceMode

# Configure DSPy with GPT-5 (max_tokens=50000, temperature=1.0)
lm = dspy.LM('openai/gpt-5', max_tokens=50000, temperature=1.0)
dspy.settings.configure(lm=lm)

# Create planner with LLM4Predict mode
planner = LLMAssistedPlanner(
    guidance_mode=GuidanceMode.PREDICT,
    max_iterations=50,
    enable_refinement=True
)

# Define your planning problem
result = planner(
    initial_state="Block A on table, Block B on A",
    goal_state="Block B on table, Block A on B",
    available_actions="pickup, putdown, stack, unstack",
    domain_description="Blocks world with moveable blocks",
    domain_constraints="Can only hold one block at a time"
)

# View results
print("Generated Plan:", result.plan)
print("Sub-tasks:", result.sub_tasks)
print("Is Valid:", result.is_valid)
```

### Running Examples

```bash
# Run the Blocks World example
python example.py
```

The example script demonstrates:
- Blocks World planning
- Logistics delivery planning
- Robot navigation planning
- Comparison of all guidance modes

## Guidance Modes

### LLM4Inspire (General Knowledge)

Uses the LLM's general world knowledge to provide heuristic guidance:

```python
planner = LLMAssistedPlanner(guidance_mode=GuidanceMode.INSPIRE)
```

**Best for:**
- Domains with limited prior knowledge
- Problems requiring creative solutions
- When domain-specific patterns are unclear

### LLM4Predict (Domain-Specific)

Uses domain-specific knowledge to predict intermediate conditions:

```python
planner = LLMAssistedPlanner(guidance_mode=GuidanceMode.PREDICT)
```

**Best for:**
- Well-defined domains with clear constraints
- Problems with known solution patterns
- When accuracy is critical

### Hybrid Mode

Combines both approaches for robust planning:

```python
planner = LLMAssistedPlanner(guidance_mode=GuidanceMode.HYBRID)
```

**Best for:**
- Complex problems requiring both creativity and precision
- Domains with partial knowledge
- When you want maximum reliability

## Optimization

Improve planner performance using DSPy's optimization framework:

```python
from optimizer import PlannerOptimizer, PlanningExample

# Create training examples
trainset = [
    PlanningExample(
        initial_state="...",
        goal_state="...",
        available_actions="...",
        domain_description="...",
        domain_constraints="...",
        expected_plan=["action1", "action2", ...],
        expected_sub_tasks=["subtask1", "subtask2", ...]
    ),
    # ... more examples
]

# Optimize the planner
optimizer = PlannerOptimizer(
    planner=planner,
    optimizer_type="mipro",  # or "bootstrap", "random_search"
    num_threads=4
)

optimized_planner = optimizer.optimize(trainset)

# Evaluate performance
results = optimizer.evaluate(testset)
print(f"Average Score: {results['average_score']:.3f}")
print(f"Validity Rate: {results['validity_rate']:.1%}")
```

### Available Optimizers

1. **BootstrapFewShot**: Generates few-shot examples automatically
2. **MIPROv2**: Advanced prompt optimization with instruction refinement
3. **BootstrapFewShotWithRandomSearch**: Random search over prompt variations

## Example Domains

### Academic Examples (example.py)

**Blocks World** - Classic AI planning domain:
```python
from example import blocks_world_example, run_planner_example
problem = blocks_world_example()
result = run_planner_example(problem, GuidanceMode.PREDICT, "Blocks World")
```

**Logistics** - Package delivery optimization:
```python
from example import logistics_example, run_planner_example
problem = logistics_example()
result = run_planner_example(problem, GuidanceMode.HYBRID, "Logistics")
```

**Robot Navigation** - Grid-based pathfinding:
```python
from example import robot_navigation_example, run_planner_example
problem = robot_navigation_example()
result = run_planner_example(problem, GuidanceMode.INSPIRE, "Robot Navigation")
```

### Real-World Business Examples (business_examples.py)

**Supply Chain Optimization** - Multi-warehouse inventory distribution:
```python
from business_examples import supply_chain_optimization, run_business_example
result = run_business_example(supply_chain_optimization, GuidanceMode.PREDICT)
```
*Use case:* Optimize inventory levels across warehouses, minimize shipping costs, meet stock thresholds

**Software Project Planning** - Feature development with dependencies:
```python
from business_examples import software_project_planning, run_business_example
result = run_business_example(software_project_planning, GuidanceMode.HYBRID)
```
*Use case:* Schedule development tasks, manage team capacity, handle technical dependencies

**Manufacturing Workflow** - Production line scheduling:
```python
from business_examples import manufacturing_workflow, run_business_example
result = run_business_example(manufacturing_workflow, GuidanceMode.PREDICT)
```
*Use case:* Schedule production orders, optimize machine utilization, meet deadlines

**Customer Service Routing** - Support ticket management:
```python
from business_examples import customer_service_routing, run_business_example
result = run_business_example(customer_service_routing, GuidanceMode.INSPIRE)
```
*Use case:* Route tickets to appropriate teams, meet SLAs, manage escalations

**Marketing Campaign Launch** - Multi-channel coordination:
```python
from business_examples import marketing_campaign_launch, run_business_example
result = run_business_example(marketing_campaign_launch, GuidanceMode.HYBRID)
```
*Use case:* Plan product launches, allocate budget across channels, coordinate content

**Event Planning** - Corporate conference logistics:
```python
from business_examples import event_planning, run_business_example
result = run_business_example(event_planning, GuidanceMode.PREDICT)
```
*Use case:* Manage venue booking, speaker coordination, catering, budget constraints

## Advanced Features

### Sub-Task Refinement

Enable automatic sub-task refinement based on search feedback:

```python
planner = LLMAssistedPlanner(
    guidance_mode=GuidanceMode.PREDICT,
    enable_refinement=True  # Refine sub-tasks if initial attempts fail
)
```

### Incremental Planning

Use the incremental planner for progressive search with history tracking:

```python
from planner import IncrementalPlanner

planner = IncrementalPlanner(
    guidance_mode=GuidanceMode.HYBRID,
    max_iterations=100
)

result = planner(...)
print("Search History:", result.search_history)
```

### Custom Metrics

Define custom evaluation metrics for optimization:

```python
def custom_metric(example, prediction, trace=None):
    score = 0.0

    # Custom scoring logic
    if prediction.is_valid:
        score += 0.5

    if len(prediction.plan) <= len(example.expected_plan):
        score += 0.5

    return score

optimizer = PlannerOptimizer(planner, optimizer_type="bootstrap")
optimized = optimizer.optimize(trainset, metric=custom_metric)
```

## Performance Tips

1. **Choose the Right Mode**:
   - Use `PREDICT` for well-defined domains
   - Use `INSPIRE` for novel/creative problems
   - Use `HYBRID` when unsure

2. **Optimize for Your Domain**:
   - Create domain-specific training examples
   - Use MIPROv2 for best results (requires more examples)
   - Start with BootstrapFewShot for quick improvements

3. **Tune Parameters**:
   - `max_iterations`: Increase for complex problems
   - `max_sub_tasks`: Adjust based on problem complexity
   - `enable_refinement`: Enable for difficult domains

4. **Model Selection**:
   - **Use GPT-5 for best results** (configured with max_tokens=50000, temperature=1.0 by default)
   - GPT-5 provides superior reasoning and planning capabilities with extended context
   - Use GPT-4 for good quality on a budget
   - Use GPT-4o-mini for faster, cost-effective planning
   - Use GPT-3.5-turbo for simple problems

## Research Background

This implementation is based on the research paper:

**"Inspire or Predict? Exploring New Paradigms in Assisting Classical Planners with Large Language Models"**
*Wenkai Yu, Jianhang Tang, Yang Zhang, Shanjiang Tang, Kebing Jin, Hankz Hankui Zhuo*
arXiv:2508.11524v1 [cs.AI], August 2025
DOI: [10.48550/arXiv.2508.11524](https://doi.org/10.48550/arXiv.2508.11524)

### Abstract Summary

> "Addressing large-scale planning problems has become one of the central challenges in the planning community, deriving from the state-space explosion caused by growing objects and actions. Recently, researchers have explored the effectiveness of leveraging Large Language Models (LLMs) to generate helpful actions and states to prune the search space. However, prior works have largely overlooked integrating LLMs with domain-specific knowledge to ensure valid plans."

The paper proposes a novel LLM-assisted planner with problem decomposition and introduces two paradigms:
- **LLM4Inspire**: Provides heuristic guidance using general knowledge
- **LLM4Predict**: Employs domain-specific knowledge to infer intermediate conditions

### Key Research Insights

1. **Problem Decomposition**: Breaking large planning problems into simpler sub-tasks effectively partitions the search space and reduces computational complexity

2. **LLM Integration**: LLMs can effectively locate feasible solutions when pruning the search space, making them valuable tools for assisting classical planners

3. **Domain Knowledge Matters**: Experimental results show that **LLM4Predict (domain-specific knowledge) holds particular promise compared with LLM4Inspire (general knowledge)** when solving well-defined planning domains

4. **Search Space Partition**: The decomposition approach demonstrates strong ability in search space partition across multiple planning domains

### Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{yu2025inspire,
  title={Inspire or Predict? Exploring New Paradigms in Assisting Classical Planners with Large Language Models},
  author={Yu, Wenkai and Tang, Jianhang and Zhang, Yang and Tang, Shanjiang and Jin, Kebing and Zhuo, Hankz Hankui},
  journal={arXiv preprint arXiv:2508.11524},
  year={2025}
}
```

## Contributing

To extend this planner:

1. **Add New Signatures**: Define new LLM interfaces in `signatures.py`
2. **Create Modules**: Implement new planning strategies in `modules.py`
3. **Add Domains**: Create domain-specific examples in `example.py`
4. **Improve Optimization**: Add custom metrics in `optimizer.py`

## License

MIT License - feel free to use and modify for your planning needs!

## Acknowledgments

This implementation faithfully reproduces the approach described in the research paper by Yu et al. We thank the authors for their groundbreaking work on integrating LLMs with classical planning through problem decomposition and the innovative LLM4Inspire/LLM4Predict paradigms.

## Troubleshooting

**Issue**: Plans are invalid
- **Solution**: Provide clearer domain constraints, enable refinement, or use PREDICT mode

**Issue**: Decomposition creates too many sub-tasks
- **Solution**: Reduce `max_sub_tasks` parameter or provide clearer problem descriptions

**Issue**: Slow planning
- **Solution**: Use a faster model (gpt-4o-mini), reduce `max_iterations`, or optimize the planner

**Issue**: Optimization not improving
- **Solution**: Create more/better training examples, try different optimizer types, or adjust metric weights

## Support

For questions or issues:
1. Check the examples in `example.py`
2. Review the module documentation in source files
3. Experiment with different guidance modes and parameters
