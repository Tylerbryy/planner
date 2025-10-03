# Contributing to LLM-Assisted Planner

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## License Agreement

By contributing to this project, you agree that your contributions will be licensed under the **GNU General Public License v3.0** (GPL-3.0), the same license as the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, DSPy version, OS)

### Suggesting Enhancements

We welcome suggestions for:
- New planning domains (business or academic)
- Improved DSPy signatures or modules
- Performance optimizations
- Better examples or documentation

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style below
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Ensure GPL-3.0 compliance**:
   - Include license header in new files
   - Document any changes
   - Maintain open-source spirit
6. **Submit a pull request** with a clear description

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Comment complex logic

## Adding New Planning Domains

To add a new planning domain:

1. **Create the domain function** in `example.py` or `business_examples.py`:
```python
def your_domain_example():
    """
    Brief description of the planning problem.
    """
    return {
        "initial_state": "...",
        "goal_state": "...",
        "available_actions": "...",
        "domain_description": "...",
        "domain_constraints": "..."
    }
```

2. **Test with different guidance modes**:
```python
run_business_example(your_domain_example, GuidanceMode.PREDICT)
run_business_example(your_domain_example, GuidanceMode.INSPIRE)
run_business_example(your_domain_example, GuidanceMode.HYBRID)
```

3. **Document the domain** in the README

## Adding New DSPy Signatures

To add a new signature:

1. **Define in `signatures.py`**:
```python
class YourSignature(dspy.Signature):
    """
    Clear description of what this signature does.
    """
    input_field = dspy.InputField(desc="description")
    output_field = dspy.OutputField(desc="description")
```

2. **Create corresponding module** in `modules.py`:
```python
class YourModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(YourSignature)

    def forward(self, input_field: str) -> dspy.Prediction:
        return self.predictor(input_field=input_field)
```

3. **Add tests** and examples

## Testing

Before submitting:
- Run existing examples to ensure they still work
- Test your new features with multiple scenarios
- Verify GPL-3.0 compliance
- Check that documentation is updated

## Questions?

If you have questions about contributing:
- Open a discussion issue
- Check existing issues and pull requests
- Review the research paper for theoretical background

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on improving the project
- Help others learn and grow

## Recognition

Contributors will be acknowledged in the project. Significant contributions may be recognized in the README.

Thank you for contributing to open-source AI planning research! 🚀
