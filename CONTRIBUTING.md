# Contributing to Block-Shock

Thank you for your interest in contributing to Block-Shock! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and professional in all interactions.

### Expected Behavior

- Be respectful of differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or discriminatory comments
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Block-Shock.git
   cd Block-Shock
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/MathewYoussef/Block-Shock.git
   ```
4. **Create a branch** for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with Tensor Core support (for full testing)
- 2 GPUs for multi-GPU experiments

### Installation

1. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. Install additional dependencies:
   ```bash
   pip install pyyaml matplotlib numpy
   ```

3. Verify installation:
   ```bash
   python -m src.main --help
   ```

### Running Tests

Run smoke tests to verify your setup:
```bash
# Timing smoke test
python -m src.timing_smoke

# Distributed smoke test (requires 2 GPUs)
torchrun --standalone --nproc_per_node=2 -m src.allreduce_smoke

# Mask generation smoke test
python -m src.mask_smoke

# Semi-structured sparse smoke test
python -m src.semistructured_smoke

# Workload generation smoke test
python -m src.workload_gen_repetition_smoke
```

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, PyTorch version, GPU model)
- Relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear description of the enhancement
- The motivation behind it (what problem does it solve?)
- Possible implementation approaches
- Any relevant examples or references

### Contributing Code

Areas where contributions are particularly welcome:

1. **New Methods**: Implementing additional baseline or sparse training methods
2. **Performance Optimizations**: Improving runtime efficiency
3. **Extended Hardware Support**: Supporting additional GPU types or configurations
4. **Testing**: Adding unit tests, integration tests, or benchmarks
5. **Documentation**: Improving code comments, docstrings, or user guides
6. **Analysis Tools**: Enhancing visualization and analysis scripts

## Pull Request Process

1. **Keep changes focused**: Each PR should address a single concern
2. **Update documentation**: Include relevant documentation updates
3. **Add tests**: Include tests for new functionality
4. **Follow coding standards**: See [Coding Standards](#coding-standards)
5. **Commit message format**:
   ```
   Brief summary (50 chars or less)

   Detailed explanation of changes, motivation, and context.
   Reference any related issues.
   ```

6. **Ensure all tests pass**: Run smoke tests before submitting
7. **Update PROGRESS.md**: Add notes about your changes if relevant
8. **Request review**: Tag maintainers for review

### PR Checklist

Before submitting your PR, ensure:
- [ ] Code follows project style guidelines
- [ ] New code has appropriate comments and docstrings
- [ ] Tests have been added or updated
- [ ] Documentation has been updated
- [ ] All smoke tests pass
- [ ] Commit messages are clear and descriptive

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Maximum line length: 100 characters (relaxed from PEP 8's 79)

### Type Hints

Use type hints for function signatures:
```python
def build(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build method state from configuration.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Method state dictionary
    """
    pass
```

### Documentation

- Add docstrings to all modules, classes, and public functions
- Use Google-style docstrings:
  ```python
  def function(arg1: int, arg2: str) -> bool:
      """Short description.
      
      Longer description if needed.
      
      Args:
          arg1: Description of arg1
          arg2: Description of arg2
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: When input is invalid
      """
  ```

### File Headers

Include descriptive headers in new files:
```python
### block_shock/src/module_name.py
## Brief description of what this module does.

from __future__ import annotations
# ... rest of file
```

### Imports

- Use `from __future__ import annotations` for forward compatibility
- Group imports: standard library, third-party, local
- Use absolute imports for project modules

### Configuration Files

- Keep YAML configs clean and well-commented
- Use consistent naming conventions
- Document all configuration options

## Testing Guidelines

### Smoke Tests

Smoke tests are quick validation tests that check basic functionality:
- Should run in seconds, not minutes
- Focus on critical paths
- Include expected vs actual comparisons

### Integration Tests

When adding new methods:
1. Create Phase 0 correctness check
2. Add Phase 1 throughput benchmark
3. Compare against existing baselines

### Distributed Tests

For multi-GPU code:
- Test with `torchrun` and appropriate `nproc_per_node`
- Verify correctness across ranks
- Check communication overhead

## Documentation

### Code Comments

- Explain **why**, not just **what**
- Document assumptions and constraints
- Mark TODOs clearly with `TODO:` prefix

### README Updates

When adding features:
- Update relevant sections in README.md
- Add usage examples
- Update project structure if files added

### PROGRESS.md

Track development milestones in PROGRESS.md:
- Document what was implemented
- Note verification commands and outputs
- Track "Definition of Done" criteria

## Community

### Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### Recognition

Contributors will be recognized in:
- Git commit history
- Release notes
- Project acknowledgments

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search existing issues and discussions
3. Create a new issue with your question

Thank you for contributing to Block-Shock! 🚀
