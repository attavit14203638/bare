# Contributing to BARE

Thank you for your interest in contributing to BARE! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/bare.git` (replace YOUR_USERNAME with your GitHub username)
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes thoroughly
6. Commit your changes: `git commit -am 'Add some feature'`
7. Push to the branch: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Check code style
flake8 *.py
```

## Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Comment complex logic

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, PyTorch version, etc.)
- Relevant error messages or logs

## Feature Requests

We welcome feature requests! Please:

- Check if the feature has already been requested
- Clearly describe the feature and its benefits
- Provide examples of how it would be used
- Consider backwards compatibility

## Pull Request Guidelines

- Keep changes focused and atomic
- Update documentation as needed
- Add tests for new functionality
- Ensure all tests pass
- Follow the existing code style
- Provide a clear description of changes

## Questions?

Feel free to open an issue for questions or discussions about the project.

Thank you for contributing!
