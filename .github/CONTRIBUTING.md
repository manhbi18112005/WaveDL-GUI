# Contributing to WaveDL

Thank you for your interest in contributing to WaveDL! This document provides guidelines for contributing to the project.

## Ways to Contribute

- **Bug Reports**: Open an issue describing the bug, including steps to reproduce
- **Feature Requests**: Open an issue describing the proposed feature and its use case
- **Code Contributions**: Submit a pull request with your changes
- **Documentation**: Improve or add documentation

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ductho-le/WaveDL.git
cd WaveDL

# Create virtual environment (or use conda)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (required for contributors)
pip install pre-commit
pre-commit install
```

## Code Quality

We use **Ruff** for linting and formatting, enforced via pre-commit hooks and CI.

### Automatic Checks (on every commit)

Once you run `pre-commit install`, these checks run automatically before each commit:

| Check | Purpose |
|-------|---------|
| **Ruff linter** | Catches errors, unused imports, style issues |
| **Ruff formatter** | Auto-formats code to consistent style |
| **File hygiene** | Trailing whitespace, end-of-file, YAML/TOML validation |

### Manual Check

To run all checks manually:

```bash
pre-commit run --all-files
```

### Code Style Guidelines

- Follow PEP 8 (enforced by Ruff)
- Use type hints for function signatures
- Include docstrings for public functions and classes
- Line length: 88 characters (Ruff default)

## Running Tests

Run the unit test suite before submitting any changes:

```bash
pytest unit_tests -v
```

All tests should pass with no warnings.

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear commit messages
4. Run `pre-commit run --all-files` to ensure code quality
5. Run `pytest unit_tests -v` to ensure tests pass
6. Submit a pull request with a clear description

## Adding New Models

See [`models/_template.py`](../models/_template.py) for a template to create new model architectures.

New models should:
- Inherit from `BaseModel`
- Use the `@register_model("model_name")` decorator
- Support 1D, 2D, and 3D input shapes (if possible)
- Include docstrings and type hints

## Questions?

Open an issue or start a [GitHub Discussion](https://github.com/ductho-le/WaveDL/discussions).
