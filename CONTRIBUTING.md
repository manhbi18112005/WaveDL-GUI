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
```

## Running Tests

Run the unit test suite before submitting any changes:

```bash
pytest unit_tests -v
```

All tests should pass with no warnings.

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Include docstrings for public functions and classes
- Keep lines under 100 characters

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear commit messages
4. Ensure code passes any existing tests
5. Submit a pull request with a clear description

## Adding New Models

See [`models/_template.py`](models/_template.py) for a template to create new model architectures.

## Questions?

Open an issue or contact the maintainer.
