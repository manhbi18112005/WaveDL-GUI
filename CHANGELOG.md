# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-01-02

### Added
- Console entry points: `wavedl-train`, `wavedl-test`, `wavedl-hpo`
- Single version source in `src/wavedl/__init__.py` with dynamic `pyproject.toml` reading
- `--single_channel` flag for explicit channel handling in data loading
- Optuna hyperparameter optimization support (`hpo.py`)

### Changed
- **BREAKING**: Restructured to `src/wavedl/` namespace package layout
  - Use `python -m wavedl.train` instead of `python train.py`
  - Use `from wavedl.models import CNN` instead of `from models import CNN`
- Migrated CI workflows to use `pyproject.toml` for dependencies
- Improved data loading robustness with lazy handles and format detection
- Optimized training loop and consolidated data utilities
- Updated Ruff linter to v0.14.10 for consistent formatting
- Enhanced contributor guidelines with pre-commit setup
- Pinned Ruff version across pre-commit, CI, and local configs
- Moved development setup instructions to CONTRIBUTING.md

### Fixed
- Worker seeding in DataLoader for diverse random augmentations

## [1.1.0] - 2025-12-28

### Added
- GitHub Actions CI/CD for automated testing and linting
- Google Colab demo notebook for easy experimentation
- Pre-commit hooks for code quality enforcement
- GitHub Discussions link for community support

### Fixed
- LaTeX rendering in diagnostic plots
- Badge spacing and display in README

## [1.0.0] - 2025-12-24

### Added
- Initial release of WaveDL framework
- Core CNN, ResNet, and Transformer model architectures
- Multi-format data loading (NPZ, HDF5, MAT)
- Training and evaluation scripts with WandB integration
- Comprehensive diagnostic plotting (10+ plot types)
- ONNX export functionality
- Mixed-precision training support
- Reproducibility features (seeding, deterministic ops)
- Example configurations and training scripts
- MIT License and citation file

[1.2.0]: https://github.com/ductho-le/WaveDL/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/ductho-le/WaveDL/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ductho-le/WaveDL/releases/tag/v1.0.0
