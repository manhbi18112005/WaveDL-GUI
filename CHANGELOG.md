# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.3] - 2026-01-10

### Changed
- **HPC**: TORCH_HOME and WandB caches now always use CWD (compute nodes lack internet access)
- **HPC**: Triton/Inductor caches set unconditionally before torch imports (prevents `--compile` permission errors)
- **Training**: Per-GPU Triton/Inductor cache directories prevent multi-process race warnings with `--compile`
- **Validation**: Replaced manual `torch.distributed.gather` with `accelerator.gather_for_metrics` (eliminates GPU memory spike)
- **Config**: `wavedl_version` metadata now dynamically reads from `__version__` instead of hardcoded `"1.0.0"`

### Fixed
- **Cross-validation**: Auto-detect optimal DataLoader workers when `--workers=-1` (matches `train.py` behavior)
- **Test data loading**: Prioritize `input_test`/`output_test` keys over training keys in `load_test_data()`
- **ResNet**: Added GroupNorm divisibility validation (prevents cryptic runtime errors)
- **Tests**: Force `pretrained=False` in architecture tests for offline CI compatibility
- **Documentation**: Updated README custom model signature and HPC environment variable notes
- **Metadata**: Synced CITATION.cff version

## [1.5.2] - 2026-01-08

### Fixed
- **Critical**: NPZ safe_load failed on data access (error occurs when reading arrays, not file open)

## [1.5.1] - 2026-01-07

### Added
- **MPS Inference**: Apple Silicon GPU support for inference (`test.py` auto-detects MPS)
- `--input_channels` flag for explicit channel override in `load_test_data()` (bypasses heuristics)

### Changed
- **NPZ Security**: Pickle now disabled by default, only enabled as fallback for sparse matrices

### Fixed
- **Input-dependent constraints**: Now properly pass inputs to loss function for `x_mean`, `x[...]` expressions
- **DDP validation memory**: Gather validation data only on rank 0 (prevents OOM on multi-GPU setups)
- **Cross-validation**: OneCycleLR now correctly steps per-batch instead of per-epoch
- **ViT patch embedding**: Added warning for non-divisible input shapes (prevents silent data loss)

## [1.5.0] - 2026-01-06

### Added
- **Physics-Constrained Training**: Enforce physical laws during training via penalty terms
  - `--constraint`: Expression constraints (`"y0 > 0"`, `"y0 - y1*y2"`)
  - `--constraint_file`: Custom Python constraint functions
  - `--constraint_weight`: Penalty weights (default: 0.1)
  - `--constraint_reduction`: Reduction mode (`mse` or `mae`)
- Expression syntax with math functions (`sin`, `cos`, `exp`, `log`, `sqrt`, etc.)
- Comparison operators (`>`, `<`, `>=`, `<=`, `==`)
- Input indexing with literal integers (`x[0]`, `x[0,5]`, `x[0,5,10]`)
- Input aggregates (`x_mean`, `x_sum`, `x_max`, `x_min`, `x_std`)
- Automatic denormalization for constraints in physical space
- 21 new unit tests for constraints (704 → 725 total)

### Removed
- `--output_transform` and `--output_bounds` (hard constraints) — redundant with soft constraints

## [1.4.6] - 2026-01-04

### Added
- **HPO**: Auto-detect GPUs and default `--n_jobs` to GPU count (maximizes resource utilization)
- **HPO**: GPU isolation for parallel trials (each trial runs on a dedicated GPU)

### Changed
- **HPC**: Launcher now passes `--multi_gpu` explicitly to suppress accelerate auto-detection warnings
- **Training**: Checkpoints now use `.bin` format (`safe_serialization=False`) for faster saves
- **Training**: Suppressed verbose accelerate checkpoint logging during saves (cleaner output)
- **HPO**: Default `--n_jobs` changed from `1` to `-1` (auto-detect GPUs)

### Fixed
- **HPC**: WandB offline sync instructions only shown when `--wandb` flag is actually used
- **Inference**: `test.py` now checks for `model.bin` in addition to `model.safetensors` and `pytorch_model.bin`
- **HPO**: Relative data paths now converted to absolute (fixes "file not found" in child processes)

## [1.4.5] - 2026-01-04

### Fixed
- **Critical**: `test.py` failed to load checkpoints from `--compile` models (`_orig_mod.` prefix not stripped)

## [1.4.4] - 2026-01-04

### Changed
- Unified HPC cache directory setup across all entry points (`train.py`, `test.py`, `hpc.py`)
- Simplified cache logic: uses CWD fallback only when home is not writable (cleaner for local development)
- Removed `tempfile` dependency from `train.py` and `hpc.py` (uses CWD-based caching instead)

### Fixed
- `torch.compile` model unwrapping during checkpoint save (handles missing `_orig_mod` gracefully)
- E402 lint errors in `test.py` from intentional HPC environment setup imports
- Unit test for HPC environment setup now properly mocks non-writable home directory

## [1.4.3] - 2026-01-03

### Added
- Smart HPC cache directory setup (`_setup_cache_dir`) - auto-detects writable paths for matplotlib/fontconfig

### Changed
- **DDP**: Switched back to `accelerator.gather()` for broader accelerate version compatibility
- Simplified Triton availability check (imports package instead of internal compiler API)

### Fixed
- E402 lint errors from intentional HPC environment setup imports in `train.py`
- Configured per-file-ignores in `pyproject.toml` to allow early `os`/`tempfile` imports
- Added pydantic warning suppression for accelerate's internal Field() usage

## [1.4.2] - 2026-01-03

### Added
- Input-only loading for HDF5/MAT files in `load_test_data()` (inference without ground truth)
- Cache metadata now includes file size and modification time for stale detection

### Changed
- **DDP**: Validation now uses `gather_object` (memory-efficient, collects only on rank 0)
- **HPO**: Reads `training_history.csv` instead of parsing stdout (reliable metric extraction)
- HPO stdout fallback uses regex pattern matching to avoid false positives

### Fixed
- **Critical**: HPO trials always returned `inf` (stdout parsing never matched trainer output)
- **Critical**: DDP validation gathered full tensors to all ranks, risking OOM on large val sets
- HDF5/MAT `load_test_data()` raised KeyError when outputs missing (now optional)
- MAT input-only fallback lacked sparse matrix handling (now uses `MATSource._load_dataset`)

## [1.4.1] - 2026-01-03

### Added
- `validate_input_shape()` method in `BaseModel` for explicit shape contract enforcement
- `--wandb_watch` flag for opt-in gradient watching (reduces overhead by default)
- `--main_process_ip` and `--main_process_port` args in `wavedl-hpc` for multi-node clusters
- Unknown config key detection with helpful warnings for typos

### Changed
- **Performance**: Enabled TF32 precision by default (~2x speedup on Ampere/Hopper GPUs)
- **Performance**: Enabled cuDNN benchmark for auto-tuned convolutions
- **Performance**: Increased DataLoader worker cap from 8 to 16 per GPU
- Improved config validation with type checking before numeric comparisons
- Made `wandb.watch()` opt-in via `--wandb_watch` flag (was always-on)

### Fixed
- **Critical**: `--machine_rank` was hardcoded to 0 in `wavedl-hpc` (multi-node now works correctly)
- `merge_config_with_args()` fragility when required args are added later
- Silent exception swallowing in cross-validation cleanup
- Documentation clarity for `--precision` vs `--mixed_precision` flags

## [1.4.0] - 2026-01-03

### Added
- 6 new model architectures (38 total variants):
  - **EfficientNetV2** (S/M/L) - modern efficient CNNs with pretrained weights
  - **MobileNetV3** (Small/Large) - mobile-optimized with pretrained weights
  - **RegNet** (Y-400MF to Y-8GF) - regularized networks with pretrained weights
  - **ResNet3D-18, MC3-18** - 3D video/volume models
  - **Swin Transformer** (T/S/B) - shifted window attention with pretrained weights
  - **TCN** (small/base/large) - temporal convolutional networks for 1D signals
- New unit tests: `test_cli.py`, `test_config_metrics.py`, `test_data_cv.py`
- 704 total unit tests (up from 422)

### Changed
- Simplified installation: `pip install wavedl` now includes all dependencies
- Removed optional extras `[all]`, `[hpo]`, `[onnx]` - all included by default
- Triton installs automatically on Linux only (via environment marker)
- Skip slow architecture tests in CI for faster builds
- Synced `wavedl-hpc` with original bash script functionality

### Fixed
- E402 lint errors in `train.py` (moved imports to top)
- Suppressed pydantic deprecation warnings

## [1.3.1] - 2026-01-02

### Fixed
- `wavedl-train --list_models` crash with `UnboundLocalError: cannot access local variable 'sys'`

## [1.3.0] - 2026-01-02

### Added
- `wavedl-hpc` command for HPC distributed training (replaces `run_training.sh`)
- `--import` flag for loading custom model modules without wrapper scripts
- PyPI package: `pip install wavedl`

### Changed
- Removed `run_training.sh` (use `wavedl-hpc` instead)
- Made `triton` dependency Linux-only for cross-platform compatibility
- Simplified custom model documentation to 2-step workflow
- Updated all CLI examples to use `wavedl-*` commands

### Fixed
- Pinned `setuptools<77` for PyPI metadata compatibility

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

[1.5.3]: https://github.com/ductho-le/WaveDL/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/ductho-le/WaveDL/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/ductho-le/WaveDL/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/ductho-le/WaveDL/compare/v1.4.6...v1.5.0
[1.4.6]: https://github.com/ductho-le/WaveDL/compare/v1.4.5...v1.4.6
[1.4.5]: https://github.com/ductho-le/WaveDL/compare/v1.4.4...v1.4.5
[1.4.4]: https://github.com/ductho-le/WaveDL/compare/v1.4.3...v1.4.4
[1.4.3]: https://github.com/ductho-le/WaveDL/compare/v1.4.2...v1.4.3
[1.4.2]: https://github.com/ductho-le/WaveDL/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/ductho-le/WaveDL/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/ductho-le/WaveDL/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/ductho-le/WaveDL/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/ductho-le/WaveDL/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/ductho-le/WaveDL/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/ductho-le/WaveDL/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ductho-le/WaveDL/releases/tag/v1.0.0
