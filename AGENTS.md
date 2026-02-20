# AGENTS.md — WaveDL-GUI

## Project Overview

WaveDL is a deep learning framework for wave-based inverse problems with two packages:
- `src/wavedl/` — Core ML framework (PyTorch training, models, utilities)
- `src/wavedl_gui/` — GUI application built with PySide6 + PyQt-Fluent-Widgets (`qfluentwidgets`)

Python **3.11+** required. Licensed under MIT.

## Build & Install

```bash
pip install -e ".[dev]"        # Core + dev tools (ruff, pytest)
pip install -e ".[dev,gui]"    # Core + dev + GUI (PySide6, qfluentwidgets)
```

## Lint & Format

Ruff is the sole linter and formatter (replaces Black, Flake8, isort).

```bash
ruff check .                       # Lint
ruff check . --fix                 # Lint with auto-fix
ruff format .                      # Format
ruff format . --check --diff       # Check formatting without modifying
```

Pre-commit hooks are configured. Install with:
```bash
pre-commit install
pre-commit run --all-files         # Manual full run
```

The CI (`lint.yml`) pins `ruff==0.14.10` and runs both `ruff check` and `ruff format --check`.

## Test Commands

```bash
pytest                                           # All tests (verbose, short traceback)
pytest unit_tests/test_losses.py                 # Single file
pytest unit_tests/test_losses.py::TestGetLoss    # Single class
pytest unit_tests/test_losses.py::TestGetLoss::test_mse_loss  # Single test
pytest -m "not slow"                             # Skip slow architecture tests (CI default)
pytest -m "not gpu"                              # Skip GPU-requiring tests
pytest -k "test_mse"                             # Keyword filter
```

Tests live in `unit_tests/`. Filenames follow `test_*.py`, classes `Test*`, functions `test_*`.
Default addopts: `-v --tb=short --strict-markers`. Markers: `slow`, `gpu`, `integration`.

## UI Style

The design has to be Apple-like - modern, intuitive and nice looking

## Code Style

### Formatting
- **Line length**: 88 (ruff/Black default)
- **Quotes**: Double quotes
- **Indentation**: 4 spaces
- **Trailing commas**: Preserved (not skipped)
- **Line endings**: LF (enforced by pre-commit)
- **Docstring code**: Formatted by ruff

### Imports
Ruff handles import sorting (isort rules). Order: stdlib, third-party, local.
- `known-first-party = ["wavedl"]`
- Two blank lines after imports
- Multi-imports use `combine-as-imports = true`
- `__init__.py` files may have unused imports (`F401` suppressed)

Typical import pattern in GUI code:
```python
# coding: utf-8
import os
import sys
from enum import Enum

from PySide6.QtCore import QObject, Signal
from qfluentwidgets import (FluentIconBase, getIconColor, Theme)

from ..common.config import cfg
from ..common.signal_bus import signalBus
```

### Naming Conventions
- **Classes**: `PascalCase` — `TrainingProgress`, `ProcessState`, `SignalBus`
- **Functions/methods**: `camelCase` for GUI layer (`connectSignalToSlot`, `initNavigation`), `snake_case` for ML layer (`load_config`, `get_loss`, `build_model`)
- **Private methods**: Single underscore prefix `_check_environment()`
- **Config items**: `camelCase` matching qfluentwidgets convention (`batchSize`, `learningRate`)
- **Constants/module singletons**: `UPPER_SNAKE` for true constants (`FEEDBACK_URL`), `camelCase` for singleton instances (`signalBus`, `cfg`, `trainingService`)
- **Enums**: `PascalCase` class name, `UPPER_SNAKE` members (`ProcessState.IDLE`)
- **Test classes**: `Test*` prefix; test methods: `test_*` with snake_case
- **Signals**: `camelCase` ending with `Sig` suffix (`trainingProgressSig`, `appErrorSig`)

### Type Annotations
- Used in the ML layer: `def load_config(config_path: str) -> dict[str, Any]:`
- Use modern syntax (`dict[str, Any]`, `list[float]`) not `typing.Dict`/`typing.List`
- `from __future__ import annotations` used in newer files
- GUI layer is less strictly typed (follows qfluentwidgets conventions)
- Dataclasses used for structured data (`@dataclass class TrainingProgress`)

### Error Handling
- Factory functions raise `ValueError` for unknown keys: `raise ValueError(f"Unknown loss function")`
- GUI uses decorator-based exception handling: `@exceptionHandler("log_name", default_value)`
- `FileNotFoundError` for missing config/data files
- GUI errors are forwarded via `signalBus.appErrorSig.emit(message)` and shown as `InfoBar`

### Docstrings
- ML layer uses detailed module-level docstrings with sections (Features, Usage, Author)
- Function docstrings use Google-style or NumPy-style (`Parameters`, `Returns`, `Raises`)
- GUI classes use short one-line docstrings
- Test docstrings describe expected behavior: `"""MSE loss should work correctly."""`

## Architecture Patterns

### GUI Layer (`wavedl_gui`)
- **Window**: `MSFluentWindow` base class (Microsoft Fluent design)
- **Signal bus**: Centralized `SignalBus(QObject)` singleton for cross-component communication
- **Config**: `QConfig` subclass with typed `ConfigItem` fields, auto-persisted to JSON
- **StyleSheets**: `StyleSheetBase` + `Enum` pattern with theme-aware `path()` method
- **Icons**: `FluentIconBase` + `Enum` with `path()` returning resource paths
- **Services**: Stateless service singletons (`trainingService`, `VersionService`)
- **Concurrency**: `TaskExecutor` for async work, `QThread` for subprocess management
- **Navigation**: `NavigationItemPosition.TOP/BOTTOM`, `addSubInterface()` pattern

### ML Layer (`wavedl`)
- **Registry pattern**: `register_model`, `get_model`, `list_models`, `get_loss`, `list_losses`
- **Config**: YAML-based with `load_config()` / `save_config()` / CLI merge
- **Data**: NPZ, MAT, HDF5 formats via `prepare_data()` / `load_training_data()`

## Key Files

| Path | Purpose |
|---|---|
| `src/wavedl_gui/main.py` | GUI entry point |
| `src/wavedl_gui/view/main_window.py` | Main window (MSFluentWindow) |
| `src/wavedl_gui/common/config.py` | QConfig with all ConfigItems |
| `src/wavedl_gui/common/signal_bus.py` | Application-wide signal bus |
| `src/wavedl_gui/common/style_sheet.py` | Theme-aware stylesheets |
| `src/wavedl_gui/common/icon.py` | Custom icon enums |
| `src/wavedl_gui/service/training_service.py` | Subprocess training manager |
| `src/wavedl/train.py` | CLI training entry point |
| `src/wavedl/models/` | 18+ model architectures |
| `src/wavedl/utils/` | Losses, metrics, optimizers, schedulers, data loading |
| `pyproject.toml` | All project config (deps, ruff, pytest) |
| `configs/config.yaml` | Example training config |

## Ruff Rules Summary

Enabled: `E` `W` `F` `I` `B` `C4` `UP` `SIM` `TCH` `RUF`

Notable ignores: `E501` (line length handled by formatter), `E741` (math variable names like `l`, `I`), `B006`/`B008` (mutable defaults intentional in some APIs), `SIM108` (ternary not always clearer).

Per-file: `__init__.py` allows unused imports. `unit_tests/` allows `assert`. `train.py`/`test.py` allow late imports (`E402`).
