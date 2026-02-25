# Implement a cross-platform GUI application for the wavedl. There should be 2 mode of working, directly on GUI or print out a list of commands for using in terminal. We should leverage the existing wavedl training scripts and CLI.

### Current CLI Entry Points

| Command | Purpose | Key Arguments |
|---------|---------|---------------|
| `wavedl-hpc` | HPC training launcher | `--model`, `--data_path`, `--num_gpus` |
| `wavedl.train` | Core training module | All hyperparameters |
| `wavedl.test` | Inference & export | `--checkpoint`, `--plot` |
| `wavedl-hpo` | Hyperparameter search | `--n_trials`, `--models` |

### Data Flow to Preserve

```
User Input → YAML Config / CLI Args → train.py → Accelerator → Model Training
                                         ↓
                              training_history.csv + checkpoints
```

---

## Technology Stack

**PySide6 with QFluentWidgets + PyQtGraph**

| Aspect | Details |
|--------|---------|
| **Real-time Plots** | PyQtGraph (GPU-accelerated, handles 1000s of points) |
| **Packaging** | PyInstaller

---

## Feature Specification

### 1. Project Setup Panel

```
┌─────────────────────────────────────────────────────────────┐
│ 📁 Project Setup                                            │
├─────────────────────────────────────────────────────────────┤
│ Data File:    [train_data.npz        ] [Browse...]          │
│ Output Dir:   [./experiments/run_001 ] [Browse...]          │
│                                                             │
│ ┌─ Data Preview ──────────────────────────────────────────┐ │
│ │ ✓ Format: NPZ                                           │ │
│ │ ✓ Samples: 10,000                                       │ │
│ │ ✓ Input Shape: (256, 256) → 2D data                     │ │
│ │ ✓ Targets: 5 parameters                                 │ │
│ │ ✓ dtype: float32                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2. Model Configuration Panel

```
┌─────────────────────────────────────────────────────────────┐
│ 🧠 Model Configuration                                      │
├─────────────────────────────────────────────────────────────┤
│ Architecture: [CNN ▼] [ResNet18] [EfficientNet-B0] ...      │
│                                                             │
│ ☑ Use Pretrained Weights (ImageNet)                         │
│                                                             │
│ Model Info:                                                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Parameters: 1.7M trainable                              │ │
│ │ Size: 6.8 MB                                            │ │
│ │ Supports: 1D ✓ | 2D ✓ | 3D ✓                            │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3. Training Hyperparameters Panel

```
┌─────────────────────────────────────────────────────────────┐
│ ⚙️ Training Configuration                                   │
├─────────────────────────────────────────────────────────────┤
│ Batch Size:    [128    ▼]     Learning Rate: [0.001    ]    │
│ Epochs:        [1000   ]      Patience:      [20       ]    │
│ Weight Decay:  [0.0001 ]      Grad Clip:     [1.0      ]    │
│                                                             │
│ Loss:       [MSE ▼] [MAE] [Huber] [Weighted MSE]            │
│ Optimizer:  [AdamW ▼] [SGD] [RAdam]                         │
│ Scheduler:  [Plateau ▼] [Cosine] [OneCycle]                 │
│                                                             │
│ ☑ Mixed Precision (BF16)    ☐ torch.compile                 │
│ ☐ Deterministic Mode        Seed: [2025    ]                │
└─────────────────────────────────────────────────────────────┘
```

### 4. Real-Time Training Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 📊 Training Progress                                      [■] [□] [×]   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Loss Curves                          Learning Rate                     │
│  ┌────────────────────────────┐      ┌────────────────────────────┐    │
│  │     \                      │      │ ────────────\              │    │
│  │      \  train              │      │              \____         │    │
│  │       \_____               │      │                   \___     │    │
│  │             \___  val      │      │                            │    │
│  └────────────────────────────┘      └────────────────────────────┘    │
│                                                                         │
│  Epoch: 142/1000   ████████████████░░░░░░░░░░░░░░░░░░  14.2%           │
│  Time: 00:23:41    ETA: 02:21:15                                        │
│                                                                         │
│  ┌─ Current Metrics ────────────────────────────────────────────────┐  │
│  │ Train Loss: 0.00234  │ Val Loss: 0.00289  │ LR: 3.2e-5           │  │
│  │ R²: 0.9847           │ Pearson: 0.9923    │ MAE: [0.12, 0.08...]  │  │
│  │ Best Val: 0.00245 @ epoch 128            │ Patience: 14/20       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  [⏸ Pause] [⏹ Stop] [💾 Save Checkpoint]                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5. Results & Export Panel

```
┌─────────────────────────────────────────────────────────────┐
│ 📈 Results & Export                                         │
├─────────────────────────────────────────────────────────────┤
│ Checkpoint: ./experiments/run_001/best_checkpoint           │
│                                                             │
│ ┌─ Test Results ────────────────────────────────────────┐   │
│ │ R²: 0.9912 | Pearson: 0.9956 | MAE: [0.08, 0.05, ...] │   │
│ └───────────────────────────────────────────────────────┘   │
│                                                             │
│ [📊 Generate Plots]  [📄 Export CSV]  [📦 Export ONNX]      │
│                                                             │
│ Plot Preview:                                               │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ [Scatter] [Residuals] [Bland-Altman] [Q-Q] [Histo]    │   │
│ │                                                       │   │
│ │              (embedded matplotlib figure)              │   │
│ │                                                       │   │
│ └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Development Pipeline

### Phase 1: Core Infrastructure

### Phase 2: Widget Implementation

#### Task Breakdown

| Widget | Priority | Complexity | Dependencies |
|--------|----------|------------|--------------|
| Data Panel | P0 | Medium | numpy, h5py |
| Model Panel | P0 | Low | wavedl.models |
| Training Panel | P0 | Medium | YAML config |
| Dashboard | P1 | High | PyQtGraph |
| Results Panel | P2 | Medium | matplotlib |

### Phase 3: Training Integration

### Phase 4: Real-Time Dashboard

## Key Technical Considerations

### 1. Process Isolation

Training runs in a **subprocess** to:

- Prevent GUI freezing during long training
- Allow clean termination
- Isolate GPU memory management

### 2. Cross-Platform Paths

````python
from pathlib import Path
import platform

def get_default_output_dir() -> Path:
    """Get platform-appropriate default output directory."""
    if platform.system() == "Windows":
        base = Path.home() / "Documents" / "WaveDL"
    elif platform.system() == "Darwin":  # macOS
        base = Path.home() / "Documents" / "WaveDL"
    else:  # Linux
        base = Path.home() / "wavedl_experiments"

    base.mkdir(parents=True, exist_ok=True)
    return base
````

### 3. GPU Detection for Windows, macOS, Linux

## Important Notes
1. Ensure all dependencies are cross-platform compatible.
2. Prioritize user experience with responsive design and intuitive workflows. The GUI should be good enough for non-technical users. Tooltips with explanations for each option are essential. Do not use emojis, prefer business and formal design instead.
3. The arguments should match those in the wavedl CLI for consistency.
4. No need to create tests, i will implement them later.
5. Focus on modularity to allow easy future extensions (e.g., adding new models or loss functions).
