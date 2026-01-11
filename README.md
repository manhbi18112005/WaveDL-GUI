<div align="center">

<img src="logos/wavedl_logo.png" alt="WaveDL Logo" width="500">

### A Scalable Deep Learning Framework for Wave-Based Inverse Problems

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?style=plastic&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?style=plastic&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Accelerate](https://img.shields.io/badge/Accelerate-Enabled-yellow.svg?style=plastic&logo=huggingface&logoColor=white)](https://huggingface.co/docs/accelerate/)
<br>
[![Tests](https://img.shields.io/github/actions/workflow/status/ductho-le/WaveDL/test.yml?branch=main&style=plastic&logo=githubactions&logoColor=white&label=Tests)](https://github.com/ductho-le/WaveDL/actions/workflows/test.yml)
[![Lint](https://img.shields.io/github/actions/workflow/status/ductho-le/WaveDL/lint.yml?branch=main&style=plastic&logo=ruff&logoColor=white&label=Lint)](https://github.com/ductho-le/WaveDL/actions/workflows/lint.yml)
[![Try it on Colab](https://img.shields.io/badge/Try_it_on_Colab-8E44AD?style=plastic&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/ductho-le/WaveDL/blob/main/notebooks/demo.ipynb)
<br>
[![Downloads](https://img.shields.io/badge/dynamic/json?url=https://pypistats.org/api/packages/wavedl/recent?period=month%26mirrors=false&query=data.last_month&style=plastic&logo=pypi&logoColor=white&color=9ACD32&label=Downloads&suffix=/month)](https://pypistats.org/packages/wavedl)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg?style=plastic)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.18012338-008080.svg?style=plastic)](https://doi.org/10.5281/zenodo.18012338)

**Production-ready • Multi-GPU DDP • Memory-Efficient • Plug-and-Play**

[Getting Started](#-getting-started) •
[Documentation](#-documentation) •
[Examples](#-examples) •
[Discussions](https://github.com/ductho-le/WaveDL/discussions) •
[Citation](#-citation)

---

 **Plug in your model, load your data, and let WaveDL do the heavy lifting 💪**

</div>

---

## 💡 What is WaveDL?

WaveDL is a **deep learning framework** built for **wave-based inverse problems** — from ultrasonic NDE and geophysics to biomedical tissue characterization. It provides a robust, scalable training pipeline for mapping multi-dimensional data (1D/2D/3D) to physical quantities.

```
Input: Waveforms, spectrograms, B-scans, dispersion curves, ...
   ↓
Output: Material properties, defect dimensions, damage locations, ...
```

The framework handles the engineering challenges of large-scale deep learning — big datasets, distributed training, and HPC deployment — so you can focus on the science, not the infrastructure.

**Built for researchers who need:**
- 📊 Multi-target regression with reproducibility and fair benchmarking
- 🚀 Seamless multi-GPU training on HPC clusters
- 💾 Memory-efficient handling of large-scale datasets
- 🔧 Easy integration of custom model architectures

---

## ✨ Features

<div align="center">
<table width="100%">
<tr>
<td width="50%" valign="top">

**⚡ Load All Data — No More Bottleneck**

Train on datasets larger than RAM:
- Memory-mapped, zero-copy streaming
- Full random shuffling at GPU speed
- Your GPU stays fed — always

</td>
<td width="50%" valign="top">

**🧠 Models? We've Got Options**

38 architectures, ready to go:
- CNNs, ResNets, ViTs, EfficientNets...
- All adapted for regression
- [Add your own](#adding-custom-models) in one line

</td>
</tr>
<tr>
<td width="50%" valign="top">

**🛡️ DDP That Actually Works**

Multi-GPU training without the pain:
- Synchronized early stopping
- Deadlock-free checkpointing
- Correct metric aggregation

</td>
<td width="50%" valign="top">

**🔬 Physics-Constrained Training**

Make your model respect the laws:
- Enforce bounds, positivity, equations
- Simple expression syntax or Python
- [Custom constraints](#physical-constraints) for various laws

</td>
</tr>
<tr>
<td width="50%" valign="top">

**🖥️ HPC-Native Design**

Built for high-performance clusters:
- Automatic GPU detection
- WandB experiment tracking
- BF16/FP16 mixed precision

</td>
<td width="50%" valign="top">

**🔄 Crash-Proof Training**

Never lose your progress:
- Full state checkpoints
- Resume from any point
- Emergency saves on interrupt

</td>
</tr>
<tr>
<td width="50%" valign="top">

**🎛️ Flexible & Reproducible Training**

Fully configurable via CLI flags or YAML:
- Loss functions, optimizers, schedulers
- K-fold cross-validation
- See [Configuration](#️-configuration) for details

</td>
<td width="50%" valign="top">

**📦 ONNX Export**

Deploy models anywhere:
- One-command export to ONNX
- LabVIEW, MATLAB, C++ compatible
- Validated PyTorch↔ONNX outputs

</td>
</tr>
</table>
</div>

---

## 🚀 Getting Started

### Installation

#### From PyPI (recommended for all users)

```bash
pip install wavedl
```

This installs everything you need: training, inference, HPO, ONNX export.

#### From Source (for development)

```bash
git clone https://github.com/ductho-le/WaveDL.git
cd WaveDL
pip install -e .
```

> [!NOTE]
> Python 3.11+ required. For development setup, see [CONTRIBUTING.md](.github/CONTRIBUTING.md).

### Quick Start

> [!TIP]
> In all examples below, replace `<...>` placeholders with your values. See [Configuration](#️-configuration) for defaults and options.

#### Option 1: Using wavedl-hpc (Recommended for HPC)

The `wavedl-hpc` command automatically configures the environment for HPC systems:

```bash
# Basic training (auto-detects available GPUs)
wavedl-hpc --model <model_name> --data_path <train_data> --batch_size <number> --output_dir <output_folder>

# Detailed configuration
wavedl-hpc --model <model_name> --data_path <train_data> --batch_size <number> \
  --lr <number> --epochs <number> --patience <number> --compile --output_dir <output_folder>

# Specify GPU count explicitly
wavedl-hpc --num_gpus 4 --model cnn --data_path train.npz --output_dir results
```

#### Option 2: Direct Accelerate Launch

```bash
# Local - auto-detects GPUs
accelerate launch -m wavedl.train --model <model_name> --data_path <train_data> --batch_size <number> --output_dir <output_folder>

# Resume training (automatic - just re-run with same output_dir)
# Manual resume from specific checkpoint:
accelerate launch -m wavedl.train --model <model_name> --data_path <train_data> --resume <checkpoint_folder> --output_dir <output_folder>

# Force fresh start (ignores existing checkpoints)
accelerate launch -m wavedl.train --model <model_name> --data_path <train_data> --output_dir <output_folder> --fresh

# List available models
wavedl-train --list_models
```

> [!TIP]
> **Auto-Resume**: If training crashes or is interrupted, simply re-run with the same `--output_dir`. The framework automatically detects incomplete training and resumes from the last checkpoint. Use `--fresh` to force a fresh start.
>
> **GPU Auto-Detection**: `wavedl-hpc` automatically detects available GPUs using `nvidia-smi`. Use `--num_gpus` to override.

### Testing & Inference

After training, use `wavedl.test` to evaluate your model on test data:

```bash
# Basic inference
python -m wavedl.test --checkpoint <checkpoint_folder> --data_path <test_data>

# With visualization, CSV export, and multiple file formats
python -m wavedl.test --checkpoint <checkpoint_folder> --data_path <test_data> \
  --plot --plot_format png pdf --save_predictions --output_dir <output_folder>

# With custom parameter names
python -m wavedl.test --checkpoint <checkpoint_folder> --data_path <test_data> \
  --param_names '$p_1$' '$p_2$' '$p_3$' --plot

# Export model to ONNX for deployment (LabVIEW, MATLAB, C++, etc.)
python -m wavedl.test --checkpoint <checkpoint_folder> --data_path <test_data> \
  --export onnx --export_path <output_file.onnx>

# For 3D volumes with small depth (e.g., 8×128×128), override auto-detection
python -m wavedl.test --checkpoint <checkpoint_folder> --data_path <test_data> \
  --input_channels 1
```

**Output:**
- **Console**: R², Pearson correlation, MAE per parameter
- **CSV** (with `--save_predictions`): True, predicted, error, and absolute error for all parameters
- **Plots** (with `--plot`): 10 publication-quality plots (scatter, histogram, residuals, Bland-Altman, Q-Q, correlation, relative error, CDF, index plot, box plot)
- **Format** (with `--plot_format`): Supported formats: `png` (default), `pdf` (vector), `svg` (vector), `eps` (LaTeX), `tiff`, `jpg`, `ps`

> [!NOTE]
> `wavedl.test` auto-detects the model architecture from checkpoint metadata. If unavailable, it falls back to folder name parsing. Use `--model` to override if needed.

### Adding Custom Models

<details>
<summary><b>Creating Your Own Architecture</b></summary>

**Requirements** (your model must):
1. Inherit from `BaseModel`
2. Accept `in_shape`, `out_size` in `__init__`
3. Return a tensor of shape `(batch, out_size)` from `forward()`

---

**Step 1: Create `my_model.py`**

```python
import torch.nn as nn
import torch.nn.functional as F
from wavedl.models import BaseModel, register_model

@register_model("my_model")  # This name is used with --model flag
class MyModel(BaseModel):
    def __init__(self, in_shape, out_size, **kwargs):
        # in_shape: spatial dimensions, e.g., (128,) or (64, 64) or (32, 32, 32)
        # out_size: number of parameters to predict (auto-detected from data)
        super().__init__(in_shape, out_size)

        # Define your layers (this is just an example for 2D)
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)  # Input always has 1 channel
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128, out_size)

    def forward(self, x):
        # Input x has shape: (batch, 1, *in_shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=[-2, -1])  # Global average pooling
        return self.fc(x)  # Output shape: (batch, out_size)
```

**Step 2: Train**

```bash
wavedl-hpc --import my_model.py --model my_model --data_path train.npz
```

WaveDL handles everything else: training loop, logging, checkpoints, multi-GPU, early stopping, etc.

</details>

---

## 📁 Project Structure

```
WaveDL/
├── src/
│   └── wavedl/                   # Main package (namespaced)
│       ├── __init__.py           # Package init with __version__
│       ├── train.py              # Training entry point
│       ├── test.py               # Testing & inference script
│       ├── hpo.py                # Hyperparameter optimization
│       ├── hpc.py                # HPC distributed training launcher
│       │
│       ├── models/               # Model architectures (38 variants)
│       │   ├── registry.py       # Model factory (@register_model)
│       │   ├── base.py           # Abstract base class
│       │   ├── cnn.py            # Baseline CNN (1D/2D/3D)
│       │   ├── resnet.py         # ResNet-18/34/50 (1D/2D/3D)
│       │   ├── resnet3d.py       # ResNet3D-18, MC3-18 (3D only)
│       │   ├── tcn.py            # TCN (1D only)
│       │   ├── efficientnet.py   # EfficientNet-B0/B1/B2 (2D)
│       │   ├── efficientnetv2.py # EfficientNetV2-S/M/L (2D)
│       │   ├── mobilenetv3.py    # MobileNetV3-Small/Large (2D)
│       │   ├── regnet.py         # RegNetY variants (2D)
│       │   ├── swin.py           # Swin Transformer (2D)
│       │   ├── vit.py            # Vision Transformer (1D/2D)
│       │   ├── convnext.py       # ConvNeXt (1D/2D/3D)
│       │   ├── densenet.py       # DenseNet-121/169 (1D/2D/3D)
│       │   └── unet.py           # U-Net Regression
│       │
│       └── utils/                # Utilities
│           ├── data.py           # Memory-mapped data pipeline
│           ├── metrics.py        # R², Pearson, visualization
│           ├── constraints.py    # Physical constraints for training
│           ├── distributed.py    # DDP synchronization
│           ├── losses.py         # Loss function factory
│           ├── optimizers.py     # Optimizer factory
│           ├── schedulers.py     # LR scheduler factory
│           └── config.py         # YAML configuration support
│
├── configs/                      # YAML config templates
├── examples/                     # Ready-to-run examples
├── notebooks/                    # Jupyter notebooks
├── unit_tests/                   # Pytest test suite (725 tests)
│
├── pyproject.toml                # Package config, dependencies
├── CHANGELOG.md                  # Version history
└── CITATION.cff                  # Citation metadata
```
---

## ⚙️ Configuration

> [!NOTE]
> All configuration options below work with **both** `wavedl-hpc` and direct `accelerate launch`. The wrapper script passes all arguments directly to `train.py`.
>
> **Examples:**
> ```bash
> # Using wavedl-hpc
> wavedl-hpc --model cnn --batch_size 256 --lr 5e-4 --compile
>
> # Using accelerate launch directly
> accelerate launch -m wavedl.train --model cnn --batch_size 256 --lr 5e-4 --compile
> ```

<details>
<summary><b>Available Models</b> — 38 architectures</summary>

| Model | Params | Dim |
|-------|--------|-----|
| **CNN** — Convolutional Neural Network |||
| `cnn` | 1.7M | 1D/2D/3D |
| **ResNet** — Residual Network |||
| `resnet18` | 11.4M | 1D/2D/3D |
| `resnet34` | 21.5M | 1D/2D/3D |
| `resnet50` | 24.6M | 1D/2D/3D |
| `resnet18_pretrained` ⭐ | 11.4M | 2D |
| `resnet50_pretrained` ⭐ | 24.6M | 2D |
| **ResNet3D** — 3D Residual Network |||
| `resnet3d_18` | 33.6M | 3D |
| `mc3_18` — Mixed Convolution 3D | 11.9M | 3D |
| **TCN** — Temporal Convolutional Network |||
| `tcn_small` | 1.0M | 1D |
| `tcn` | 7.0M | 1D |
| `tcn_large` | 10.2M | 1D |
| **EfficientNet** — Efficient Neural Network |||
| `efficientnet_b0` ⭐ | 4.7M | 2D |
| `efficientnet_b1` ⭐ | 7.2M | 2D |
| `efficientnet_b2` ⭐ | 8.4M | 2D |
| **EfficientNetV2** — Efficient Neural Network V2 |||
| `efficientnet_v2_s` ⭐ | 21.0M | 2D |
| `efficientnet_v2_m` ⭐ | 53.6M | 2D |
| `efficientnet_v2_l` ⭐ | 118.0M | 2D |
| **MobileNetV3** — Mobile Neural Network V3 |||
| `mobilenet_v3_small` ⭐ | 1.1M | 2D |
| `mobilenet_v3_large` ⭐ | 3.2M | 2D |
| **RegNet** — Regularized Network |||
| `regnet_y_400mf` ⭐ | 4.0M | 2D |
| `regnet_y_800mf` ⭐ | 5.8M | 2D |
| `regnet_y_1_6gf` ⭐ | 10.5M | 2D |
| `regnet_y_3_2gf` ⭐ | 18.3M | 2D |
| `regnet_y_8gf` ⭐ | 37.9M | 2D |
| **Swin** — Shifted Window Transformer |||
| `swin_t` ⭐ | 28.0M | 2D |
| `swin_s` ⭐ | 49.4M | 2D |
| `swin_b` ⭐ | 87.4M | 2D |
| **ConvNeXt** — Convolutional Next |||
| `convnext_tiny` | 28.2M | 1D/2D/3D |
| `convnext_small` | 49.8M | 1D/2D/3D |
| `convnext_base` | 88.1M | 1D/2D/3D |
| `convnext_tiny_pretrained` ⭐ | 28.2M | 2D |
| **DenseNet** — Densely Connected Network |||
| `densenet121` | 7.5M | 1D/2D/3D |
| `densenet169` | 13.3M | 1D/2D/3D |
| `densenet121_pretrained` ⭐ | 7.5M | 2D |
| **ViT** — Vision Transformer |||
| `vit_tiny` | 5.5M | 1D/2D |
| `vit_small` | 21.6M | 1D/2D |
| `vit_base` | 85.6M | 1D/2D |
| **U-Net** — U-shaped Network |||
| `unet_regression` | 31.1M | 1D/2D/3D |

⭐ = **Pretrained on ImageNet** (recommended for smaller datasets). Weights are downloaded automatically on first use.
- **Cache location**: `~/.cache/torch/hub/checkpoints/` (or `./.torch_cache/` on HPC if home is not writable)
- **Size**: ~20–350 MB per model depending on architecture

**💡 HPC Users**: If compute nodes block internet, pre-download weights on the login node:

```bash
# Run once on login node (with internet) — downloads ALL pretrained weights (~1.5 GB total)
python -c "
import os
os.environ['TORCH_HOME'] = '.torch_cache'  # Match WaveDL's HPC cache location

from torchvision import models as m
from torchvision.models import video as v

# Model name -> Weights class mapping
weights = {
    'resnet18': m.ResNet18_Weights, 'resnet50': m.ResNet50_Weights,
    'efficientnet_b0': m.EfficientNet_B0_Weights, 'efficientnet_b1': m.EfficientNet_B1_Weights,
    'efficientnet_b2': m.EfficientNet_B2_Weights, 'efficientnet_v2_s': m.EfficientNet_V2_S_Weights,
    'efficientnet_v2_m': m.EfficientNet_V2_M_Weights, 'efficientnet_v2_l': m.EfficientNet_V2_L_Weights,
    'mobilenet_v3_small': m.MobileNet_V3_Small_Weights, 'mobilenet_v3_large': m.MobileNet_V3_Large_Weights,
    'regnet_y_400mf': m.RegNet_Y_400MF_Weights, 'regnet_y_800mf': m.RegNet_Y_800MF_Weights,
    'regnet_y_1_6gf': m.RegNet_Y_1_6GF_Weights, 'regnet_y_3_2gf': m.RegNet_Y_3_2GF_Weights,
    'regnet_y_8gf': m.RegNet_Y_8GF_Weights, 'swin_t': m.Swin_T_Weights, 'swin_s': m.Swin_S_Weights,
    'swin_b': m.Swin_B_Weights, 'convnext_tiny': m.ConvNeXt_Tiny_Weights, 'densenet121': m.DenseNet121_Weights,
}
for name, w in weights.items():
    getattr(m, name)(weights=w.DEFAULT); print(f'✓ {name}')

# 3D video models
v.r3d_18(weights=v.R3D_18_Weights.DEFAULT); print('✓ r3d_18')
v.mc3_18(weights=v.MC3_18_Weights.DEFAULT); print('✓ mc3_18')
print('\\n✓ All pretrained weights cached!')
"
```


</details>

<details>
<summary><b>Training Parameters</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `cnn` | Model architecture |
| `--import` | - | Python file(s) to import for custom models (supports multiple) |
| `--batch_size` | `128` | Per-GPU batch size |
| `--lr` | `1e-3` | Learning rate |
| `--epochs` | `1000` | Maximum epochs |
| `--patience` | `20` | Early stopping patience |
| `--weight_decay` | `1e-4` | AdamW regularization |
| `--grad_clip` | `1.0` | Gradient clipping |

</details>

<details>
<summary><b>Data & I/O</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `train_data.npz` | Dataset path |
| `--workers` | `-1` | DataLoader workers per GPU (-1=auto-detect) |
| `--seed` | `2025` | Random seed |
| `--output_dir` | `.` | Output directory for checkpoints |
| `--resume` | `None` | Checkpoint to resume (auto-detected if not set) |
| `--save_every` | `50` | Checkpoint frequency |
| `--fresh` | `False` | Force fresh training, ignore existing checkpoints |
| `--single_channel` | `False` | Confirm data is single-channel (for shallow 3D volumes like `(8, 128, 128)`) |

</details>

<details>
<summary><b>Performance</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--compile` | `False` | Enable `torch.compile` (recommended for long runs) |
| `--precision` | `bf16` | Mixed precision mode (`bf16`, `fp16`, `no`) |
| `--workers` | `-1` | DataLoader workers per GPU (-1=auto, up to 16) |
| `--wandb` | `False` | Enable W&B logging |
| `--wandb_watch` | `False` | Enable W&B gradient watching (adds overhead) |
| `--project_name` | `DL-Training` | W&B project name |
| `--run_name` | `None` | W&B run name (auto-generated if not set) |

**Automatic GPU Optimizations:**

WaveDL automatically enables performance optimizations for modern GPUs:

| Optimization | Effect | GPU Support |
|--------------|--------|-------------|
| **TF32 precision** | ~2x speedup for float32 matmul | A100, H100 (Ampere+) |
| **cuDNN benchmark** | Auto-tuned convolutions | All NVIDIA GPUs |
| **Worker scaling** | Up to 16 workers per GPU | All systems |

> [!NOTE]
> These optimizations are **backward compatible** — they have no effect on older GPUs (V100, T4, GTX) or CPU-only systems. No configuration needed.

**HPC Best Practices:**
- Stage data to `$SLURM_TMPDIR` (local NVMe) for maximum I/O throughput
- Use `--compile` for training runs > 50 epochs
- Increase `--workers` manually if auto-detection is suboptimal

</details>

<details>
<summary><b>HPC CLI Arguments (wavedl-hpc)</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_gpus` | **Auto-detected** | Number of GPUs to use. By default, automatically detected via `nvidia-smi`. Set explicitly to override |
| `--num_machines` | `1` | Number of machines in distributed setup |
| `--mixed_precision` | `bf16` | Precision mode: `bf16`, `fp16`, or `no` |
| `--dynamo_backend` | `no` | PyTorch Dynamo backend |

**Environment Variables (for logging):**

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_MODE` | `offline` | WandB mode: `offline` or `online` |

</details>

<details>
<summary><b>Loss Functions</b></summary>

| Loss | Flag | Best For | Notes |
|------|------|----------|-------|
| `mse` | `--loss mse` | Default, smooth gradients | Standard Mean Squared Error |
| `mae` | `--loss mae` | Outlier-robust, linear penalty | Mean Absolute Error (L1) |
| `huber` | `--loss huber --huber_delta 1.0` | Best of MSE + MAE | Robust, smooth transition |
| `smooth_l1` | `--loss smooth_l1` | Similar to Huber | PyTorch native implementation |
| `log_cosh` | `--loss log_cosh` | Smooth approximation to MAE | Differentiable everywhere |
| `weighted_mse` | `--loss weighted_mse --loss_weights "2.0,1.0,1.0"` | Prioritize specific targets | Per-target weighting |

**Example:**
```bash
# Use Huber loss for noisy NDE data
accelerate launch -m wavedl.train --model cnn --loss huber --huber_delta 0.5

# Weighted MSE: prioritize thickness (first target)
accelerate launch -m wavedl.train --model cnn --loss weighted_mse --loss_weights "2.0,1.0,1.0"
```

</details>

<details>
<summary><b>Optimizers</b></summary>

| Optimizer | Flag | Best For | Key Parameters |
|-----------|------|----------|----------------|
| `adamw` | `--optimizer adamw` | Default, most cases | `--betas "0.9,0.999"` |
| `adam` | `--optimizer adam` | Legacy compatibility | `--betas "0.9,0.999"` |
| `sgd` | `--optimizer sgd` | Better generalization | `--momentum 0.9 --nesterov` |
| `nadam` | `--optimizer nadam` | Adam + Nesterov | Faster convergence |
| `radam` | `--optimizer radam` | Variance-adaptive | More stable training |
| `rmsprop` | `--optimizer rmsprop` | RNN/LSTM models | `--momentum 0.9` |

**Example:**
```bash
# SGD with Nesterov momentum (often better generalization)
accelerate launch -m wavedl.train --model cnn --optimizer sgd --lr 0.01 --momentum 0.9 --nesterov

# RAdam for more stable training
accelerate launch -m wavedl.train --model cnn --optimizer radam --lr 1e-3
```

</details>

<details>
<summary><b>Learning Rate Schedulers</b></summary>

| Scheduler | Flag | Best For | Key Parameters |
|-----------|------|----------|----------------|
| `plateau` | `--scheduler plateau` | Default, adaptive | `--scheduler_patience 10 --scheduler_factor 0.5` |
| `cosine` | `--scheduler cosine` | Long training, smooth decay | `--min_lr 1e-6` |
| `cosine_restarts` | `--scheduler cosine_restarts` | Escape local minima | Warm restarts |
| `onecycle` | `--scheduler onecycle` | Fast convergence | Super-convergence |
| `step` | `--scheduler step` | Simple decay | `--step_size 30 --scheduler_factor 0.1` |
| `multistep` | `--scheduler multistep` | Custom milestones | `--milestones "30,60,90"` |
| `exponential` | `--scheduler exponential` | Continuous decay | `--scheduler_factor 0.95` |
| `linear_warmup` | `--scheduler linear_warmup` | Warmup phase | `--warmup_epochs 5` |

**Example:**
```bash
# Cosine annealing for 1000 epochs
accelerate launch -m wavedl.train --model cnn --scheduler cosine --epochs 1000 --min_lr 1e-7

# OneCycleLR for super-convergence
accelerate launch -m wavedl.train --model cnn --scheduler onecycle --lr 1e-2 --epochs 50

# MultiStep with custom milestones
accelerate launch -m wavedl.train --model cnn --scheduler multistep --milestones "100,200,300"
```

</details>

<details>
<summary><b>Cross-Validation</b></summary>

For robust model evaluation, simply add the `--cv` flag:

```bash
# 5-fold cross-validation (works with both methods!)
wavedl-hpc --model cnn --cv 5 --data_path train_data.npz
# OR
accelerate launch -m wavedl.train --model cnn --cv 5 --data_path train_data.npz

# Stratified CV (recommended for unbalanced data)
wavedl-hpc --model cnn --cv 5 --cv_stratify --loss huber --epochs 100

# Full configuration
wavedl-hpc --model cnn --cv 5 --cv_stratify \
    --loss huber --optimizer adamw --scheduler cosine \
    --output_dir ./cv_results
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--cv` | `0` | Number of CV folds (0=disabled, normal training) |
| `--cv_stratify` | `False` | Use stratified splitting (bins targets) |
| `--cv_bins` | `10` | Number of bins for stratified CV |

**Output:**
- `cv_summary.json`: Aggregated metrics (mean ± std)
- `cv_results.csv`: Per-fold detailed results
- `fold_*/`: Individual fold models and scalers

</details>

<details>
<summary><b>Configuration Files (YAML)</b></summary>

Use YAML files for reproducible experiments. CLI arguments can override any config value.

```bash
# Use a config file
accelerate launch -m wavedl.train --config configs/config.yaml --data_path train.npz

# Override specific values from config
accelerate launch -m wavedl.train --config configs/config.yaml --lr 5e-4 --epochs 500
```

**Example config (`configs/config.yaml`):**
```yaml
# Model & Training
model: cnn
batch_size: 128
lr: 0.001
epochs: 1000
patience: 20

# Loss, Optimizer, Scheduler
loss: mse
optimizer: adamw
scheduler: plateau

# Cross-Validation (0 = disabled)
cv: 0

# Performance
precision: bf16
compile: false
seed: 2025
```

> See [`configs/config.yaml`](configs/config.yaml) for the complete template with all available options documented.

</details>

<details>
<summary><b>Physical Constraints</b> — Enforce Physics During Training</summary>

Add penalty terms to the loss function to enforce physical laws:

```
Total Loss = Data Loss + weight × penalty(violation)
```

### Expression Constraints

```bash
# Positivity
--constraint "y0 > 0"

# Bounds
--constraint "y0 >= 0" "y0 <= 1"

# Equations (penalize deviations from zero)
--constraint "y2 - y0 * y1"

# Input-dependent constraints
--constraint "y0 - 2*x[0]"

# Multiple constraints with different weights
--constraint "y0 > 0" "y1 - y2" --constraint_weight 0.1 1.0
```

### Custom Python Constraints

For complex physics (matrix operations, implicit equations):

```python
# my_constraint.py
import torch

def constraint(pred, inputs=None):
    """
    Args:
        pred:   (batch, num_outputs)
        inputs: (batch, features) or (batch, C, H, W) or (batch, C, D, H, W)
    Returns:
        (batch,) — violation per sample (0 = satisfied)
    """
    # Outputs (same for all data types)
    y0, y1, y2 = pred[:, 0], pred[:, 1], pred[:, 2]

    # Inputs — Tabular: (batch, features)
    # x0 = inputs[:, 0]                    # Feature 0
    # x_sum = inputs.sum(dim=1)            # Sum all features

    # Inputs — Images: (batch, C, H, W)
    # pixel = inputs[:, 0, 3, 5]           # Pixel at (3,5), channel 0
    # img_mean = inputs.mean(dim=(1,2,3))  # Mean over C,H,W

    # Inputs — 3D Volumes: (batch, C, D, H, W)
    # voxel = inputs[:, 0, 2, 3, 5]        # Voxel at (2,3,5), channel 0

    # Example constraints:
    # return y2 - y0 * y1                                    # Wave equation
    # return y0 - 2 * inputs[:, 0]                           # Output = 2×input
    # return inputs[:, 0, 3, 5] * y0 + inputs[:, 0, 6, 7] * y1  # Mixed

    return y0 - y1 * y2
```

```bash
--constraint_file my_constraint.py --constraint_weight 1.0
```

---

### Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--constraint` | — | Expression(s): `"y0 > 0"`, `"y0 - y1*y2"` |
| `--constraint_file` | — | Python file with `constraint(pred, inputs)` |
| `--constraint_weight` | `0.1` | Penalty weight(s) |
| `--constraint_reduction` | `mse` | `mse` (squared) or `mae` (linear) |

#### Expression Syntax

| Variable | Meaning |
|----------|---------|
| `y0`, `y1`, ... | Model outputs |
| `x[0]`, `x[1]`, ... | Input values (1D tabular) |
| `x[i,j]`, `x[i,j,k]` | Input values (2D/3D: images, volumes) |
| `x_mean`, `x_sum`, `x_max`, `x_min`, `x_std` | Input aggregates |

**Operators:** `+`, `-`, `*`, `/`, `**`, `>`, `<`, `>=`, `<=`, `==`

**Functions:** `sin`, `cos`, `exp`, `log`, `sqrt`, `sigmoid`, `softplus`, `tanh`, `relu`, `abs`

</details>



<details>
<summary><b>Hyperparameter Search (HPO)</b></summary>

Automatically find the best training configuration using [Optuna](https://optuna.org/).

**Run HPO:**

```bash
# Basic HPO (auto-detects GPUs for parallel trials)
wavedl-hpo --data_path train.npz --models cnn --n_trials 100

# Search multiple models
wavedl-hpo --data_path train.npz --models cnn resnet18 efficientnet_b0 --n_trials 200

# Quick mode (fewer parameters, faster)
wavedl-hpo --data_path train.npz --models cnn --n_trials 50 --quick
```

> [!TIP]
> **Auto GPU Detection**: HPO automatically detects available GPUs and runs one trial per GPU in parallel. On a 4-GPU system, 4 trials run simultaneously. Use `--n_jobs 1` to force serial execution.

**Train with best parameters**

After HPO completes, it prints the optimal command:
```bash
accelerate launch -m wavedl.train --data_path train.npz --model cnn --lr 3.2e-4 --batch_size 128 ...
```

---

**What Gets Searched:**

| Parameter | Default | You Can Override With |
|-----------|---------|----------------------|
| Models | cnn, resnet18, resnet34 | `--models X Y Z` |
| Optimizers | [all 6](#optimizers) | `--optimizers X Y` |
| Schedulers | [all 8](#learning-rate-schedulers) | `--schedulers X Y` |
| Losses | [all 6](#loss-functions) | `--losses X Y` |
| Learning rate | 1e-5 → 1e-2 | (always searched) |
| Batch size | 16, 32, 64, 128 | (always searched) |

**Quick Mode** (`--quick`):
- Uses minimal defaults: cnn + adamw + plateau + mse
- Faster for testing your setup before running full search
- You can still override any option with the flags above

---

**All Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | (required) | Training data file |
| `--models` | 3 defaults | Models to search (specify any number) |
| `--n_trials` | `50` | Number of trials to run |
| `--quick` | `False` | Use minimal defaults (faster) |
| `--optimizers` | all 6 | Optimizers to search |
| `--schedulers` | all 8 | Schedulers to search |
| `--losses` | all 6 | Losses to search |
| `--n_jobs` | `-1` | Parallel trials (-1 = auto-detect GPUs) |
| `--max_epochs` | `50` | Max epochs per trial |
| `--output` | `hpo_results.json` | Output file |


> See [Available Models](#available-models) for all 38 architectures you can search.

</details>

---

## 📈 Data Preparation

WaveDL supports multiple data formats for training and inference:

| Format | Extension | Key Advantages |
|--------|-----------|----------------|
| **NPZ** | `.npz` | Native NumPy, fast loading, recommended |
| **HDF5** | `.h5`, `.hdf5` | Large datasets, hierarchical, cross-platform |
| **MAT** | `.mat` | MATLAB compatibility (**v7.3+ only**, saved with `-v7.3` flag) |

**The framework automatically detects file format and data dimensionality** (1D, 2D, or 3D) — you only need to provide the appropriate model architecture.

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `input_train` / `input_test` | `(N, L)`, `(N, H, W)`, or `(N, D, H, W)` | `float32` | N samples of 1D/2D/3D representations |
| `output_train` / `output_test` | `(N, T)` | `float32` | N samples with T regression targets |

> [!TIP]
> - **Flexible Key Names**: WaveDL auto-detects common key pairs:
>   - `input_train`/`output_train`, `input_test`/`output_test` (WaveDL standard)
>   - `X`/`Y`, `x`/`y` (ML convention)
>   - `data`/`labels`, `inputs`/`outputs`, `features`/`targets`
> - **Automatic Dimension Detection**: Channel dimension is added automatically. No manual reshaping required!
> - **Sparse Matrix Support**: NPZ and MAT v7.3 files with scipy/MATLAB sparse matrices are automatically converted to dense arrays.
> - **Auto-Normalization**: Target values are automatically standardized during training. MAE is reported in original physical units.

> [!IMPORTANT]
> **MATLAB Users**: MAT files must be saved with the `-v7.3` flag for memory-efficient loading:
> ```matlab
> save('data.mat', 'input_train', 'output_train', '-v7.3')
> ```
> Older MAT formats (v5/v7) are not supported. Convert to NPZ for best compatibility.

<details>
<summary><b>Example: Basic Preparation</b></summary>

```python
import numpy as np

X = np.array(images, dtype=np.float32)  # (N, H, W)
y = np.array(labels, dtype=np.float32)  # (N, T)

np.savez('train_data.npz', input_train=X, output_train=y)
```

</details>

<details>
<summary><b>Example: From Image Files + CSV</b></summary>

```python
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

# Load images
images = [np.array(Image.open(f).convert('L'), dtype=np.float32)
          for f in sorted(Path("images/").glob("*.png"))]
X = np.stack(images)

# Load labels
y = pd.read_csv("labels.csv").values.astype(np.float32)

np.savez('train_data.npz', input_train=X, output_train=y)
```

</details>

<details>
<summary><b>Example: From MATLAB (.mat)</b></summary>

```python
import numpy as np
from scipy.io import loadmat

data = loadmat('simulation_data.mat')
X = data['spectrograms'].astype(np.float32)  # Adjust key
y = data['parameters'].astype(np.float32)

# Transpose if needed: (H, W, N) → (N, H, W)
if X.ndim == 3 and X.shape[2] < X.shape[0]:
    X = np.transpose(X, (2, 0, 1))

np.savez('train_data.npz', input_train=X, output_train=y)
```

</details>

<details>
<summary><b>Example: Synthetic Test Data</b></summary>

```python
import numpy as np

X = np.random.randn(1000, 256, 256).astype(np.float32)
y = np.random.randn(1000, 5).astype(np.float32)

np.savez('test_data.npz', input_test=X, output_test=y)
```

</details>

<details>
<summary><b>Validation Script</b></summary>

```python
import numpy as np

data = np.load('train_data.npz')
assert data['input_train'].ndim >= 2, "Input must be at least 2D: (N, ...) "
assert data['output_train'].ndim == 2, "Output must be 2D: (N, T)"
assert len(data['input_train']) == len(data['output_train']), "Sample mismatch"

print(f"✓ Input:  {data['input_train'].shape} {data['input_train'].dtype}")
print(f"✓ Output: {data['output_train'].shape} {data['output_train'].dtype}")
```

</details>


---

## 📦 Examples [![Try it on Colab](https://img.shields.io/badge/Try_it_on_Colab-8E44AD?style=plastic&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/ductho-le/WaveDL/blob/main/notebooks/demo.ipynb)

The `examples/` folder contains a **complete, ready-to-run example** for **material characterization of isotropic plates**. The pre-trained CNN predicts three physical parameters from Lamb wave dispersion curves:

| Parameter | Unit | Description |
|-----------|------|-------------|
| *h* | mm | Plate thickness |
| √(*E*/ρ) | km/s | Square root of Young's modulus over density |
| *ν* | — | Poisson's ratio |

> [!NOTE]
> This example is based on our paper at **SPIE Smart Structures + NDE 2026**: [*"Deep learning-based ultrasonic assessment of plate thickness and elasticity"*](https://spie.org/spie-smart-structures-and-materials-nondestructive-evaluation/presentation/Deep-learningbased-ultrasonic-assessment-of-plate-thickness-and-elasticity/13951-4) (Paper 13951-4, to appear).

**Try it yourself:**

```bash
# Run inference on the example data
python -m wavedl.test --checkpoint ./examples/elastic_cnn_example/best_checkpoint \
  --data_path ./examples/elastic_cnn_example/Test_data_500.mat \
  --plot --save_predictions --output_dir ./examples/elastic_cnn_example/test_results

# Export to ONNX (already included as model.onnx)
python -m wavedl.test --checkpoint ./examples/elastic_cnn_example/best_checkpoint \
  --data_path ./examples/elastic_cnn_example/Test_data_500.mat \
  --export onnx --export_path ./examples/elastic_cnn_example/model.onnx
```

**What's Included:**

| File | Description |
|------|-------------|
| `best_checkpoint/` | Pre-trained CNN checkpoint |
| `Test_data_500.mat` | 500 sample test set (500×500 dispersion curves → *h*, √(*E*/ρ), *ν*) |
| `model.onnx` | ONNX export with embedded de-normalization |
| `training_history.csv` | Epoch-by-epoch training metrics (loss, R², LR, etc.) |
| `training_curves.png` | Training/validation loss and learning rate plot |
| `test_results/` | Example predictions and diagnostic plots |
| `WaveDL_ONNX_Inference.m` | MATLAB script for ONNX inference |

**Training Progress:**

<p align="center">
  <img src="examples/elastic_cnn_example/training_curves.png" alt="Training curves" width="600"><br>
  <em>Training and validation loss over 227 epochs with <code>onecycle</code> learning rate schedule</em>
</p>

**Inference Results:**

<p align="center">
  <img src="examples/elastic_cnn_example/test_results/scatter_all.png" alt="Scatter plot" width="700"><br>
  <em>Figure 1: Predictions vs ground truth for all three elastic parameters</em>
</p>

<p align="center">
  <img src="examples/elastic_cnn_example/test_results/error_histogram.png" alt="Error histogram" width="700"><br>
  <em>Figure 2: Distribution of prediction errors showing near-zero mean bias</em>
</p>

<p align="center">
  <img src="examples/elastic_cnn_example/test_results/residuals.png" alt="Residual plot" width="700"><br>
  <em>Figure 3: Residuals vs predicted values (no heteroscedasticity detected)</em>
</p>

<p align="center">
  <img src="examples/elastic_cnn_example/test_results/bland_altman.png" alt="Bland-Altman plot" width="700"><br>
  <em>Figure 4: Bland-Altman analysis with ±1.96 SD limits of agreement</em>
</p>

<p align="center">
  <img src="examples/elastic_cnn_example/test_results/qq_plot.png" alt="Q-Q plot" width="700"><br>
  <em>Figure 5: Q-Q plots confirming normally distributed prediction errors</em>
</p>

<p align="center">
  <img src="examples/elastic_cnn_example/test_results/error_correlation.png" alt="Error correlation" width="300"><br>
  <em>Figure 6: Error correlation matrix between parameters</em>
</p>

<p align="center">
  <img src="examples/elastic_cnn_example/test_results/relative_error.png" alt="Relative error" width="700"><br>
  <em>Figure 7: Relative error (%) vs true value for each parameter</em>
</p>

<p align="center">
  <img src="examples/elastic_cnn_example/test_results/error_cdf.png" alt="Error CDF" width="500"><br>
  <em>Figure 8: Cumulative error distribution — 95% of predictions within indicated bounds</em>
</p>

<p align="center">
  <img src="examples/elastic_cnn_example/test_results/prediction_vs_index.png" alt="Prediction vs index" width="700"><br>
  <em>Figure 9: True vs predicted values by sample index</em>
</p>

<p align="center">
  <img src="examples/elastic_cnn_example/test_results/error_boxplot.png" alt="Error box plot" width="400"><br>
  <em>Figure 10: Error distribution summary (median, quartiles, outliers)</em>
</p>

---

## 🔬 Broader Applications

Beyond the material characterization example above, the WaveDL pipeline can be adapted for a wide range of **wave-based inverse problems** across multiple domains:

### 🏗️ Non-Destructive Evaluation & Structural Health Monitoring

| Application | Input | Output |
|-------------|-------|--------|
| Defect Sizing | A-scans, phased array images, FMC/TFM, ... | Crack length, depth, ... |
| Corrosion Estimation | Thickness maps, resonance spectra, ... | Wall thickness, corrosion rate, ... |
| Weld Quality Assessment | Phased array images, TOFD, ... | Porosity %, penetration depth, ... |
| RUL Prediction | Acoustic emission (AE), vibration spectra, ... | Cycles to failure, ... |
| Damage Localization | Wavefield images, DAS/DVS data, ... | Damage coordinates (x, y, z) |

### 🌍 Geophysics & Seismology

| Application | Input | Output |
|-------------|-------|--------|
| Seismic Inversion | Shot gathers, seismograms, ... | Velocity models, density profiles, ... |
| Subsurface Characterization | Surface wave dispersion, receiver functions, ... | Layer thickness, shear modulus, ... |
| Earthquake Source Parameters | Waveforms, spectrograms, ... | Magnitude, depth, focal mechanism, ... |
| Reservoir Characterization | Reflection seismic, AVO attributes, ... | Porosity, fluid saturation, ... |

### 🩺 Biomedical Ultrasound & Elastography

| Application | Input | Output |
|-------------|-------|--------|
| Tissue Elastography | Shear wave data, strain images, ... | Shear modulus, Young's modulus, ... |
| Liver Fibrosis Staging | Elastography images, US RF data, ... | Stiffness (kPa), fibrosis score, ... |
| Tumor Characterization | B-mode + elastography, ARFI data, ... | Lesion stiffness, size, ... |
| Bone QUS | Axial-transmission signals, ... | Porosity, cortical thickness, elastic modulus ... |

> [!NOTE]
> Adapting WaveDL to these applications requires preparing your own dataset and choosing a suitable model architecture to match your input dimensionality.

---

## 📚 Documentation

| Resource | Description |
|----------|-------------|
| Technical Paper | In-depth framework description *(coming soon)* |
| [`_template.py`](src/wavedl/models/_template.py) | Template for custom architectures |

---

## 📜 Citation

If you use WaveDL in your research, please cite:

```bibtex
@software{le2025wavedl,
  author = {Le, Ductho},
  title = {{WaveDL}: A Scalable Deep Learning Framework for Wave-Based Inverse Problems},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18012338},
  url = {https://doi.org/10.5281/zenodo.18012338}
}
```

Or in APA format:
> Le, D. (2025). *WaveDL: A Scalable Deep Learning Framework for Wave-Based Inverse Problems*. Zenodo. https://doi.org/10.5281/zenodo.18012338

---

## 🙏 Acknowledgments

Ductho Le would like to acknowledge [NSERC](https://www.nserc-crsng.gc.ca/) and [Alberta Innovates](https://albertainnovates.ca/) for supporting his study and research by means of a research assistantship and a graduate doctoral fellowship.

This research was enabled in part by support provided by [Compute Ontario](https://www.computeontario.ca/), [Calcul Québec](https://www.calculquebec.ca/), and the [Digital Research Alliance of Canada](https://alliancecan.ca/).

<br>

<p align="center">
  <a href="https://www.ualberta.ca/"><img src="logos/ualberta.png" alt="University of Alberta" height="60"></a>
  &emsp;&emsp;
  <a href="https://albertainnovates.ca/"><img src="logos/alberta_innovates.png" alt="Alberta Innovates" height="60"></a>
  &emsp;&emsp;
  <a href="https://www.nserc-crsng.gc.ca/"><img src="logos/nserc.png" alt="NSERC" height="60"></a>
</p>

<p align="center">
  <a href="https://alliancecan.ca/"><img src="logos/drac.png" alt="Digital Research Alliance of Canada" height="50"></a>
</p>

---

<div align="center">

**[Ductho Le](mailto:ductho.le@outlook.com)** · University of Alberta

[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--3073--1416-a6ce39?style=plastic&logo=orcid&logoColor=white)](https://orcid.org/0000-0002-3073-1416)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-4285F4?style=plastic&logo=google-scholar&logoColor=white)](https://scholar.google.ca/citations?user=OlwMr9AAAAAJ)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=plastic&logo=researchgate&logoColor=white)](https://www.researchgate.net/profile/Ductho-Le)

<sub>May your signals be strong and your attenuation low 👋</sub>

</div>
