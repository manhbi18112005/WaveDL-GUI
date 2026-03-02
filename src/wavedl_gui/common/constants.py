"""
WaveDL GUI - Constants and Training Configuration

Centralized definitions for application-wide constants, default values,
and configuration options that match the WaveDL CLI arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# DEFAULT VALUES (matching wavedl CLI defaults)
# =============================================================================

# Training hyperparameters
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 1000
DEFAULT_PATIENCE = 20
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_SEED = 2025

# Loss functions available in wavedl
LOSS_FUNCTIONS = [
    ("mse", "Mean Squared Error", "Standard MSE loss for regression tasks"),
    ("mae", "Mean Absolute Error", "L1 loss, more robust to outliers"),
    ("huber", "Huber Loss", "Smooth L1 loss, combines MSE and MAE"),
    ("smooth_l1", "Smooth L1", "Similar to Huber with delta=1"),
    ("log_cosh", "Log-Cosh", "Smooth approximation of MAE"),
    ("weighted_mse", "Weighted MSE", "MSE with per-output weights"),
]

# Optimizers available in wavedl
OPTIMIZERS = [
    ("adamw", "AdamW", "Adam with decoupled weight decay (recommended)"),
    ("adam", "Adam", "Standard Adam optimizer"),
    ("sgd", "SGD", "Stochastic Gradient Descent with momentum"),
    ("nadam", "NAdam", "Adam with Nesterov momentum"),
    ("radam", "RAdam", "Rectified Adam for stable training"),
    ("rmsprop", "RMSprop", "Root Mean Square Propagation"),
]

# Learning rate schedulers available in wavedl
SCHEDULERS = [
    ("plateau", "Reduce on Plateau", "Reduce LR when validation loss plateaus"),
    ("cosine", "Cosine Annealing", "Smooth cosine decay to minimum LR"),
    ("cosine_restarts", "Cosine with Restarts", "Warm restarts for exploration"),
    ("onecycle", "One Cycle", "Super-convergence with LR warmup and decay"),
    ("step", "Step LR", "Reduce LR by factor every N epochs"),
    ("multistep", "Multi-Step LR", "Reduce LR at specific milestones"),
    ("exponential", "Exponential", "Exponential decay of learning rate"),
    ("linear_warmup", "Linear Warmup", "Linear warmup then constant"),
]

# Mixed precision options
PRECISION_OPTIONS = [
    ("bf16", "BFloat16", "Best for Ampere+ GPUs (A100, H100)"),
    ("fp16", "Float16", "Compatible with older GPUs"),
    ("no", "Full Precision (FP32)", "Maximum accuracy, slower training"),
]

# Cache validation options
CACHE_VALIDATION_OPTIONS = [
    ("sha256", "SHA256 Hash", "Full file hash validation (most secure)"),
    ("fast", "Fast Hash", "Partial file hash (faster, less thorough)"),
    ("size", "Size Only", "Quick file size check (fastest)"),
]

# =============================================================================
# MODEL CATEGORIES FOR UI ORGANIZATION
# =============================================================================
MODEL_CATEGORIES = {
    "Basic CNN": ["cnn"],
    "ResNet": [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet18_pretrained",
        "resnet50_pretrained",
    ],
    "ResNet 3D": ["resnet3d_18", "mc3_18"],
    "TCN": ["tcn_small", "tcn", "tcn_large"],
    "EfficientNet": ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"],
    "EfficientNetV2": ["efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"],
    "MobileNetV3": ["mobilenet_v3_small", "mobilenet_v3_large"],
    "RegNet": [
        "regnet_y_400mf",
        "regnet_y_800mf",
        "regnet_y_1_6gf",
        "regnet_y_3_2gf",
        "regnet_y_8gf",
    ],
    "Swin Transformer": ["swin_t", "swin_s", "swin_b"],
    "Vision Transformer": ["vit_tiny", "vit_small", "vit_base"],
    "ConvNeXt": [
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_tiny_pretrained",
    ],
    "DenseNet": ["densenet121", "densenet169", "densenet121_pretrained"],
    "U-Net": ["unet_regression"],
}

# Model information (parameters, size, supported dimensions)
MODEL_INFO = {
    "cnn": {"params": "~500K", "size": "2 MB", "dims": ["1D", "2D", "3D"]},
    "resnet18": {"params": "~11M", "size": "45 MB", "dims": ["2D"]},
    "resnet18_pretrained": {"params": "~11M", "size": "45 MB", "dims": ["2D"]},
    "resnet34": {"params": "~21M", "size": "85 MB", "dims": ["2D"]},
    "resnet50": {"params": "~25M", "size": "100 MB", "dims": ["2D"]},
    "resnet50_pretrained": {"params": "~25M", "size": "100 MB", "dims": ["2D"]},
    "resnet3d_18": {"params": "~33M", "size": "130 MB", "dims": ["3D"]},
    "mc3_18": {"params": "~11M", "size": "45 MB", "dims": ["3D"]},
    "tcn_small": {"params": "~300K", "size": "1.2 MB", "dims": ["1D"]},
    "tcn": {"params": "~1M", "size": "4 MB", "dims": ["1D"]},
    "tcn_large": {"params": "~4M", "size": "16 MB", "dims": ["1D"]},
    "efficientnet_b0": {"params": "~5M", "size": "20 MB", "dims": ["2D"]},
    "efficientnet_b1": {"params": "~8M", "size": "32 MB", "dims": ["2D"]},
    "efficientnet_b2": {"params": "~9M", "size": "36 MB", "dims": ["2D"]},
    "efficientnet_v2_s": {"params": "~21M", "size": "85 MB", "dims": ["2D"]},
    "efficientnet_v2_m": {"params": "~54M", "size": "220 MB", "dims": ["2D"]},
    "efficientnet_v2_l": {"params": "~118M", "size": "480 MB", "dims": ["2D"]},
    "mobilenet_v3_small": {"params": "~2.5M", "size": "10 MB", "dims": ["2D"]},
    "mobilenet_v3_large": {"params": "~5.4M", "size": "22 MB", "dims": ["2D"]},
    "regnet_y_400mf": {"params": "~4M", "size": "16 MB", "dims": ["2D"]},
    "regnet_y_800mf": {"params": "~6M", "size": "24 MB", "dims": ["2D"]},
    "regnet_y_1_6gf": {"params": "~11M", "size": "45 MB", "dims": ["2D"]},
    "regnet_y_3_2gf": {"params": "~19M", "size": "76 MB", "dims": ["2D"]},
    "regnet_y_8gf": {"params": "~39M", "size": "156 MB", "dims": ["2D"]},
    "swin_t": {"params": "~28M", "size": "112 MB", "dims": ["2D"]},
    "swin_s": {"params": "~50M", "size": "200 MB", "dims": ["2D"]},
    "swin_b": {"params": "~88M", "size": "352 MB", "dims": ["2D"]},
    "vit_tiny": {"params": "~5.7M", "size": "23 MB", "dims": ["2D"]},
    "vit_small": {"params": "~22M", "size": "88 MB", "dims": ["2D"]},
    "vit_base": {"params": "~86M", "size": "344 MB", "dims": ["2D"]},
    "convnext_tiny": {"params": "~28M", "size": "112 MB", "dims": ["2D"]},
    "convnext_small": {"params": "~50M", "size": "200 MB", "dims": ["2D"]},
    "convnext_base": {"params": "~88M", "size": "352 MB", "dims": ["2D"]},
    "convnext_tiny_pretrained": {"params": "~28M", "size": "112 MB", "dims": ["2D"]},
    "densenet121": {"params": "~8M", "size": "32 MB", "dims": ["2D"]},
    "densenet169": {"params": "~14M", "size": "56 MB", "dims": ["2D"]},
    "densenet121_pretrained": {"params": "~8M", "size": "32 MB", "dims": ["2D"]},
    "unet_regression": {"params": "~31M", "size": "124 MB", "dims": ["2D"]},
}

# Flat list of all models
ALL_MODELS = [model for models in MODEL_CATEGORIES.values() for model in models]

# Models that support pretrained weights
PRETRAINED_MODELS = [m for m in ALL_MODELS if "pretrained" in m]

# =============================================================================
# TOOLTIPS FOR UI ELEMENTS
# =============================================================================
TOOLTIPS = {
    # Project Setup
    "data_file": (
        "Path to training data file.\n"
        "Supported formats: NPZ, MAT, HDF5\n"
        "Must contain 'X' (inputs) and 'Y' (targets) arrays."
    ),
    "output_dir": (
        "Directory where training outputs will be saved.\n"
        "Includes: checkpoints, training curves, logs."
    ),
    # Model Configuration
    "model": (
        "Neural network architecture to train.\n"
        "Choose based on your data dimensionality and size."
    ),
    "pretrained": (
        "Use weights pre-trained on ImageNet.\n"
        "Recommended for 2D image-like data with 3 channels.\n"
        "Transfer learning can significantly improve results."
    ),
    # Training Hyperparameters
    "batch_size": (
        "Number of samples per training batch.\n"
        "Larger batches = faster training but more GPU memory.\n"
        "Typical values: 32, 64, 128, 256"
    ),
    "learning_rate": (
        "Initial learning rate for optimization.\n"
        "Too high: unstable training. Too low: slow convergence.\n"
        "Typical range: 1e-4 to 1e-2"
    ),
    "epochs": (
        "Maximum number of training epochs.\n"
        "Training will stop early if validation loss doesn't improve."
    ),
    "patience": (
        "Early stopping patience.\n"
        "Stop training if no improvement for this many epochs."
    ),
    "weight_decay": (
        "L2 regularization strength.\nHelps prevent overfitting. Typical: 1e-4 to 1e-2"
    ),
    "grad_clip": (
        "Gradient clipping norm threshold.\nPrevents exploding gradients. 0 = disabled."
    ),
    "grad_accum_steps": (
        "Gradient accumulation steps.\n"
        "Simulates larger batch sizes with limited GPU memory.\n"
        "Effective batch = batch_size x steps x num_GPUs."
    ),
    "loss": ("Loss function for training.\nMSE is standard for regression."),
    "optimizer": ("Optimization algorithm.\nAdamW is recommended for most cases."),
    "scheduler": (
        "Learning rate scheduler.\nAdjusts LR during training for better convergence."
    ),
    "precision": (
        "Mixed precision training mode.\n"
        "BF16 is best for modern GPUs (Ampere+).\n"
        "FP16 for older GPUs, FP32 for maximum accuracy."
    ),
    "compile": (
        "Enable torch.compile() for faster training.\n"
        "Requires PyTorch 2.0+. First epoch may be slower."
    ),
    "deterministic": (
        "Enable deterministic mode for reproducibility.\n"
        "Makes training exactly reproducible but slower."
    ),
    "seed": (
        "Random seed for reproducibility.\nUse the same seed to reproduce results."
    ),
    # Cross-validation
    "cv": (
        "Enable K-fold cross-validation.\n0 = disabled. 5 or 10 are common choices."
    ),
    "cv_stratify": (
        "Use stratified splitting for cross-validation.\n"
        "Ensures each fold has similar target distributions."
    ),
    # Data
    "workers": (
        "Number of data loading workers.\n"
        "-1 = auto-detect based on CPU cores.\n"
        "More workers can speed up data loading."
    ),
    "cache_validate": (
        "Cache validation mode for data loading.\nSHA256: most secure. Size: fastest."
    ),
    "no_cache": (
        "Delete cached data before each training run.\n"
        "Forces data to be reloaded and preprocessed from scratch.\n"
        "Useful when the data file has changed."
    ),
}


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================
@dataclass
class TrainingConfig:
    """Configuration for a training run, matching wavedl CLI arguments."""

    # Data
    data_path: str = ""
    output_dir: str = ""

    # Model
    model: str = "cnn"
    pretrained: bool = True

    # Training hyperparameters
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LEARNING_RATE
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_clip: float = DEFAULT_GRAD_CLIP
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM_STEPS

    # Loss and optimizer
    loss: str = "mse"
    huber_delta: float = 1.0
    loss_weights: str = ""
    optimizer: str = "adamw"
    momentum: float = 0.9
    nesterov: bool = False
    betas: str = "0.9,0.999"

    # Scheduler
    scheduler: str = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    step_size: int = 30
    milestones: str = ""

    # Performance
    precision: str = "bf16"
    compile: bool = False
    deterministic: bool = False
    seed: int = DEFAULT_SEED

    # Data loading
    workers: int = -1
    cache_validate: str = "fast"
    single_channel: bool = False
    no_cache: bool = False

    # Cross-validation
    cv: int = 0
    cv_stratify: bool = False
    cv_bins: int = 10

    # Checkpointing
    resume: str = ""
    save_every: int = 50
    fresh: bool = False

    # Logging
    wandb: bool = False

    # HPC options
    num_gpus: int = 1
    mixed_precision: str = "bf16"

    def to_cli_args(self) -> list[str]:
        """Convert configuration to CLI arguments list."""
        args = []

        # Required arguments
        if self.data_path:
            args.extend(["--data_path", self.data_path])
        if self.output_dir:
            args.extend(["--output_dir", self.output_dir])

        # Model
        args.extend(["--model", self.model])
        if not self.pretrained:
            args.append("--no_pretrained")

        # Training hyperparameters
        args.extend(["--batch_size", str(self.batch_size)])
        args.extend(["--lr", str(self.lr)])
        args.extend(["--epochs", str(self.epochs)])
        args.extend(["--patience", str(self.patience)])
        args.extend(["--weight_decay", str(self.weight_decay)])
        args.extend(["--grad_clip", str(self.grad_clip)])
        if self.grad_accum_steps > 1:
            args.extend(["--grad_accum_steps", str(self.grad_accum_steps)])

        # Loss
        args.extend(["--loss", self.loss])
        if self.loss == "huber":
            args.extend(["--huber_delta", str(self.huber_delta)])
        if self.loss == "weighted_mse" and self.loss_weights:
            args.extend(["--loss_weights", self.loss_weights])

        # Optimizer
        args.extend(["--optimizer", self.optimizer])
        if self.optimizer in ("sgd", "rmsprop"):
            args.extend(["--momentum", str(self.momentum)])
        if self.optimizer == "sgd" and self.nesterov:
            args.append("--nesterov")
        if self.optimizer in ("adam", "adamw", "nadam", "radam"):
            args.extend(["--betas", self.betas])

        # Scheduler
        args.extend(["--scheduler", self.scheduler])
        if self.scheduler == "plateau":
            args.extend(["--scheduler_patience", str(self.scheduler_patience)])
            args.extend(["--scheduler_factor", str(self.scheduler_factor)])
        args.extend(["--min_lr", str(self.min_lr)])
        if self.scheduler == "linear_warmup":
            args.extend(["--warmup_epochs", str(self.warmup_epochs)])
        if self.scheduler == "step":
            args.extend(["--step_size", str(self.step_size)])
        if self.scheduler == "multistep" and self.milestones:
            args.extend(["--milestones", self.milestones])

        # Performance
        args.extend(["--precision", self.precision])

        if self.compile:
            args.append("--compile")
        if self.deterministic:
            args.append("--deterministic")
        args.extend(["--seed", str(self.seed)])

        # Data loading
        args.extend(["--workers", str(self.workers)])
        args.extend(["--cache_validate", self.cache_validate])
        if self.single_channel:
            args.append("--single_channel")

        # Cross-validation
        if self.cv > 0:
            args.extend(["--cv", str(self.cv)])
            if self.cv_stratify:
                args.append("--cv_stratify")
            args.extend(["--cv_bins", str(self.cv_bins)])

        # Checkpointing
        if self.resume:
            args.extend(["--resume", self.resume])
        args.extend(["--save_every", str(self.save_every)])
        if self.fresh:
            args.append("--fresh")

        # Logging
        if self.wandb:
            args.append("--wandb")

        return args

    def to_command(self) -> str:
        """Generate full command string."""
        cmd = ["wavedl-train"]
        cmd.extend(self.to_cli_args())
        return " ".join(cmd)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML export."""
        return {
            "model": self.model,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "patience": self.patience,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "grad_accum_steps": self.grad_accum_steps,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "precision": self.precision,
            "compile": self.compile,
            "deterministic": self.deterministic,
            "seed": self.seed,
            "workers": self.workers,
            "cache_validate": self.cache_validate,
            "cv": self.cv,
            "cv_stratify": self.cv_stratify,
            "cv_bins": self.cv_bins,
            "save_every": self.save_every,
        }


@dataclass
class TestConfig:
    """Configuration for testing/inference, matching wavedl-test CLI arguments."""

    checkpoint: str = ""
    data_path: str = ""
    model: str = ""
    format: str = "auto"
    input_key: str = ""
    output_key: str = ""
    param_names: list[str] = field(default_factory=list)
    batch_size: int = 128
    workers: int = 0
    output_dir: str = ""
    save_predictions: bool = True
    plot: bool = True
    plot_format: list[str] = field(default_factory=lambda: ["png"])
    export_format: str = ""
    export_path: str = ""

    def to_cli_args(self) -> list[str]:
        """Convert configuration to CLI arguments list."""
        args = []

        # Required
        if self.checkpoint:
            args.extend(["--checkpoint", self.checkpoint])
        if self.data_path:
            args.extend(["--data_path", self.data_path])

        # Optional model specification
        if self.model:
            args.extend(["--model", self.model])

        # Data format
        if self.format != "auto":
            args.extend(["--format", self.format])
        if self.input_key:
            args.extend(["--input_key", self.input_key])
        if self.output_key:
            args.extend(["--output_key", self.output_key])
        if self.param_names:
            args.extend(["--param_names"] + self.param_names)

        # Inference options
        args.extend(["--batch_size", str(self.batch_size)])
        args.extend(["--workers", str(self.workers)])

        # Output options
        if self.output_dir:
            args.extend(["--output_dir", self.output_dir])
        if self.save_predictions:
            args.append("--save_predictions")
        if self.plot:
            args.append("--plot")
            if self.plot_format:
                args.extend(["--plot_format"] + self.plot_format)

        # Export
        if self.export_format:
            args.extend(["--export", self.export_format])
        if self.export_path:
            args.extend(["--export_path", self.export_path])

        return args

    def to_command(self) -> str:
        """Generate full command string."""
        cmd = ["wavedl-test"]
        cmd.extend(self.to_cli_args())
        return " ".join(cmd)
