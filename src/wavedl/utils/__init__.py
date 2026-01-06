"""
Utility Functions and Classes
=============================

Centralized exports for all utility modules.

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from .config import (
    create_default_config,
    load_config,
    merge_config_with_args,
    save_config,
    validate_config,
)
from .constraints import (
    ExpressionConstraint,
    FileConstraint,
    PhysicsConstrainedLoss,
    build_constraints,
)
from .cross_validation import (
    CVDataset,
    run_cross_validation,
    train_fold,
)
from .data import (
    # Multi-format data loading
    DataSource,
    HDF5Source,
    MATSource,
    MemmapDataset,
    NPZSource,
    get_data_source,
    load_outputs_only,
    load_test_data,
    load_training_data,
    memmap_worker_init_fn,
    prepare_data,
)
from .distributed import (
    broadcast_early_stop,
    broadcast_value,
    sync_tensor,
)
from .losses import (
    LogCoshLoss,
    WeightedMSELoss,
    get_loss,
    list_losses,
)
from .metrics import (
    COLORS,
    FIGURE_DPI,
    FIGURE_WIDTH_CM,
    # Style constants
    FIGURE_WIDTH_INCH,
    FONT_SIZE_TEXT,
    FONT_SIZE_TICKS,
    MetricTracker,
    calc_pearson,
    calc_per_target_r2,
    configure_matplotlib_style,
    create_training_curves,
    get_lr,
    plot_bland_altman,
    plot_correlation_heatmap,
    plot_error_boxplot,
    plot_error_cdf,
    plot_error_histogram,
    plot_prediction_vs_index,
    plot_qq,
    plot_relative_error,
    plot_residuals,
    plot_scientific_scatter,
)
from .optimizers import (
    get_optimizer,
    get_optimizer_with_param_groups,
    list_optimizers,
)
from .schedulers import (
    get_scheduler,
    get_scheduler_with_warmup,
    is_epoch_based,
    list_schedulers,
)


__all__ = [
    "COLORS",
    "FIGURE_DPI",
    "FIGURE_WIDTH_CM",
    # Style constants
    "FIGURE_WIDTH_INCH",
    "FONT_SIZE_TEXT",
    "FONT_SIZE_TICKS",
    # Constraints
    "CVDataset",
    "DataSource",
    "ExpressionConstraint",
    "FileConstraint",
    "HDF5Source",
    "LogCoshLoss",
    "MATSource",
    # Data
    "MemmapDataset",
    # Metrics
    "MetricTracker",
    "NPZSource",
    "PhysicsConstrainedLoss",
    "WeightedMSELoss",
    # Distributed
    "broadcast_early_stop",
    "broadcast_value",
    "build_constraints",
    "calc_pearson",
    "calc_per_target_r2",
    "configure_matplotlib_style",
    "create_default_config",
    "create_training_curves",
    "get_data_source",
    # Losses
    "get_loss",
    "get_lr",
    # Optimizers
    "get_optimizer",
    "get_optimizer_with_param_groups",
    # Schedulers
    "get_scheduler",
    "get_scheduler_with_warmup",
    "is_epoch_based",
    "list_losses",
    "list_optimizers",
    "list_schedulers",
    # Config
    "load_config",
    "load_outputs_only",
    "load_test_data",
    "load_training_data",
    "memmap_worker_init_fn",
    "merge_config_with_args",
    "plot_bland_altman",
    "plot_correlation_heatmap",
    "plot_error_boxplot",
    "plot_error_cdf",
    "plot_error_histogram",
    "plot_prediction_vs_index",
    "plot_qq",
    "plot_relative_error",
    "plot_residuals",
    "plot_scientific_scatter",
    "prepare_data",
    # Cross-Validation
    "run_cross_validation",
    "save_config",
    "sync_tensor",
    "train_fold",
    "validate_config",
]
