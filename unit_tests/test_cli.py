"""
Unit Tests for CLI Entry Points
===============================

Consolidated tests for all CLI entry points in WaveDL:
    - wavedl-train (wavedl.train)
    - wavedl-test (wavedl.test)
    - wavedl-hpo (wavedl.hpo)
    - wavedl-hpc (wavedl.hpc)

**Tested Components**:
    - Argument parsing and validation
    - Configuration defaults
    - Helper functions (metrics, GPU detection, environment setup)

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn


# ==============================================================================
# TRAINING MODULE TESTS (wavedl.train)
# ==============================================================================
class TestTrainParseArgs:
    """Tests for training argument parsing."""

    def test_default_values(self):
        """Test default argument values."""
        from wavedl.train import parse_args

        with patch.object(
            sys,
            "argv",
            ["wavedl-train", "--model", "cnn", "--data_path", "/fake/path.npz"],
        ):
            args, _parser = parse_args()

            assert args.model == "cnn"
            assert args.data_path == "/fake/path.npz"
            assert args.epochs == 1000
            assert args.batch_size == 128
            assert args.lr == 1e-3
            assert args.optimizer == "adamw"
            assert args.scheduler == "plateau"

    def test_model_argument(self):
        """Test that model argument is parsed correctly."""
        from wavedl.train import parse_args

        with patch.object(
            sys,
            "argv",
            ["wavedl-train", "--model", "resnet18", "--data_path", "/fake/path.npz"],
        ):
            args, _ = parse_args()
            assert args.model == "resnet18"

    def test_hyperparameter_arguments(self):
        """Test hyperparameter argument parsing."""
        from wavedl.train import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "wavedl-train",
                "--model",
                "cnn",
                "--data_path",
                "/fake/path.npz",
                "--epochs",
                "200",
                "--batch_size",
                "64",
                "--lr",
                "0.01",
                "--weight_decay",
                "0.05",
                "--patience",
                "15",
            ],
        ):
            args, _ = parse_args()
            assert args.epochs == 200
            assert args.batch_size == 64
            assert args.lr == 0.01
            assert args.weight_decay == 0.05
            assert args.patience == 15

    def test_optimizer_choices(self):
        """Test optimizer argument parsing."""
        from wavedl.train import parse_args

        for optimizer in ["adamw", "adam", "sgd", "nadam", "radam", "rmsprop"]:
            with patch.object(
                sys,
                "argv",
                [
                    "wavedl-train",
                    "--model",
                    "cnn",
                    "--data_path",
                    "/fake/path.npz",
                    "--optimizer",
                    optimizer,
                ],
            ):
                args, _ = parse_args()
                assert args.optimizer == optimizer

    def test_scheduler_choices(self):
        """Test scheduler argument parsing."""
        from wavedl.train import parse_args

        for scheduler in ["plateau", "cosine", "step", "onecycle"]:
            with patch.object(
                sys,
                "argv",
                [
                    "wavedl-train",
                    "--model",
                    "cnn",
                    "--data_path",
                    "/fake/path.npz",
                    "--scheduler",
                    scheduler,
                ],
            ):
                args, _ = parse_args()
                assert args.scheduler == scheduler

    def test_loss_choices(self):
        """Test loss function argument parsing."""
        from wavedl.train import parse_args

        for loss in ["mse", "mae", "huber", "smooth_l1"]:
            with patch.object(
                sys,
                "argv",
                [
                    "wavedl-train",
                    "--model",
                    "cnn",
                    "--data_path",
                    "/fake/path.npz",
                    "--loss",
                    loss,
                ],
            ):
                args, _ = parse_args()
                assert args.loss == loss

    def test_precision_choices(self):
        """Test precision argument parsing."""
        from wavedl.train import parse_args

        for precision in ["bf16", "fp16", "no"]:
            with patch.object(
                sys,
                "argv",
                [
                    "wavedl-train",
                    "--model",
                    "cnn",
                    "--data_path",
                    "/fake/path.npz",
                    "--precision",
                    precision,
                ],
            ):
                args, _ = parse_args()
                assert args.precision == precision

    def test_compile_flag(self):
        """Test compile flag parsing."""
        from wavedl.train import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "wavedl-train",
                "--model",
                "cnn",
                "--data_path",
                "/fake/path.npz",
                "--compile",
            ],
        ):
            args, _ = parse_args()
            assert args.compile is True

    def test_list_models_flag(self):
        """Test that --list_models flag is recognized."""
        from wavedl.train import parse_args

        with patch.object(sys, "argv", ["wavedl-train", "--list_models"]):
            args, _ = parse_args()
            assert args.list_models is True

    def test_output_and_seed_arguments(self):
        """Test output directory and seed argument parsing."""
        from wavedl.train import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "wavedl-train",
                "--model",
                "cnn",
                "--data_path",
                "/fake/path.npz",
                "--output_dir",
                "/custom/output",
                "--seed",
                "42",
                "--workers",
                "8",
            ],
        ):
            args, _ = parse_args()
            assert args.output_dir == "/custom/output"
            assert args.seed == 42
            assert args.workers == 8


# ==============================================================================
# INFERENCE MODULE TESTS (wavedl.test)
# ==============================================================================
class TestComputeMetrics:
    """Tests for regression metrics computation."""

    def test_basic_metrics(self):
        """Test that basic metrics are computed correctly."""
        from wavedl.test import compute_metrics

        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])

        metrics = compute_metrics(y_true, y_pred)

        assert "r2_score" in metrics
        assert "pearson_corr" in metrics
        assert "mae_avg" in metrics
        assert "rmse" in metrics

    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        from wavedl.test import compute_metrics

        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = y_true.copy()

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["r2_score"] == pytest.approx(1.0)
        assert metrics["mae_avg"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)

    def test_per_parameter_mae(self):
        """Test that per-parameter MAE is computed."""
        from wavedl.test import compute_metrics

        y_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_pred = np.array([[1.1, 2.2, 3.3], [4.1, 5.2, 6.3]])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mae_p0"] == pytest.approx(0.1, abs=1e-6)
        assert metrics["mae_p1"] == pytest.approx(0.2, abs=1e-6)
        assert metrics["mae_p2"] == pytest.approx(0.3, abs=1e-6)

    def test_single_sample_handling(self):
        """Test that single sample case is handled gracefully."""
        from wavedl.test import compute_metrics

        y_true = np.array([[1.0, 2.0]])
        y_pred = np.array([[1.5, 2.5]])

        metrics = compute_metrics(y_true, y_pred)

        assert np.isnan(metrics["r2_score"])
        assert np.isnan(metrics["pearson_corr"])
        assert metrics["mae_avg"] == pytest.approx(0.5)


class TestModelWithDenormalization:
    """Tests for the ONNX denormalization wrapper."""

    def test_wraps_model(self):
        """Test that wrapper correctly wraps a model."""
        from wavedl.test import ModelWithDenormalization

        base_model = nn.Linear(10, 3)
        scaler_mean = np.array([0.0, 1.0, 2.0])
        scaler_scale = np.array([1.0, 2.0, 3.0])

        wrapper = ModelWithDenormalization(base_model, scaler_mean, scaler_scale)

        assert hasattr(wrapper, "model")
        assert hasattr(wrapper, "scaler_mean")
        assert hasattr(wrapper, "scaler_scale")

    def test_denormalization_output(self):
        """Test that forward pass applies denormalization correctly."""
        from wavedl.test import ModelWithDenormalization

        class ConstantModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0), 3)

        wrapper = ModelWithDenormalization(
            ConstantModel(),
            np.array([10.0, 20.0, 30.0]),
            np.array([1.0, 1.0, 1.0]),
        )

        output = wrapper(torch.randn(2, 10))
        expected = torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        assert torch.allclose(output, expected)

    def test_buffers_are_registered(self):
        """Test that scaler values are registered as buffers."""
        from wavedl.test import ModelWithDenormalization

        wrapper = ModelWithDenormalization(
            nn.Linear(10, 3),
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        )

        buffer_names = [name for name, _ in wrapper.named_buffers()]
        assert "scaler_mean" in buffer_names
        assert "scaler_scale" in buffer_names


# ==============================================================================
# HPO MODULE TESTS (wavedl.hpo - requires optuna)
# ==============================================================================
try:
    import optuna  # noqa: F401

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
class TestHPOConstants:
    """Tests for HPO default configuration constants."""

    @pytest.fixture(autouse=True)
    def import_hpo(self):
        """Import HPO module for tests."""
        from wavedl.hpo import (
            DEFAULT_LOSSES,
            DEFAULT_MODELS,
            DEFAULT_OPTIMIZERS,
            DEFAULT_SCHEDULERS,
            QUICK_LOSSES,
            QUICK_MODELS,
            QUICK_OPTIMIZERS,
            QUICK_SCHEDULERS,
        )

        self.DEFAULT_MODELS = DEFAULT_MODELS
        self.DEFAULT_OPTIMIZERS = DEFAULT_OPTIMIZERS
        self.DEFAULT_SCHEDULERS = DEFAULT_SCHEDULERS
        self.DEFAULT_LOSSES = DEFAULT_LOSSES
        self.QUICK_MODELS = QUICK_MODELS
        self.QUICK_OPTIMIZERS = QUICK_OPTIMIZERS
        self.QUICK_SCHEDULERS = QUICK_SCHEDULERS
        self.QUICK_LOSSES = QUICK_LOSSES

    def test_default_lists_not_empty(self):
        """Test that all default lists are populated."""
        assert len(self.DEFAULT_MODELS) > 0
        assert len(self.DEFAULT_OPTIMIZERS) > 0
        assert len(self.DEFAULT_SCHEDULERS) > 0
        assert len(self.DEFAULT_LOSSES) > 0

    def test_quick_lists_are_subsets(self):
        """Test that quick lists are subsets of default lists."""
        for opt in self.QUICK_OPTIMIZERS:
            assert opt in self.DEFAULT_OPTIMIZERS
        for sched in self.QUICK_SCHEDULERS:
            assert sched in self.DEFAULT_SCHEDULERS
        for loss in self.QUICK_LOSSES:
            assert loss in self.DEFAULT_LOSSES


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
class TestHPOObjective:
    """Tests for Optuna objective function creation."""

    def test_returns_callable(self):
        """Test that create_objective returns a callable."""
        from wavedl.hpo import create_objective

        args = MagicMock()
        args.data_path = "/fake/path.npz"
        args.max_epochs = 10
        args.quick = False
        args.seed = 2025
        args.timeout = 3600
        args.models = None
        args.optimizers = None
        args.schedulers = None
        args.losses = None

        objective = create_objective(args)
        assert callable(objective)


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
class TestHPOIntegration:
    """Integration tests for HPO configuration against registries."""

    def test_all_default_optimizers_are_valid(self):
        """Test that all default optimizers exist in registry."""
        from wavedl.hpo import DEFAULT_OPTIMIZERS
        from wavedl.utils import list_optimizers

        available = list_optimizers()
        for opt in DEFAULT_OPTIMIZERS:
            assert opt in available

    def test_all_default_schedulers_are_valid(self):
        """Test that all default schedulers exist in registry."""
        from wavedl.hpo import DEFAULT_SCHEDULERS
        from wavedl.utils import list_schedulers

        available = list_schedulers()
        for sched in DEFAULT_SCHEDULERS:
            assert sched in available

    def test_all_default_losses_are_valid(self):
        """Test that all default losses exist in registry."""
        from wavedl.hpo import DEFAULT_LOSSES
        from wavedl.utils import list_losses

        available = list_losses()
        for loss in DEFAULT_LOSSES:
            assert loss in available


# ==============================================================================
# HPC MODULE TESTS (wavedl.hpc)
# ==============================================================================
class TestHPCDetectGPUs:
    """Tests for GPU auto-detection functionality."""

    def test_no_nvidia_smi(self):
        """Test fallback when nvidia-smi is not available."""
        from wavedl.hpc import detect_gpus

        with patch("shutil.which", return_value=None):
            assert detect_gpus() == 1

    def test_nvidia_smi_success(self):
        """Test successful GPU detection."""
        from wavedl.hpc import detect_gpus

        mock_result = MagicMock()
        mock_result.stdout = "GPU 0: A100\nGPU 1: A100\nGPU 2: A100\n"

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_result),
        ):
            assert detect_gpus() == 3

    def test_nvidia_smi_failure(self):
        """Test fallback when nvidia-smi fails."""
        import subprocess

        from wavedl.hpc import detect_gpus

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
            ),
        ):
            assert detect_gpus() == 1


class TestHPCEnvironment:
    """Tests for HPC environment configuration."""

    def test_sets_default_env_vars(self):
        """Test that default environment variables are set when home is not writable."""
        from wavedl.hpc import setup_hpc_environment

        env_vars = ["MPLCONFIGDIR", "WANDB_MODE", "WANDB_DIR"]
        original = {v: os.environ.pop(v, None) for v in env_vars}

        try:
            # Mock home directory as non-writable (simulates HPC environment)
            with patch("os.access", return_value=False):
                setup_hpc_environment()
                assert "MPLCONFIGDIR" in os.environ
                assert os.environ["WANDB_MODE"] == "offline"
        finally:
            for var, val in original.items():
                if val is not None:
                    os.environ[var] = val
                elif var in os.environ:
                    del os.environ[var]


class TestHPCParseArgs:
    """Tests for HPC argument parsing."""

    def test_default_values(self):
        """Test default argument values."""
        from wavedl.hpc import parse_args

        with patch.object(sys, "argv", ["wavedl-hpc"]):
            args, remaining = parse_args()

            assert args.num_gpus is None
            assert args.num_machines == 1
            assert args.mixed_precision == "bf16"
            assert remaining == []

    def test_passthrough_args(self):
        """Test that unknown args are passed through to train.py."""
        from wavedl.hpc import parse_args

        with patch.object(
            sys,
            "argv",
            ["wavedl-hpc", "--num_gpus", "2", "--model", "cnn", "--epochs", "100"],
        ):
            args, remaining = parse_args()

            assert args.num_gpus == 2
            assert "--model" in remaining
            assert "cnn" in remaining


class TestHPCPrintSummary:
    """Tests for training summary output."""

    def test_success_message(self, capsys):
        """Test success summary output."""
        from wavedl.hpc import print_summary

        print_summary(
            exit_code=0,
            wandb_enabled=True,
            wandb_mode="offline",
            wandb_dir="/tmp/wandb",
        )

        captured = capsys.readouterr()
        assert "Training completed successfully" in captured.out
        assert "wandb sync" in captured.out

    def test_failure_message(self, capsys):
        """Test failure summary output."""
        from wavedl.hpc import print_summary

        print_summary(
            exit_code=1,
            wandb_enabled=True,
            wandb_mode="offline",
            wandb_dir="/tmp/wandb",
        )

        captured = capsys.readouterr()
        assert "Training failed" in captured.out
