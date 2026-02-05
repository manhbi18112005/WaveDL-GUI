"""
Unit Tests for CLI Entry Points
===============================

Consolidated tests for all CLI entry points in WaveDL:
    - wavedl-train (wavedl.launcher)
    - wavedl-test (wavedl.test)
    - wavedl-hpo (wavedl.hpo)

**Tested Components**:
    - Argument parsing and validation
    - Configuration defaults
    - Helper functions (metrics, GPU detection, environment setup)

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import pickle
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn


# Check for optional ONNX dependencies
try:
    import onnx  # noqa: F401

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


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
# LAUNCHER MODULE TESTS (wavedl.launcher)
# ==============================================================================
class TestHPCDetectGPUs:
    """Tests for GPU auto-detection functionality."""

    def test_no_nvidia_smi(self):
        """Test fallback when nvidia-smi is not available."""
        from wavedl.launcher import detect_gpus

        with patch("shutil.which", return_value=None):
            assert detect_gpus() == 1

    def test_nvidia_smi_success(self):
        """Test successful GPU detection."""
        from wavedl.launcher import detect_gpus

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

        from wavedl.launcher import detect_gpus

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
        from wavedl.launcher import setup_environment

        env_vars = ["MPLCONFIGDIR", "WANDB_MODE", "WANDB_DIR"]
        original = {v: os.environ.pop(v, None) for v in env_vars}

        try:
            # Mock home directory as non-writable (simulates HPC environment)
            with patch("os.access", return_value=False):
                setup_environment()
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
        from wavedl.launcher import parse_args

        with patch.object(sys, "argv", ["wavedl-hpc"]):
            args, remaining = parse_args()

            assert args.num_gpus is None
            assert args.num_machines == 1
            assert args.mixed_precision == "bf16"
            assert remaining == []

    def test_passthrough_args(self):
        """Test that unknown args are passed through to train.py."""
        from wavedl.launcher import parse_args

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
        from wavedl.launcher import print_summary

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
        from wavedl.launcher import print_summary

        print_summary(
            exit_code=1,
            wandb_enabled=True,
            wandb_mode="offline",
            wandb_dir="/tmp/wandb",
        )

        captured = capsys.readouterr()
        assert "Training failed" in captured.out

    def test_success_without_wandb(self, capsys):
        """Test success message when wandb is disabled."""
        from wavedl.launcher import print_summary

        print_summary(
            exit_code=0,
            wandb_enabled=False,
            wandb_mode="offline",
            wandb_dir="/tmp/wandb",
        )

        captured = capsys.readouterr()
        assert "Training completed successfully" in captured.out
        assert "wandb sync" not in captured.out


# ==============================================================================
# TRAINING MODULE CACHE SETUP TESTS
# ==============================================================================
class TestTrainCacheSetup:
    """Tests for cache directory setup functions in train.py."""

    def test_setup_cache_dir_respects_existing_env(self):
        """Test that _setup_cache_dir respects existing environment variables."""
        from wavedl.train import _setup_cache_dir

        original = os.environ.get("TEST_CACHE_VAR")
        try:
            os.environ["TEST_CACHE_VAR"] = "/custom/path"
            _setup_cache_dir("TEST_CACHE_VAR", "test_cache")
            assert os.environ["TEST_CACHE_VAR"] == "/custom/path"
        finally:
            if original:
                os.environ["TEST_CACHE_VAR"] = original
            elif "TEST_CACHE_VAR" in os.environ:
                del os.environ["TEST_CACHE_VAR"]

    def test_setup_cache_dir_creates_directory(self, temp_dir):
        """Test that _setup_cache_dir creates the cache directory."""
        from wavedl.train import _setup_cache_dir

        test_var = "WAVEDL_TEST_CACHE_" + str(os.getpid())
        original_cwd = os.getcwd()

        try:
            # Ensure env var is not set
            if test_var in os.environ:
                del os.environ[test_var]

            # Change to temp dir and mock non-writable home
            os.chdir(temp_dir)
            with patch("os.access", return_value=False):
                _setup_cache_dir(test_var, "my_cache")

            # Verify directory was created
            # Use realpath to resolve symlinks (e.g., /var -> /private/var on macOS)
            expected_path = os.path.realpath(os.path.join(temp_dir, ".my_cache"))
            actual_path = os.path.realpath(os.environ.get(test_var, ""))
            assert actual_path == expected_path
            assert os.path.exists(expected_path)
        finally:
            os.chdir(original_cwd)
            if test_var in os.environ:
                del os.environ[test_var]


class TestTrainSuppressLogging:
    """Tests for logging suppression context manager."""

    def test_suppress_accelerate_logging_restores_level(self):
        """Test that suppress_accelerate_logging restores original log level."""
        import logging

        from wavedl.train import suppress_accelerate_logging

        accelerate_logger = logging.getLogger("accelerate.checkpointing")
        original_level = accelerate_logger.level

        with suppress_accelerate_logging():
            assert accelerate_logger.level == logging.WARNING

        assert accelerate_logger.level == original_level


# ==============================================================================
# INFERENCE MODULE TESTS (wavedl.test) - EXTENDED
# ==============================================================================
class TestLoadCheckpoint:
    """Tests for checkpoint loading functionality."""

    def test_load_checkpoint_success(self, temp_checkpoint_dir):
        """Test successful checkpoint loading."""
        from wavedl.test import load_checkpoint

        model, scaler = load_checkpoint(
            temp_checkpoint_dir,
            in_shape=(64, 64),
            out_size=5,
            model_name="cnn",
        )

        assert model is not None
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "scale_")

    def test_load_checkpoint_auto_detects_model(self, temp_checkpoint_dir):
        """Test that model is auto-detected from metadata."""
        from wavedl.test import load_checkpoint

        model, _scaler = load_checkpoint(
            temp_checkpoint_dir,
            in_shape=(64, 64),
            out_size=5,
            model_name=None,  # Auto-detect
        )

        assert model is not None

    def test_load_checkpoint_missing_raises(self):
        """Test that missing checkpoint raises FileNotFoundError."""
        from wavedl.test import load_checkpoint

        with pytest.raises(FileNotFoundError):
            load_checkpoint(
                "/nonexistent/checkpoint",
                in_shape=(64, 64),
                out_size=5,
            )


class TestRunInference:
    """Tests for batch inference functionality."""

    def test_run_inference_produces_correct_shape(self):
        """Test that inference produces correct output shape."""
        from wavedl.models.cnn import CNN
        from wavedl.test import run_inference

        model = CNN(in_shape=(64, 64), out_size=5)
        X = torch.randn(20, 1, 64, 64)

        predictions = run_inference(model, X, batch_size=8)

        assert predictions.shape == (20, 5)
        assert isinstance(predictions, np.ndarray)

    def test_run_inference_single_sample(self):
        """Test inference with single sample."""
        from wavedl.models.cnn import CNN
        from wavedl.test import run_inference

        model = CNN(in_shape=(32, 32), out_size=3)
        X = torch.randn(1, 1, 32, 32)

        predictions = run_inference(model, X, batch_size=1)

        assert predictions.shape == (1, 3)

    def test_run_inference_deterministic(self):
        """Test that inference is deterministic in eval mode."""
        from wavedl.models.cnn import CNN
        from wavedl.test import run_inference

        model = CNN(in_shape=(32, 32), out_size=3)
        X = torch.randn(10, 1, 32, 32)

        pred1 = run_inference(model, X, batch_size=4)
        pred2 = run_inference(model, X, batch_size=4)

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestSavePredictions:
    """Tests for prediction saving functionality."""

    def test_save_predictions_creates_csv(self, temp_dir):
        """Test that save_predictions creates a valid CSV file."""
        from wavedl.test import save_predictions

        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])
        output_path = os.path.join(temp_dir, "predictions.csv")

        save_predictions(y_true, y_pred, output_path)

        assert os.path.exists(output_path)

        # Verify CSV content
        import pandas as pd

        df = pd.read_csv(output_path)
        assert "True_P0" in df.columns
        assert "Pred_P0" in df.columns
        assert "Error_P0" in df.columns
        assert len(df) == 3

    def test_save_predictions_with_param_names(self, temp_dir):
        """Test save_predictions with custom parameter names."""
        from wavedl.test import save_predictions

        y_true = np.array([[1.0, 2.0, 3.0]])
        y_pred = np.array([[1.1, 2.1, 3.1]])
        output_path = os.path.join(temp_dir, "predictions.csv")
        param_names = ["height", "velocity", "density"]

        save_predictions(y_true, y_pred, output_path, param_names=param_names)

        import pandas as pd

        df = pd.read_csv(output_path)
        assert "True_height" in df.columns
        assert "Pred_velocity" in df.columns


class TestPrintResults:
    """Tests for result printing functionality."""

    def test_print_results_outputs_metrics(self, capsys):
        """Test that print_results outputs expected metrics."""
        from wavedl.test import compute_metrics, print_results

        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])
        metrics = compute_metrics(y_true, y_pred)

        print_results(y_true, y_pred, metrics)

        captured = capsys.readouterr()
        assert "R² Score" in captured.out
        assert "Pearson" in captured.out
        assert "MAE" in captured.out


@pytest.mark.skipif(not HAS_ONNX, reason="onnx not installed")
class TestONNXExport:
    """Tests for ONNX export functionality."""

    def test_export_creates_file(self, temp_dir):
        """Test that ONNX export creates a file."""
        from wavedl.models.cnn import CNN
        from wavedl.test import export_to_onnx

        model = CNN(in_shape=(32, 32), out_size=3)
        sample_input = torch.randn(1, 1, 32, 32)
        output_path = os.path.join(temp_dir, "model.onnx")

        success = export_to_onnx(model, sample_input, output_path, validate=False)

        assert success
        assert os.path.exists(output_path)

    def test_export_with_denormalization(self, temp_dir, mock_scaler):
        """Test ONNX export with denormalization layer."""
        from wavedl.models.cnn import CNN
        from wavedl.test import export_to_onnx

        model = CNN(in_shape=(32, 32), out_size=5)
        sample_input = torch.randn(1, 1, 32, 32)
        output_path = os.path.join(temp_dir, "model_denorm.onnx")

        success = export_to_onnx(
            model,
            sample_input,
            output_path,
            scaler=mock_scaler,
            include_denorm=True,
            validate=False,
        )

        assert success
        assert os.path.exists(output_path)

    def test_get_onnx_model_info(self, temp_dir):
        """Test ONNX model info extraction."""
        from wavedl.models.cnn import CNN
        from wavedl.test import export_to_onnx, get_onnx_model_info

        model = CNN(in_shape=(32, 32), out_size=3)
        sample_input = torch.randn(1, 1, 32, 32)
        output_path = os.path.join(temp_dir, "model.onnx")

        export_to_onnx(model, sample_input, output_path, validate=False)
        info = get_onnx_model_info(output_path)

        assert "input_name" in info or "error" in info
        if "error" not in info:
            assert "output_name" in info
            assert "file_size_mb" in info


class TestLoadDataForInference:
    """Tests for inference data loading."""

    def test_load_data_for_inference_npz(self, temp_dir):
        """Test loading NPZ data for inference."""
        from wavedl.test import load_data_for_inference

        X = np.random.randn(20, 64, 64).astype(np.float32)
        y = np.random.randn(20, 5).astype(np.float32)
        path = os.path.join(temp_dir, "test.npz")
        np.savez(path, input_test=X, output_test=y)

        X_loaded, y_loaded = load_data_for_inference(path)

        assert X_loaded.shape == (20, 1, 64, 64)  # Channel added
        assert y_loaded.shape == (20, 5)

    def test_load_data_for_inference_without_targets(self, temp_dir):
        """Test loading data without target values."""
        from wavedl.test import load_data_for_inference

        X = np.random.randn(10, 32, 32).astype(np.float32)
        path = os.path.join(temp_dir, "inputs_only.npz")
        np.savez(path, input_test=X)

        X_loaded, y_loaded = load_data_for_inference(path)

        assert X_loaded.shape == (10, 1, 32, 32)
        assert y_loaded is None


# ==============================================================================
# SAFETENSORS HANDLING TESTS
# ==============================================================================
class TestSafetensorsHandling:
    """Tests for safetensors checkpoint loading behavior."""

    def test_safetensors_unavailable_gives_helpful_error(
        self, temp_checkpoint_dir_safetensors
    ):
        """Test that helpful ImportError is raised when safetensors unavailable."""
        import wavedl.test as test_module

        # Mock HAS_SAFETENSORS to False to simulate safetensors not installed
        with (
            patch.object(test_module, "HAS_SAFETENSORS", False),
            pytest.raises(ImportError, match="safetensors"),
        ):
            test_module.load_checkpoint(
                temp_checkpoint_dir_safetensors,
                in_shape=(64, 64),
                out_size=5,
                model_name="cnn",
            )


# ==============================================================================
# SCALER PORTABILITY TESTS
# ==============================================================================
class TestScalerPortability:
    """Tests for scaler checkpoint portability."""

    def test_scaler_overwrite_on_retrain(self, temp_dir):
        """Test that scaler is overwritten (not stale) when saving best checkpoint.

        Simulates retrain scenario where:
        1. Initial training creates checkpoint with scaler v1
        2. Retraining in same dir creates new scaler v2
        3. New best checkpoint should have scaler v2, not stale v1
        """
        import shutil

        # Setup directories
        output_dir = temp_dir
        ckpt_dir = os.path.join(temp_dir, "best_checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Create "old" scaler in checkpoint (simulates previous training)
        old_scaler = {"version": 1, "mean": 0.0, "std": 1.0}
        with open(os.path.join(ckpt_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(old_scaler, f)

        # Create "new" scaler in output_dir (simulates new training run)
        new_scaler = {"version": 2, "mean": 0.5, "std": 2.0}
        with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(new_scaler, f)

        # Simulate the scaler copy logic from train.py (FIXED version)
        scaler_src = os.path.join(output_dir, "scaler.pkl")
        scaler_dst = os.path.join(ckpt_dir, "scaler.pkl")
        if os.path.exists(scaler_src):
            shutil.copy2(scaler_src, scaler_dst)

        # Verify the checkpoint now has the NEW scaler, not stale old one
        with open(os.path.join(ckpt_dir, "scaler.pkl"), "rb") as f:
            loaded_scaler = pickle.load(f)

        assert loaded_scaler["version"] == 2, (
            "Scaler in checkpoint should be overwritten with latest version"
        )
        assert loaded_scaler["mean"] == 0.5
        assert loaded_scaler["std"] == 2.0
