"""
Pytest Configuration and Shared Fixtures
=========================================

Provides reusable fixtures for all test modules, including:
- Sample data generation for various dimensionalities
- Model instantiation helpers
- Temporary file management
- Mock objects for distributed training

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

# Set matplotlib backend before importing pyplot (prevents display issues in tests)
import matplotlib


matplotlib.use("Agg")

import numpy as np
import pytest
import torch


# Ensure the parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# RANDOM SEED CONTROL
# ==============================================================================
@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ==============================================================================
# SAMPLE DATA FIXTURES
# ==============================================================================
@pytest.fixture
def sample_1d_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample 1D signal data (e.g., waveforms).

    Returns:
        Tuple of (inputs, targets) where inputs are 1D signals.
    """
    n_samples = 100
    signal_length = 256
    n_targets = 3

    X = np.random.randn(n_samples, signal_length).astype(np.float32)
    y = np.random.randn(n_samples, n_targets).astype(np.float32)

    return X, y


@pytest.fixture
def sample_2d_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample 2D image data (e.g., spectrograms).

    Returns:
        Tuple of (inputs, targets) where inputs are 2D images.
    """
    n_samples = 50
    height, width = 64, 64
    n_targets = 5

    X = np.random.randn(n_samples, height, width).astype(np.float32)
    y = np.random.randn(n_samples, n_targets).astype(np.float32)

    return X, y


@pytest.fixture
def sample_3d_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample 3D volumetric data.

    Returns:
        Tuple of (inputs, targets) where inputs are 3D volumes.
    """
    n_samples = 20
    depth, height, width = 16, 32, 32
    n_targets = 4

    X = np.random.randn(n_samples, depth, height, width).astype(np.float32)
    y = np.random.randn(n_samples, n_targets).astype(np.float32)

    return X, y


@pytest.fixture
def sample_batch_2d() -> torch.Tensor:
    """Generate a batch of 2D tensors with channel dimension.

    Returns:
        Tensor of shape (B, C, H, W).
    """
    batch_size = 8
    channels = 1
    height, width = 64, 64

    return torch.randn(batch_size, channels, height, width)


@pytest.fixture
def sample_batch_1d() -> torch.Tensor:
    """Generate a batch of 1D tensors with channel dimension.

    Returns:
        Tensor of shape (B, C, L).
    """
    batch_size = 8
    channels = 1
    length = 256

    return torch.randn(batch_size, channels, length)


# ==============================================================================
# TEMPORARY FILE FIXTURES
# ==============================================================================
@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    tmpdir = tempfile.mkdtemp(prefix="wavedl_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_npz_file(temp_dir, sample_2d_data):
    """Create a temporary NPZ file with sample data."""
    X, y = sample_2d_data
    filepath = os.path.join(temp_dir, "test_data.npz")
    np.savez(filepath, input_train=X, output_train=y)
    return filepath


# ==============================================================================
# MOCK FIXTURES FOR DISTRIBUTED TRAINING
# ==============================================================================
@pytest.fixture
def mock_accelerator():
    """Create a mock Accelerator for testing distributed utilities."""
    accelerator = MagicMock()
    accelerator.device = torch.device("cpu")
    accelerator.num_processes = 1
    accelerator.process_index = 0
    accelerator.local_process_index = 0
    accelerator.is_main_process = True
    return accelerator


@pytest.fixture
def mock_accelerator_multi_gpu():
    """Create a mock Accelerator simulating multi-GPU setup."""
    accelerator = MagicMock()
    accelerator.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    accelerator.num_processes = 4
    accelerator.process_index = 0
    accelerator.local_process_index = 0
    accelerator.is_main_process = True
    return accelerator


# ==============================================================================
# MODEL TESTING HELPERS
# ==============================================================================
@pytest.fixture
def model_input_shapes():
    """Common input shapes for model testing."""
    return {
        "1d": (256,),
        "2d": (64, 64),
        "3d": (16, 32, 32),
    }


@pytest.fixture
def model_output_sizes():
    """Common output sizes for model testing."""
    return [1, 3, 5, 10]


# ==============================================================================
# PYTEST CONFIGURATION
# ==============================================================================
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")


# ==============================================================================
# CHECKPOINT AND SCALER FIXTURES
# ==============================================================================
@pytest.fixture
def mock_scaler():
    """Create a mock StandardScaler for testing."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    # Fit on dummy data to populate mean_ and scale_
    scaler.fit(np.random.randn(100, 5))
    return scaler


@pytest.fixture
def temp_checkpoint_dir(temp_dir, mock_scaler):
    """Create a temporary checkpoint directory with model weights and scaler.

    Returns:
        Path to checkpoint directory containing:
        - model.bin: Random model weights
        - scaler.pkl: Fitted StandardScaler
        - training_meta.pkl: Training metadata
    """
    import pickle

    from wavedl.models.cnn import CNN

    checkpoint_dir = os.path.join(temp_dir, "best_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create and save model weights
    model = CNN(in_shape=(64, 64), out_size=5)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.bin"))

    # Save scaler
    with open(os.path.join(checkpoint_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(mock_scaler, f)

    # Save training metadata
    meta = {
        "epoch": 50,
        "best_val_loss": 0.0123,
        "patience_ctr": 0,
        "model_name": "cnn",
        "in_shape": (64, 64),
        "out_dim": 5,
    }
    with open(os.path.join(checkpoint_dir, "training_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    return checkpoint_dir


@pytest.fixture
def temp_training_data(temp_dir):
    """Create temporary training data files.

    Returns:
        Dict with paths to NPZ and cache files.
    """
    n_samples = 100
    X = np.random.randn(n_samples, 64, 64).astype(np.float32)
    y = np.random.randn(n_samples, 5).astype(np.float32)

    npz_path = os.path.join(temp_dir, "train_data.npz")
    np.savez(npz_path, input_train=X, output_train=y)

    return {
        "npz_path": npz_path,
        "n_samples": n_samples,
        "in_shape": (64, 64),
        "out_dim": 5,
        "temp_dir": temp_dir,
    }


@pytest.fixture
def sample_predictions():
    """Generate sample predictions and targets for metrics testing."""
    n_samples = 100
    n_targets = 5

    # Create correlated predictions (not random) for meaningful metrics
    y_true = np.random.randn(n_samples, n_targets).astype(np.float32)
    noise = np.random.randn(n_samples, n_targets).astype(np.float32) * 0.1
    y_pred = y_true + noise  # High correlation with small noise

    return y_true, y_pred
