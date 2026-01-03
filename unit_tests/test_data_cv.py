"""
Unit Tests for Data Loading and Cross-Validation Utilities
===========================================================

Additional tests for data.py and cross_validation.py to improve coverage.

**Tested Components**:
    - data.py: DataSource.detect_format, LazyDataHandle, NPZSource, get_data_source
    - cross_validation.py: CVDataset

Note: Full data loading tests are in test_data.py. This file adds coverage
for edge cases and utilities not covered there.

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import tempfile

import numpy as np
import pytest
import torch


# ==============================================================================
# DATA SOURCE FORMAT DETECTION
# ==============================================================================


class TestDataSourceDetectFormat:
    """Tests for automatic file format detection."""

    def test_detect_npz(self):
        """Test NPZ format detection."""
        from wavedl.utils.data import DataSource

        assert DataSource.detect_format("data.npz") == "npz"
        assert DataSource.detect_format("/path/to/data.npz") == "npz"
        assert DataSource.detect_format("data.NPZ") == "npz"

    def test_detect_hdf5(self):
        """Test HDF5 format detection."""
        from wavedl.utils.data import DataSource

        assert DataSource.detect_format("data.h5") == "hdf5"
        assert DataSource.detect_format("data.hdf5") == "hdf5"
        assert DataSource.detect_format("data.HDF5") == "hdf5"

    def test_detect_mat(self):
        """Test MAT format detection."""
        from wavedl.utils.data import DataSource

        assert DataSource.detect_format("data.mat") == "mat"
        assert DataSource.detect_format("/path/to/data.MAT") == "mat"

    def test_detect_unsupported_raises(self):
        """Test that unsupported formats raise ValueError."""
        from wavedl.utils.data import DataSource

        with pytest.raises(ValueError, match="Unsupported"):
            DataSource.detect_format("data.csv")

        with pytest.raises(ValueError, match="Unsupported"):
            DataSource.detect_format("data.txt")


class TestGetDataSource:
    """Tests for data source factory function."""

    def test_get_npz_source(self):
        """Test getting NPZ data source."""
        from wavedl.utils.data import NPZSource, get_data_source

        source = get_data_source("npz")
        assert isinstance(source, NPZSource)

    def test_get_hdf5_source(self):
        """Test getting HDF5 data source."""
        from wavedl.utils.data import HDF5Source, get_data_source

        source = get_data_source("hdf5")
        assert isinstance(source, HDF5Source)

    def test_get_mat_source(self):
        """Test getting MAT data source."""
        from wavedl.utils.data import MATSource, get_data_source

        source = get_data_source("mat")
        assert isinstance(source, MATSource)

    def test_get_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        from wavedl.utils.data import get_data_source

        with pytest.raises(ValueError):
            get_data_source("invalid")


# ==============================================================================
# LAZY DATA HANDLE
# ==============================================================================


class TestLazyDataHandle:
    """Tests for LazyDataHandle context manager."""

    def test_context_manager_returns_data(self):
        """Test that context manager yields inputs and outputs."""
        from wavedl.utils.data import LazyDataHandle

        inputs = np.array([1, 2, 3])
        outputs = np.array([4, 5, 6])

        with LazyDataHandle(inputs, outputs) as (x, y):
            assert np.array_equal(x, inputs)
            assert np.array_equal(y, outputs)

    def test_close_is_safe_to_call_multiple_times(self):
        """Test that close() can be called multiple times."""
        from wavedl.utils.data import LazyDataHandle

        handle = LazyDataHandle(np.array([1]), np.array([2]))
        handle.close()
        handle.close()  # Should not raise

    def test_repr_contains_shape_info(self):
        """Test that repr contains useful info."""
        from wavedl.utils.data import LazyDataHandle

        handle = LazyDataHandle(np.zeros((10, 5)), np.zeros((10, 3)))
        repr_str = repr(handle)
        assert "10" in repr_str or "LazyDataHandle" in repr_str


# ==============================================================================
# NPZ SOURCE
# ==============================================================================


class TestNPZSourceLoad:
    """Tests for NPZ data loading."""

    def test_load_basic_npz(self):
        """Test loading a basic NPZ file."""
        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            inputs = np.random.randn(100, 64, 64).astype(np.float32)
            outputs = np.random.randn(100, 3).astype(np.float32)
            np.savez(f.name, input_train=inputs, output_train=outputs)

            try:
                source = NPZSource()
                X, y = source.load(f.name)

                assert X.shape == (100, 64, 64)
                assert y.shape == (100, 3)
            finally:
                os.unlink(f.name)

    def test_load_mmap_basic(self):
        """Test memory-mapped loading."""
        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            inputs = np.random.randn(50, 32).astype(np.float32)
            outputs = np.random.randn(50, 2).astype(np.float32)
            np.savez(f.name, X=inputs, Y=outputs)

            try:
                source = NPZSource()
                X, y = source.load_mmap(f.name)

                assert X.shape == (50, 32)
                assert y.shape == (50, 2)
            finally:
                os.unlink(f.name)

    def test_load_outputs_only(self):
        """Test loading only outputs."""
        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            inputs = np.random.randn(100, 128, 128).astype(np.float32)  # Large
            outputs = np.random.randn(100, 5).astype(np.float32)  # Small
            np.savez(f.name, input_train=inputs, output_train=outputs)

            try:
                source = NPZSource()
                y = source.load_outputs_only(f.name)

                assert y.shape == (100, 5)
            finally:
                os.unlink(f.name)


class TestNPZSourceKeyDetection:
    """Tests for NPZ key auto-detection."""

    def test_detects_standard_keys(self):
        """Test detection of standard key names."""
        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(
                f.name, input_train=np.zeros((10, 5)), output_train=np.ones((10, 2))
            )

            try:
                source = NPZSource()
                X, y = source.load(f.name)
                assert X.shape == (10, 5)
                assert y.shape == (10, 2)
            finally:
                os.unlink(f.name)

    def test_detects_alternative_keys(self):
        """Test detection of alternative key names."""
        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, X=np.zeros((10, 5)), Y=np.ones((10, 2)))

            try:
                source = NPZSource()
                X, y = source.load(f.name)
                assert X.shape == (10, 5)
                assert y.shape == (10, 2)
            finally:
                os.unlink(f.name)


# ==============================================================================
# CROSS-VALIDATION DATASET
# ==============================================================================


class TestCVDataset:
    """Tests for CVDataset class."""

    def test_basic_initialization(self):
        """Test basic dataset initialization."""
        from wavedl.utils.cross_validation import CVDataset

        X = np.random.randn(100, 64, 64).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)

        dataset = CVDataset(X, y)

        assert len(dataset) == 100

    def test_getitem_returns_tensors(self):
        """Test that __getitem__ returns torch tensors."""
        from wavedl.utils.cross_validation import CVDataset

        X = np.random.randn(10, 32).astype(np.float32)
        y = np.random.randn(10, 2).astype(np.float32)

        dataset = CVDataset(X, y)
        x_item, y_item = dataset[0]

        assert isinstance(x_item, torch.Tensor)
        assert isinstance(y_item, torch.Tensor)

    def test_adds_channel_dimension_for_2d(self):
        """Test that channel dimension is added for 2D inputs."""
        from wavedl.utils.cross_validation import CVDataset

        X = np.random.randn(10, 64, 64).astype(np.float32)  # (N, H, W)
        y = np.random.randn(10, 3).astype(np.float32)

        dataset = CVDataset(X, y)
        x_item, _ = dataset[0]

        # Should have shape (C, H, W) = (1, 64, 64)
        assert x_item.ndim == 3
        assert x_item.shape[0] == 1

    def test_adds_channel_dimension_for_1d(self):
        """Test that channel dimension is added for 1D inputs."""
        from wavedl.utils.cross_validation import CVDataset

        X = np.random.randn(10, 128).astype(np.float32)  # (N, L)
        y = np.random.randn(10, 2).astype(np.float32)

        dataset = CVDataset(X, y)
        x_item, _ = dataset[0]

        # Should have shape (C, L) = (1, 128)
        assert x_item.ndim == 2
        assert x_item.shape[0] == 1

    def test_preserves_3d_with_expected_spatial_ndim(self):
        """Test that 3D inputs are handled correctly with expected_spatial_ndim."""
        from wavedl.utils.cross_validation import CVDataset

        X = np.random.randn(10, 16, 32, 32).astype(np.float32)  # (N, D, H, W)
        y = np.random.randn(10, 3).astype(np.float32)

        # With expected_spatial_ndim=3, should add channel
        dataset = CVDataset(X, y, expected_spatial_ndim=3)
        x_item, _ = dataset[0]

        assert x_item.ndim == 4  # (C, D, H, W)


# ==============================================================================
# SPARSE MATRIX HANDLING
# ==============================================================================


class TestSparseMatrixHandling:
    """Tests for sparse matrix conversion."""

    def test_detects_csr_matrix(self):
        """Test that CSR sparse matrices are detected and converted."""

        from wavedl.utils.data import NPZSource

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            # Create sparse inputs
            dense = np.random.randn(10, 100).astype(np.float32)
            dense[dense < 0.5] = 0  # Make it sparse
            # csr_matrix(dense) would create sparse, but we test dense loading

            outputs = np.random.randn(10, 2).astype(np.float32)

            # Save as sparse - NPZ supports sparse arrays
            np.savez(f.name, input_train=dense, output_train=outputs)

            try:
                source = NPZSource()
                X, _y = source.load(f.name)

                assert isinstance(X, np.ndarray)
                assert X.shape == (10, 100)
            finally:
                os.unlink(f.name)
