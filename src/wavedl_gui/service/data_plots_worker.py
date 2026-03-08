"""
WaveDL GUI - Data Plots Worker

Background QThread that loads a SUBSAMPLE of a data file and pre-computes
statistics for visualisation.  Only ~5 000 rows are ever materialised,
keeping memory usage minimal even for very large datasets.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from PySide6.QtCore import QThread, Signal
from scipy.sparse import issparse

from wavedl.utils.data import DataSource, get_data_source


# ── Constants ────────────────────────────────────────────────────────────────

_MAX_SAMPLES = 5_000
_MAX_HIST_FEATURES = 8
_MAX_CORR_COLS = 30
_THREAD_STACK_SIZE = 16 * 1024 * 1024


# ── Data bundle ──────────────────────────────────────────────────────────────


@dataclass
class DataBundle:
    """Pre-computed data ready for plotting on the main thread."""

    x_flat: np.ndarray  # (n, features) float32
    y_flat: np.ndarray  # (n, outputs)  float32
    x_raw: np.ndarray  # original subsampled X (for sample preview)
    input_feat_indices: np.ndarray
    corr_matrix: np.ndarray | None = None
    input_means: np.ndarray = field(default_factory=lambda: np.array([]))
    input_stds: np.ndarray = field(default_factory=lambda: np.array([]))
    stat_feat_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    scatter_feat_indices: np.ndarray = field(default_factory=lambda: np.array([]))


# ── Helpers ──────────────────────────────────────────────────────────────────


def _row_to_float(row) -> np.ndarray:
    """Convert a single row/element to a flat float32 array."""
    if issparse(row) or hasattr(row, "toarray"):
        return np.asarray(row.toarray(), dtype=np.float32).ravel()
    return np.asarray(row, dtype=np.float32).ravel()


def _subsample_from_handle(handle_data, n_total: int) -> np.ndarray:
    """Read at most _MAX_SAMPLES rows from a lazy handle into dense float32.

    Only the selected rows are materialised — the rest stays on disk.
    Works with: np.memmap, h5py.Dataset, _TransposedH5Dataset, np.ndarray,
    and object arrays containing sparse matrices.
    """
    rng = np.random.default_rng(42)
    n = min(n_total, _MAX_SAMPLES)
    if n_total <= _MAX_SAMPLES:
        indices = np.arange(n_total)
    else:
        indices = np.sort(rng.choice(n_total, size=n, replace=False))

    # Read rows one-by-one to handle heterogeneous types (sparse, object, etc.)
    rows = []
    for idx in indices:
        row = handle_data[int(idx)]
        rows.append(_row_to_float(row))

    # Stack — all rows now have the same length after _row_to_float
    first_len = rows[0].shape[0]
    if all(r.shape[0] == first_len for r in rows):
        return np.stack(rows)

    # Ragged rows (rare) — zero-pad
    max_len = max(r.shape[0] for r in rows)
    out = np.zeros((len(rows), max_len), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, : r.shape[0]] = r
    return out


def _prepare_bundle(X: np.ndarray, Y: np.ndarray) -> DataBundle:
    """Compute statistics from already-subsampled arrays."""
    n = min(X.shape[0], Y.shape[0])
    X = X[:n]
    Y = Y[:n]

    x_flat = X.reshape(X.shape[0], -1)
    y_flat = Y.reshape(Y.shape[0], -1)

    # Input histogram feature indices
    n_feats = x_flat.shape[1]
    n_hist = min(n_feats, _MAX_HIST_FEATURES)
    if n_feats > _MAX_HIST_FEATURES:
        input_feat_indices = np.linspace(0, n_feats - 1, n_hist, dtype=int)
    else:
        input_feat_indices = np.arange(n_hist)

    # Correlation matrix
    n_out = y_flat.shape[1]
    if n_out >= 2:
        corr_flat = y_flat
        if n_out > _MAX_CORR_COLS:
            corr_idx = np.linspace(0, n_out - 1, _MAX_CORR_COLS, dtype=int)
            corr_flat = y_flat[:, corr_idx]
        corr_matrix = np.corrcoef(corr_flat, rowvar=False)
    else:
        corr_matrix = None

    # Input statistics
    max_stat = 40
    if n_feats > max_stat:
        stat_feat_indices = np.linspace(0, n_feats - 1, max_stat, dtype=int)
        stat_flat = x_flat[:, stat_feat_indices]
    else:
        stat_feat_indices = np.arange(n_feats)
        stat_flat = x_flat
    input_means = stat_flat.mean(axis=0)
    input_stds = stat_flat.std(axis=0)

    # Scatter feature indices
    n_scatter = min(6, n_feats)
    if n_feats > n_scatter:
        scatter_feat_indices = np.linspace(0, n_feats - 1, n_scatter, dtype=int)
    else:
        scatter_feat_indices = np.arange(n_scatter)

    return DataBundle(
        x_flat=x_flat,
        y_flat=y_flat,
        x_raw=X,
        input_feat_indices=input_feat_indices,
        corr_matrix=corr_matrix,
        input_means=input_means,
        input_stds=input_stds,
        stat_feat_indices=stat_feat_indices,
        scatter_feat_indices=scatter_feat_indices,
    )


# ── Worker thread ────────────────────────────────────────────────────────────


class DataPlotsWorker(QThread):
    """Load a *subsample* of data and pre-compute statistics.

    Only ~5 000 rows are ever read from disk, keeping memory usage
    minimal regardless of dataset size.

    Signals
    -------
    dataReady(DataBundle)
        Emitted with pre-computed data for the main thread to plot.
    errorOccurred(str)
        Emitted if loading or computation fails.
    """

    dataReady = Signal(object)  # DataBundle
    errorOccurred = Signal(str)

    PLOT_NAMES: ClassVar[list[str]] = [
        "Input Distribution",
        "Output Distribution",
        "Sample Preview",
        "Correlation Heatmap",
        "Input Statistics",
        "Input vs Output",
    ]

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.setStackSize(_THREAD_STACK_SIZE)
        self._file_path = file_path

    def run(self):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        try:
            fmt = DataSource.detect_format(self._file_path)
            source = get_data_source(fmt)

            handle = source.load_mmap(self._file_path)
            try:
                inp, outp = handle.inputs, handle.outputs
                n_total = len(inp)

                # Subsample directly from the lazy handle — only ~5 000 rows
                # are ever materialised, keeping memory usage minimal.
                X_sub = _subsample_from_handle(inp, n_total)

                n_total_y = len(outp)
                Y_sub = _subsample_from_handle(outp, n_total_y)
            finally:
                handle.close()

            # Handle 1D targets: (N,) shape after subsampling
            if Y_sub.ndim == 1:
                Y_sub = Y_sub.reshape(-1, 1)

            bundle = _prepare_bundle(X_sub, Y_sub)

            # Explicitly free the subsampled arrays — bundle has its own copies
            del X_sub, Y_sub
            gc.collect()

            self.dataReady.emit(bundle)

        except Exception as e:
            self.errorOccurred.emit(str(e))
