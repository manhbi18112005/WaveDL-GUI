"""
Data Loading and Preprocessing Utilities
=========================================

Provides memory-efficient data loading for large-scale datasets with:
- Memory-mapped file support for datasets exceeding RAM
- DDP-safe data preparation with proper synchronization
- Thread-safe DataLoader worker initialization

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

import os
import gc
import pickle
import logging
from typing import Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from accelerate import Accelerator


# ==============================================================================
# WORKER INITIALIZATION
# ==============================================================================
def memmap_worker_init_fn(worker_id: int):
    """
    Worker initialization function for proper memmap handling in multi-worker DataLoader.
    
    Each DataLoader worker process runs this function after forking. It resets the 
    memmap file handle to None, forcing each worker to open its own read-only handle.
    This prevents file descriptor sharing issues and race conditions.
    
    Args:
        worker_id: Worker index (0 to num_workers-1), provided by DataLoader
    
    Usage:
        DataLoader(dataset, num_workers=8, worker_init_fn=memmap_worker_init_fn)
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Force re-initialization of memmap in each worker
        dataset.data = None


# ==============================================================================
# MEMORY-MAPPED DATASET
# ==============================================================================
class MemmapDataset(Dataset):
    """
    Zero-copy memory-mapped dataset for large-scale training.
    
    Uses numpy memory mapping to load data directly from disk, allowing training
    on datasets that exceed available RAM. The memmap is only opened when first
    accessed (lazy initialization), and each DataLoader worker maintains its own
    file handle for thread safety.
    
    Args:
        memmap_path: Path to the memory-mapped data file
        targets: Pre-loaded target tensor (small enough to fit in memory)
        shape: Full shape of the memmap array (N, C, H, W)
        indices: Indices into the memmap for this split (train/val)
    
    Thread Safety:
        When using with DataLoader num_workers > 0, must use memmap_worker_init_fn
        as the worker_init_fn to ensure each worker gets its own file handle.
    
    Example:
        dataset = MemmapDataset("cache.dat", y_tensor, (10000, 1, 500, 500), train_indices)
        loader = DataLoader(dataset, num_workers=8, worker_init_fn=memmap_worker_init_fn)
    """
    
    def __init__(
        self, 
        memmap_path: str, 
        targets: torch.Tensor, 
        shape: Tuple[int, ...], 
        indices: np.ndarray
    ):
        self.memmap_path = memmap_path
        self.targets = targets
        self.shape = shape
        self.indices = indices
        self.data: Optional[np.memmap] = None  # Lazy initialization
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            # Mode 'r' = read-only, prevents accidental data modification
            self.data = np.memmap(
                self.memmap_path, 
                dtype='float32', 
                mode='r', 
                shape=self.shape
            )
        
        real_idx = self.indices[idx]
        
        # .copy() detaches from mmap buffer - essential for PyTorch pinned memory
        x = torch.from_numpy(self.data[real_idx].copy()).contiguous()
        y = self.targets[real_idx]
        
        return x, y
    
    def __repr__(self) -> str:
        return (
            f"MemmapDataset(path='{self.memmap_path}', "
            f"samples={len(self)}, shape={self.shape})"
        )


# ==============================================================================
# DATA PREPARATION
# ==============================================================================
def prepare_data(
    args: Any,
    logger: logging.Logger,
    accelerator: Accelerator,
    cache_dir: str = ".",
    val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, StandardScaler, Tuple[int, ...], int]:
    """
    Prepare DataLoaders with DDP synchronization guarantees.
    
    This function handles:
    1. Loading raw data and creating memmap cache (Rank 0 only)
    2. Fitting StandardScaler on training set only (no data leakage)
    3. Synchronizing all ranks before proceeding
    4. Creating thread-safe DataLoaders for DDP training
    
    Supports any input dimensionality:
        - 1D: (N, L) → returns in_shape = (L,)
        - 2D: (N, H, W) → returns in_shape = (H, W)
        - 3D: (N, D, H, W) → returns in_shape = (D, H, W)
    
    Args:
        args: Argument namespace with data_path, seed, batch_size, workers
        logger: Logger instance for status messages
        accelerator: Accelerator instance for DDP coordination
        cache_dir: Directory for cache files (default: current directory)
        val_split: Validation set fraction (default: 0.2)
    
    Returns:
        Tuple of:
            - train_dl: Training DataLoader
            - val_dl: Validation DataLoader
            - scaler: Fitted StandardScaler (for inverse transforms)
            - in_shape: Input spatial dimensions - (L,), (H, W), or (D, H, W)
            - out_dim: Number of output targets
    
    Cache Files Created:
        - train_data_cache.dat: Memory-mapped input data
        - scaler.pkl: Fitted StandardScaler
        - data_metadata.pkl: Shape and dimension metadata
    """
    CACHE_FILE = os.path.join(cache_dir, "train_data_cache.dat")
    SCALER_FILE = os.path.join(cache_dir, "scaler.pkl")
    META_FILE = os.path.join(cache_dir, "data_metadata.pkl")
    
    # ==========================================================================
    # PHASE 1: DATA GENERATION (Rank 0 Only)
    # ==========================================================================
    with accelerator.main_process_first():
        cache_exists = (
            os.path.exists(CACHE_FILE) and 
            os.path.exists(SCALER_FILE) and 
            os.path.exists(META_FILE)
        )
        
        if not cache_exists:
            logger.info(f"⚡ [Rank 0] Initializing Data Processing from: {args.data_path}")
            
            # Validate data file exists
            if not os.path.exists(args.data_path):
                raise FileNotFoundError(f"CRITICAL: Data file not found: {args.data_path}")
            
            # Load raw data
            try:
                raw = np.load(args.data_path, allow_pickle=True)
                inp = raw['input_train']
                outp = raw['output_train']
            except Exception as e:
                logger.error(f"Failed to load NPZ file: {e}")
                raise
            
            # Detect shape (handle sparse matrices) - DIMENSION AGNOSTIC
            num_samples = len(inp)
            out_dim = outp.shape[1]
            sample_0 = inp[0]
            if issparse(sample_0) or hasattr(sample_0, 'toarray'):
                sample_0 = sample_0.toarray()
            
            # Automatically detect dimensionality from sample shape
            spatial_shape = sample_0.shape  # Could be (L,), (H, W), or (D, H, W)
            full_shape = (num_samples, 1) + spatial_shape  # Add channel dim: (N, 1, ...)
            
            dim_names = {1: "1D (L)", 2: "2D (H, W)", 3: "3D (D, H, W)"}
            dim_type = dim_names.get(len(spatial_shape), f"{len(spatial_shape)}D")
            logger.info(f"   Shape Detected: {full_shape} [{dim_type}] | Output Dim: {out_dim}")
            
            # Save metadata
            with open(META_FILE, 'wb') as f:
                pickle.dump({'shape': full_shape, 'out_dim': out_dim}, f)
            
            # Create memmap cache
            if not os.path.exists(CACHE_FILE):
                logger.info("   Writing Memmap Cache (one-time operation)...")
                fp = np.memmap(CACHE_FILE, dtype='float32', mode='w+', shape=full_shape)
                
                chunk_size = 2000
                pbar = tqdm(
                    range(0, num_samples, chunk_size),
                    desc="Caching",
                    disable=not accelerator.is_main_process
                )
                
                for i in pbar:
                    end = min(i + chunk_size, num_samples)
                    batch = inp[i:end]
                    
                    # Handle sparse/dense conversion
                    if issparse(batch[0]) or hasattr(batch[0], 'toarray'):
                        data_chunk = np.stack([x.toarray().astype(np.float32) for x in batch])
                    else:
                        data_chunk = np.array(batch).astype(np.float32)
                    
                    # Add channel dimension if needed (handles 1D, 2D, and 3D spatial data)
                    # data_chunk shape: (batch, *spatial) -> need (batch, 1, *spatial)
                    # full_shape is (N, 1, *spatial), so expected ndim = len(full_shape)
                    if data_chunk.ndim == len(full_shape) - 1:
                        # Missing channel dim: (batch, *spatial) -> (batch, 1, *spatial)
                        data_chunk = np.expand_dims(data_chunk, 1)
                    
                    fp[i:end] = data_chunk
                    
                    # Periodic flush to disk
                    if i % 10000 == 0:
                        fp.flush()
                
                fp.flush()
                del fp
                gc.collect()
            
            # Train/Val split and scaler fitting
            indices = np.arange(num_samples)
            tr_idx, val_idx = train_test_split(
                indices, 
                test_size=val_split, 
                random_state=args.seed
            )
            
            if not os.path.exists(SCALER_FILE):
                logger.info("   Fitting StandardScaler (training set only)...")
                scaler = StandardScaler()
                scaler.fit(outp[tr_idx])
                with open(SCALER_FILE, 'wb') as f:
                    pickle.dump(scaler, f)
            
            # Cleanup
            del inp, outp, raw
            gc.collect()
    
    # ==========================================================================
    # PHASE 2: SYNCHRONIZED LOADING (All Ranks)
    # ==========================================================================
    accelerator.wait_for_everyone()
    
    # Load metadata
    with open(META_FILE, 'rb') as f:
        meta = pickle.load(f)
        full_shape = meta['shape']
        out_dim = meta['out_dim']
    
    # Load and validate scaler
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    
    if not hasattr(scaler, 'scale_') or scaler.scale_ is None:
        raise RuntimeError("CRITICAL: Scaler is not properly fitted (scale_ is None)")
    
    # Load targets (lightweight compared to input data)
    outp = np.load(args.data_path, allow_pickle=True)['output_train']
    y_scaled = scaler.transform(outp).astype(np.float32)
    y_tensor = torch.tensor(y_scaled)
    
    # Regenerate indices (deterministic with same seed)
    indices = np.arange(full_shape[0])
    tr_idx, val_idx = train_test_split(
        indices, 
        test_size=val_split, 
        random_state=args.seed
    )
    
    # Create datasets
    tr_ds = MemmapDataset(CACHE_FILE, y_tensor, full_shape, tr_idx)
    val_ds = MemmapDataset(CACHE_FILE, y_tensor, full_shape, val_idx)
    
    # Create DataLoaders with thread-safe configuration
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
        worker_init_fn=memmap_worker_init_fn if args.workers > 0 else None
    )
    
    train_dl = DataLoader(tr_ds, shuffle=True, **loader_kwargs)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    
    # Return spatial shape (H, W) for model initialization
    in_shape = full_shape[2:]
    
    return train_dl, val_dl, scaler, in_shape, out_dim
