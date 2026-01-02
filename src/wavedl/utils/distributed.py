"""
Distributed Training Utilities
==============================

Provides DDP-safe utilities for multi-GPU training including:
- Early stopping synchronization across ranks
- Value broadcasting from rank 0
- Tensor synchronization

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

import torch
import torch.distributed as dist
from accelerate import Accelerator


def broadcast_early_stop(should_stop: bool, accelerator: Accelerator) -> bool:
    """
    Broadcast early stopping decision from rank 0 to all processes.

    In DDP training, early stopping state (patience counter, best loss) is typically
    tracked only on rank 0. This function ensures all ranks receive the same stop signal
    to prevent deadlocks from inconsistent termination.

    Args:
        should_stop: Whether to stop training (only meaningful on rank 0)
        accelerator: Accelerator instance for device and process info

    Returns:
        True if training should stop (synchronized across all ranks)

    Example:
        # On main process: patience_ctr >= patience
        # On other processes: unknown/stale value
        should_stop = patience_ctr >= args.patience if accelerator.is_main_process else False
        if broadcast_early_stop(should_stop, accelerator):
            break  # All ranks exit loop together
    """
    stop_tensor = torch.tensor(
        1 if should_stop else 0, device=accelerator.device, dtype=torch.int32
    )

    if accelerator.num_processes > 1:
        dist.broadcast(stop_tensor, src=0)

    return stop_tensor.item() == 1


def broadcast_value(value: int | float, accelerator: Accelerator) -> int | float:
    """
    Broadcast a scalar value from rank 0 to all processes.

    Useful for synchronizing hyperparameters or computed values across ranks.

    Args:
        value: Scalar value to broadcast (only rank 0's value is used)
        accelerator: Accelerator instance for device and process info

    Returns:
        Value from rank 0 (synchronized across all ranks)
    """
    is_int = isinstance(value, int)
    dtype = torch.int64 if is_int else torch.float32

    tensor = torch.tensor(value, device=accelerator.device, dtype=dtype)

    if accelerator.num_processes > 1:
        dist.broadcast(tensor, src=0)

    result = tensor.item()
    return int(result) if is_int else result


def sync_tensor(
    tensor: torch.Tensor, accelerator: Accelerator, reduction: str = "sum"
) -> torch.Tensor:
    """
    Synchronize a tensor across all processes with specified reduction.

    Wrapper around accelerator.reduce with additional validation.

    Args:
        tensor: Tensor to synchronize
        accelerator: Accelerator instance
        reduction: Reduction operation ("sum", "mean", "max", "min")

    Returns:
        Reduced tensor (synchronized across all ranks)

    Raises:
        ValueError: If reduction type is not recognized
    """
    valid_reductions = {"sum", "mean", "max", "min"}
    if reduction not in valid_reductions:
        raise ValueError(
            f"Invalid reduction '{reduction}'. Must be one of {valid_reductions}"
        )

    return accelerator.reduce(tensor, reduction=reduction)


def get_world_info(accelerator: Accelerator) -> dict:
    """
    Get distributed training world information.

    Args:
        accelerator: Accelerator instance

    Returns:
        Dictionary with world_size, rank, local_rank, is_main, device
    """
    return {
        "world_size": accelerator.num_processes,
        "rank": accelerator.process_index,
        "local_rank": accelerator.local_process_index,
        "is_main": accelerator.is_main_process,
        "device": str(accelerator.device),
    }


def print_rank0(message: str, accelerator: Accelerator, logger=None):
    """
    Print message only on rank 0.

    Convenience function for logging in distributed setting.

    Args:
        message: Message to print
        accelerator: Accelerator instance
        logger: Optional logger (uses print if None)
    """
    if accelerator.is_main_process:
        if logger:
            logger.info(message)
        else:
            print(message)
