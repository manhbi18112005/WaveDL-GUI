"""
Base Model Abstract Class
==========================

Defines the interface contract that all models must implement for compatibility
with the training pipeline. Provides common utilities and enforces consistency.

Author: Ductho Le (ductho.le@outlook.com)
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all regression models.

    All models in this framework must inherit from BaseModel and implement
    the required abstract methods. This ensures compatibility with the
    training pipeline and provides a consistent interface.

    Supports any input dimensionality:
        - 1D: in_shape = (L,) for signals/waveforms
        - 2D: in_shape = (H, W) for images/spectrograms
        - 3D: in_shape = (D, H, W) for volumes

    Attributes:
        in_shape: Input spatial dimensions (varies by dimensionality)
        out_size: Number of output targets

    Example:
        from wavedl.models.base import BaseModel
        from wavedl.models.registry import register_model

        @register_model("my_model")
        class MyModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                # Build layers...

            def forward(self, x):
                # Forward pass...
                return output
    """

    @abstractmethod
    def __init__(
        self,
        in_shape: tuple[int] | tuple[int, int] | tuple[int, int, int],
        out_size: int,
        **kwargs,
    ):
        """
        Initialize the model.

        Args:
            in_shape: Input spatial dimensions, excluding batch and channel dims:
                      - 1D: (L,) for signal length
                      - 2D: (H, W) for image dimensions
                      - 3D: (D, H, W) for volume dimensions
            out_size: Number of regression output targets
            **kwargs: Model-specific hyperparameters
        """
        super().__init__()
        self.in_shape = in_shape
        self.out_size = out_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, out_size)
        """
        pass

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_summary(self) -> dict[str, Any]:
        """
        Generate a summary of model parameters.

        Returns:
            Dictionary with parameter statistics
        """
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "total_mb": total * 4 / (1024 * 1024),  # Assuming float32
        }

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Return default configuration for this model.
        Override in subclasses to provide model-specific defaults.

        Returns:
            Dictionary of default hyperparameters
        """
        return {}

    def get_optimizer_groups(self, base_lr: float, weight_decay: float = 1e-4) -> list:
        """
        Get parameter groups for optimizer with optional layer-wise learning rates.
        Override in subclasses for custom parameter grouping (e.g., no decay on biases).

        Args:
            base_lr: Base learning rate
            weight_decay: Weight decay coefficient

        Returns:
            List of parameter group dictionaries
        """
        # Default: no weight decay on bias and normalization layers
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Skip weight decay for bias and normalization parameters
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Handle empty parameter lists gracefully
        groups = []
        if decay_params:
            groups.append(
                {"params": decay_params, "lr": base_lr, "weight_decay": weight_decay}
            )
        if no_decay_params:
            groups.append(
                {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0}
            )

        return (
            groups
            if groups
            else [
                {
                    "params": self.parameters(),
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            ]
        )
