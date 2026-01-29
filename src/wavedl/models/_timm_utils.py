"""
Shared Utilities for Model Architectures
=========================================

Common components used across multiple models:
- GRN (Global Response Normalization) for ConvNeXt V2
- Dimension-agnostic layer factories
- Regression head builders
- Input channel adaptation for pretrained models

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# DIMENSION-AGNOSTIC LAYER FACTORIES
# =============================================================================


def get_conv_layer(dim: int) -> type[nn.Module]:
    """Get dimension-appropriate Conv class."""
    layers = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    if dim not in layers:
        raise ValueError(f"Unsupported dimension: {dim}")
    return layers[dim]


def get_norm_layer(dim: int) -> type[nn.Module]:
    """Get dimension-appropriate BatchNorm class."""
    layers = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
    if dim not in layers:
        raise ValueError(f"Unsupported dimension: {dim}")
    return layers[dim]


def get_pool_layer(dim: int) -> type[nn.Module]:
    """Get dimension-appropriate AdaptiveAvgPool class."""
    layers = {1: nn.AdaptiveAvgPool1d, 2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}
    if dim not in layers:
        raise ValueError(f"Unsupported dimension: {dim}")
    return layers[dim]


# =============================================================================
# GLOBAL RESPONSE NORMALIZATION (GRN) - ConvNeXt V2
# =============================================================================


class GRN1d(nn.Module):
    """
    Global Response Normalization for 1D inputs.

    GRN enhances inter-channel feature competition and promotes diversity.
    Replaces LayerScale in ConvNeXt V2.

    Reference: ConvNeXt V2 (CVPR 2023)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        Gx = torch.norm(x, p=2, dim=2, keepdim=True)  # (B, C, 1)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)  # (B, C, 1)
        return self.gamma * (x * Nx) + self.beta + x


class GRN2d(nn.Module):
    """
    Global Response Normalization for 2D inputs.

    GRN enhances inter-channel feature competition and promotes diversity.
    Replaces LayerScale in ConvNeXt V2.

    Reference: ConvNeXt V2 (CVPR 2023)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)  # (B, C, 1, 1)
        return self.gamma * (x * Nx) + self.beta + x


class GRN3d(nn.Module):
    """
    Global Response Normalization for 3D inputs.

    GRN enhances inter-channel feature competition and promotes diversity.
    Replaces LayerScale in ConvNeXt V2.

    Reference: ConvNeXt V2 (CVPR 2023)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        Gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)  # (B, C, 1, 1, 1)
        return self.gamma * (x * Nx) + self.beta + x


def get_grn_layer(dim: int) -> type[nn.Module]:
    """Get dimension-appropriate GRN class."""
    layers = {1: GRN1d, 2: GRN2d, 3: GRN3d}
    if dim not in layers:
        raise ValueError(f"Unsupported dimension: {dim}")
    return layers[dim]


# =============================================================================
# LAYER NORMALIZATION (Channels Last for CNNs)
# =============================================================================


class LayerNormNd(nn.Module):
    """
    LayerNorm that works with channels-first tensors of any dimension.
    Applies normalization over the channel dimension.
    """

    def __init__(self, normalized_shape: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.dim = dim
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move channels to last, apply LN, move back
        if self.dim == 1:
            # (B, C, L) -> (B, L, C) -> LN -> (B, C, L)
            x = x.permute(0, 2, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 2, 1)
        elif self.dim == 2:
            # (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
        elif self.dim == 3:
            # (B, C, D, H, W) -> (B, D, H, W, C) -> LN -> (B, C, D, H, W)
            x = x.permute(0, 2, 3, 4, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 4, 1, 2, 3)
        return x


# =============================================================================
# REGRESSION HEAD BUILDERS
# =============================================================================


def build_regression_head(
    in_features: int,
    out_size: int,
    dropout_rate: float = 0.3,
    hidden_dim: int = 512,
) -> nn.Sequential:
    """
    Build a standard regression head for pretrained models.

    Args:
        in_features: Input feature dimension
        out_size: Number of regression targets
        dropout_rate: Dropout rate
        hidden_dim: Hidden layer dimension

    Returns:
        nn.Sequential regression head
    """
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Dropout(dropout_rate * 0.5),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim // 2, out_size),
    )


def adapt_input_channels(
    conv_layer: nn.Module,
    new_in_channels: int = 1,
    pretrained: bool = True,
) -> nn.Module:
    """
    Adapt a convolutional layer for different input channels.

    For pretrained models, averages RGB weights to grayscale.

    Args:
        conv_layer: Original conv layer (expects 3 input channels)
        new_in_channels: New number of input channels (default: 1)
        pretrained: Whether to adapt pretrained weights

    Returns:
        New conv layer with adapted input channels
    """
    if isinstance(conv_layer, nn.Conv2d):
        new_conv = nn.Conv2d(
            new_in_channels,
            conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=conv_layer.bias is not None,
        )
        if pretrained and conv_layer.in_channels == 3:
            with torch.no_grad():
                # Average RGB weights
                new_conv.weight.copy_(conv_layer.weight.mean(dim=1, keepdim=True))
                if conv_layer.bias is not None:
                    new_conv.bias.copy_(conv_layer.bias)
        return new_conv
    else:
        raise NotImplementedError(f"Unsupported layer type: {type(conv_layer)}")
