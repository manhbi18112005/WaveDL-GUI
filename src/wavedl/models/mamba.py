"""
Vision Mamba: Efficient Visual Representation Learning with State Space Models
===============================================================================

Vision Mamba (Vim) adapts the Mamba selective state space model for vision tasks.
Provides O(n) linear complexity vs O(n²) for transformers, making it efficient
for long sequences and high-resolution images.

**Key Features**:
    - Bidirectional SSM for image understanding
    - O(n) linear complexity
    - 2.8x faster than ViT, 86.8% less GPU memory
    - Works for 1D (time-series) and 2D (images)

**Variants**:
    - mamba_1d: For 1D time-series (alternative to TCN)
    - vim_tiny: 7M params for 2D images
    - vim_small: 26M params for 2D images
    - vim_base: 98M params for 2D images

**Dependencies**:
    - Optional: mamba-ssm (for optimized CUDA kernels)
    - Fallback: Pure PyTorch implementation

Reference:
    Zhu, L., et al. (2024). Vision Mamba: Efficient Visual Representation
    Learning with Bidirectional State Space Model. ICML 2024.
    https://arxiv.org/abs/2401.09417

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


# Type aliases
SpatialShape = tuple[int] | tuple[int, int]

__all__ = [
    "Mamba1D",
    "Mamba1DBase",
    "MambaBlock",
    "VimBase",
    "VimSmall",
    "VimTiny",
    "VisionMambaBase",
]


# =============================================================================
# SELECTIVE SSM CORE (Pure PyTorch Implementation)
# =============================================================================


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Core of Mamba.

    The key innovation is making the SSM parameters (B, C, Δ) input-dependent,
    allowing the model to selectively focus on or ignore inputs.

    This is a simplified pure-PyTorch implementation. For production use,
    consider the optimized mamba-ssm package.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable SSM matrices
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input sequence

        Returns:
            y: (B, L, D) output sequence
        """
        _B, L, _D = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # Conv for local context
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]  # Causal
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)

        # SSM parameters from input
        x_proj = self.x_proj(x)  # (B, L, d_state*2 + 1)
        delta = F.softplus(self.dt_proj(x_proj[:, :, :1]))  # (B, L, d_inner)
        B_param = x_proj[:, :, 1 : self.d_state + 1]  # (B, L, d_state)
        C_param = x_proj[:, :, self.d_state + 1 :]  # (B, L, d_state)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_state,)

        # Selective scan (simplified, not optimized)
        y = self._selective_scan(x, delta, A, B_param, C_param, self.D)

        # Gating
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified selective scan.

        For real applications, use the CUDA-optimized version from mamba-ssm.
        This implementation is for understanding and testing only.
        """
        B_batch, L, d_inner = x.shape
        d_state = A.shape[0]

        # Initialize state
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            x_t = x[:, t, :]  # (B, d_inner)
            delta_t = delta[:, t, :]  # (B, d_inner)
            B_t = B[:, t, :]  # (B, d_state)
            C_t = C[:, t, :]  # (B, d_state)

            # Discretize: A_bar = exp(delta * A)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A)  # (B, d_inner, d_state)

            # Update state: h = A_bar * h + delta * B * x
            h = A_bar * h + delta_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(-1)

            # Output: y = C * h + D * x
            y_t = (C_t.unsqueeze(1) * h).sum(-1) + D * x_t  # (B, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, d_inner)


# =============================================================================
# MAMBA BLOCK
# =============================================================================


class MambaBlock(nn.Module):
    """
    Mamba Block with residual connection.

    Architecture:
        Input → Norm → SelectiveSSM → Residual
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# =============================================================================
# BIDIRECTIONAL MAMBA (For Vision)
# =============================================================================


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block for vision tasks.

    Processes sequence in both forward and backward directions
    to capture global context in images.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm_forward = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ssm_backward = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.merge = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)

        # Forward pass
        y_forward = self.ssm_forward(x_norm)

        # Backward pass (flip, process, flip back)
        y_backward = self.ssm_backward(x_norm.flip(dims=[1])).flip(dims=[1])

        # Merge
        y = self.merge(torch.cat([y_forward, y_backward], dim=-1))

        return x + y


# =============================================================================
# MAMBA 1D (For Time-Series)
# =============================================================================


class Mamba1DBase(BaseModel):
    """
    Mamba for 1D time-series data.

    Alternative to TCN with theoretically infinite receptive field
    and linear complexity.
    """

    def __init__(
        self,
        in_shape: tuple[int],
        out_size: int,
        d_model: int = 256,
        n_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 1:
            raise ValueError(f"Mamba1D requires 1D input (L,), got {len(in_shape)}D")

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, in_shape[0], d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mamba blocks
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)]
        )

        # Final norm
        self.norm = nn.LayerNorm(d_model)

        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(d_model // 2, out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, L) input signal

        Returns:
            (B, out_size) regression output
        """
        _B, _C, L = x.shape

        # Reshape to sequence
        x = x.transpose(1, 2)  # (B, L, 1)
        x = self.input_proj(x)  # (B, L, d_model)

        # Add positional encoding
        x = x + self.pos_embed[:, :L, :]

        # Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling (mean over sequence)
        x = x.mean(dim=1)  # (B, d_model)

        # Final norm and head
        x = self.norm(x)
        return self.head(x)


# =============================================================================
# VISION MAMBA (For 2D Images)
# =============================================================================


class VisionMambaBase(BaseModel):
    """
    Vision Mamba (Vim) for 2D images.

    Uses bidirectional SSM to capture global context efficiently.
    O(n) complexity instead of O(n²) for transformers.
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        patch_size: int = 16,
        d_model: int = 192,
        n_layers: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 2:
            raise ValueError(
                f"VisionMamba requires 2D input (H, W), got {len(in_shape)}D"
            )

        self.patch_size = patch_size
        self.d_model = d_model

        H, W = in_shape
        self.num_patches = (H // patch_size) * (W // patch_size)
        self.grid_size = (H // patch_size, W // patch_size)

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            1, d_model, kernel_size=patch_size, stride=patch_size
        )

        # CLS token for classification/regression
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Bidirectional Mamba blocks
        self.blocks = nn.ModuleList(
            [
                BidirectionalMambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ]
        )

        # Final norm
        self.norm = nn.LayerNorm(d_model)

        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(d_model // 2, out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) input image

        Returns:
            (B, out_size) regression output
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + num_patches, d_model)

        # Add positional embedding
        x = x + self.pos_embed

        # Bidirectional Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Extract CLS token
        cls_output = x[:, 0]  # (B, d_model)

        # Final norm and head
        cls_output = self.norm(cls_output)
        return self.head(cls_output)


# =============================================================================
# REGISTERED VARIANTS
# =============================================================================


@register_model("mamba_1d")
class Mamba1D(Mamba1DBase):
    """
    Mamba 1D: ~3.4M backbone parameters (for time-series regression).

    8 layers, 256 dim. Alternative to TCN for time-series.
    Pure PyTorch implementation.

    Example:
        >>> model = Mamba1D(in_shape=(4096,), out_size=3)
        >>> x = torch.randn(4, 1, 4096)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int], out_size: int, **kwargs):
        kwargs.setdefault("d_model", 256)
        kwargs.setdefault("n_layers", 8)
        super().__init__(in_shape=in_shape, out_size=out_size, **kwargs)

    def __repr__(self) -> str:
        return f"Mamba1D(in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("vim_tiny")
class VimTiny(VisionMambaBase):
    """
    Vision Mamba Tiny: ~6.6M backbone parameters.

    12 layers, 192 dim. For 2D images.
    Pure PyTorch implementation with O(n) complexity.

    Example:
        >>> model = VimTiny(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("d_model", 192)
        kwargs.setdefault("n_layers", 12)
        super().__init__(in_shape=in_shape, out_size=out_size, **kwargs)

    def __repr__(self) -> str:
        return f"VisionMamba_Tiny(in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("vim_small")
class VimSmall(VisionMambaBase):
    """
    Vision Mamba Small: ~51.1M backbone parameters.

    24 layers, 384 dim. For 2D images.
    Pure PyTorch implementation with O(n) complexity.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("d_model", 384)
        kwargs.setdefault("n_layers", 24)
        super().__init__(in_shape=in_shape, out_size=out_size, **kwargs)

    def __repr__(self) -> str:
        return f"VisionMamba_Small(in_shape={self.in_shape}, out_size={self.out_size})"


@register_model("vim_base")
class VimBase(VisionMambaBase):
    """
    Vision Mamba Base: ~201.4M backbone parameters.

    24 layers, 768 dim. For 2D images.
    Pure PyTorch implementation with O(n) complexity.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("d_model", 768)
        kwargs.setdefault("n_layers", 24)
        super().__init__(in_shape=in_shape, out_size=out_size, **kwargs)

    def __repr__(self) -> str:
        return f"VisionMamba_Base(in_shape={self.in_shape}, out_size={self.out_size})"
