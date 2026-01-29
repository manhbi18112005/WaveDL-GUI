"""
MaxViT: Multi-Axis Vision Transformer
======================================

MaxViT combines local and global attention with O(n) complexity using
multi-axis attention: block attention (local) + grid attention (global sparse).

**Key Features**:
    - Multi-axis attention for both local and global context
    - Hybrid design with MBConv + attention
    - Linear O(n) complexity
    - Hierarchical multi-scale features

**Variants**:
    - maxvit_tiny: 31M params
    - maxvit_small: 69M params
    - maxvit_base: 120M params

**Requirements**:
    - timm (for pretrained models and architecture)
    - torchvision (fallback, limited support)

Reference:
    Tu, Z., et al. (2022). MaxViT: Multi-Axis Vision Transformer.
    ECCV 2022. https://arxiv.org/abs/2204.01697

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch
import torch.nn as nn

from wavedl.models._timm_utils import build_regression_head
from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


__all__ = [
    "MaxViTBase",
    "MaxViTBaseLarge",
    "MaxViTSmall",
    "MaxViTTiny",
]


# =============================================================================
# MAXVIT BASE CLASS
# =============================================================================


class MaxViTBase(BaseModel):
    """
    MaxViT base class wrapping timm implementation.

    Multi-axis attention with local block and global grid attention.
    2D only due to attention structure.
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        model_name: str = "maxvit_tiny_tf_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 2:
            raise ValueError(f"MaxViT requires 2D input (H, W), got {len(in_shape)}D")

        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.model_name = model_name

        # Try to load from timm
        try:
            import timm

            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
            )

            # Get feature dimension
            with torch.no_grad():
                dummy = torch.zeros(1, 3, *in_shape)
                features = self.backbone(dummy)
                in_features = features.shape[-1]

        except ImportError:
            raise ImportError(
                "timm is required for MaxViT. Install with: pip install timm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load MaxViT model '{model_name}': {e}")

        # Adapt input channels (3 -> 1)
        self._adapt_input_channels()

        # Regression head
        self.head = build_regression_head(in_features, out_size, dropout_rate)

        if freeze_backbone:
            self._freeze_backbone()

    def _adapt_input_channels(self):
        """Adapt first conv layer for single-channel input."""
        # MaxViT uses stem.conv1 (Conv2dSame from timm)
        adapted = False

        # Find the first Conv2d with 3 input channels
        for name, module in self.backbone.named_modules():
            if hasattr(module, "in_channels") and module.in_channels == 3:
                # Get parent and child names
                parts = name.split(".")
                parent = self.backbone
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                child_name = parts[-1]

                # Create new conv with 1 input channel
                new_conv = self._make_new_conv(module)
                setattr(parent, child_name, new_conv)
                adapted = True
                break

        if not adapted:
            import warnings

            warnings.warn(
                "Could not adapt MaxViT input channels. Model may fail.", stacklevel=2
            )

    def _make_new_conv(self, old_conv: nn.Module) -> nn.Module:
        """Create new conv layer with 1 input channel."""
        # Handle both Conv2d and Conv2dSame from timm
        type(old_conv)

        # Get common parameters
        kwargs = {
            "out_channels": old_conv.out_channels,
            "kernel_size": old_conv.kernel_size,
            "stride": old_conv.stride,
            "padding": old_conv.padding if hasattr(old_conv, "padding") else 0,
            "bias": old_conv.bias is not None,
        }

        # Create new conv (use regular Conv2d for simplicity)
        new_conv = nn.Conv2d(1, **kwargs)

        if self.pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
        return new_conv

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# =============================================================================
# REGISTERED VARIANTS
# =============================================================================


@register_model("maxvit_tiny")
class MaxViTTiny(MaxViTBase):
    """
    MaxViT Tiny: ~30.1M backbone parameters.

    Multi-axis attention with local+global context.
    2D only.

    Example:
        >>> model = MaxViTTiny(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="maxvit_tiny_tf_224",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"MaxViT_Tiny(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("maxvit_small")
class MaxViTSmall(MaxViTBase):
    """
    MaxViT Small: ~67.6M backbone parameters.

    Multi-axis attention with local+global context.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="maxvit_small_tf_224",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"MaxViT_Small(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("maxvit_base")
class MaxViTBaseLarge(MaxViTBase):
    """
    MaxViT Base: ~118.1M backbone parameters.

    Multi-axis attention with local+global context.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="maxvit_base_tf_224",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"MaxViT_Base(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )
