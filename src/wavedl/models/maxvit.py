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
import torch.nn.functional as F

from wavedl.models._pretrained_utils import build_regression_head
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

    Note:
        MaxViT requires input dimensions divisible by 28 (4x stem downsample × 7 window).
        This implementation automatically resizes inputs to the nearest compatible size.
    """

    # MaxViT stem downsamples by 4x, then requires divisibility by 7 (window size)
    # So original input must be divisible by 4 * 7 = 28
    _DIVISOR = 28

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

        # Compute compatible input size for MaxViT attention windows
        self._target_size = self._compute_compatible_size(in_shape)

        # Try to load from timm
        try:
            import timm

            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
            )

            # Get feature dimension using compatible size
            with torch.no_grad():
                dummy = torch.zeros(1, 3, *self._target_size)
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
        from wavedl.models._pretrained_utils import find_and_adapt_input_convs

        adapted_count = find_and_adapt_input_convs(
            self.backbone, pretrained=self.pretrained, adapt_all=False
        )

        if adapted_count == 0:
            import warnings

            warnings.warn(
                "Could not adapt MaxViT input channels. Model may fail.", stacklevel=2
            )

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _compute_compatible_size(self, in_shape: tuple[int, int]) -> tuple[int, int]:
        """
        Compute the nearest input size compatible with MaxViT attention windows.

        MaxViT requires input dimensions divisible by 28 (4x stem downsample × 7 window).
        This rounds up to the nearest compatible size.

        Args:
            in_shape: Original (H, W) input shape

        Returns:
            Compatible (H, W) shape divisible by 28
        """
        import math

        h, w = in_shape
        target_h = math.ceil(h / self._DIVISOR) * self._DIVISOR
        target_w = math.ceil(w / self._DIVISOR) * self._DIVISOR
        return (target_h, target_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize input to compatible size if needed
        _, _, h, w = x.shape
        if (h, w) != self._target_size:
            x = F.interpolate(
                x,
                size=self._target_size,
                mode="bilinear",
                align_corners=False,
            )
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
