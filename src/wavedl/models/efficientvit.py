"""
EfficientViT: Memory-Efficient Vision Transformer with Cascaded Group Attention
================================================================================

EfficientViT (MIT) achieves state-of-the-art speed-accuracy trade-off by using
cascaded group attention (CGA) which reduces computational redundancy in
multi-head self-attention while maintaining model capability.

**Key Features**:
    - Cascaded Group Attention (CGA): Linear complexity attention
    - Memory-efficient design for edge deployment
    - Faster than Swin Transformer with similar accuracy
    - Excellent for real-time NDE applications

**Variants**:
    - efficientvit_m0: 2.3M params (mobile, fastest)
    - efficientvit_m1: 2.9M params (mobile)
    - efficientvit_m2: 4.2M params (mobile)
    - efficientvit_b0: 3.4M params (balanced)
    - efficientvit_b1: 9.1M params (balanced)
    - efficientvit_b2: 24M params (balanced)
    - efficientvit_b3: 49M params (balanced)
    - efficientvit_l1: 53M params (large)
    - efficientvit_l2: 64M params (large)

**Requirements**:
    - timm >= 0.9.0 (for EfficientViT models)

Reference:
    Liu, X., et al. (2023). EfficientViT: Memory Efficient Vision Transformer
    with Cascaded Group Attention. CVPR 2023.
    https://arxiv.org/abs/2305.07027

Author: Ductho Le (ductho.le@outlook.com)
"""

import torch

from wavedl.models._pretrained_utils import build_regression_head
from wavedl.models.base import BaseModel
from wavedl.models.registry import register_model


__all__ = [
    "EfficientViTB0",
    "EfficientViTB1",
    "EfficientViTB2",
    "EfficientViTB3",
    "EfficientViTBase",
    "EfficientViTL1",
    "EfficientViTL2",
    "EfficientViTM0",
    "EfficientViTM1",
    "EfficientViTM2",
]


# =============================================================================
# EFFICIENTVIT BASE CLASS
# =============================================================================


class EfficientViTBase(BaseModel):
    """
    EfficientViT base class wrapping timm implementation.

    Uses Cascaded Group Attention for efficient multi-head attention with
    linear complexity. 2D only due to attention structure.

    Args:
        in_shape: (H, W) input shape (2D only)
        out_size: Number of regression targets
        model_name: timm model name
        pretrained: Whether to load pretrained weights
        freeze_backbone: Whether to freeze backbone for fine-tuning
        dropout_rate: Dropout rate for regression head
    """

    def __init__(
        self,
        in_shape: tuple[int, int],
        out_size: int,
        model_name: str = "efficientvit_b0",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        **kwargs,
    ):
        super().__init__(in_shape, out_size)

        if len(in_shape) != 2:
            raise ValueError(
                f"EfficientViT requires 2D input (H, W), got {len(in_shape)}D"
            )

        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.model_name = model_name

        # Load from timm
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
                "timm >= 0.9.0 is required for EfficientViT. "
                "Install with: pip install timm>=0.9.0"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load EfficientViT model '{model_name}': {e}")

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
                "Could not adapt EfficientViT input channels. Model may fail.",
                stacklevel=2,
            )

    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# =============================================================================
# MOBILE VARIANTS (Ultra-lightweight)
# =============================================================================


@register_model("efficientvit_m0")
class EfficientViTM0(EfficientViTBase):
    """
    EfficientViT-M0: ~2.2M backbone parameters (fastest mobile variant).

    Cascaded group attention for efficient inference.
    Ideal for edge deployment and real-time NDE applications.
    2D only.

    Example:
        >>> model = EfficientViTM0(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_m0",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_M0(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("efficientvit_m1")
class EfficientViTM1(EfficientViTBase):
    """
    EfficientViT-M1: ~2.6M backbone parameters.

    Slightly larger mobile variant with better accuracy.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_m1",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_M1(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("efficientvit_m2")
class EfficientViTM2(EfficientViTBase):
    """
    EfficientViT-M2: ~3.8M backbone parameters.

    Largest mobile variant, best accuracy among M-series.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_m2",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_M2(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


# =============================================================================
# BALANCED VARIANTS (B-series)
# =============================================================================


@register_model("efficientvit_b0")
class EfficientViTB0(EfficientViTBase):
    """
    EfficientViT-B0: ~2.1M backbone parameters.

    Smallest balanced variant. Good accuracy-speed trade-off.
    2D only.

    Example:
        >>> model = EfficientViTB0(in_shape=(224, 224), out_size=3)
        >>> x = torch.randn(4, 1, 224, 224)
        >>> out = model(x)  # (4, 3)
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_b0",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_B0(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("efficientvit_b1")
class EfficientViTB1(EfficientViTBase):
    """
    EfficientViT-B1: ~7.5M backbone parameters.

    Medium balanced variant with improved capacity.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_b1",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_B1(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("efficientvit_b2")
class EfficientViTB2(EfficientViTBase):
    """
    EfficientViT-B2: ~21.8M backbone parameters.

    Larger balanced variant for complex patterns.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_b2",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_B2(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("efficientvit_b3")
class EfficientViTB3(EfficientViTBase):
    """
    EfficientViT-B3: ~46.1M backbone parameters.

    Largest balanced variant, highest accuracy in B-series.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_b3",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_B3(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


# =============================================================================
# LARGE VARIANTS (L-series)
# =============================================================================


@register_model("efficientvit_l1")
class EfficientViTL1(EfficientViTBase):
    """
    EfficientViT-L1: ~49.5M backbone parameters.

    Large variant for maximum accuracy.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_l1",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_L1(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )


@register_model("efficientvit_l2")
class EfficientViTL2(EfficientViTBase):
    """
    EfficientViT-L2: ~60.5M backbone parameters.

    Largest variant, best accuracy.
    2D only.
    """

    def __init__(self, in_shape: tuple[int, int], out_size: int, **kwargs):
        super().__init__(
            in_shape=in_shape,
            out_size=out_size,
            model_name="efficientvit_l2",
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"EfficientViT_L2(in_shape={self.in_shape}, out_size={self.out_size}, "
            f"pretrained={self.pretrained})"
        )
