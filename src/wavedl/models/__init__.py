"""
Model Registry and Factory Pattern for Deep Learning Architectures
===================================================================

This module provides a centralized registry for neural network architectures,
enabling dynamic model selection via command-line arguments.

**Dimensionality Coverage**:
    - 1D (waveforms): TCN, CNN, ResNet, ConvNeXt, DenseNet, ViT
    - 2D (images): CNN, ResNet, ConvNeXt, DenseNet, ViT, UNet,
                   EfficientNet, MobileNetV3, RegNet, Swin
    - 3D (volumes): ResNet3D, CNN, ResNet, ConvNeXt, DenseNet

Usage:
    from wavedl.models import get_model, list_models, MODEL_REGISTRY

    # List available models
    print(list_models())

    # Get a model class by name
    ModelClass = get_model("cnn")
    model = ModelClass(in_shape=(500, 500), out_size=5)

Adding New Models:
    1. Create a new file in models/ (e.g., models/my_model.py)
    2. Inherit from BaseModel
    3. Use the @register_model decorator

    Example:
        from wavedl.models.base import BaseModel
        from wavedl.models.registry import register_model

        @register_model("my_model")
        class MyModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                ...

Author: Ductho Le (ductho.le@outlook.com)
"""

# Import registry first (no dependencies)
# Import base class (depends only on torch)
from .base import BaseModel

# Import model implementations (triggers registration via decorators)
from .cnn import CNN
from .convnext import ConvNeXtBase_, ConvNeXtSmall, ConvNeXtTiny
from .densenet import DenseNet121, DenseNet169
from .efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2
from .efficientnetv2 import EfficientNetV2L, EfficientNetV2M, EfficientNetV2S
from .mobilenetv3 import MobileNetV3Large, MobileNetV3Small
from .registry import (
    MODEL_REGISTRY,
    build_model,
    get_model,
    list_models,
    register_model,
)
from .regnet import RegNetY1_6GF, RegNetY3_2GF, RegNetY8GF, RegNetY400MF, RegNetY800MF
from .resnet import ResNet18, ResNet34, ResNet50
from .resnet3d import MC3_18, ResNet3D18
from .swin import SwinBase, SwinSmall, SwinTiny
from .tcn import TCN, TCNLarge, TCNSmall
from .unet import UNetRegression
from .vit import ViTBase_, ViTSmall, ViTTiny


# Export public API (sorted alphabetically per RUF022)
# See module docstring for dimensionality support details
__all__ = [
    "CNN",
    "MC3_18",
    "MODEL_REGISTRY",
    "TCN",
    "BaseModel",
    "ConvNeXtBase_",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
    "DenseNet121",
    "DenseNet169",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetV2L",
    "EfficientNetV2M",
    "EfficientNetV2S",
    "MobileNetV3Large",
    "MobileNetV3Small",
    "RegNetY1_6GF",
    "RegNetY3_2GF",
    "RegNetY8GF",
    "RegNetY400MF",
    "RegNetY800MF",
    "ResNet3D18",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "SwinBase",
    "SwinSmall",
    "SwinTiny",
    "TCNLarge",
    "TCNSmall",
    "UNetRegression",
    "ViTBase_",
    "ViTSmall",
    "ViTTiny",
    "build_model",
    "get_model",
    "list_models",
    "register_model",
]
