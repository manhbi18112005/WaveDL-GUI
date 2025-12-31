"""
Model Registry and Factory Pattern for Deep Learning Architectures
===================================================================

This module provides a centralized registry for neural network architectures,
enabling dynamic model selection via command-line arguments.

Usage:
    from models import get_model, list_models, MODEL_REGISTRY

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
        from models.base import BaseModel
        from models.registry import register_model

        @register_model("my_model")
        class MyModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                ...

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

# Import registry first (no dependencies)
# Import base class (depends only on torch)
from .base import BaseModel

# Import model implementations (triggers registration via decorators)
from .cnn import CNN
from .convnext import ConvNeXtBase_, ConvNeXtSmall, ConvNeXtTiny
from .densenet import DenseNet121, DenseNet169
from .efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2
from .registry import (
    MODEL_REGISTRY,
    build_model,
    get_model,
    list_models,
    register_model,
)
from .resnet import ResNet18, ResNet34, ResNet50
from .unet import UNet, UNetRegression
from .vit import ViTBase_, ViTSmall, ViTTiny


# Export public API
__all__ = [
    # Models
    "CNN",
    # Registry
    "MODEL_REGISTRY",
    # Base class
    "BaseModel",
    "ConvNeXtBase_",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
    "DenseNet121",
    "DenseNet169",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "UNet",
    "UNetRegression",
    "ViTBase_",
    "ViTSmall",
    "ViTTiny",
    "build_model",
    "get_model",
    "list_models",
    "register_model",
]
