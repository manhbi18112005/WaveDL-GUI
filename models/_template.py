"""
Model Template for New Architectures
=====================================

Copy this file and modify to add new model architectures to the framework.
The model will be automatically registered and available via --model flag.

Steps to Add a New Model:
    1. Copy this file to models/your_model.py
    2. Rename the class and update @register_model("your_model")
    3. Implement the __init__ and forward methods
    4. Import your model in models/__init__.py:
       from models.your_model import YourModel
    5. Run: accelerate launch train.py --model your_model --wandb

Author: Ductho Le (ductho.le@outlook.com)
Version: 1.0.0
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.registry import register_model


# Uncomment the decorator to register this model
# @register_model("template")
class TemplateModel(BaseModel):
    """
    Template Model Architecture.
    
    Replace this docstring with your model description.
    The first line will appear in --list_models output.
    
    Args:
        in_shape: Input spatial dimensions (H, W)
        out_size: Number of regression output targets
        hidden_dim: Size of hidden layers (default: 256)
        num_layers: Number of convolutional layers (default: 4)
        dropout: Dropout rate (default: 0.1)
    
    Input Shape:
        (B, 1, H, W) - Single-channel images
        
    Output Shape:
        (B, out_size) - Regression predictions
    """
    
    def __init__(
        self,
        in_shape: Tuple[int, int],
        out_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        **kwargs  # Accept extra kwargs for flexibility
    ):
        # REQUIRED: Call parent __init__ with in_shape and out_size
        super().__init__(in_shape, out_size)
        
        # Store hyperparameters as attributes (optional but recommended)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # =================================================================
        # BUILD YOUR ARCHITECTURE HERE
        # =================================================================
        
        # Example: Simple CNN encoder
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )
        
        # Example: Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        REQUIRED: Must accept (B, C, H, W) and return (B, out_size)
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
            
        Returns:
            Output tensor of shape (B, out_size)
        """
        # Encode
        features = self.encoder(x)
        
        # Predict
        output = self.head(features)
        
        return output
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Return default hyperparameters for this model.
        
        OPTIONAL: Override to provide model-specific defaults.
        These can be used for documentation or config files.
        """
        return {
            "hidden_dim": 256,
            "num_layers": 4,
            "dropout": 0.1,
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Quick test of the model
    model = TemplateModel(in_shape=(500, 500), out_size=5)
    
    # Print model summary
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Default config: {model.get_default_config()}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 500, 500)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
