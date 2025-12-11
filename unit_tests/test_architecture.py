"""
Unit Tests for Model Architectures
===================================

Comprehensive tests for:
- BaseModel: abstract base class functionality
- CNN: baseline convolutional network

Tests cover:
- Model instantiation with various input shapes
- Forward pass correctness
- Output shape validation
- Parameter counting
- Gradient flow

Author: Ductho Le (ductho.le@outlook.com)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from models.base import BaseModel
from models.cnn import CNN
from models.registry import register_model


# ==============================================================================
# BASE MODEL TESTS
# ==============================================================================
class TestBaseModel:
    """Tests for the BaseModel abstract base class."""
    
    def test_cannot_instantiate_directly(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel(in_shape=(64, 64), out_size=5)
    
    def test_subclass_must_implement_forward(self):
        """Test that subclass must implement forward method."""
        # This should work - implementing required methods
        @register_model("test_valid_subclass")
        class ValidSubclass(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                self.fc = nn.Linear(10, out_size)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1)[:, :10])
        
        model = ValidSubclass(in_shape=(64, 64), out_size=5)
        assert model.in_shape == (64, 64)
        assert model.out_size == 5
    
    def test_count_parameters(self):
        """Test parameter counting functionality."""
        @register_model("test_param_count")
        class ParamCountModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                self.fc = nn.Linear(100, out_size)  # 100*out_size + out_size params
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1)[:, :100])
        
        model = ParamCountModel(in_shape=(10, 10), out_size=5)
        
        # Expected: 100*5 (weights) + 5 (bias) = 505
        assert model.count_parameters() == 505
        assert model.count_parameters(trainable_only=True) == 505
        assert model.count_parameters(trainable_only=False) == 505
    
    def test_parameter_summary(self):
        """Test parameter summary generation."""
        @register_model("test_param_summary")
        class ParamSummaryModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                self.fc = nn.Linear(100, out_size)
            
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1)[:, :100])
        
        model = ParamSummaryModel(in_shape=(10, 10), out_size=5)
        summary = model.parameter_summary()
        
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "frozen_parameters" in summary
        assert "total_mb" in summary
        assert summary["total_parameters"] == 505
        assert summary["frozen_parameters"] == 0
    
    def test_get_default_config(self):
        """Test default config returns empty dict for base."""
        @register_model("test_default_config")
        class ConfigModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
            
            def forward(self, x):
                return torch.zeros(x.size(0), self.out_size)
        
        config = ConfigModel.get_default_config()
        assert isinstance(config, dict)
    
    def test_get_optimizer_groups(self):
        """Test optimizer parameter groups with weight decay."""
        @register_model("test_optim_groups")
        class OptimGroupsModel(BaseModel):
            def __init__(self, in_shape, out_size, **kwargs):
                super().__init__(in_shape, out_size)
                self.fc = nn.Linear(100, out_size)
                self.norm = nn.LayerNorm(out_size)
            
            def forward(self, x):
                out = self.fc(x.view(x.size(0), -1)[:, :100])
                return self.norm(out)
        
        model = OptimGroupsModel(in_shape=(10, 10), out_size=5)
        groups = model.get_optimizer_groups(base_lr=0.001, weight_decay=0.01)
        
        assert isinstance(groups, list)
        assert len(groups) >= 1
        
        # Check that groups have required keys
        for group in groups:
            assert "params" in group
            assert "lr" in group
            assert "weight_decay" in group


# ==============================================================================
# SIMPLE CNN TESTS
# ==============================================================================
class TestCNN:
    """Tests for the CNN model."""
    
    def test_instantiation(self):
        """Test CNN can be instantiated."""
        model = CNN(in_shape=(64, 64), out_size=5)
        
        assert model.in_shape == (64, 64)
        assert model.out_size == 5
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        model = CNN(in_shape=(64, 64), out_size=5)
        x = torch.randn(4, 1, 64, 64)
        
        output = model(x)
        
        assert output.shape == (4, 5)
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with various batch sizes."""
        model = CNN(in_shape=(64, 64), out_size=3)
        
        for batch_size in [1, 2, 8, 16, 32]:
            x = torch.randn(batch_size, 1, 64, 64)
            output = model(x)
            assert output.shape == (batch_size, 3)
    
    def test_forward_pass_different_input_sizes(self):
        """Test forward pass with various input dimensions."""
        test_cases = [
            ((32, 32), 3),
            ((64, 64), 5),
            ((128, 128), 7),
            ((256, 256), 10),
        ]
        
        for in_shape, out_size in test_cases:
            model = CNN(in_shape=in_shape, out_size=out_size)
            x = torch.randn(2, 1, *in_shape)
            output = model(x)
            assert output.shape == (2, out_size)
    
    def test_forward_pass_non_square(self):
        """Test forward pass with non-square input."""
        model = CNN(in_shape=(64, 128), out_size=5)
        x = torch.randn(4, 1, 64, 128)
        
        output = model(x)
        
        assert output.shape == (4, 5)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = CNN(in_shape=(64, 64), out_size=5)
        x = torch.randn(4, 1, 64, 64, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check all parameters received gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CNN.get_default_config()
        
        assert "base_channels" in config
        assert "dropout_rate" in config
        assert config["base_channels"] == 16
        assert config["dropout_rate"] == 0.1
    
    def test_custom_parameters(self):
        """Test model with custom parameters."""
        model = CNN(
            in_shape=(64, 64),
            out_size=5,
            base_channels=32,
            dropout_rate=0.2
        )
        
        assert model.base_channels == 32
        assert model.dropout_rate == 0.2
    
    def test_eval_mode(self):
        """Test model behavior in eval mode."""
        model = CNN(in_shape=(64, 64), out_size=5)
        model.eval()
        
        x = torch.randn(4, 1, 64, 64)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        # In eval mode, outputs should be deterministic
        assert torch.allclose(output1, output2)
    
    def test_train_mode_dropout(self):
        """Test that dropout is active in train mode."""
        model = CNN(in_shape=(64, 64), out_size=5, dropout_rate=0.5)
        model.train()
        
        x = torch.randn(4, 1, 64, 64)
        
        # Multiple forward passes should give different results due to dropout
        outputs = [model(x) for _ in range(10)]
        
        # Check that not all outputs are identical
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout doesn't seem to be active in train mode"


# ==============================================================================
# NUMERICAL STABILITY TESTS
# ==============================================================================
class TestNumericalStability:
    """Tests for numerical stability of models."""
    
    def test_no_nan_output(self):
        """Test that models don't produce NaN outputs."""
        model = CNN(in_shape=(64, 64), out_size=5)
        x = torch.randn(4, 1, 64, 64)
        
        output = model(x)
        assert not torch.isnan(output).any(), f"{model.__class__.__name__} produced NaN"
    
    def test_no_inf_output(self):
        """Test that models don't produce Inf outputs."""
        model = CNN(in_shape=(64, 64), out_size=5)
        x = torch.randn(4, 1, 64, 64)
        
        output = model(x)
        assert not torch.isinf(output).any(), f"{model.__class__.__name__} produced Inf"
    
    def test_handles_large_input_values(self):
        """Test models handle large input values."""
        model = CNN(in_shape=(64, 64), out_size=5)
        x = torch.randn(4, 1, 64, 64) * 100  # Large values
        
        output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_handles_small_input_values(self):
        """Test models handle very small input values."""
        model = CNN(in_shape=(64, 64), out_size=5)
        x = torch.randn(4, 1, 64, 64) * 1e-6  # Small values
        
        output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# ==============================================================================
# GPU TESTS (Conditional)
# ==============================================================================
@pytest.mark.gpu
class TestGPUModels:
    """Tests that require GPU."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_to_gpu(self):
        """Test model can be moved to GPU."""
        model = CNN(in_shape=(64, 64), out_size=5).cuda()
        x = torch.randn(4, 1, 64, 64).cuda()
        
        output = model(x)
        
        assert output.device.type == "cuda"
        assert output.shape == (4, 5)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision(self):
        """Test model works with mixed precision."""
        model = CNN(in_shape=(64, 64), out_size=5).cuda()
        x = torch.randn(4, 1, 64, 64).cuda()
        
        with torch.amp.autocast('cuda'):
            output = model(x)
        
        assert output.shape == (4, 5)
        assert not torch.isnan(output).any()
