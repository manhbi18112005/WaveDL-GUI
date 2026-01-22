"""
Unit Tests for Model Architectures
===================================

Comprehensive tests for all model architectures in WaveDL.

**Tested Architectures**:
    - CNN: Dimension-agnostic baseline (1D/2D/3D)
    - ResNet: ResNet-18/34/50 (1D/2D/3D)
    - ResNet3D: Video-style ResNet3D-18/MC3-18 (3D only)
    - EfficientNet: B0/B1/B2 with pretrained weights (2D only)
    - EfficientNetV2: S/M/L variants (2D only)
    - MobileNetV3: Small/Large (2D only)
    - RegNet: Y-400MF to Y-8GF variants (2D only)
    - ViT: Vision Transformer Tiny/Small/Base (1D/2D)
    - ConvNeXt: Modern CNN Tiny/Small/Base (1D/2D/3D)
    - DenseNet: DenseNet-121/169 (1D/2D/3D)
    - Swin: Swin Transformer Tiny/Small/Base (2D only)
    - TCN: Temporal Convolutional Network (1D only)
    - U-Net: Encoder-decoder for regression (1D/2D/3D)

**Test Coverage**:
    - Model instantiation with appropriate input shapes per dimensionality
    - Forward pass correctness and output validation
    - Gradient flow through all layers
    - Numerical stability (no NaN/Inf outputs)
    - Eval mode determinism
    - Parameter counting and optimizer groups

**Dimensionality-Aware Testing**:
    Tests automatically select appropriate input shapes based on each model's
    supported dimensionality. This prevents false failures from passing
    incompatible input dimensions to dimension-specific models.

Author: Ductho Le (ductho.le@outlook.com)
"""

import pytest
import torch
import torch.nn as nn

from wavedl.models.base import BaseModel
from wavedl.models.cnn import CNN
from wavedl.models.registry import build_model, list_models, register_model


# ==============================================================================
# MODEL DIMENSIONALITY MAPPING
# ==============================================================================
# Maps model name prefixes to their supported input dimensionalities.
# This is the single source of truth for test input shape selection.

MODEL_DIMS = {
    # 1D only (specialized for temporal signals)
    "tcn": [1],
    # 2D only (pretrained torchvision models)
    "efficientnet": [2],
    "mobilenet": [2],
    "regnet": [2],
    "swin": [2],
    # 3D only (video models from torchvision)
    "resnet3d": [3],
    "mc3": [3],
    # 1D/2D (transformers - no 3D due to attention complexity)
    "vit": [1, 2],
    # 1D/2D/3D (dimension-agnostic architectures that work with small test volumes)
    "resnet": [1, 2, 3],  # Standard ResNet (not ResNet3D)
    "unet": [1, 2, 3],  # U-Net works well in all dimensions
    # 1D/2D only for testing (3D requires very large volumes due to pooling layers)
    # These architectures technically support 3D but need impractically large inputs
    "cnn": [1, 2],
    "convnext": [1, 2],
    "densenet": [1, 2],
}

# Default for models not in the mapping
DEFAULT_DIMS = [2]


def get_supported_dims(model_name: str) -> list[int]:
    """
    Get supported input dimensionalities for a model.

    Returns list of supported dimensions (1, 2, or 3) based on model name.
    Uses prefix matching against MODEL_DIMS mapping.

    Note: Pretrained models (ending with '_pretrained') are always 2D-only
    because they use torchvision weights which require RGB input format.
    """
    model_lower = model_name.lower()

    # Pretrained models are always 2D-only (torchvision pretrained weights)
    if model_lower.endswith("_pretrained"):
        return [2]

    # Check for specific prefixes (order matters: longer prefixes first)
    # ResNet3D must be checked before ResNet
    if model_lower.startswith("resnet3d") or model_lower.startswith("mc3"):
        return [3]

    for prefix, dims in MODEL_DIMS.items():
        if model_lower.startswith(prefix):
            return dims

    return DEFAULT_DIMS


def get_primary_dim(model_name: str) -> int:
    """Get the primary (first) supported dimension for a model."""
    return get_supported_dims(model_name)[0]


# ==============================================================================
# TEST INPUT CONFIGURATION
# ==============================================================================
def get_test_config(model_name: str, dim: int | None = None) -> tuple:
    """
    Get appropriate test configuration for a model.

    Args:
        model_name: Name of the model to test
        dim: Optional dimension override. If None, uses model's primary dimension.

    Returns:
        (in_shape, kwargs) tuple suitable for testing.
    """
    # Use primary dimension if not specified
    if dim is None:
        dim = get_primary_dim(model_name)

    # Standard test shapes for each dimensionality
    # Shapes chosen to be compatible with all model kernel sizes
    if dim == 1:
        in_shape = (256,)  # 1D signal length
    elif dim == 2:
        in_shape = (64, 64)  # 2D image
    else:
        in_shape = (16, 64, 64)  # 3D volume (larger to avoid kernel size issues)

    kwargs = {}

    # Model-specific adjustments for valid configurations
    model_lower = model_name.lower()

    # Force pretrained=False for all pretrained models to enable offline CI
    # This prevents network downloads during tests for faster, reliable execution
    if model_lower.endswith("_pretrained"):
        kwargs["pretrained"] = False

    if model_lower.startswith("vit"):
        # Smaller patches for test input size
        kwargs["patch_size"] = 8 if dim == 2 else 16
    elif model_lower.startswith("unet"):
        # Smaller depth for faster tests
        kwargs["depth"] = 3

    return in_shape, kwargs


# ==============================================================================
# UNIVERSAL MODEL TESTS (Auto-tests ALL registered models)
# ==============================================================================
@pytest.mark.slow
class TestAllModels:
    """
    Universal tests that automatically run on ALL registered models.

    When you add a new model to the registry, it will automatically
    be included in these tests - no manual updates needed!

    Tests use dimension-aware input shapes: each model is tested with its
    primary supported dimensionality to avoid false failures from
    dimension mismatches.
    """

    @pytest.mark.parametrize("model_name", list_models())
    def test_model_forward(self, model_name):
        """Test that all models can perform a forward pass."""
        in_shape, kwargs = get_test_config(model_name)

        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()

        x = torch.randn(2, 1, *in_shape)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 2, f"{model_name}: Batch size mismatch"
        assert not torch.isnan(out).any(), f"{model_name}: Output contains NaN"
        assert not torch.isinf(out).any(), f"{model_name}: Output contains Inf"

    @pytest.mark.parametrize("model_name", list_models())
    def test_model_gradient_flow(self, model_name):
        """Test that gradients flow through all models."""
        in_shape, kwargs = get_test_config(model_name)

        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.train()

        x = torch.randn(2, 1, *in_shape, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, f"{model_name}: No gradient on input"
        assert not torch.isnan(x.grad).any(), f"{model_name}: Gradient contains NaN"

    @pytest.mark.parametrize("model_name", list_models())
    def test_model_output_shape(self, model_name):
        """Test that all models produce correct output shape."""
        in_shape, kwargs = get_test_config(model_name)
        out_size = 5

        model = build_model(model_name, in_shape=in_shape, out_size=out_size, **kwargs)
        model.eval()

        x = torch.randn(4, 1, *in_shape)
        with torch.no_grad():
            out = model(x)

        # UNet with spatial_output=True has different shape
        model_lower = model_name.lower()
        if model_lower.startswith("unet"):
            assert out.shape[0] == 4
            assert out.shape[1] == out_size
        else:
            assert out.shape == (4, out_size), (
                f"{model_name}: Expected {(4, out_size)}, got {out.shape}"
            )

    @pytest.mark.parametrize("model_name", list_models())
    def test_model_eval_deterministic(self, model_name):
        """Test that models are deterministic in eval mode."""
        in_shape, kwargs = get_test_config(model_name)

        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()

        x = torch.randn(2, 1, *in_shape)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2), (
            f"{model_name}: Not deterministic in eval mode"
        )

    @pytest.mark.parametrize("model_name", list_models())
    def test_model_has_parameters(self, model_name):
        """Test that all models have trainable parameters."""
        in_shape, kwargs = get_test_config(model_name)

        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert param_count > 0, f"{model_name}: No trainable parameters"

    @pytest.mark.parametrize("model_name", list_models())
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_model_different_batch_sizes(self, model_name, batch_size):
        """Test that all models handle various batch sizes."""
        in_shape, kwargs = get_test_config(model_name)

        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()

        x = torch.randn(batch_size, 1, *in_shape)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == batch_size, (
            f"{model_name}: Batch size {batch_size} failed"
        )

    @pytest.mark.parametrize("model_name", list_models())
    def test_model_all_params_have_gradients(self, model_name):
        """Test that gradients reach all trainable parameters."""
        in_shape, kwargs = get_test_config(model_name)

        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.train()

        x = torch.randn(2, 1, *in_shape, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{model_name}: No gradient for {name}"


# ==============================================================================
# MULTI-DIMENSIONALITY TESTS
# ==============================================================================
@pytest.mark.slow
class TestMultiDimensionality:
    """
    Tests for models that support multiple input dimensionalities.

    Only tests models with their explicitly supported dimensions to avoid
    false failures from dimension mismatches.
    """

    def _get_models_supporting_dim(self, dim: int) -> list[str]:
        """Get list of models that support a specific dimension."""
        return [m for m in list_models() if dim in get_supported_dims(m)]

    @pytest.mark.parametrize(
        "model_name",
        [m for m in list_models() if 1 in get_supported_dims(m)],
    )
    def test_1d_input(self, model_name):
        """Test models that support 1D input."""
        in_shape, kwargs = get_test_config(model_name, dim=1)

        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()

        x = torch.randn(2, 1, *in_shape)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 2, f"{model_name}: 1D forward pass failed"
        assert not torch.isnan(out).any(), f"{model_name}: 1D output contains NaN"

    @pytest.mark.parametrize(
        "model_name",
        [m for m in list_models() if 2 in get_supported_dims(m)],
    )
    def test_2d_input(self, model_name):
        """Test models that support 2D input."""
        in_shape, kwargs = get_test_config(model_name, dim=2)

        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()

        x = torch.randn(2, 1, *in_shape)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 2, f"{model_name}: 2D forward pass failed"
        assert not torch.isnan(out).any(), f"{model_name}: 2D output contains NaN"

    @pytest.mark.parametrize(
        "model_name",
        [m for m in list_models() if 3 in get_supported_dims(m)],
    )
    @pytest.mark.slow
    def test_3d_input(self, model_name):
        """Test models that support 3D input (marked slow due to memory)."""
        in_shape, kwargs = get_test_config(model_name, dim=3)

        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)
        model.eval()

        x = torch.randn(2, 1, *in_shape)
        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == 2, f"{model_name}: 3D forward pass failed"
        assert not torch.isnan(out).any(), f"{model_name}: 3D output contains NaN"


# ==============================================================================
# BASE MODEL ABSTRACT CLASS TESTS
# ==============================================================================
class TestBaseModel:
    """Tests for the BaseModel abstract base class interface and utilities."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel(in_shape=(64, 64), out_size=5)

    def test_subclass_must_implement_forward(self):
        """Test that subclass must implement forward method."""

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

        for group in groups:
            assert "params" in group
            assert "lr" in group
            assert "weight_decay" in group


# ==============================================================================
# GPU TESTS (Optional - require CUDA)
# ==============================================================================
@pytest.mark.gpu
class TestGPUModels:
    """Tests for GPU deployment and mixed precision training."""

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

        with torch.amp.autocast("cuda"):
            output = model(x)

        assert output.shape == (4, 5)
        assert not torch.isnan(output).any()


# ==============================================================================
# PARAMETER SUMMARY TESTS FOR ALL MODELS
# ==============================================================================
@pytest.mark.slow
class TestModelParameterSummary:
    """Tests for parameter_summary() method across all models."""

    @pytest.mark.parametrize("model_name", list_models())
    def test_parameter_summary_returns_dict(self, model_name):
        """Test that parameter_summary returns a properly structured dict."""
        in_shape, kwargs = get_test_config(model_name)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)

        summary = model.parameter_summary()

        assert isinstance(summary, dict)
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "frozen_parameters" in summary
        assert "total_mb" in summary

        # Verify values are reasonable
        assert summary["total_parameters"] >= 0
        assert summary["trainable_parameters"] >= 0
        assert summary["frozen_parameters"] >= 0
        assert summary["total_mb"] >= 0

    @pytest.mark.parametrize("model_name", list_models())
    def test_optimizer_groups_returns_list(self, model_name):
        """Test that get_optimizer_groups returns valid parameter groups."""
        in_shape, kwargs = get_test_config(model_name)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)

        groups = model.get_optimizer_groups(base_lr=0.001, weight_decay=0.01)

        assert isinstance(groups, list)
        assert len(groups) >= 1

        for group in groups:
            assert "params" in group
            assert "lr" in group
            assert "weight_decay" in group

    @pytest.mark.parametrize("model_name", list_models())
    def test_count_parameters_matches_summary(self, model_name):
        """Test that count_parameters matches parameter_summary."""
        in_shape, kwargs = get_test_config(model_name)
        model = build_model(model_name, in_shape=in_shape, out_size=3, **kwargs)

        count = model.count_parameters(trainable_only=True)
        summary = model.parameter_summary()

        assert count == summary["trainable_parameters"]


# ==============================================================================
# MODEL EDGE CASE TESTS
# ==============================================================================
class TestModelEdgeCases:
    """Tests for edge cases in model behavior."""

    def test_single_output_dimension(self):
        """Test model with single output dimension."""
        model = CNN(in_shape=(32, 32), out_size=1)
        x = torch.randn(4, 1, 32, 32)

        output = model(x)

        assert output.shape == (4, 1)

    def test_large_output_dimension(self):
        """Test model with many output dimensions."""
        model = CNN(in_shape=(32, 32), out_size=100)
        x = torch.randn(2, 1, 32, 32)

        output = model(x)

        assert output.shape == (2, 100)

    def test_very_small_input(self):
        """Test model with minimum viable input size."""
        # CNN has 5 pooling layers (each halves spatial dims), so minimum is 32x32
        # Input: 32 -> 16 -> 8 -> 4 -> 2 -> 1 (valid spatial output)
        model = CNN(in_shape=(32, 32), out_size=3)
        x = torch.randn(2, 1, 32, 32)

        output = model(x)

        assert output.shape == (2, 3)
        assert not torch.isnan(output).any()

    def test_non_square_input(self):
        """Test model with non-square input dimensions."""
        model = CNN(in_shape=(32, 64), out_size=5)
        x = torch.randn(4, 1, 32, 64)

        output = model(x)

        assert output.shape == (4, 5)

    def test_model_state_dict_save_load(self):
        """Test that model state can be saved and loaded."""
        import tempfile

        model1 = CNN(in_shape=(32, 32), out_size=3)
        x = torch.randn(2, 1, 32, 32)

        # Get output from original model
        model1.eval()
        with torch.no_grad():
            out1 = model1(x)

        # Save and load state dict
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model1.state_dict(), f.name)

            model2 = CNN(in_shape=(32, 32), out_size=3)
            model2.load_state_dict(torch.load(f.name, weights_only=True))
            model2.eval()

            with torch.no_grad():
                out2 = model2(x)

            import os

            os.unlink(f.name)

        assert torch.allclose(out1, out2)

    def test_model_train_eval_mode_difference(self):
        """Test that train and eval modes behave differently for models with dropout/batchnorm."""
        model = CNN(in_shape=(32, 32), out_size=3)
        x = torch.randn(4, 1, 32, 32)

        # In eval mode, output should be deterministic
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

        # In train mode with dropout, outputs may differ
        # (but this depends on model architecture)
        model.train()


# ==============================================================================
# PRETRAINED MODEL TESTS
# ==============================================================================
class TestPretrainedModels:
    """Tests for pretrained model loading (with pretrained=False for CI)."""

    @pytest.mark.parametrize(
        "model_name",
        [
            m
            for m in list_models()
            if "pretrained" in m.lower() or m in ["resnet18", "efficientnet_b0"]
        ],
    )
    def test_pretrained_model_builds(self, model_name):
        """Test that pretrained models can be built (with pretrained=False)."""
        in_shape, kwargs = get_test_config(model_name)
        kwargs["pretrained"] = False  # Disable download for CI

        model = build_model(model_name, in_shape=in_shape, out_size=5, **kwargs)

        assert model is not None
        assert model.count_parameters() > 0
