"""
Integration Tests for WaveDL
=============================

End-to-end tests that verify components work together correctly:
- Model training simulation
- Data pipeline integration
- Full forward/backward passes

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from wavedl.models.cnn import CNN
from wavedl.models.registry import build_model
from wavedl.utils.metrics import MetricTracker, calc_pearson, calc_per_target_r2


# ==============================================================================
# END-TO-END TRAINING SIMULATION
# ==============================================================================
@pytest.mark.integration
class TestTrainingSimulation:
    """Integration tests simulating training workflows."""

    @pytest.fixture
    def training_setup(self):
        """Create a complete training setup."""
        # Generate synthetic data
        n_samples = 100
        in_shape = (32, 32)
        out_size = 3
        batch_size = 16

        X = torch.randn(n_samples, 1, *in_shape)
        y = torch.randn(n_samples, out_size)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = CNN(in_shape=in_shape, out_size=out_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        return {
            "model": model,
            "optimizer": optimizer,
            "criterion": criterion,
            "dataloader": dataloader,
            "in_shape": in_shape,
            "out_size": out_size,
        }

    def test_single_epoch_training(self, training_setup):
        """Test that a single epoch of training completes successfully."""
        model = training_setup["model"]
        optimizer = training_setup["optimizer"]
        criterion = training_setup["criterion"]
        dataloader = training_setup["dataloader"]

        model.train()
        loss_tracker = MetricTracker()

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            loss_tracker.update(loss.item(), n=batch_x.size(0))

        # Should have processed all batches
        assert loss_tracker.count > 0
        assert loss_tracker.avg > 0

    def test_multiple_epochs_training(self, training_setup):
        """Test training over multiple epochs with loss decreasing."""
        model = training_setup["model"]
        optimizer = training_setup["optimizer"]
        criterion = training_setup["criterion"]
        dataloader = training_setup["dataloader"]

        model.train()
        epoch_losses = []

        for epoch in range(5):
            epoch_tracker = MetricTracker()

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                epoch_tracker.update(loss.item(), n=batch_x.size(0))

            epoch_losses.append(epoch_tracker.avg)

        # Loss should generally decrease (allowing some fluctuation)
        assert epoch_losses[-1] < epoch_losses[0]

    def test_eval_mode_inference(self, training_setup):
        """Test inference in eval mode."""
        model = training_setup["model"]
        dataloader = training_setup["dataloader"]

        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                output = model(batch_x)
                all_predictions.append(output)
                all_targets.append(batch_y)

        predictions = torch.cat(all_predictions, dim=0).numpy()
        targets = torch.cat(all_targets, dim=0).numpy()

        # Should have predictions for all samples
        assert predictions.shape == targets.shape

        # Calculate metrics
        r2_scores = calc_per_target_r2(targets, predictions)
        pearson = calc_pearson(targets, predictions)

        # Metrics should be calculable (not NaN)
        assert not np.isnan(r2_scores).any()
        assert not np.isnan(pearson)


# ==============================================================================
# MODEL PIPELINE TESTS
# ==============================================================================
@pytest.mark.integration
class TestModelPipeline:
    """Tests for the complete model pipeline."""

    def test_build_train_eval_pipeline(self):
        """Test building a model from registry, training, and evaluating."""
        # Import to ensure registration

        # Build model from registry
        model = build_model("cnn", in_shape=(32, 32), out_size=5)

        # Create sample data
        X = torch.randn(20, 1, 32, 32)
        y = torch.randn(20, 5)

        # Training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Evaluation step
        model.eval()
        with torch.no_grad():
            predictions = model(X)

        assert predictions.shape == (20, 5)

    def test_model_checkpointing(self):
        """Test saving and loading model checkpoints."""
        model = CNN(in_shape=(32, 32), out_size=3)

        # Train briefly
        X = torch.randn(10, 1, 32, 32)
        y = torch.randn(10, 3)

        optimizer = torch.optim.Adam(model.parameters())
        loss = nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.step()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": 1,
            }
            torch.save(checkpoint, checkpoint_path)

            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)

            new_model = CNN(in_shape=(32, 32), out_size=3)
            new_model.load_state_dict(loaded_checkpoint["model_state_dict"])

            # Verify weights match
            for (name1, param1), (name2, param2) in zip(
                model.named_parameters(), new_model.named_parameters()
            ):
                assert torch.allclose(param1, param2), f"Mismatch at {name1}"

    def test_different_models_same_interface(self):
        """Test that different models work with the same training code."""
        in_shape = (32, 32)
        out_size = 3

        # Test with CNN (add more models here as they become available)
        model = CNN(in_shape=in_shape, out_size=out_size)

        X = torch.randn(8, 1, *in_shape)
        y = torch.randn(8, out_size)

        model.train()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        # Should work the same way for all models
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        assert output.shape == (8, out_size)


# ==============================================================================
# DATA PIPELINE INTEGRATION
# ==============================================================================
@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for data loading pipeline."""

    def test_npz_to_model_pipeline(self):
        """Test complete pipeline from NPZ file to model predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NPZ file
            n_samples = 50
            X = np.random.randn(n_samples, 32, 32).astype(np.float32)
            y = np.random.randn(n_samples, 3).astype(np.float32)

            npz_path = os.path.join(tmpdir, "data.npz")
            np.savez(npz_path, input_train=X, output_train=y)

            # Load data (explicit close for Windows compatibility)
            data = np.load(npz_path)
            X_loaded = data["input_train"].copy()  # Copy to release file handle
            y_loaded = data["output_train"].copy()
            data.close()  # Explicitly close the file

            # Add channel dimension
            X_tensor = torch.from_numpy(X_loaded[:, np.newaxis, :, :])
            torch.from_numpy(y_loaded)

            # Create model and run inference
            model = CNN(in_shape=(32, 32), out_size=3)
            model.eval()

            with torch.no_grad():
                predictions = model(X_tensor)

            assert predictions.shape == (n_samples, 3)

    def test_batch_processing(self):
        """Test that batch processing produces consistent results."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.eval()

        # Create data
        X = torch.randn(32, 1, 32, 32)

        # Process all at once
        with torch.no_grad():
            full_output = model(X)

        # Process in batches
        batch_outputs = []
        for i in range(0, 32, 8):
            with torch.no_grad():
                batch_out = model(X[i : i + 8])
            batch_outputs.append(batch_out)

        batched_output = torch.cat(batch_outputs, dim=0)

        # Results should be identical
        assert torch.allclose(full_output, batched_output, atol=1e-5)


# ==============================================================================
# METRICS INTEGRATION
# ==============================================================================
@pytest.mark.integration
class TestMetricsIntegration:
    """Integration tests for metrics with models."""

    def test_metrics_on_model_output(self):
        """Test calculating metrics on actual model outputs."""
        model = CNN(in_shape=(32, 32), out_size=5)
        model.eval()

        X = torch.randn(100, 1, 32, 32)
        y_true = torch.randn(100, 5)

        with torch.no_grad():
            y_pred = model(X)

        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()

        # Calculate all metrics
        r2_scores = calc_per_target_r2(y_true_np, y_pred_np)
        pearson = calc_pearson(y_true_np, y_pred_np)

        assert len(r2_scores) == 5
        assert isinstance(pearson, float)

    def test_metric_tracker_in_training_loop(self):
        """Test MetricTracker integration in training loop."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.train()

        X = torch.randn(32, 1, 32, 32)
        y = torch.randn(32, 3)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        loss_tracker = MetricTracker()

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            loss_tracker.update(loss.item(), n=batch_x.size(0))

        # Verify tracker accumulated correctly
        assert loss_tracker.count == 32
        assert loss_tracker.avg > 0


# ==============================================================================
# GRADIENT FLOW INTEGRATION
# ==============================================================================
@pytest.mark.integration
class TestGradientFlowIntegration:
    """Integration tests for gradient flow through models."""

    def test_gradient_accumulation(self):
        """Test gradient accumulation works correctly."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.train()

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        accumulation_steps = 4

        # Simulate gradient accumulation
        for i in range(accumulation_steps):
            X = torch.randn(4, 1, 32, 32)
            y = torch.randn(4, 3)

            output = model(X)
            loss = criterion(output, y) / accumulation_steps
            loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

        optimizer.step()
        optimizer.zero_grad()

    def test_gradient_clipping(self):
        """Test gradient clipping works correctly."""
        model = CNN(in_shape=(32, 32), out_size=3)
        model.train()

        criterion = nn.MSELoss()
        max_norm = 1.0

        X = torch.randn(8, 1, 32, 32)
        y = torch.randn(8, 3)

        output = model(X)
        loss = criterion(output, y)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Verify gradients are clipped
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        assert total_norm <= max_norm * 1.01  # Small tolerance
