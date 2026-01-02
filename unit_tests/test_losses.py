"""
Unit Tests for Loss Functions
=============================

Tests for the loss function factory and custom loss implementations.

Author: Ductho Le (ductho.le@outlook.com)
"""

import os
import sys

import pytest
import torch
import torch.nn as nn


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavedl.utils.losses import LogCoshLoss, WeightedMSELoss, get_loss, list_losses


class TestListLosses:
    """Tests for list_losses function."""

    def test_returns_list(self):
        """list_losses should return a list."""
        result = list_losses()
        assert isinstance(result, list)

    def test_contains_expected_losses(self):
        """list_losses should contain all expected loss names."""
        result = list_losses()
        expected = ["mse", "mae", "huber", "smooth_l1", "log_cosh", "weighted_mse"]
        for loss_name in expected:
            assert loss_name in result


class TestGetLoss:
    """Tests for get_loss factory function."""

    @pytest.mark.parametrize(
        "loss_name", ["mse", "mae", "huber", "smooth_l1", "log_cosh", "weighted_mse"]
    )
    def test_all_losses_instantiate(self, loss_name):
        """All registered losses should instantiate without error."""
        loss = get_loss(loss_name)
        assert isinstance(loss, nn.Module)

    def test_mse_loss(self):
        """MSE loss should work correctly."""
        loss = get_loss("mse")
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert loss(pred, target).item() == 0.0

    def test_mae_loss(self):
        """MAE loss should work correctly."""
        loss = get_loss("mae")
        pred = torch.tensor([[2.0], [4.0]])
        target = torch.tensor([[1.0], [2.0]])
        expected = 1.5  # (1 + 2) / 2
        assert abs(loss(pred, target).item() - expected) < 1e-5

    def test_huber_loss_with_delta(self):
        """Huber loss should accept delta parameter."""
        loss = get_loss("huber", delta=0.5)
        assert isinstance(loss, nn.HuberLoss)
        assert loss.delta == 0.5

    def test_weighted_mse_with_weights(self):
        """Weighted MSE should accept weights parameter."""
        weights = [2.0, 1.0, 1.0]
        loss = get_loss("weighted_mse", weights=weights)
        assert isinstance(loss, WeightedMSELoss)

    def test_unknown_loss_raises_error(self):
        """Unknown loss name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss function"):
            get_loss("unknown_loss")

    def test_case_insensitive(self):
        """Loss names should be case insensitive."""
        loss1 = get_loss("MSE")
        loss2 = get_loss("mse")
        assert type(loss1) == type(loss2)


class TestLogCoshLoss:
    """Tests for LogCoshLoss custom implementation."""

    def test_instantiation(self):
        """LogCoshLoss should instantiate."""
        loss = LogCoshLoss()
        assert isinstance(loss, nn.Module)

    def test_zero_error(self):
        """LogCoshLoss should be near zero for identical inputs."""
        loss = LogCoshLoss()
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])
        result = loss(pred, target)
        assert result.item() < 1e-5

    def test_positive_output(self):
        """LogCoshLoss should always be non-negative."""
        loss = LogCoshLoss()
        pred = torch.randn(10, 5)
        target = torch.randn(10, 5)
        result = loss(pred, target)
        assert result.item() >= 0

    def test_reduction_modes(self):
        """LogCoshLoss should support different reduction modes."""
        pred = torch.randn(4, 3)
        target = torch.randn(4, 3)

        loss_none = LogCoshLoss(reduction="none")
        result_none = loss_none(pred, target)
        assert result_none.shape == (4, 3)

        loss_sum = LogCoshLoss(reduction="sum")
        result_sum = loss_sum(pred, target)
        assert result_sum.shape == ()

        loss_mean = LogCoshLoss(reduction="mean")
        result_mean = loss_mean(pred, target)
        assert result_mean.shape == ()

    def test_invalid_reduction_raises(self):
        """Invalid reduction mode should raise ValueError."""
        with pytest.raises(ValueError):
            LogCoshLoss(reduction="invalid")


class TestWeightedMSELoss:
    """Tests for WeightedMSELoss custom implementation."""

    def test_instantiation(self):
        """WeightedMSELoss should instantiate."""
        loss = WeightedMSELoss()
        assert isinstance(loss, nn.Module)

    def test_without_weights_equals_mse(self):
        """WeightedMSELoss without weights should equal MSE."""
        pred = torch.randn(10, 3)
        target = torch.randn(10, 3)

        weighted_loss = WeightedMSELoss()
        mse_loss = nn.MSELoss()

        result_weighted = weighted_loss(pred, target)
        result_mse = mse_loss(pred, target)

        assert torch.allclose(result_weighted, result_mse, atol=1e-5)

    def test_weights_affect_loss(self):
        """Weights should affect the loss calculation."""
        pred = torch.tensor([[0.0, 0.0]])
        target = torch.tensor([[1.0, 1.0]])

        # Equal weights
        loss_equal = WeightedMSELoss(weights=[1.0, 1.0])
        result_equal = loss_equal(pred, target)

        # Unequal weights - weight first target more
        loss_unequal = WeightedMSELoss(weights=[2.0, 1.0])
        result_unequal = loss_unequal(pred, target)

        # With weights [2, 1], loss = (2*1 + 1*1)/2 = 1.5
        # With weights [1, 1], loss = (1*1 + 1*1)/2 = 1.0
        assert result_unequal.item() > result_equal.item()

    def test_weights_tensor_input(self):
        """Weights can be provided as tensor."""
        weights = torch.tensor([1.0, 2.0, 3.0])
        loss = WeightedMSELoss(weights=weights)

        pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        result = loss(pred, target)

        assert result.shape == ()


class TestLossGradients:
    """Tests for gradient computation through loss functions."""

    @pytest.mark.parametrize(
        "loss_name", ["mse", "mae", "huber", "log_cosh", "weighted_mse"]
    )
    def test_gradients_flow(self, loss_name):
        """Gradients should flow through all loss functions."""
        loss = get_loss(loss_name)

        pred = torch.randn(4, 3, requires_grad=True)
        target = torch.randn(4, 3)

        result = loss(pred, target)
        result.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape
