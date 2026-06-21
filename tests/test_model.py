"""Tests for the value network."""

import torch

from causal_chess.model import ValueNetwork


class TestValueNetwork:
    """Test ValueNetwork forward pass and properties."""

    def test_output_shape(self) -> None:
        model = ValueNetwork()
        x = torch.randn(1, 15, 8, 8)
        out = model(x)
        assert out.shape == (1, 1)

    def test_batch_output_shape(self) -> None:
        model = ValueNetwork()
        x = torch.randn(8, 15, 8, 8)
        out = model(x)
        assert out.shape == (8, 1)

    def test_output_range(self) -> None:
        """Output should be in [0, 1] due to Sigmoid."""
        model = ValueNetwork()
        # Test with various inputs including extreme values
        for _ in range(10):
            x = torch.randn(4, 15, 8, 8) * 10
            out = model(x)
            assert (out >= 0.0).all(), f"Output below 0: {out.min()}"
            assert (out <= 1.0).all(), f"Output above 1: {out.max()}"

    def test_gradient_flows(self) -> None:
        """A backward pass should not error and should produce gradients."""
        model = ValueNetwork()
        x = torch.randn(1, 15, 8, 8)
        out = model(x)
        loss = out.sum()
        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients were computed"

    def test_param_count_reasonable(self) -> None:
        """Model should be small (< 200K params)."""
        model = ValueNetwork()
        count = model.param_count()
        assert count < 200_000, f"Model too large: {count} params"
        assert count > 10_000, f"Model too small: {count} params"

    def test_deterministic_eval_mode(self) -> None:
        """Same input should produce same output in eval mode."""
        model = ValueNetwork()
        model.eval()
        x = torch.randn(1, 15, 8, 8)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)
