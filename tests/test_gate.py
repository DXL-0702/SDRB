import pytest
import torch
from ops.gate import ScaleGate, entropy_regularization_loss, gate_entropy


def test_scale_gate_output_shape() -> None:
    gate = ScaleGate(channels=8, num_scales=3)
    x = torch.randn(2, 8, 16, 16)

    weights = gate(x)

    assert weights.shape == (2, 3, 16, 16)


def test_scale_gate_weights_sum_to_one() -> None:
    gate = ScaleGate(channels=8, num_scales=3)
    x = torch.randn(2, 8, 16, 16)

    weights = gate(x)

    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, 16, 16))


def test_scale_gate_bias_init_prefers_first_branch() -> None:
    gate = ScaleGate(channels=8, num_scales=3, bias_init=(1.0, 0.0, 0.0))
    x = torch.zeros(4, 8, 8, 8)

    weights = gate(x)
    branch_means = weights.mean(dim=(0, 2, 3))

    assert branch_means[0] > branch_means[1]
    assert branch_means[0] > branch_means[2]


def test_gate_entropy_and_regularization_are_finite() -> None:
    gate = ScaleGate(channels=8, num_scales=3)
    x = torch.randn(2, 8, 16, 16)
    weights = gate(x)

    entropy = gate_entropy(weights)
    regularization = entropy_regularization_loss(weights)

    assert torch.isfinite(entropy)
    assert torch.isfinite(regularization)
    torch.testing.assert_close(regularization, -entropy)


def test_scale_gate_invalid_temperature_raises() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        ScaleGate(channels=8, temperature=0.0)

    gate = ScaleGate(channels=8)
    x = torch.randn(2, 8, 16, 16)
    with pytest.raises(ValueError, match="temperature must be positive"):
        gate(x, temperature=0.0)


def test_scale_gate_gumbel_path_shape_finite_and_normalized() -> None:
    gate = ScaleGate(channels=8, num_scales=3, use_gumbel=True, hard=True)
    gate.train()
    x = torch.randn(2, 8, 16, 16)

    weights = gate(x)

    assert weights.shape == (2, 3, 16, 16)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, 16, 16))
