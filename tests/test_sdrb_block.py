import torch
from ops.sdrb_block import SDRBBlock
from torch import nn


def test_sdrb_block_identity_residual_shape() -> None:
    block = SDRBBlock(in_channels=16, out_channels=16, stride=1)
    x = torch.randn(2, 16, 32, 32)

    out = block(x)

    assert out.shape == (2, 16, 32, 32)


def test_sdrb_block_projection_residual_shape() -> None:
    block = SDRBBlock(in_channels=16, out_channels=32, stride=2)
    x = torch.randn(2, 16, 32, 32)

    out = block(x)

    assert out.shape == (2, 32, 16, 16)


def test_sdrb_block_one_hot_gate_matches_depthwise_separable_reference() -> None:
    block = SDRBBlock(in_channels=16, out_channels=16, stride=1, dilations=(1, 2, 3))
    reference_dw = nn.Conv2d(
        16,
        16,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=16,
        bias=False,
    )
    reference_pw = nn.Conv2d(16, 16, kernel_size=1, bias=False)
    reference_bn = nn.BatchNorm2d(16)

    reference_dw.load_state_dict(block.branches[0].state_dict())
    reference_pw.load_state_dict(block.fuse.state_dict())
    reference_bn.load_state_dict(block.bn.state_dict())

    block.eval()
    reference_dw.eval()
    reference_pw.eval()
    reference_bn.eval()

    x = torch.randn(2, 16, 16, 16)
    gate_override = torch.tensor([1.0, 0.0, 0.0])

    out = block(x, gate_override=gate_override)
    reference = reference_bn(reference_pw(reference_dw(x))) + x

    torch.testing.assert_close(out, reference)


def test_sdrb_block_backward_gradients_are_finite() -> None:
    block = SDRBBlock(in_channels=8, out_channels=16, stride=2, use_gumbel=True)
    block.train()
    x = torch.randn(2, 8, 16, 16, requires_grad=True)

    out = block(x)
    loss = out.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_sdrb_block_return_gate_shape_and_normalization() -> None:
    block = SDRBBlock(in_channels=8, out_channels=8, stride=1)
    x = torch.randn(2, 8, 16, 16)

    out, gate = block(x, return_gate=True)

    assert out.shape == (2, 8, 16, 16)
    assert gate.shape == (2, 3, 16, 16)
    torch.testing.assert_close(gate.sum(dim=1), torch.ones(2, 16, 16))
