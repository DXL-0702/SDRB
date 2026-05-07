"""Per-pixel scale gate for SDRB operators."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ScaleGate(nn.Module):
    """Generate per-pixel scale-routing weights over dilation branches."""

    def __init__(
        self,
        channels: int,
        num_scales: int = 3,
        stride: int = 1,
        temperature: float = 1.0,
        use_gumbel: bool = False,
        hard: bool = False,
        bias_init: Sequence[float] | None = (1.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        if channels <= 0:
            msg = "channels must be positive"
            raise ValueError(msg)
        if num_scales <= 0:
            msg = "num_scales must be positive"
            raise ValueError(msg)
        if stride <= 0:
            msg = "stride must be positive"
            raise ValueError(msg)
        if temperature <= 0:
            msg = "temperature must be positive"
            raise ValueError(msg)
        if bias_init is not None and len(bias_init) != num_scales:
            msg = "bias_init length must match num_scales"
            raise ValueError(msg)

        self.channels = channels
        self.num_scales = num_scales
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.hard = hard

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pwconv = nn.Conv2d(channels, num_scales, kernel_size=1, bias=True)

        if bias_init is not None:
            bias = self.pwconv.bias
            assert bias is not None
            with torch.no_grad():
                bias.copy_(torch.tensor(bias_init, dtype=bias.dtype, device=bias.device))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized scale-routing logits."""
        return self.pwconv(self.dwconv(x))

    def forward(self, x: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
        """Return scale weights normalized along the scale dimension."""
        tau = self.temperature if temperature is None else temperature
        if tau <= 0:
            msg = "temperature must be positive"
            raise ValueError(msg)

        logits = self.logits(x)
        if self.training and self.use_gumbel:
            return F.gumbel_softmax(logits, tau=tau, hard=self.hard, dim=1)
        return F.softmax(logits / tau, dim=1)


def gate_entropy(weights: torch.Tensor, eps: float = 1e-8, normalize: bool = True) -> torch.Tensor:
    """Return positive entropy of scale weights."""
    safe_weights = weights.clamp_min(eps)
    entropy = -(safe_weights * safe_weights.log()).sum(dim=1)
    if normalize:
        entropy = entropy.mean()
    return entropy


def entropy_regularization_loss(
    weights: torch.Tensor,
    eps: float = 1e-8,
    normalize: bool = True,
) -> torch.Tensor:
    """Return negative entropy so minimizing it encourages high-entropy gates."""
    return -gate_entropy(weights, eps=eps, normalize=normalize)
