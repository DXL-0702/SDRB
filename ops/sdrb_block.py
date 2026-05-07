"""Scale-decoupled residual block operator."""

from collections.abc import Sequence

import torch
from torch import nn

from ops.gate import ScaleGate


class SDRBBlock(nn.Module):
    """Depthwise multi-dilation residual block with per-pixel scale routing."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        stride: int = 1,
        dilations: Sequence[int] = (1, 1, 2),
        temperature: float = 1.0,
        use_gumbel: bool = False,
        hard_gate: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        gate_bias_init: Sequence[float] | None = (1.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            msg = "in_channels must be positive"
            raise ValueError(msg)
        out_channels = in_channels if out_channels is None else out_channels
        if out_channels <= 0:
            msg = "out_channels must be positive"
            raise ValueError(msg)
        if stride <= 0:
            msg = "stride must be positive"
            raise ValueError(msg)
        if len(dilations) != 3:
            msg = "dilations must contain exactly three values"
            raise ValueError(msg)
        if any(dilation <= 0 for dilation in dilations):
            msg = "dilations must be positive"
            raise ValueError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilations = tuple(dilations)

        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    dilation=dilation,
                    groups=in_channels,
                    bias=False,
                )
                for dilation in self.dilations
            ]
        )
        self.gate = ScaleGate(
            channels=in_channels,
            num_scales=len(self.dilations),
            stride=stride,
            temperature=temperature,
            use_gumbel=use_gumbel,
            hard=hard_gate,
            bias_init=gate_bias_init,
        )
        self.fuse = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = norm_layer(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.projection: nn.Module | None = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels),
            )
        else:
            self.projection = None

    def _resolve_gate_override(
        self,
        gate_override: torch.Tensor,
        branch_stack: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_scales, _, height, width = branch_stack.shape
        weights = gate_override.to(device=branch_stack.device, dtype=branch_stack.dtype)

        if weights.ndim == 1:
            if weights.shape[0] != num_scales:
                msg = "gate_override scale dimension must match number of branches"
                raise ValueError(msg)
            weights = weights.view(1, num_scales, 1, 1)
        elif weights.ndim != 4:
            msg = "gate_override must have shape [K] or [N, K, H, W]"
            raise ValueError(msg)

        if weights.shape[1] != num_scales:
            msg = "gate_override scale dimension must match number of branches"
            raise ValueError(msg)
        if weights.shape[0] not in (1, batch_size):
            msg = "gate_override batch dimension must be 1 or match input batch size"
            raise ValueError(msg)
        if weights.shape[2] not in (1, height) or weights.shape[3] not in (1, width):
            msg = "gate_override spatial dimensions must be 1 or match branch output"
            raise ValueError(msg)

        return weights.expand(batch_size, num_scales, height, width)

    def forward(
        self,
        x: torch.Tensor,
        gate_override: torch.Tensor | None = None,
        return_gate: bool = False,
        temperature: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply SDRB and optionally return per-pixel gate weights.

        gate_override is intended for one-hot branch-selection tests, ablations,
        and visualization probes. It accepts either [K] or [N, K, H, W].
        """
        branch_stack = torch.stack([branch(x) for branch in self.branches], dim=1)
        if gate_override is None:
            gate_weights = self.gate(x, temperature=temperature)
        else:
            gate_weights = self._resolve_gate_override(gate_override, branch_stack)

        weighted = (branch_stack * gate_weights.unsqueeze(2)).sum(dim=1)
        out = self.bn(self.fuse(weighted))
        residual = x if self.projection is None else self.projection(x)
        out = out + residual

        if return_gate:
            return out, gate_weights
        return out
