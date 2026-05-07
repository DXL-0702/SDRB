"""CIFAR ResNet-20 and SDRB-ResNet-20 model definitions."""

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ops.sdrb_block import SDRBBlock

StageName = Literal["stage2", "stage3", "stage4"]
GateDict = dict[str, list[torch.Tensor]]

DEFAULT_STAGE_DILATIONS: dict[StageName, tuple[int, int, int]] = {
    "stage2": (1, 1, 2),
    "stage3": (1, 2, 3),
    "stage4": (1, 2, 5),
}


class CifarBasicBlock(nn.Module):
    """Post-activation BasicBlock for CIFAR-style ResNet."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            msg = "in_channels must be positive"
            raise ValueError(msg)
        if out_channels <= 0:
            msg = "out_channels must be positive"
            raise ValueError(msg)
        if stride <= 0:
            msg = "stride must be positive"
            raise ValueError(msg)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_layer(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CifarSDRBBasicBlock(nn.Module):
    """SDRB-backed post-activation block for CIFAR-style ResNet."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilations: Sequence[int] = (1, 1, 2),
        temperature: float = 1.0,
        use_gumbel: bool = False,
        hard_gate: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        gate_bias_init: Sequence[float] | None = (1.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        self.sdrb = SDRBBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilations=dilations,
            temperature=temperature,
            use_gumbel=use_gumbel,
            hard_gate=hard_gate,
            norm_layer=norm_layer,
            gate_bias_init=gate_bias_init,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False,
        temperature: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.sdrb(x, return_gate=return_gate, temperature=temperature)
        if return_gate:
            features, gate = out
            return F.relu(features, inplace=True), gate
        return F.relu(out, inplace=True)


class CifarResNet(nn.Module):
    """CIFAR ResNet scaffold shared by baseline and SDRB variants."""

    def __init__(
        self,
        block: type[CifarBasicBlock] | type[CifarSDRBBasicBlock],
        num_blocks: Sequence[int],
        num_classes: int = 100,
        stage_dilations: Mapping[str, Sequence[int]] | None = None,
        temperature: float = 1.0,
        use_gumbel: bool = False,
        hard_gate: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 3:
            msg = "num_blocks must contain exactly three stage depths"
            raise ValueError(msg)
        if any(depth <= 0 for depth in num_blocks):
            msg = "all stage depths must be positive"
            raise ValueError(msg)
        if num_classes <= 0:
            msg = "num_classes must be positive"
            raise ValueError(msg)

        self.in_channels = 16
        self.is_sdrb = issubclass(block, CifarSDRBBasicBlock)
        resolved_dilations = self._resolve_stage_dilations(stage_dilations)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(16)
        self.stage2 = self._make_stage(
            block,
            out_channels=16,
            num_blocks=num_blocks[0],
            stride=1,
            stage_name="stage2",
            dilations=resolved_dilations["stage2"],
            temperature=temperature,
            use_gumbel=use_gumbel,
            hard_gate=hard_gate,
            norm_layer=norm_layer,
        )
        self.stage3 = self._make_stage(
            block,
            out_channels=32,
            num_blocks=num_blocks[1],
            stride=2,
            stage_name="stage3",
            dilations=resolved_dilations["stage3"],
            temperature=temperature,
            use_gumbel=use_gumbel,
            hard_gate=hard_gate,
            norm_layer=norm_layer,
        )
        self.stage4 = self._make_stage(
            block,
            out_channels=64,
            num_blocks=num_blocks[2],
            stride=2,
            stage_name="stage4",
            dilations=resolved_dilations["stage4"],
            temperature=temperature,
            use_gumbel=use_gumbel,
            hard_gate=hard_gate,
            norm_layer=norm_layer,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._init_weights()

    @staticmethod
    def _resolve_stage_dilations(
        stage_dilations: Mapping[str, Sequence[int]] | None,
    ) -> dict[StageName, tuple[int, int, int]]:
        if stage_dilations is None:
            return dict(DEFAULT_STAGE_DILATIONS)

        resolved = dict(DEFAULT_STAGE_DILATIONS)
        unknown = set(stage_dilations) - set(resolved)
        if unknown:
            msg = f"unknown stage dilation keys: {sorted(unknown)}"
            raise ValueError(msg)

        for stage_name, dilations in stage_dilations.items():
            if len(dilations) != 3:
                msg = f"{stage_name} dilations must contain exactly three values"
                raise ValueError(msg)
            if any(dilation <= 0 for dilation in dilations):
                msg = f"{stage_name} dilations must be positive"
                raise ValueError(msg)
            resolved[stage_name] = tuple(int(dilation) for dilation in dilations)
        return resolved

    def _make_stage(
        self,
        block: type[CifarBasicBlock] | type[CifarSDRBBasicBlock],
        out_channels: int,
        num_blocks: int,
        stride: int,
        stage_name: StageName,
        dilations: Sequence[int],
        temperature: float,
        use_gumbel: bool,
        hard_gate: bool,
        norm_layer: type[nn.Module],
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        for block_index in range(num_blocks):
            block_stride = stride if block_index == 0 else 1
            if issubclass(block, CifarSDRBBasicBlock):
                layers.append(
                    block(
                        self.in_channels,
                        out_channels,
                        stride=block_stride,
                        dilations=dilations,
                        temperature=temperature,
                        use_gumbel=use_gumbel,
                        hard_gate=hard_gate,
                        norm_layer=norm_layer,
                    )
                )
            else:
                layers.append(
                    block(
                        self.in_channels,
                        out_channels,
                        stride=block_stride,
                        norm_layer=norm_layer,
                    )
                )
            self.in_channels = out_channels * block.expansion
        return layers

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def _forward_stage(
        self,
        x: torch.Tensor,
        stage: nn.ModuleList,
        stage_name: StageName,
        gates_by_stage: GateDict,
        return_gates: bool,
        temperature: float | None,
    ) -> torch.Tensor:
        out = x
        for block in stage:
            if return_gates and isinstance(block, CifarSDRBBasicBlock):
                out_with_gate = block(out, return_gate=True, temperature=temperature)
                out, gate = out_with_gate
                gates_by_stage[stage_name].append(gate)
            else:
                out = block(out)
        return out

    def forward(
        self,
        x: torch.Tensor,
        return_gates: bool = False,
        temperature: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, GateDict]:
        if return_gates and not self.is_sdrb:
            msg = "return_gates=True is only supported for SDRB models"
            raise ValueError(msg)

        gates_by_stage: GateDict = {"stage2": [], "stage3": [], "stage4": []}
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self._forward_stage(
            out,
            self.stage2,
            "stage2",
            gates_by_stage,
            return_gates,
            temperature,
        )
        out = self._forward_stage(
            out,
            self.stage3,
            "stage3",
            gates_by_stage,
            return_gates,
            temperature,
        )
        out = self._forward_stage(
            out,
            self.stage4,
            "stage4",
            gates_by_stage,
            return_gates,
            temperature,
        )
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        logits = self.fc(out)

        if return_gates:
            return logits, gates_by_stage
        return logits


def cifar_resnet20(num_classes: int = 100) -> CifarResNet:
    """Build CIFAR ResNet-20 baseline."""
    return CifarResNet(CifarBasicBlock, [3, 3, 3], num_classes=num_classes)


def cifar_sdrb_resnet20(
    num_classes: int = 100,
    stage_dilations: Mapping[str, Sequence[int]] | None = None,
    temperature: float = 1.0,
    use_gumbel: bool = False,
    hard_gate: bool = False,
) -> CifarResNet:
    """Build CIFAR SDRB-ResNet-20."""
    return CifarResNet(
        CifarSDRBBasicBlock,
        [3, 3, 3],
        num_classes=num_classes,
        stage_dilations=stage_dilations,
        temperature=temperature,
        use_gumbel=use_gumbel,
        hard_gate=hard_gate,
    )


def build_cifar_model(name: str, num_classes: int = 100, **kwargs: object) -> CifarResNet:
    """Build a CIFAR model by registry name."""
    normalized_name = name.lower()
    if normalized_name == "resnet20":
        return cifar_resnet20(num_classes=num_classes)
    if normalized_name == "sdrb_resnet20":
        return cifar_sdrb_resnet20(num_classes=num_classes, **kwargs)

    msg = f"unknown CIFAR model: {name}"
    raise ValueError(msg)
