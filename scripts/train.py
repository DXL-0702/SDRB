"""Train CIFAR-100 ResNet-20 and SDRB-ResNet-20 experiments."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import build_cifar_model
from ops.gate import entropy_regularization_loss

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
STAGE_NAMES = ("stage2", "stage3", "stage4")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    root: str
    download: bool
    batch_size: int
    num_workers: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int
    stage_dilations: dict[str, tuple[int, int, int]] | None = None


@dataclass(frozen=True)
class GateConfig:
    temperature_start: float = 1.0
    temperature_end: float = 1.0
    entropy_weight: float = 0.0
    metric_batches: int = 0
    image_interval: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    log_interval: int
    eval_interval: int


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    momentum: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    eta_min: float


@dataclass(frozen=True)
class OutputConfig:
    runs_dir: str
    checkpoint_dir: str


@dataclass(frozen=True)
class TrainConfig:
    experiment: ExperimentConfig
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    train: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    output: OutputConfig
    gate: GateConfig | None = None


GateDict = dict[str, list[torch.Tensor]]


def resolve_repo_root() -> Path:
    """Return the SDRB repository root."""
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        msg = "PyYAML is required to load training configs; install requirements.txt first"
        raise RuntimeError(msg) from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        msg = f"config must be a YAML mapping: {path}"
        raise ValueError(msg)
    return raw


def _as_mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{section} must be a mapping"
        raise ValueError(msg)
    return value


def _require_keys(
    mapping: Mapping[str, Any],
    section: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = set() if optional is None else optional
    keys = set(mapping)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        msg = f"{section} missing required keys: {sorted(missing)}"
        raise ValueError(msg)
    if unknown:
        msg = f"{section} has unknown keys: {sorted(unknown)}"
        raise ValueError(msg)


def _as_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{field} must be a non-empty string"
        raise ValueError(msg)
    return value


def _as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{field} must be a boolean"
        raise ValueError(msg)
    return value


def _as_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{field} must be an integer"
        raise ValueError(msg)
    if minimum is not None and value < minimum:
        msg = f"{field} must be >= {minimum}"
        raise ValueError(msg)
    return value


def _as_float(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{field} must be a number"
        raise ValueError(msg)
    number = float(value)
    if minimum is not None and number < minimum:
        msg = f"{field} must be >= {minimum}"
        raise ValueError(msg)
    return number


def _parse_stage_dilations(value: Any) -> dict[str, tuple[int, int, int]]:
    mapping = _as_mapping(value, "model.stage_dilations")
    allowed = set(STAGE_NAMES)
    unknown = set(mapping) - allowed
    missing = allowed - set(mapping)
    if unknown:
        msg = f"model.stage_dilations has unknown stages: {sorted(unknown)}"
        raise ValueError(msg)
    if missing:
        msg = f"model.stage_dilations missing stages: {sorted(missing)}"
        raise ValueError(msg)

    parsed: dict[str, tuple[int, int, int]] = {}
    for stage_name in STAGE_NAMES:
        raw_dilations = mapping[stage_name]
        if not isinstance(raw_dilations, Sequence) or isinstance(raw_dilations, str):
            msg = f"model.stage_dilations.{stage_name} must be a sequence"
            raise ValueError(msg)
        if len(raw_dilations) != 3:
            msg = f"model.stage_dilations.{stage_name} must contain exactly three values"
            raise ValueError(msg)
        dilations = tuple(
            _as_int(dilation, f"model.stage_dilations.{stage_name}[{index}]", minimum=1)
            for index, dilation in enumerate(raw_dilations)
        )
        parsed[stage_name] = cast(tuple[int, int, int], dilations)
    return parsed


def load_config(path: Path) -> TrainConfig:
    """Load and strictly validate a Stage 2 YAML config."""
    raw = _load_yaml(path)
    required_top_level = {
        "experiment",
        "seed",
        "device",
        "data",
        "model",
        "train",
        "optimizer",
        "scheduler",
        "output",
    }
    _require_keys(raw, "config", required_top_level, optional={"gate"})

    experiment_raw = _as_mapping(raw["experiment"], "experiment")
    _require_keys(experiment_raw, "experiment", {"name"})
    experiment = ExperimentConfig(name=_as_str(experiment_raw["name"], "experiment.name"))

    data_raw = _as_mapping(raw["data"], "data")
    _require_keys(data_raw, "data", {"dataset", "root", "download", "batch_size", "num_workers"})
    data = DataConfig(
        dataset=_as_str(data_raw["dataset"], "data.dataset"),
        root=_as_str(data_raw["root"], "data.root"),
        download=_as_bool(data_raw["download"], "data.download"),
        batch_size=_as_int(data_raw["batch_size"], "data.batch_size", minimum=1),
        num_workers=_as_int(data_raw["num_workers"], "data.num_workers", minimum=0),
    )
    if data.dataset != "cifar100":
        msg = "data.dataset must be cifar100"
        raise ValueError(msg)

    model_raw = _as_mapping(raw["model"], "model")
    _require_keys(model_raw, "model", {"name", "num_classes"}, optional={"stage_dilations"})
    model_name = _as_str(model_raw["name"], "model.name")
    if model_name not in {"resnet20", "sdrb_resnet20"}:
        msg = "model.name must be resnet20 or sdrb_resnet20"
        raise ValueError(msg)
    stage_dilations = None
    if "stage_dilations" in model_raw:
        stage_dilations = _parse_stage_dilations(model_raw["stage_dilations"])
    model = ModelConfig(
        name=model_name,
        num_classes=_as_int(model_raw["num_classes"], "model.num_classes", minimum=1),
        stage_dilations=stage_dilations,
    )

    gate = None
    if "gate" in raw:
        gate_raw = _as_mapping(raw["gate"], "gate")
        _require_keys(
            gate_raw,
            "gate",
            {
                "temperature_start",
                "temperature_end",
                "entropy_weight",
                "metric_batches",
                "image_interval",
            },
        )
        gate = GateConfig(
            temperature_start=_as_float(
                gate_raw["temperature_start"],
                "gate.temperature_start",
                minimum=0.0,
            ),
            temperature_end=_as_float(
                gate_raw["temperature_end"],
                "gate.temperature_end",
                minimum=0.0,
            ),
            entropy_weight=_as_float(gate_raw["entropy_weight"], "gate.entropy_weight"),
            metric_batches=_as_int(gate_raw["metric_batches"], "gate.metric_batches", minimum=0),
            image_interval=_as_int(gate_raw["image_interval"], "gate.image_interval", minimum=0),
        )
        if gate.temperature_start <= 0 or gate.temperature_end <= 0:
            msg = "gate temperatures must be positive"
            raise ValueError(msg)

    train_raw = _as_mapping(raw["train"], "train")
    _require_keys(train_raw, "train", {"epochs", "log_interval", "eval_interval"})
    train = TrainingConfig(
        epochs=_as_int(train_raw["epochs"], "train.epochs", minimum=1),
        log_interval=_as_int(train_raw["log_interval"], "train.log_interval", minimum=1),
        eval_interval=_as_int(train_raw["eval_interval"], "train.eval_interval", minimum=1),
    )

    optimizer_raw = _as_mapping(raw["optimizer"], "optimizer")
    _require_keys(optimizer_raw, "optimizer", {"name", "lr", "momentum", "weight_decay"})
    optimizer_name = _as_str(optimizer_raw["name"], "optimizer.name")
    if optimizer_name != "sgd":
        msg = "optimizer.name must be sgd"
        raise ValueError(msg)
    optimizer = OptimizerConfig(
        name=optimizer_name,
        lr=_as_float(optimizer_raw["lr"], "optimizer.lr", minimum=0.0),
        momentum=_as_float(optimizer_raw["momentum"], "optimizer.momentum", minimum=0.0),
        weight_decay=_as_float(optimizer_raw["weight_decay"], "optimizer.weight_decay", minimum=0.0),
    )

    scheduler_raw = _as_mapping(raw["scheduler"], "scheduler")
    _require_keys(scheduler_raw, "scheduler", {"name", "eta_min"})
    scheduler_name = _as_str(scheduler_raw["name"], "scheduler.name")
    if scheduler_name != "cosine":
        msg = "scheduler.name must be cosine"
        raise ValueError(msg)
    scheduler = SchedulerConfig(
        name=scheduler_name,
        eta_min=_as_float(scheduler_raw["eta_min"], "scheduler.eta_min", minimum=0.0),
    )

    output_raw = _as_mapping(raw["output"], "output")
    _require_keys(output_raw, "output", {"runs_dir", "checkpoint_dir"})
    output = OutputConfig(
        runs_dir=_as_str(output_raw["runs_dir"], "output.runs_dir"),
        checkpoint_dir=_as_str(output_raw["checkpoint_dir"], "output.checkpoint_dir"),
    )

    return TrainConfig(
        experiment=experiment,
        seed=_as_int(raw["seed"], "seed", minimum=0),
        device=_as_str(raw["device"], "device"),
        data=data,
        model=model,
        gate=gate,
        train=train,
        optimizer=optimizer,
        scheduler=scheduler,
        output=output,
    )


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve auto/mps/cpu/cuda device strings."""
    if device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        msg = "CUDA was requested but is not available"
        raise RuntimeError(msg)
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        msg = "MPS was requested but is not available"
        raise RuntimeError(msg)
    return resolved


def _resolve_path(path: str, repo_root: Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return repo_root / resolved


def _cifar100_transforms() -> tuple[Any, Any]:
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return train_transform, test_transform


def build_dataloaders(config: TrainConfig) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build CIFAR-100 train/test dataloaders."""
    from torchvision import datasets

    repo_root = resolve_repo_root()
    data_root = _resolve_path(config.data.root, repo_root)
    train_transform, test_transform = _cifar100_transforms()
    train_dataset = datasets.CIFAR100(
        root=str(data_root),
        train=True,
        download=config.data.download,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=str(data_root),
        train=False,
        download=config.data.download,
        transform=test_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


def build_test_dataloader(config: TrainConfig) -> DataLoader[Any]:
    """Build only the CIFAR-100 test dataloader."""
    from torchvision import datasets

    repo_root = resolve_repo_root()
    data_root = _resolve_path(config.data.root, repo_root)
    _, test_transform = _cifar100_transforms()
    test_dataset = datasets.CIFAR100(
        root=str(data_root),
        train=False,
        download=config.data.download,
        transform=test_transform,
    )
    return DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=False,
    )


def build_model(config: TrainConfig) -> nn.Module:
    """Build the configured CIFAR model."""
    kwargs: dict[str, Any] = {}
    if config.model.name == "sdrb_resnet20":
        kwargs["stage_dilations"] = config.model.stage_dilations
        kwargs["use_gumbel"] = True
        if config.gate is not None:
            kwargs["temperature"] = config.gate.temperature_start
    return build_cifar_model(config.model.name, num_classes=config.model.num_classes, **kwargs)


def build_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Build the SGD optimizer."""
    return torch.optim.SGD(
        model.parameters(),
        lr=config.optimizer.lr,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
    )


def build_scheduler(
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Build the cosine LR scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=config.scheduler.eta_min,
    )


def temperature_at_epoch(epoch: int, epochs: int, start: float, end: float) -> float:
    """Linearly anneal gate temperature from start to end."""
    if epochs <= 1:
        return start
    progress = (epoch - 1) / (epochs - 1)
    return start + progress * (end - start)


def compute_top1(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Return top-1 accuracy in percent for one batch."""
    predictions = logits.argmax(dim=1)
    return float(predictions.eq(target).float().mean().item() * 100.0)


def _count_correct(logits: torch.Tensor, target: torch.Tensor) -> int:
    return int(logits.argmax(dim=1).eq(target).sum().item())


def _empty_gate_dict() -> GateDict:
    return {stage_name: [] for stage_name in STAGE_NAMES}


def _collect_gate_samples(samples: GateDict, gates_by_stage: GateDict) -> None:
    for stage_name in STAGE_NAMES:
        samples[stage_name].extend(gate.detach().cpu() for gate in gates_by_stage.get(stage_name, []))


def _mean_entropy_regularization(gates_by_stage: GateDict) -> torch.Tensor:
    losses = [
        entropy_regularization_loss(gate)
        for gates in gates_by_stage.values()
        for gate in gates
    ]
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def summarize_gate_metrics(gates_by_stage: GateDict) -> dict[str, float]:
    """Summarize SDRB gate tensors into scalar metrics."""
    metrics: dict[str, float] = {}
    for stage_name in STAGE_NAMES:
        gates = gates_by_stage.get(stage_name, [])
        if not gates:
            continue
        weights = torch.cat([gate.detach().float().cpu() for gate in gates], dim=0)
        num_branches = weights.shape[1]
        safe_weights = weights.clamp_min(1e-8)
        entropy = -(safe_weights * safe_weights.log()).sum(dim=1).mean()
        metrics[f"gate/{stage_name}/entropy_norm"] = float(
            (entropy / math.log(num_branches)).item()
        )
        branch_prob = weights.mean(dim=(0, 2, 3))
        argmax = weights.argmax(dim=1)
        spatial_std = weights.std(dim=(2, 3), unbiased=False).mean(dim=0)
        for branch_index in range(num_branches):
            metrics[f"gate/{stage_name}/prob_branch{branch_index}"] = float(
                branch_prob[branch_index].item()
            )
            metrics[f"gate/{stage_name}/argmax_frac_branch{branch_index}"] = float(
                argmax.eq(branch_index).float().mean().item()
            )
            metrics[f"gate/{stage_name}/spatial_std_branch{branch_index}"] = float(
                spatial_std[branch_index].item()
            )
    return metrics


def _model_is_sdrb(model: nn.Module) -> bool:
    return bool(getattr(model, "is_sdrb", False))


def _forward_sdrb(
    model: nn.Module,
    images: torch.Tensor,
    temperature: float | None,
) -> tuple[torch.Tensor, GateDict]:
    output = model(images, return_gates=True, temperature=temperature)
    logits, gates_by_stage = cast(tuple[torch.Tensor, GateDict], output)
    return logits, gates_by_stage


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    gate_config: GateConfig | None = None,
    log_interval: int = 50,
    writer: Any | None = None,
) -> dict[str, float]:
    """Train one epoch and return aggregate metrics."""
    model.train()
    is_sdrb = _model_is_sdrb(model)
    temperature = None
    if is_sdrb and gate_config is not None:
        temperature = temperature_at_epoch(
            epoch,
            epochs,
            gate_config.temperature_start,
            gate_config.temperature_end,
        )

    total_loss = 0.0
    total_ce_loss = 0.0
    total_entropy_loss = 0.0
    total_correct = 0
    total_samples = 0
    entropy_batches = 0
    gate_samples = _empty_gate_dict()

    for batch_index, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        if is_sdrb:
            logits, gates_by_stage = _forward_sdrb(model, images, temperature)
            ce_loss = criterion(logits, targets)
            entropy_loss = _mean_entropy_regularization(gates_by_stage)
            entropy_weight = 0.0 if gate_config is None else gate_config.entropy_weight
            loss = ce_loss + entropy_weight * entropy_loss
            if gate_config is not None and batch_index <= gate_config.metric_batches:
                _collect_gate_samples(gate_samples, gates_by_stage)
            total_entropy_loss += float(entropy_loss.item()) * images.size(0)
            entropy_batches += images.size(0)
        else:
            logits = model(images)
            ce_loss = criterion(logits, targets)
            loss = ce_loss

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_ce_loss += float(ce_loss.item()) * batch_size
        total_correct += _count_correct(logits, targets)
        total_samples += batch_size

        if writer is not None and batch_index % log_interval == 0:
            global_step = (epoch - 1) * len(dataloader) + batch_index
            writer.add_scalar("train/batch_loss", float(loss.item()), global_step)

    if total_samples == 0:
        msg = "dataloader produced no samples"
        raise ValueError(msg)

    metrics = {
        "loss": total_loss / total_samples,
        "ce_loss": total_ce_loss / total_samples,
        "top1": 100.0 * total_correct / total_samples,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    if entropy_batches > 0:
        metrics["entropy_loss"] = total_entropy_loss / entropy_batches
    if temperature is not None:
        metrics["gate/temperature"] = temperature
    metrics.update(summarize_gate_metrics(gate_samples))
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    gate_config: GateConfig | None = None,
    temperature: float | None = None,
) -> dict[str, float]:
    """Evaluate a model and return aggregate metrics."""
    model.eval()
    is_sdrb = _model_is_sdrb(model)
    collect_gates = is_sdrb and gate_config is not None and gate_config.metric_batches > 0
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gate_samples = _empty_gate_dict()

    for batch_index, (images, targets) in enumerate(dataloader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        if is_sdrb and collect_gates:
            logits, gates_by_stage = _forward_sdrb(model, images, temperature)
            if gate_config is not None and batch_index <= gate_config.metric_batches:
                _collect_gate_samples(gate_samples, gates_by_stage)
        else:
            logits = model(images)

        loss = criterion(logits, targets)
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += _count_correct(logits, targets)
        total_samples += batch_size

    if total_samples == 0:
        msg = "dataloader produced no samples"
        raise ValueError(msg)

    metrics = {
        "loss": total_loss / total_samples,
        "top1": 100.0 * total_correct / total_samples,
    }
    metrics.update(summarize_gate_metrics(gate_samples))
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_top1: float,
    config: TrainConfig,
    metrics: Mapping[str, Any],
) -> None:
    """Save model, optimizer, scheduler, config, and metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_top1": best_top1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": asdict(config),
            "metrics": dict(metrics),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint into model and optionally optimizer/scheduler."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return cast(dict[str, Any], checkpoint)


def _log_scalars(writer: Any, prefix: str, metrics: Mapping[str, float], epoch: int) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", value, epoch)


def _append_metrics(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _apply_overrides(
    config: TrainConfig,
    epochs: int | None,
    num_workers: int | None,
    run_name: str | None,
    device: str | None,
) -> TrainConfig:
    updated = config
    if epochs is not None:
        updated = replace(updated, train=replace(updated.train, epochs=epochs))
    if num_workers is not None:
        updated = replace(updated, data=replace(updated.data, num_workers=num_workers))
    if run_name is not None:
        updated = replace(updated, experiment=replace(updated.experiment, name=run_name))
    if device is not None:
        updated = replace(updated, device=device)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = _apply_overrides(
        config,
        epochs=args.epochs,
        num_workers=args.num_workers,
        run_name=args.run_name,
        device=args.device,
    )
    set_seed(config.seed)
    device = resolve_device(config.device)
    train_loader, test_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    repo_root = resolve_repo_root()
    run_dir = _resolve_path(config.output.runs_dir, repo_root) / config.experiment.name
    checkpoint_dir = _resolve_path(config.output.checkpoint_dir, repo_root) / config.experiment.name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(run_dir))
    start_epoch = 1
    best_top1 = 0.0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_top1 = float(checkpoint.get("best_top1", best_top1))

    try:
        for epoch in range(start_epoch, config.train.epochs + 1):
            temperature = None
            if _model_is_sdrb(model) and config.gate is not None:
                temperature = temperature_at_epoch(
                    epoch,
                    config.train.epochs,
                    config.gate.temperature_start,
                    config.gate.temperature_end,
                )
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                config.train.epochs,
                gate_config=config.gate,
                log_interval=config.train.log_interval,
                writer=writer,
            )
            scheduler.step()
            _log_scalars(writer, "train", train_metrics, epoch)

            eval_metrics: dict[str, float] = {}
            if epoch % config.train.eval_interval == 0:
                eval_metrics = evaluate(
                    model,
                    test_loader,
                    criterion,
                    device,
                    gate_config=config.gate,
                    temperature=temperature,
                )
                _log_scalars(writer, "eval", eval_metrics, epoch)
                if eval_metrics["top1"] > best_top1:
                    best_top1 = eval_metrics["top1"]
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best_top1,
                        config,
                        {"train": train_metrics, "eval": eval_metrics},
                    )

            save_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_top1,
                config,
                {"train": train_metrics, "eval": eval_metrics},
            )
            record = {
                "epoch": epoch,
                "best_top1": best_top1,
                "train": train_metrics,
                "eval": eval_metrics,
            }
            _append_metrics(metrics_path, record)
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "train_top1": train_metrics["top1"],
                        "eval_top1": eval_metrics.get("top1"),
                        "best_top1": best_top1,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
