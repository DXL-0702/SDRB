"""Evaluate CIFAR-100 checkpoints for Stage 2 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts.train import (
    build_model,
    build_test_dataloader,
    evaluate,
    load_checkpoint,
    load_config,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config = config.__class__(
            experiment=config.experiment,
            seed=config.seed,
            device=args.device,
            data=config.data,
            model=config.model,
            train=config.train,
            optimizer=config.optimizer,
            scheduler=config.scheduler,
            output=config.output,
            gate=config.gate,
        )
    device = resolve_device(config.device)
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    test_loader = build_test_dataloader(config)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device, gate_config=config.gate)
    summary: dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "model": config.model.name,
        "epoch": checkpoint.get("epoch"),
        "best_top1": checkpoint.get("best_top1"),
        "metrics": metrics,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
