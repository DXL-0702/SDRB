<p align="center">
   <a href="./README.zh-CN.md">中文文档</a>
</p>


# SDRB: Scale-Decoupled Residual Block

SDRB is a research-oriented PyTorch prototype for a scale-aware residual block. The long-term goal is to design a plug-in replacement block for the ResNet family that supports pretrained-weight transfer and improves small-object-friendly backbone representations.

This repository is currently in the **M4 local prototype stage**. The focus is not full-scale ImageNet or COCO training yet. The current goal is to build a small, reproducible development baseline on Apple Silicon MPS before moving to larger hardware.

## Current Stage

**Stage:** M4 short-term deliverables

The M4 stage is limited to:

1. Project scaffold and lightweight code organization.
2. SDRB block implementation and unit tests.
3. Static theoretical analysis before training.
4. CIFAR-100 training for baseline vs SDRB comparison.
5. Empirical analysis after training, including gate visualization and latency notes.

The following are intentionally out of scope for this stage:

- ImageNet-1k full training.
- COCO downstream detection experiments.
- TensorRT deployment.
- CUDA-only optimization.
- MMDetection integration.

These will be considered after moving to more suitable GPU hardware.

## Short-Term Deliverables

The current short-term deliverables are:

1. **Module code**
   - `ops/gate.py`
   - `ops/sdrb_block.py`
   - Unit tests under `tests/`

2. **CIFAR-100 validation**
   - Baseline ResNet-20 training.
   - SDRB-ResNet-20 training.
   - Top-1 accuracy comparison.
   - TensorBoard logs and selected checkpoints.

3. **Theory and empirical analysis**
   - Static analysis: parameters, FLOPs, receptive field, gradient path, equivalence condition, theoretical latency.
   - Empirical analysis: gate distribution, entropy dynamics, training curves, measured latency on MPS.

## Locked M4 Direction

For the current local prototype, the implementation direction is:

| Item | Decision |
|---|---|
| Target block | BasicBlock |
| CIFAR model | ResNet-20 |
| Dataset | CIFAR-100 |
| Training recipe | 200 epochs, SGD momentum 0.9, cosine learning-rate schedule, batch size 128 |
| Code organization | Lightweight custom PyTorch implementation |
| Local device | MacBook Air M4, Apple Silicon MPS |

The choice of BasicBlock follows the standard CIFAR ResNet design. ResNet-20 is used first to reduce local training cost and to establish a complete baseline-vs-SDRB validation loop.

## Planned Repository Structure

```text
SDRB/
├── configs/          # CIFAR-100 experiment configs
├── docs/             # Public theory notes and analysis reports
├── models/           # CIFAR ResNet and SDRB-ResNet models
├── ops/              # SDRB block and gate operators
├── scripts/          # Training, evaluation, analysis scripts
├── tests/            # Unit tests
├── README.md         # English project overview
└── README.zh-CN.md      # Chinese project overview
```

## SDRB Design Summary

The planned SDRB block routes each spatial position across multiple dilation scales inside a residual block:

```text
input x
 ├─ depthwise 3x3, dilation=d1
 ├─ depthwise 3x3, dilation=d2
 └─ depthwise 3x3, dilation=d3
        ↓
 per-pixel scale gate
        ↓
 weighted sum → shared pointwise 1x1 fuse → BN → residual add
```

The gate is planned as:

```text
depthwise 3x3 → pointwise 1x1 → softmax / Gumbel-Softmax
```

The current design uses stage-specific dilation groups:

| Stage | Dilations |
|---|---|
| stage2 | `(1, 1, 2)` |
| stage3 | `(1, 2, 3)` |
| stage4 | `(1, 2, 5)` |

## Environment

The M4 stage targets:

- Python 3.x
- PyTorch 2.x
- torchvision
- timm
- fvcore
- tensorboard
- matplotlib
- tqdm
- onnx
- onnxruntime
- pytest
- pytest-cov
- ruff
- mypy

Apple Silicon MPS should be used when available:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

A locked `requirements.txt` will be added during Stage 0 dependency setup.

## Current Status

This repository is at the scaffold / prototype-planning stage.

Completed:

- M4 scope defined.
- Target block and CIFAR validation direction selected.
- Initial public README files added.
- Minimal directory structure prepared.

Not completed yet:

- SDRBBlock implementation.
- Gate implementation.
- CIFAR ResNet-20 baseline.
- Unit tests.
- CIFAR-100 training.
- Static and empirical theory reports.

## Development Roadmap

| Step | Description | Output |
|---|---|---|
| Stage 0 | Scaffold, dependencies, README, lint/test setup | Project foundation |
| Step B | SDRBBlock and gate implementation with tests | `ops/`, `tests/` |
| Step C | Static theoretical analysis | `docs/theory_static.md` |
| Step D | CIFAR-100 training and result collection | Logs, metrics, checkpoints |
| Step E | Empirical analysis and gate visualization | `docs/theory_empirical.md`, figures |

## Notes

This is an early research prototype. Results from CIFAR-100 on M4 are intended for method validation only. Final claims require larger-scale ImageNet and downstream detection experiments on suitable GPU hardware.
