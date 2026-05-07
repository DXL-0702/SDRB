# SDRB：Scale-Decoupled Residual Block

SDRB 是一个面向研究验证的 PyTorch 原型项目，全称为 **Scale-Decoupled Residual Block**。项目长期目标是为 ResNet 家族设计一个可即插即用、可加载预训练权重、对小目标检测更友好的残差块，并在后续阶段形成论文、开源代码与开源权重。

当前仓库处于 **M4 本地原型开发阶段**。本阶段不追求 ImageNet / COCO 级别的大规模结论，而是先在 Apple Silicon MPS 环境下完成短期可复现闭环，为后续换设备深度训练打基础。

## 当前阶段

**阶段定位：M4 短期交付物**

M4 阶段只覆盖：

1. 项目脚手架与轻量代码组织。
2. SDRBBlock 模块实现与单元测试。
3. 训练前的第一次静态理论分析。
4. CIFAR-100 上的 baseline vs SDRB 小规模验证。
5. 训练后的第二次动态理论分析，包括 gate 可视化与延迟归因。

本阶段明确不承担：

- ImageNet-1k 全量训练；
- COCO 检测下游实验；
- TensorRT 部署；
- CUDA 专属优化；
- MMDetection 接入。

以上内容将在后续更换合适 GPU 设备后再启动。

## 短期交付物

当前短期交付物分为三类。

### 1. 模块代码

计划包含：

```text
ops/gate.py
ops/sdrb_block.py
tests/
```

核心目标是让 SDRBBlock 可运行、可测试、可导出，并能在特定条件下验证与原始 ResNet BasicBlock 的数值等价关系。

### 2. CIFAR-100 训练与结果收集

计划完成：

- ResNet-20 baseline 训练；
- SDRB-ResNet-20 训练；
- Top-1 accuracy 对比；
- TensorBoard 日志；
- 关键 checkpoint；
- gate 分布与 entropy 统计。

### 3. 理论与经验分析

第一次静态分析关注：

- 参数量；
- FLOPs；
- 理论感受野；
- 梯度路径；
- 数值等价条件；
- 理论推理延迟。

第二次动态分析关注：

- gate heatmap；
- gate entropy 演化；
- loss / accuracy 曲线；
- M4 MPS 实测延迟；
- 理论延迟与实测延迟的偏差归因。

## M4 阶段已锁定方向

当前本地原型阶段锁定如下方向：

| 决策项 | 结论 |
|---|---|
| 目标 block | BasicBlock |
| CIFAR 模型 | ResNet-20 |
| 数据集 | CIFAR-100 |
| 训练 recipe | 200 epoch / SGD momentum 0.9 / cosine lr / batch 128 |
| 代码组织 | 自研轻量 PyTorch 结构 |
| 本地设备 | MacBook Air M4，Apple Silicon MPS |

选择 BasicBlock 的原因是 CIFAR ResNet-20/32/56 标准结构本身采用 BasicBlock，而不是 ImageNet ResNet-50 常见的 Bottleneck。选择 ResNet-20 的原因是它更适合在 M4 上先建立完整验证闭环，降低训练成本与调试风险。

## 计划仓库结构

```text
SDRB/
├── configs/          # CIFAR-100 实验配置
├── docs/             # 公开理论分析与实验分析文档
├── models/           # CIFAR ResNet 与 SDRB-ResNet 模型
├── ops/              # SDRBBlock 与 gate 算子
├── scripts/          # 训练、评估、分析脚本
├── tests/            # 单元测试
├── README.md         # 英文项目说明
└── README.zh-CN.md      # 中文项目说明
```

## SDRB 结构概述

SDRB 的核心思想是在残差块内部，对每个空间位置进行尺度路由，让不同位置动态选择不同 dilation 分支。

计划结构如下：

```text
输入 x
 ├─ Depthwise 3×3, dilation=d1
 ├─ Depthwise 3×3, dilation=d2
 └─ Depthwise 3×3, dilation=d3
        ↓
 per-pixel scale gate
        ↓
 加权求和 → 共享 Pointwise 1×1 fuse → BN → residual add
```

Gate 计划采用：

```text
Depthwise 3×3 → Pointwise 1×1 → softmax / Gumbel-Softmax
```

当前 dilation 分组设计为：

| Stage | Dilation 配置 |
|---|---|
| stage2 | `(1, 1, 2)` |
| stage3 | `(1, 2, 3)` |
| stage4 | `(1, 2, 5)` |

## 运行环境

M4 阶段目标环境：

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

Apple Silicon 环境优先使用 MPS：

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

锁定版本的 `requirements.txt` 将在 Stage 0 依赖整理阶段补充。

## 当前状态

当前仓库处于脚手架 / 原型规划阶段。

已完成：

- M4 阶段范围明确；
- 目标 block 与 CIFAR 验证方向已确认；
- 初始 README 文件已添加；
- 最小目录结构已准备。

尚未完成：

- SDRBBlock 实现；
- Gate 实现；
- CIFAR ResNet-20 baseline；
- 单元测试；
- CIFAR-100 训练；
- 静态理论分析报告；
- 动态经验分析报告。

## 开发路线

| 阶段 | 内容 | 输出 |
|---|---|---|
| Stage 0 | 脚手架、依赖、README、lint/test 基础 | 项目基础结构 |
| Step B | SDRBBlock 与 gate 实现，配套单测 | `ops/`, `tests/` |
| Step C | 第一次静态理论分析 | `docs/theory_static.md` |
| Step D | CIFAR-100 训练与指标收集 | 日志、指标、checkpoint |
| Step E | 第二次动态分析与 gate 可视化 | `docs/theory_empirical.md`、图表 |

## 说明

当前 CIFAR-100 + M4 MPS 的结果只作为方法验证，不作为最终论文结论。最终结论仍需要后续在更合适的 GPU 设备上完成 ImageNet 训练与 COCO 下游检测实验。
