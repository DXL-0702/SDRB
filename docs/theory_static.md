# Stage 1.5 静态理论分析

## 1. 分析目标与范围

### 1.1 目标
在投入 CIFAR-100 训练算力前，完成 SDRB 相对于原 ResNet 的静态理论分析，验证"理论上不会更差"的前提假设，为后续 Stage 2 训练提供决策依据。

### 1.2 范围
- **对比对象**：SDRBBlock vs ResNet BasicBlock（CIFAR-100 变体使用的结构）。
- **分析维度**：参数量、FLOPs、理论感受野、梯度路径、数值等价性边界、推理延迟（理论）。
- **输入规格**：以 CIFAR-100 的 32×32 分辨率为主要分析场景。

### 1.3 边界声明（重要）

> ⚠️ **当前 Stage 1 实现边界**
>
> 1. **数值等价性测试只覆盖 depthwise-separable reference**：`test_sdrb_block_one_hot_gate_matches_depthwise_separable_reference` 验证的是 SDRBBlock 在 `gate=[1,0,0]` 下等价于同权重的 `DWConv(dilation=1) + PWConv + BN + residual` 路径，**不验证**与原 ResNet full 3×3 conv BasicBlock 的数值等价。
> 2. **固定 3 分支设计**：SDRBBlock 当前强制 `len(dilations) == 3`，`ScaleGate` 的 `num_scales` 参数虽暴露为任意整数，但实际默认 `bias_init` 和 SDRBBlock 构造均锁定 3 分支。Stage 1.5 分析收敛为固定 3 分支语义，不泛化为 K 分支。
> 3. **接口语义分层**：`gate_override`、`return_gate`、`temperature` 是分析/测试/训练调度专用接口，不是标准推理路径；`hard_gate` 仅在 `training and use_gumbel` 时生效，eval 期回退 softmax。

---

## 2. 参数量对比

### 2.1 原 ResNet BasicBlock（CIFAR 变体）

CIFAR-100 的 ResNet-20/32/56 使用 BasicBlock（非 Bottleneck），结构为：

```
Conv2d(in_c, out_c, 3×3, stride=stride, bias=False)
BN(out_c)
ReLU
Conv2d(out_c, out_c, 3×3, stride=1, bias=False)
BN(out_c)
+ residual
```

**参数量公式**：
- 第一个 3×3 conv: `in_c × out_c × 3 × 3`
- 第二个 3×3 conv: `out_c × out_c × 3 × 3`
- 两个 BN: `2 × out_c`（weight + bias，running stats 不计入可学习参数）
- projection shortcut（如果 stride≠1 或 in_c≠out_c）: `in_c × out_c × 1 × 1 + out_c`

**identity 情况**（stride=1, in_c=out_c=C）：
```
Params_BasicBlock = C×C×9 + C×C×9 + 2×C = 18×C² + 2×C
```

### 2.2 SDRBBlock

当前 Stage 1 实现结构为：

```
# 三分支 depthwise
[DWConv3×3(dilation=d1), DWConv3×3(dilation=d2), DWConv3×3(dilation=d3)]
# gate
DWConv3×3 + PWConv1×1(num_scales=3)
# weighted sum + fuse + bn
PWConv1×1(in_c→out_c) + BN(out_c)
+ residual(projection if needed)
```

**参数量公式**：
- 三个 depthwise 分支（每组 in_c 个 3×3 filter）：`3 × in_c × 3 × 3 = 27 × in_c`
- gate: `in_c × 3 × 3 (DW) + in_c × 3 (PW bias) = 9×in_c + 3`
- fuse: `in_c × out_c × 1 × 1`
- BN: `2 × out_c`
- projection（如果需要）: `in_c × out_c + out_c`

**identity 情况**（stride=1, in_c=out_c=C, dilation=(1,1,2)）：
```
Params_SDRBBlock = 27×C + 9×C + 3 + C×C + 2×C
                 = C² + 38×C + 3
```

### 2.3 数值对比表（identity 情况）

| C | BasicBlock (18C²+2C) | SDRBBlock (C²+38C+3) | 比例 |
|---|------------------------|----------------------|------|
| 16 | 4,640 | 867 | 0.19× |
| 32 | 18,496 | 2,211 | 0.12× |
| 64 | 73,856 | 6,339 | 0.09× |
| 128 | 295,168 | 20,227 | 0.07× |

**结论**：SDRBBlock 参数量显著小于原 BasicBlock，且通道数越大优势越明显。这是因为 SDRB 用 depthwise separable 结构替代了 full 3×3 conv。

### 2.4 fvcore 实测验证

```python
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from ops import SDRBBlock

block = SDRBBlock(64, 64, stride=1)
print(parameter_count_table(block))
```

---

## 3. FLOPs 对比

### 3.1 计算方法

FLOPs 计算基于乘法-加法操作数，输入分辨率 H×W=32×32。

### 3.2 原 ResNet BasicBlock（identity, C=64）

- 第一个 3×3 conv: `2 × C × C × 3 × 3 × H × W = 2 × 64 × 64 × 9 × 1024 = 75,497,472`
- 第二个 3×3 conv: 同上
- BN: 可忽略（或计为 `O(C×H×W)`）
- **总计**: ~151M FLOPs

### 3.3 SDRBBlock（identity, C=64, dilation=(1,1,2)）

- 三个 depthwise 分支: `3 × 2 × C × 3 × 3 × H × W = 3 × 2 × 64 × 9 × 1024 = 3,538,944`
- gate DW: `2 × C × 3 × 3 × H × W = 1,179,648`
- gate PW: `2 × C × 3 × H × W = 393,216`（3 个输出通道，每个与 C 个输入做 1×1）
- fuse PW: `2 × C × C × 1 × 1 × H × W = 8,388,608`
- weighted sum: `3 × C × H × W = 6,291,456`（3 个分支加权，每个像素 3 次 mul + 2 次 add）
- **总计**: ~19.6M FLOPs

### 3.4 比例

| C | BasicBlock FLOPs | SDRBBlock FLOPs | 比例 |
|---|------------------|-----------------|------|
| 16 | ~2.4M | ~1.2M | 0.50× |
| 32 | ~9.4M | ~4.9M | 0.52× |
| 64 | ~151M | ~19.6M | 0.13× |
| 128 | ~604M | ~78.6M | 0.13× |

> 注：小通道数时比例较高是因为 gate 和分支管理开销占比大；大通道数时更接近参数量比例。

### 3.5 实现层开销声明

当前 Stage 1 实现会物化完整的 5D `branch_stack` `[N, 3, C, H, W]` 再进行加权，这带来额外内存带宽开销。上述 FLOPs 计算只统计数学操作，**不包含内存访问和 tensor 调度开销**。在 M4/MPS 等 memory-bound 设备上，实际 latency 可能与 FLOPs 预测存在显著偏差。

---

## 4. 理论感受野

### 4.1 计算方法

对于 `Conv2d(kernel=3, dilation=d, padding=d)`，感受野大小为：
```
RF = 1 + d × (3 - 1) = 1 + 2d
```

### 4.2 SDRB 三分支配置

| 分支 | dilation | RF | 备注 |
|------|----------|-----|------|
| 0 | 1 | 3 | 局部特征 |
| 1 | 1 | 3 | **与分支 0 相同 RF** |
| 2 | 2 | 5 | 更大上下文 |

### 4.3 等效感受野

gate 加权后，等效感受野是三分支的加权组合。对于默认配置 `(1,1,2)`：
- 两个 dilation=1 分支覆盖相同 RF=3 区域。
- 它们带来的是 **same-RF capacity**（同感受野下的额外表达能力），而非 **RF diversity**（不同感受野的多尺度覆盖）。

### 4.4 不同 Stage 的推荐配置

| Stage | 推荐 dilations | RF 覆盖 | 说明 |
|-------|---------------|---------|------|
| stage2 (浅层) | (1,1,2) | 3,3,5 | 避免过大 RF 引入无关背景 |
| stage3 (中层) | (1,2,3) | 3,5,7 | 逐步扩展 |
| stage4 (深层) | (1,2,5) | 3,5,11 | 更大上下文，检测友好 |

> 当前 Stage 1 默认 `(1,1,2)` 是为 stage2 浅层设计的保守配置。

---

## 5. 梯度路径分析

### 5.1 Gumbel-Softmax 梯度性质

当 `use_gumbel=True` 且 `training=True` 时：
```python
weights = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
```

- **Straight-through estimator**：前向使用 hard sample（one-hot），反向使用 softmax 的梯度。
- 梯度流经 logits，允许端到端训练离散 routing 决策。
- 温度 τ 控制离散程度：高 τ 接近均匀分布，低 τ 接近 one-hot。

### 5.2 共享 BN 对梯度流的影响

当前实现中，三个分支的输出加权后进入**共享的** `fuse + bn`：
```python
weighted = (branch_stack * gate_weights.unsqueeze(2)).sum(dim=1)
out = self.bn(self.fuse(weighted))
```

- 共享 BN 意味着三个分支的梯度通过 BN 的 running stats 和可学习参数产生耦合。
- 这与 conditional computation（如 MoE 中每个 expert 有独立 BN）不同，SDRB 的设计有意保持分支间的参数共享以减小参数量。
- 潜在影响：不同分支的梯度在 BN 层混合，可能影响分支专门化（specialization）的速度。

### 5.3 与原始 ResNet 梯度路径对比

| 特性 | ResNet BasicBlock | SDRBBlock |
|------|-------------------|-----------|
| 梯度路径 | 固定两条 3×3 conv | 动态三分支选择 |
| 可学习参数 | 独立卷积核 | 共享 gate + 分支卷积 |
| BN 分布 | 固定 | 随 gate 动态混合 |
| 梯度稀疏性 | 无 | gate 接近 one-hot 时近似稀疏 |

---

## 6. 数值等价性边界

### 6.1 当前已验证范围

**已覆盖**：
- `test_sdrb_block_one_hot_gate_matches_depthwise_separable_reference` 验证当 `gate_override=[1,0,0]` 时，SDRBBlock 的输出与同权重的 `DWConv(dilation=1) + PWConv + BN + residual` reference 数值差 < 1e-6（默认测试容差）。

**未覆盖**：
- 原 ResNet full 3×3 conv BasicBlock 的数值等价性。
- 加载 ImageNet 预训练权重后的数值差。
- 非 one-hot gate 分布下的数值行为。

### 6.2 为什么不能直接等价 full 3×3 conv

SDRBBlock 使用 depthwise-separable 结构：
- `DWConv(C→C) + PWConv(C→C)` 的表达能力**严格小于** `Conv2d(C→C, 3×3, groups=1)`。
- 因此即使权重初始化相同，两者的输出也不数值等价。

### 6.3 Stage 2/3 的验证建议

若要实现"加载预训练权重后数值等价"，需要：
1. 设计 full 3×3 → depthwise-separable 的权重分解/迁移策略。
2. 或者修改 SDRBBlock 结构，在 gate=[1,0,0] 时退化为可学习的 full 3×3 等价路径。

当前 Stage 1 选择保持结构简洁，将权重迁移问题后置到 Stage 3（ImageNet 训练）或专门的权重迁移实验。

---

## 7. 推理延迟（理论）

### 7.1 DWConv vs 标准 Conv 的计算特性

| 特性 | 标准 Conv (groups=1) | Depthwise Conv (groups=C) |
|------|----------------------|---------------------------|
| 计算密度 | 高（channel 间混合） | 低（每个 channel 独立） |
| 内存访问 | 权重小，但需多次读写 | 输入输出多次遍历 |
| 硬件优化 | 高度优化（cuDNN） | 小 kernel dispatch 开销大 |

### 7.2 SDRB 的延迟预测复杂性

SDRBBlock 包含多个小 op：
- 3 个 depthwise conv
- 1 个 gate depthwise + pointwise
- 1 个 fuse pointwise
- BN、加权、residual

在 M4/MPS 上：
- **理论 FLOPs 优势**（约 0.1×-0.5× BasicBlock）**不必然转化为 latency 优势**。
- 主要开销来源：
  1. **branch_stack 物化**：`[N, 3, C, H, W]` 中间张量的内存分配和带宽。
  2. **kernel dispatch**：多个小 op 的 launch overhead。
  3. **gate 计算**：额外的 memory-bound 操作。

### 7.3 Stage 1.5 的延迟分析建议

- 理论 FLOPs 只作 **上限参考**。
- 实际延迟应以 **M4 MPS 实测** 为准（Stage 2.5 收集）。
- 若理论 FLOPs 与实测延迟偏差显著，归因于上述实现层开销。
- 后续优化方向：
  - 流式加权（不物化 5D stack）。
  - 算子融合（fuse gate + weighted sum）。
  - 针对 MPS backend 的特定优化。

---

## 8. 接口语义与实现边界

### 8.1 固定 3 分支设计

- `SDRBBlock` 强制 `len(dilations) == 3`。
- `ScaleGate` 默认 `bias_init=(1.0, 0.0, 0.0)` 锁定 3 分支。
- **Stage 1.5 结论**：SDRB 是固定 3 分支结构，不应在报告/论文中泛化为"K 分支多尺度模块"。

### 8.2 分析/测试专用接口

| 接口 | 用途 | 标准推理？ |
|------|------|-----------|
| `gate_override` | one-hot 测试、消融、probe | 否 |
| `return_gate` | 可视化、entropy 记录 | 否 |
| `temperature` | 训练期 annealing | 否（eval 期固定或忽略）|
| `hard_gate` | 训练期 Gumbel straight-through | 否（eval 回退 softmax）|

**建议**：在训练脚本和模型封装中，这些参数不应暴露为公共推理 API。

### 8.3 术语澄清

- `gate_entropy(..., normalize=True)`：实际计算的是 **mean entropy**（在 batch×spatial 上平均），不是按 `log(num_scales)` 归一化的相对熵。
- 文档中应称其为 "mean gate entropy"，避免与信息论中的 "normalized entropy" 混淆。

---

## 9. 结论与 Go/No-Go 判据

### 9.1 静态分析结论

| 维度 | 结论 | 风险等级 |
|------|------|---------|
| 参数量 | SDRBBlock 显著小于 BasicBlock（约 0.07×-0.19×） | 低 |
| FLOPs | 显著小于 BasicBlock（约 0.13×-0.50×） | 低 |
| 理论感受野 | 默认配置产生 same-RF capacity，非 RF diversity | 中（需文档澄清） |
| 梯度路径 | Gumbel straight-through 可行，共享 BN 有耦合 | 中（需观察训练稳定性） |
| 数值等价性 | 仅验证 depthwise-separable reference | 高（需 Stage 2/3 补充预训练验证） |
| 推理延迟 | FLOPs 优势不保证 latency 优势 | 中（需 Stage 2.5 实测） |

### 9.2 Go/No-Go 判据

**Go 条件（Stage 2 CIFAR-100 训练启动条件）**：
- [x] 参数量/FLOPs 理论验证通过（无明显劣势）。
- [x] 核心算子实现完成且单测通过。
- [x] 理论分析文档（本文档）确认并归档。
- [ ] gate 可视化 pipeline 就绪（Stage 2 期间收集）。
- [ ] 训练 recipe 确认（lr schedule、entropy reg λ、温度退火策略）。

**No-Go 风险**：
- 若 CIFAR-100 训练中出现 gate collapse（所有像素选择同一分支），需暂停并调整 gate 正则化策略。
- 若 SDRB Top-1 低于 BasicBlock baseline 超过 0.5%，需回退到方案讨论。

### 9.3 Stage 1.5 → Stage 2 交接清单

| 项目 | 状态 | 位置 |
|------|------|------|
| SDRBBlock 实现 | ✅ | `ops/sdrb_block.py` |
| ScaleGate 实现 | ✅ | `ops/gate.py` |
| 单元测试 | ✅ | `tests/test_*.py` |
| 静态分析文档 | ✅ | `docs/theory_static.md` |
| 参数量/FLOPs 脚本 | 📝 可选 | 附录 |

---

## 附录

### A. 参数量计算脚本

```python
from ops import SDRBBlock

for c in [16, 32, 64, 128]:
    block = SDRBBlock(c, c, stride=1)
    params = sum(p.numel() for p in block.parameters())
    print(f"C={c}: SDRBBlock={params}, BasicBlock~{18*c*c + 2*c}")
```

### B. fvcore 使用示例

```python
from fvcore.nn import FlopCountAnalysis
from ops import SDRBBlock
import torch

block = SDRBBlock(64, 64, stride=1)
x = torch.randn(1, 64, 32, 32)

flops = FlopCountAnalysis(block, x)
print(flops.total())
print(flops.by_operator())
```

### C. 版本记录

- 文档版本：Stage 1.5
- 基于代码版本：`5ddbf47 implement SDRB core ops and tests`
- 更新日期：2026-05-07
