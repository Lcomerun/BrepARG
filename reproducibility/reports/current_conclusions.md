# 当前科学结论与下一步

## 结论分级

### 已确认

1. **端到端链路能执行，但复杂 CAD 质量不足。** 当前结果不能作为“高质量复杂 BRep 生成已解决”的证据。[E006, E007, E012]
2. **FSQ/decoder/assembly 链路是最强上游瓶颈。** 50 个 complex-curved shapes、3,399 patches 上，整体 Chamfer p95 为 `0.15012168`，surface p95 为 `0.41238499`；真实 token 重建仅 `27/50` 写出 STEP，`9/50` 当前 BRep-valid。[E006, E007]
3. **AR 有独立次级问题。** 复杂子集 teacher-forcing CE 明显高于全局 validation CE，自由生成仍向简单拓扑集中，历史上还出现 OOM 后 loss 爆炸和 nonfinite checkpoint。[E008, E011, E012]
4. **排序是次级因素。** matched medium control 中 DFS token-weighted CE `1.25869`，RCM `1.31302`；差异存在，但不足以解释 true-token 重建失败。[E010]
5. **历史 split 存在 parent-CAD 泄露。** V13 validation 记录约 `56.75%`、test 记录约 `57.17%` 与其他 split 共享 parent CAD；same-data BrepARG 也存在同类问题。[E004, E005]
6. **更长 BrepARG 自训改善但不根治。** 短训复杂 `5/92`、strict `0/92`；长训 VQ400/AR300 后复杂 `13/100`、strict `6/100`，拓扑中位数仍约 6 faces/12 edges。[E013, E014]
7. **生成约束只能改变选择结果。** 调温度、关闭 bbox 单调约束、最小面/边阈值、原始 BrepARG sampler 和质量门均未修复真实 token 重建失败或生成分布坍塌。[E009, E012]
8. **50 faces 是协议内边界。** 当前 sequence 的 faces 范围 2-50，global edges 范围 2-150。50-face 失败应单列高难度桶，不能排除为越界。[E003, E006]

### 强推断

1. validation CE 长期低于 train CE 主要由 parent-CAD 泄露、训练态 dropout 和 batch 等权 CE 共同造成，而不是单一的数据难度差异。[E003, E004, E015]
2. 当前 surface Chamfer heavy tail 和 true-token 装配失败说明仅提高 FSQ level 不足；decoder/assembly、loss 形态、曲面参数化与 shape-level 一致性都可能参与。[E006, E007, E016]
3. 长序列 exposure bias 会放大 AR 错误，但长度不是唯一主因；所有 face/length 桶均有明显重建失败。[E006, E007, E008]

### 证据不足

1. FSQ 量化、连续 decoder 和 OCC assembly 各自贡献多少，尚未被三个 oracle 完全拆开。
2. 官方 BrepARG 公开权重的实际质量尚未在兼容词表和官方协议下复现。
3. DFS 是否在全数据、同 epoch、同 context 下稳定优于 RCM，当前 medium control 不足以定论。
4. 多随机种子下的模型与生成波动范围未知。
5. parent-CAD 隔离重训后的绝对指标未知。

## 根因优先级

| 优先级 | 根因候选 | 当前判断 | 主要依据 |
| --- | --- | --- | --- |
| P0 | decoder/assembly 与复杂曲面表示 | 最强证据 | true-token 仍仅 9/50 BRep-valid |
| P0 | parent-CAD 泄露和评测协议 | 已确认 | val/test 约 57% 记录跨 split 共享 parent |
| P1 | AR 复杂子集建模和 exposure bias | 已确认独立问题 | complex CE、简单拓扑集中、历史发散 |
| P1 | context/truncation | 部分贡献 | 1024 对复杂序列覆盖不足，2048 仍非全覆盖 |
| P2 | RCM/GNN 排序 | 次级退化 | DFS CE 小幅领先，无法解释装配主失败 |
| P2 | 单纯 FSQ level 容量 | 当前实验不支持单独归因 | 候选 overall p95 反而恶化约 4.19% |
| P3 | sampling temperature/constraint | 已排除为主修复 | 只改变幸存者和 invalid 比例 |
| P3 | 单纯延长 epoch | 已排除为充分修复 | BrepARG 长训改善但仍坍塌 |

## 下一轮执行顺序

1. **冻结当前 selected checkpoint 和 protocol。** 不再继续无门控长训，不使用 nonfinite latest。
2. **重建 parent-CAD 隔离 split。** 先做身份审计，再从零训练；旧 validation CE 仅保留为开发曲线。
3. **实现 ground-truth assembly oracle。** 不经过 VQ/FSQ/AR，直接将解析几何送入同一装配与 OCC 验证链路。
4. **实现 continuous-latent bypass oracle。** 相同 decoder 和 assembly，唯一变量是绕过 FSQ quantization。
5. **实现 teacher-forced argmax reconstruction。** 每个真实前缀取一步 argmax，重建完整预测 token，隔离条件预测错误和自由运行 exposure。
6. **按 shape 等权报告复杂曲面指标。** 同时保留 patch 等权；按 surface type、faces、global edges、sequence length 分桶。
7. **只有 oracle 指向 FSQ 容量时才重训 capacity。** 一次只改 levels 或 latent dimension，并保持 split、样本、decoder、loss、seed 不变。
8. **在同一 parent-isolated protocol 下重训 V13 与 BrepARG。** 官方权重若词表不兼容，只能做官方协议复现或明确标为 unavailable。

## 已排除假设

- 只提高 sampling temperature 就能恢复复杂度和有效性。
- 只关闭 bbox monotonic / face uniqueness 就能修复质量。
- 只设置最小 faces/edges 就能让模型学会复杂拓扑。
- 只做生成后 water-tight/valid filtering 就能提高真实 Valid。
- 只换回 BrepARG 原始生成逻辑就能解决坍塌。
- 只增加 AR batch size 或训练 epoch 就能修复上游几何。
- 50-face 样本属于当前协议外。
- validation loss 较低足以证明泛化良好。
- same-data 自训 BrepARG 等价于官方模型复现。
