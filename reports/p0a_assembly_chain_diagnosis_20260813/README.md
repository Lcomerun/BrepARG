# P0-A：100-CAD 原始控制组装配链诊断

## 一页结论

本实验冻结既有 100-CAD assembly calibration 中的 **16 个 original/GT strict-invalid CAD**，没有重新抽样。每个 CAD 运行 `joint_iterations ∈ {200, 0}` 与 sewing tolerance `{1e-4, 1e-3, 1e-2}` 的笛卡尔积，共 **96/96 attempts**。结果为 **16/16 明确归因，attribution rate=100.0%**，P0-A 归因门（≥80%）通过。

主因分布：

- wire self-intersection：**10** 个；
- curve B-spline fit 在三档 fallback 后仍失败：**3** 个；
- OCC wire build 失败：**2** 个；
- non-unit/empty solid：**1** 个。

这证明 16 个 GT-invalid 不是同一种问题，其中 trim/wire self-intersection 是首要族群。现有聚合结论 `ASSEMBLY_DOMINATED` 得到了逐例、逐阶段证据支持。

## 消融解释

- `joint_iterations=0` 使 **4** 个 CAD 的完整 outcome signature 发生变化；
- sewing tolerance 三档扫描使 **1** 个 CAD 的完整 signature 发生变化；
- 但只有 **1/16** 个 CAD 在任一变体达到 both-valid：`00051587_446e8810d6884cae80689579_step_000`；该 CAD 的三个 `joint=0` tolerance 变体均有效。

因此，**“签名敏感”不等于“修复成功”**。它可能只是 native/strict、wire self-intersection 数量、shell/solid 计数或失败 stage 改变。三档 tolerance 没有形成可推广的恢复证据，不能据此全局放宽容差；关掉 joint optimize 也不是通用修复。

## 修复优先级

1. **P0-A1：trim/wire self-intersection（10/16）**。逐 face/wire 记录自交实体，检查 edge orientation、pcurve 与 outer/inner loop 语义；不得先用宽松 ShapeFix 掩盖根因。
2. **P0-A2：退化 curve fit（3/16）**。利用已记录的 edge index 与 `5e-3 → 8e-3 → 5e-2` fallback 证据，区分重复点/零长度/病态曲线，再加入有界 lower-degree 或 degenerate-edge 策略。
3. **P0-A3：wire build（2/16）**。在送入 OCC 前验证端点连续性、拓扑次序和方向，并把 builder error 与 face/loop 对齐。
4. **P0-A4：shell→solid cardinality（1/16）**。sewing 后显式枚举 shell，只允许单一闭合 shell 进入 `MakeSolid`，empty/compound/multi-shell 分开报告。
5. **横向项：joint offset**。对 4 个签名敏感 CAD 比较 surface-edge residual 与偏移量，将它视为放大器而不是已证实主因。

## 门控结论

P0-A 的“≥80% 可归因”验收已经通过（实际 100%），但这只关闭了诊断门，不代表装配已修复。`advance_to_boundary_consistency` 仍为 `false`：必须先完成 P0-B 的 0-nonfinite 稳定性重测和健康 VQ 的固定 100-CAD 装配测量，再决定是否启动 boundary-consistency loss。序列重生成和 AR 继续阻塞。

## 证据索引

- `assembly_chain_summary.json`：规范化总体 gate、cause 和 sensitivity 计数；
- `assembly_chain_cases.json`：16 个 CAD 的 baseline signature、主因和二级证据；
- `attempts_compact.csv`：96 个 CAD×variant 的阶段、validity 组件和 STEP 绑定；
- `repair_checklist.md`：由 case 分类生成的修复清单；
- `step_sha256.csv`：本地保存的 STEP 大小和 SHA-256；**不包含 STEP 文件本体**；
- `artifact_manifest.json`：Git 归档内所有轻量文件的大小和 SHA-256。

未上传内容包括 STEP、原始 pickle、模型 checkpoint、重建数组和整个 upstream `BrepARG/`。
