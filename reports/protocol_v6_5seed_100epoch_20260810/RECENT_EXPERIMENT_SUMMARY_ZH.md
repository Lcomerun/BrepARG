# 最近实验总体总结

更新时间：2026-08-13（Asia/Shanghai）

仓库：`Lcomerun/BrepARG`；分支：`experiment/protocol-v5-scaling-ladder`

基础证据快照 commit：`0654171e01127efb2cb85d3145e41a268595dc0d`

当前阶段：表示层与装配链路根因定位，尚未进入序列重生成或 AR 训练

本文面向需要在其他设备上了解项目进展的项目成员。它汇总 Protocol V5 数据与 scaling 实验、100 CAD 装配校准，以及 Protocol V6 四臂五种子训练的最新状态。本文引用 Git 中已归档的轻量证据；模型权重、重建数组和原始数据不在仓库中。

## 一页结论

1. **数据协议已基本闭环。** 当前使用论文口径的拓扑过滤、parent-CAD 隔离划分、坏 pickle fail-closed 策略和多 chunk 唯一性检查。已审计的 parent split、拓扑过滤、坏文件和多 chunk identity 检查均通过；目前没有证据表明这些已审计问题是首要根因。
2. **现有 FSQ 训练配置不健康。** Protocol V6 中已经运行的 8 个 FSQ 实例（两个 arm、seed 0 至 3）全部出现 nonfinite batch/epoch。继续原样增加 FSQ epoch 或 seed 的科学收益很低。
3. **Learned VQ 是目前更可信的量化主臂，但量化问题尚未解决。** 健康 seed 1/2 的 VQ-4096/64D perplexity 均值为 `1093.81`，落在此前采用的 `800–1500` 启发式健康区间；但其 best curved parent MSE 均值为 `9.4673e-4`，仍是 continuous bypass 的约 `2.81x`，未达到“VQ 距 bypass 小于 `2x`”的解除量化瓶颈条件。
4. **问题不只在量化器。** Continuous bypass 的 best curved parent MSE 均值为 `3.3641e-4`，仍是原参考门 `5e-5` 的约 `6.73x`。这说明即使绕过离散量化，decoder、连续表示容量或优化稳定性仍限制曲面重建。
5. **装配链路存在不可忽略的独立失败下限，并与表示误差共同构成瓶颈。** 既有 100 CAD 校准中，原始 patch 的 strict BRep-valid 也只有 `84%`，continuous bypass 为 `70%`，FSQ-8192/4D 为 `49%`；误差与 invalid 的相关性较弱或仅中等，正式判决为 `ASSEMBLY_DOMINATED`。CAD 不可用应判断为“表示误差 + decoder/优化下限 + 装配/拓扑链路”共同造成，而不是单一 AR 或 FSQ 问题。
6. **当前不能进入 AR。** Protocol V6 的五种子矩阵未完成，健康结果尚不足；其固定 100 CAD 曲面重建也尚未开始。序列重生成和 AR gate 必须继续保持关闭。
7. **本次训练因系统重启中断，而不是 Python 报错。** Windows 计划更新在 2026-08-13 03:29–03:32 触发重启；seed 3 stderr 为空，无 traceback。当前 GPU 利用率低是因为训练进程已经不存在，不是 dataloader 正在慢速运行。

## 实验脉络

| 阶段 | 目的 | 状态 | 主要结论 |
| --- | --- | --- | --- |
| Protocol V2–V5 数据协议 | 修复拓扑过滤、parent 泄露、坏 pickle 和多 chunk identity | 已完成 | 数据协议通过硬门，parent overlap 为 0 |
| Protocol V5 scaling | 12k/60k/300k 比较 FSQ，并在 60k 加入 learned VQ | 已完成 | 扩数据有改善，但外推仍远离 `5e-5`；不得进入 AR |
| Continuous bypass oracle | 隔离量化误差与 decoder/连续表示下限 | 已完成 | bypass 优于量化臂，但自身仍明显高于目标 |
| 100 CAD assembly calibration | 将 patch 曲面误差与 STEP/BRep-valid 对齐 | 已完成 | strict valid 主要受装配链路影响，判定 `ASSEMBLY_DOMINATED` |
| Protocol V6 四臂五种子 | 四 arm、五 seed、300k patch、每臂 100 epoch | 已中断 | 仅 4 个 arm/seed 组合健康完整，10 个数值失稳 |
| Protocol V6 固定 100 CAD 重建 | 在完成矩阵后做统一曲面与装配复核 | 未开始 | 不能从当前 patch validation 指标直接推断 CAD 可用率 |
| 序列重生成与 AR | 下游生成模型训练 | 阻塞 | 表示层和装配层尚未过门 |

## 1. 数据协议现状

Protocol V5 在前 10 个 ABC archive 上建立共享 master protocol，并在更大范围做过 archive/member identity 预检。以下统计已归档为轻量 [Protocol V5 master 摘要](protocol_summary_v5_master.json)，并与 V5 运行状态中的协议哈希绑定：

- master protocol 扫描 10 个 archive、66,879 个 pickle member；加载失败为 0；
- 35,847 条记录满足几何过滤，最终确定性选择 15,000 条；
- split 为 train 12,000、validation 1,500、test 1,500；
- 同一 parent CAD 的所有记录只进入一个 split，三组 pairwise parent overlap 均为 0；
- faces 过滤为 `10 <= faces <= 50`；每个 face 的 edges `<= 30`；全局 edges `<= 150`；
- 坏 pickle 不进入 split。未知坏 member 或失败数/比例越过阈值时，协议 fail closed，并保留 quarantine/manifest 证据；
- archive-qualified source identity 和 materialization target 必须全局唯一，避免多 chunk 同名 member 覆盖。

因此，本阶段应继续保留数据审计，但不应再把主要算力投入到重复验证已经通过的 split/过滤假设。当前更强的证据指向表示稳定性、decoder 下限和装配链路。

## 2. Protocol V5 scaling 结果

以下均为 parent-equal curved MSE 汇总。V5 在 12k/60k/300k 比较 FSQ，并只在 60k 加入 learned VQ；60k/300k 使用同一 master protocol 和相同 seed 设计。

| 训练 patch | Arm | Curved parent MSE | Perplexity |
| ---: | --- | ---: | ---: |
| 60k | FSQ-4096/6D | `4.4233e-3` | `482.5` |
| 60k | FSQ-8192/4D | `4.7179e-3` | `1412.2` |
| 60k | Learned VQ-4096/64D | `2.6254e-3` | `1546.3` |
| 300k | FSQ-4096/6D | `3.4970e-3` | `586.6` |
| 300k | FSQ-8192/4D | `2.0179e-3` | `1604.6` |

60k 到 300k 的两点 power-law 诊断外推到约 3M patch 时，curved parent MSE 为 `2.4985e-3`，约为 `5e-5` 参考目标的 `50x`。两点外推不是全量训练保证，但足以说明“直接扩数据并增加 epoch”没有证据支持进入 AR，因而 V5 的正式决策是 `CONTINUE_CAPACITY_INVESTIGATION`。

V5 还提供了一个重要方向性结果：60k learned VQ 的曲面误差低于两个 FSQ arm，说明 64D 自由学习码字比低维 FSQ 网格更有潜力。后续 V6 因此同时纳入 learned VQ 和 continuous bypass，并在 300k 固定 100 epoch 矩阵中继续验证。

## 3. 100 CAD 装配校准

该实验在 100 个确定性、parent-isolated validation CAD 上，用相同解析拓扑比较 original patch、continuous bypass 和 FSQ-8192/4D。所有失败都保留在 attempts 分母中。

| Arm | Strict BRep-valid | STEP 保存率 | Curved MSE 与 invalid 的 Spearman 相关 |
| --- | ---: | ---: | ---: |
| Original patch | `84/100 = 84%` | `94%` | 不适用（误差为 0） |
| Continuous bypass | `70/100 = 70%` | `96%` | `0.300` |
| FSQ-8192/4D | `49/100 = 49%` | `94%` | `0.377` |

关键解释：

- 即使输入未经模型重建，original patch 仍有 16 个 CAD 不能通过 strict BRep-valid，证明装配/拓扑链路本身存在不可忽略的失败下限；
- bypass valid 率明显高于 FSQ，但其分桶 valid 曲线并不单调，curved MSE 不能单独预测装配成功；
- 经验 `80% valid` 对应的 curved MSE 约为 bypass `1.0395e-4`、FSQ `4.4129e-4`，只能作为该 100 CAD cohort 的校准结果，不能当作跨模型的固定理论门槛；
- 正式状态为 `ASSEMBLY_DOMINATED`，所以降低 patch MSE 和修复 assembly 都需要做，不能用其中一项替代另一项。

注意：这是 V5 checkpoint 上已经完成的 assembly calibration。它不等于 Protocol V6 要求的五种子训练后固定 100 CAD 曲面重建；后者仍为 pending。

## 4. Protocol V6 设计与实际进度

固定配置：

- arms：`fsq_8192_4d`、`fsq_4096_6d`、`vq_4096_64d_random`、`continuous_bypass_64d`；
- seeds：0、1、2、3、4；
- 每个 arm 目标 100 epoch；train 300,000 patch，validation 12,000 patch；
- batch size 128，learning rate `3e-4`；
- 训练完成后原计划先对固定 100 CAD 做统一 surface reconstruction；如需判断 CAD 可用性，再单独运行 joint optimization、STEP 保存和 OCC/strict BRep validity 评估。

截至 2026-08-13 快照，20 个 arm/seed 组合的有效分类为：

| Seed | FSQ-8192/4D | FSQ-4096/6D | VQ-4096/64D | Continuous bypass |
| ---: | --- | --- | --- | --- |
| 0 | 100 ep，数值失稳 | 100 ep，数值失稳 | 100 ep，数值失稳 | 100 ep，数值失稳 |
| 1 | 100 ep，数值失稳 | 100 ep，数值失稳 | 100 ep，健康 | 100 ep，健康 |
| 2 | 100 ep，数值失稳 | 100 ep，数值失稳 | 100 ep，健康 | 100 ep，健康 |
| 3 | 100 ep，数值失稳 | 100 ep，数值失稳 | 44 ep，finite 后中断 | 未开始 |
| 4 | 未开始 | 未开始 | 未开始 | 未开始 |

汇总计数：

- 14/20 个组合的训练循环写到 epoch 99，但“循环跑完”不等于“数值健康”；
- 4/20 为 `HEALTHY_COMPLETE`；
- 10/20 为 `NUMERICALLY_UNSTABLE`；
- 1/20 为 `INTERRUPTED_FINITE`；
- 5/20 为 `PENDING`；
- V6 surface reconstruction 尚未启动，`downstream_ar_allowed=false`。

### 健康结果

| Seed | Arm | Best val recon | Best curved parent MSE | Final curved parent MSE | Final perplexity |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | VQ-4096/64D | `4.4739e-4` | `9.7540e-4` | `1.0168e-3` | `1164.92` |
| 2 | VQ-4096/64D | `4.3288e-4` | `9.1806e-4` | `9.7159e-4` | `1022.69` |
| 1 | Continuous bypass | `1.7434e-4` | `3.4355e-4` | `3.5308e-4` | 不适用 |
| 2 | Continuous bypass | `1.6351e-4` | `3.2926e-4` | `4.2082e-4` | 不适用 |

两组健康 seed 的聚合判断：

- VQ best curved MSE 均值：`9.4673e-4`；
- bypass best curved MSE 均值：`3.3641e-4`；
- VQ/bypass 比值：seed 1 为 `2.84x`，seed 2 为 `2.79x`；
- 相对 `5e-5` 参考门：VQ 高约 `18.9x`，bypass 高约 `6.7x`；
- VQ perplexity 均值 `1093.81`，码本利用率不再是最明显的失败信号，但几何精度仍未过门。

这些数字只是 validation patch/CAD-equal 重建指标。V6 固定 100 CAD surface reconstruction 尚未执行；其后的 joint optimization、STEP 保存和 OCC/strict BRep validity 也尚未执行，**不能从这张表推断最终 CAD 可用率**。

### Seed 3 中断点

Seed 3 learned VQ 完成 epoch 0–43，44 个 epoch 的 train/validation batch 均为 finite：

- best val recon：`6.6613e-4`，epoch 40；
- best curved parent MSE：`1.4413e-3`，epoch 42；
- epoch 43 curved parent MSE：`1.4941e-3`；
- epoch 43 perplexity：`1276.69`。

健康 seed 1/2 的 VQ 最佳点分别出现在约 epoch 96/95，因此 seed 3 的 44 epoch 不能替代完整 100 epoch 结果。

## 5. 中断原因与 GPU 利用率

本次中断证据指向外部系统重启：

- 2026-08-13 03:29:09，`MoUsoCoreWorker` 记录计划重启；
- 03:31:27，`TrustedInstaller` 因操作系统升级记录计划重启；
- 03:32:04，操作系统重新启动；
- seed 3 stderr 为 0 bytes，没有 Python traceback；
- 磁盘上的 `cohort_state.json` 仍记录 `RUNNING` 和 PID `22496`，但该 PID 与所有训练 Python 进程均已不存在，因此它是陈旧状态，实际状态为 `INTERRUPTED`。

所以此前观察到的低 GPU 利用率并不是训练效率问题：训练已经停止，GPU 处于空闲状态。

Seed 3 的 rolling checkpoint 只含 model state 和部分指标，不含 optimizer、AMP scaler、scheduler 或 RNG state。它可以用于诊断，但不能从 epoch 44 做与原协议严格等价的续训。

## 6. 总体根因判断

目前证据支持以下优先级，而不是“AR 没训好”这一单一解释：

1. **装配/拓扑链路：已证实的重要瓶颈。** Original patch 也只有 84% strict valid，且 bypass 的 MSE 与 invalid 相关性不强。装配失败会把已经较好的局部曲面重建变成不可用 CAD。
2. **Decoder/连续表示或优化下限：高优先级。** Bypass 绕过量化后仍比 `5e-5` 高约 6.7 倍，说明仅更换或扩大码本无法达到原目标。
3. **量化误差：仍存在，但 learned VQ 优于当前 FSQ。** VQ 比 bypass 差约 2.8 倍，尚未达到 `<2x` 条件；与此同时 VQ perplexity 已健康，下一步不能只追求更高 code usage。
4. **FSQ 数值稳定性：阻断当前 FSQ 路线。** 两种 FSQ 在 seed 0–3 的全部 8 次运行中均失稳。原配置继续跑更多 seed 不能形成可靠容量结论。
5. **数据协议：当前优先级降低。** 过滤、parent isolation、坏文件策略和 identity 检查已经通过；除非后续全量协议审计出现新证据，否则它不应继续作为主要根因。
6. **AR：尚无资格评估。** 表示层和装配层都未过门，提前训练 AR 会把上游失败与序列建模误差重新混在一起。

换言之，最近实验没有证明“只要再训练久一点就能生成可用 CAD”。它证明的是：learned VQ 值得保留，当前 FSQ 不值得原样扩算力；同时必须把 decoder/连续下限和 assembly/topology 当作两个独立问题处理。

## 7. 是否继续训练

### 如果目标是判断技术方向

现有证据已经足够做出方向性决定：不再原样扩展 FSQ，不进入 AR，优先处理 assembly 与 continuous-bypass 下限。为这个目的，无需为了凑齐矩阵而立即重跑所有 20 个组合。

### 如果目标是完成正式五种子验收矩阵

仍有必要继续，但不能直接重启旧 launcher：旧 launcher 会从 seed 3 的第一个 FSQ arm 开始重跑并覆盖日志，而 checkpoint 又不足以严格恢复 seed 3 VQ。

推荐恢复范围：

1. 完整保留当前 seed 3 中断目录作为证据；
2. 在新的 recovery 输出目录中，从初始化重训 seed 3 learned VQ 100 epoch，再补跑 seed 3 continuous bypass 100 epoch；
3. 补跑 seed 4 learned VQ 与 continuous bypass 100 epoch；
4. 只有在报告形式明确要求“四臂 x 五种子”时才考虑 seed 4 FSQ；在修复反复 nonfinite 之前，不建议继续相同 FSQ 配置；
5. 如果 FSQ 稳定性修复改变了训练定义，应建立新的 FSQ cohort，不能把修复前后的 seed 混成同一统计组；
6. 完成选定 recovery 后，先对固定 100 CAD 运行 V6 surface reconstruction；如需判断 CAD 可用性，再单独运行 joint optimization、STEP 保存和 OCC/strict BRep validity 评估；
7. 在表示与装配验收通过前，继续阻塞序列重生成和 AR。

## 8. 证据索引

建议先读以下文件：

- [Protocol V6 快照说明](README.md)
- [V6 训练健康明细 CSV](training_health_summary.csv)
- [V6 训练健康明细 JSON](training_health_summary.json)
- [中断证据](interruption_evidence.json)
- [续训价值与恢复范围评估](continuation_assessment.md)
- [本地 checkpoint SHA-256 清单（不含权重）](checkpoint_manifest.json)
- [V6 轻量归档工件清单](artifact_manifest.json)
- [Protocol V5 scaling 说明](../protocol_v5_scaling_20260807/README.md)
- [Protocol V5 scaling 数值汇总](../protocol_v5_scaling_20260807/analysis/scaling_summary.json)
- [Protocol V5 master 数据协议摘要](protocol_summary_v5_master.json)
- [100 CAD 装配校准说明](../assembly_calibration_100cad_20260809/README.md)
- [100 CAD 装配校准数值汇总](../assembly_calibration_100cad_20260809/analysis/assembly_calibration_summary.json)

仓库保留已运行 arm 的 history JSON、轻量 TensorBoard event 和 stdout/stderr；尚未开始的 arm 不产生这些文件。为避免仓库膨胀，以下内容没有上传：

- 模型 checkpoint：`*.pt`、`*.pth`、`*.ckpt`；
- 重建数组：`*.npz`；
- 原始 ABC 数据、pickle 和 materialized CAD；
- PID 文件；
- 整个 `BrepARG/` 上游源码目录；
- `papers/`。

所有结论应以本页注明的快照日期为界。后续 recovery、V6 固定 100 CAD 重建或 assembly 修复产生新证据后，应新增结果并更新本页，而不是覆盖本次中断证据。
