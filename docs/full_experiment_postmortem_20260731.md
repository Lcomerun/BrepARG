# V13 / BrepARG 全链路实验复盘与工程治理报告

审计日期：2026-07-31（Asia/Shanghai）
审计对象：`D:\luolin\V13`、`D:\V13_rootcause_recovery_20260717` 及当前本机运行环境
执行计划：`plans/full_experiment_postmortem_and_workspace_governance_20260731_execplan.md`

## 阅读约定

本报告按三档表达结论，避免把相关性误写成因果性：

- **已确认**：由代码、日志、JSON、checkpoint 或可重复统计直接支持。
- **强推断**：有多项一致证据，但仍缺少严格单变量对照。
- **证据不足**：当前材料不能支持肯定或否定结论。

“可写出 STEP”“BRep-valid”“复杂”“严格通过”和论文中的 `Valid` 不是同一个指标。除非明确写明，报告中的比例都保留原始分母，不把生成幸存样本比例当作总尝试成功率。

## 执行摘要

### 当前结论

1. **项目曾多次完整跑通，但没有取得合格的复杂 CAD 生成质量。** VQ-VAE、sequence、AR、STEP/PNG 生成和验证链路均已有成功运行记录；因此当前不是单纯“程序跑不起来”，而是“流程可执行但复杂几何质量和评测可信度不足”。依据：`plans/stable_vqvae_retraining_execplan.md`、`plans/ar_training_v13_execplan.md`、`plans/complex_curved_fsq_ar_diagnostics_execplan.md`。
2. **复杂曲面几何表征与 BRep 重建链路是当前最强的质量瓶颈证据。** 在 50 个复杂曲面样本、3,399 个 patch 上，整体 Chamfer p95 为 `0.15012`，surface p95 为 `0.41238`；完全绕过自由运行 AR、直接使用真实 token 时，仅 `27/50` 写出 STEP，`9/50` 通过当前 BRep-valid 检查。依据：`local_runs/complex_curved_rootcause_suite_20260715/experiments/00_fsq_only_patch_metrics/complex_curved_diagnostics_report.json` 和 `local_runs/complex_curved_rootcause_suite_20260715/experiments/01_teacher_forcing_true_token_reconstruction/complex_curved_diagnostics_report.json`。
3. **AR 还有独立的次级瓶颈。** 复杂曲面子集的 teacher-forcing CE 显著高于全局验证 CE；1024 context 只覆盖约 `64%` 的复杂序列；自由运行生成又出现简单拓扑坍塌、截断、曝光偏差和历史训练发散。AR 不是唯一根因，但也不能视为已经解决。依据：`docs/audits/v13_ar_distribution_coverage_20260731.json`、`local_reports/generated100_lr5e6_epoch120_best_20260705.md`、用户提供的服务器 AR OOM/NaN 日志。
4. **RCM/GNN 排序有退化信号，但不是主根因。** 同一复杂曲面 cohort 上，DFS token-weighted CE `1.25869`，RCM `1.31302`，DFS 小幅领先；两者共享相同 FSQ patch 报告，而 true-token BRep 失败远大于排序差距。依据：`D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_teacher_forcing\ar_teacher_forcing.json` 和 `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\rcm_teacher_forcing\ar_teacher_forcing.json`。
5. **更长 BrepARG 同数据训练有改善，但没有解决简单拓扑坍塌。** 短 baseline 的复杂样本为 `5/92`、严格通过 `0/92`；VQ 400 / AR 300 后变为 `13/100` 和 `6/100`，但拓扑中位数仍为 6 faces / 12 edges。依据：`D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\breparg_same_data_quality_summary.json`、`D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\breparg_same_data_resume_best_quality_summary_20260726.json` 和 `plans/breparg_long_baseline_and_v13_diagnosis_20260720_execplan.md`。
6. **不能声称“官方 BrepARG 也效果不好”。** 当前官方 ABC AR 权重 embedding 为 `[7222,256]`，本地协议 vocab 为 `10294`，无法直接加载；已完成的是 same-data 自训 fallback，不是官方权重的协议复现。官方 README 的 VQ 3000 / AR 500 是命令示例，不证明公开权重的真实训练轮次。依据：`local_reports/breparg_official_weight_probe_20260717.md`、`BrepARG/README.md`。
7. **当前 train/val/test 存在 parent-CAD 级泄露，验证 CE 不能作为独立泛化证据。** V13 验证记录中 `56.75%`、测试记录中 `57.17%` 与其他 split 共享 parent CAD；exact path 没有重复不等于 CAD 家族独立。依据：`docs/audits/v13_sequence_split_integrity_20260731.json`。same-data BrepARG 也有相同问题。
8. **验证损失低于训练损失并不说明验证集更简单。** train/val/test 的长度、face、edge 和 complex 比例非常接近；更合理的解释是 parent-CAD 泄露、训练态 dropout=0.1 而验证态关闭 dropout，以及 V13 当前按 batch 等权平均 CE。依据：`docs/audits/v13_ar_distribution_coverage_20260731.json`、`BrepARG/model.py:11-25`、`breparg_improvements/train.py:718-759`。
9. **生成时约束和质量门只能筛选，不能修复分布。** 提高温度、关闭 bbox monotonic、强制最小面数、换用 BrepARG 原始生成逻辑、生成后拒绝非水密结果均已做过；它们或增加 invalid，或需要大量尝试才能收集幸存者，没有消除 true-token 重建失败。依据：`local_runs/ubuntu_generation_eval_20260713/deep_root_cause_50_each_20260713`、`local_runs/breparg_logic_compare_20260715` 和复杂曲面 suite。
10. **当前没有 V13 或 BrepARG 训练/生成任务。** RTX 3060 审计时利用率约 0-1%；唯一 Python 进程是无关的 `LLM-CAD` HTTP server。低 GPU 利用率不是训练被 CPU 卡住，而是训练已经结束。

### 当前已具备信息

- V13 本地 VQ-VAE 与 AR 历史、checkpoint、生成 STEP/PNG 和质量汇总。
- 服务器 V13 checkpoint/sequence 的本地回传副本与用户提供的训练终端记录。
- 50 样本复杂曲面 FSQ、AR teacher-forcing、真实 token 重建和 Chamfer 相关性实验。
- FSQ levels 容量候选、DFS/RCM 局部对照、BrepARG 原始生成逻辑对照。
- same-data BrepARG 10k/1k/1k 短训练和 VQ400/AR300 长训练及 100 个 STEP/PNG 结果。
- 当前完整 sequence 的长度、face、edge、complex 分布与 parent-CAD split 审计。
- 本机环境、GPU、磁盘健康、工作树和大文件/重复文件盘点。
- 生成与验证指标的实现级审计。

### 仍缺失的关键信息

- **证据不足**：服务器训练时的完整源码快照、`pip freeze`、PyTorch/CUDA/cuDNN 组合；当前仓库代码不能自动代表当时 5090 进程加载的代码。
- **证据不足**：官方 BrepARG 权重对应的完整词表/config、真实训练轮次和官方生成协议复现结果。
- **证据不足**：按 parent CAD 完全隔离后的 VQ/AR 重训与测试结果。
- **证据不足**：当前 ubuntu sequence package 与历史本地 AR 训练 snapshot 的 record-level 对应关系；二者总量相同但 split counts 不同，不能共享 split 结论。
- **证据不足**：相同代码、数据、参数下至少 3 个随机种子的方差。
- **证据不足**：完整同数据、同 epoch、同 context 的 DFS 与 RCM 从零训练对照。
- **证据不足**：官方 COV/MMD/JSD/Novelty/Uniqueness/Valid 全套指标；当前本地结果不能替代论文协议。
- **证据不足**：连续 latent（绕过 FSQ）与 ground-truth 几何直接送入 BRep assembler 的对照，因此“FSQ 量化、decoder、OCC assembly”三者尚未完全拆开。
- **证据不足**：真正的 teacher-forced argmax 重建。现有实验同时做了“真实前缀 CE”和“真实 token 直接重建”，但没有收集 AR 在每个真实前缀下的 argmax token 再重建。

---

## 零、实验时间线与版本对比

### 0.1 是否曾成功跑通

**已确认：流程跑通过，质量目标没有跑通。**

- 2026-06-27，本地稳定 VQ-VAE 完成 40 epochs，全部 batch finite，best validation reconstruction 约 `0.00056`，checkpoint 可加载。依据：`plans/stable_vqvae_retraining_execplan.md`。
- 随后 VQ-VAE continuation 到绝对 epoch 85，best epoch 73、best val 约 `0.00037`，正常 early stop。依据：`plans/vqvae_epoch100_continuation_execplan.md`。
- 本地 V13 AR 分支最终到 epoch 120，best val CE `0.2949333`，可生成 STEP；100 次生成中 87 个 STEP、78 个当前 BRep-valid。依据：`plans/ar_training_v13_execplan.md` 和 `local_reports/generated100_lr5e6_epoch120_best_20260705.md`。
- 但同一 100 样本中 `62/87` 集中在 6F/12E 与 4F/6E，complex strict-valid 为 `0`，质量门结论为 `hold_for_failure_analysis`。因此“能跑通”不能等价为“实验成功”。依据：`local_reports/v13_generated_quality_gate_20260705.md`。

### 0.2 关键时间线

| 时间 | 版本/事件 | 直接结果 | 审计判断 | 主要依据 |
| --- | --- | --- | --- | --- |
| 2026-06-24 至 06-27 | 初始 FSQ-VQ-VAE 全流程 | 旧 run best val 约 `0.00082`，末期 `val=inf` 仍曾标记 VERIFIED | 训练稳定性回归，后续已加 finite/early-stop 防护 | `plans/stable_vqvae_retraining_execplan.md` |
| 2026-06-27 | stable VQ 40 epochs | best `0.00056`，无 NaN/Inf | 数值稳定问题得到阶段性修复 | 同上 |
| 2026-06-27 | VQ continuation | 到 epoch 85 early stop，best epoch 73、约 `0.00037` | 全局 MSE 改善，但尚未证明复杂曲面改善 | `plans/vqvae_epoch100_continuation_execplan.md` |
| 2026-06-30 至 07-05 | 本地 AR 多 LR continuation | 最终 epoch 120，best val CE `0.2949333` | 训练可收敛；小样本“valid”掩盖简单拓扑坍塌 | `plans/ar_training_v13_execplan.md` |
| 2026-07-05 至 07-07 | G20/G100、VQ 四分桶诊断 | G100 78 BRep-valid，但 0 complex strict-valid；longest VQ slice 3/10 valid | 首次明确质量失败早于自由生成 | `local_reports/v13_generation_quality_root_cause_20260705.md` |
| 2026-07-10 至 07-13 | RTX 5090 scratch/resume VQ、全量 sequence、2048 AR | resume VQ 的独立 50k patch 指标优于当时 scratch；AR 后期发生 loss 爆炸、OOM、NaN latest | 服务器流程扩展成功，但存在 checkpoint 污染和训练稳定性问题 | 用户提供终端日志；本地 `ABC/processed/train_outputs/ubuntu` 回传产物 |
| 2026-07-13 | 深度生成对比 | 调温度、bbox、min faces 能改变复杂度但不能稳定改善 BRep 质量 | 生成约束不是主修复路线 | `local_runs/ubuntu_generation_eval_20260713/deep_root_cause_50_each_20260713` |
| 2026-07-15 至 07-16 | complex-curved 50 样本 suite | surface Chamfer p95 `0.41238`；真实 token 仅 9/50 BRep-valid | 几何表征/解码/assembly 成为最强根因证据 | `local_runs/complex_curved_rootcause_suite_20260715` |
| 2026-07-15 至 07-17 | FSQ levels、DFS/RCM、BrepARG logic | levels 候选整体 p95 变差；DFS CE 小幅优于 RCM；换生成逻辑仍简单 | 单纯扩 levels、换 sampler、换排序均不足以根治 | suite 与 `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval` |
| 2026-07-17 至 07-20 | 官方权重 probe、same-data BrepARG 短 baseline | 官方 vocab 不兼容；自训 VQ best epoch 70、AR best 77；0/92 strict accepted | baseline 可执行但不是官方复现，且简单坍塌 | `local_reports/breparg_official_weight_probe_20260717.md`、`D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\breparg_same_data_quality_summary.json` |
| 2026-07-20 至 07-28 | BrepARG VQ400 / AR300 长 baseline | VQ best 269；AR best val `0.765318`；100 个 survivor 中 13 complex、6 strict accepted | 延长 epoch 有有限改善，不能解释或解决全部失败 | `plans/breparg_long_baseline_and_v13_diagnosis_20260720_execplan.md` |
| 2026-07-31 | 本次全链路审计 | 发现 parent-CAD 泄露、指标分母不一致、E 盘健康告警、目录权威漂移 | 现有结果可用于内部诊断，尚不足以形成公平论文结论 | `docs/audits/*_20260731.json` 与本报告 |

### 0.3 从“能跑”到“跑坏”的专项回归

1. **旧 VQ-VAE nonfinite 回归**：旧代码会在后期 validation 为 `inf` 时仍产生误导性 VERIFIED；2026-06-27 增加 finite batch 统计、early stop 和 history 后修复。依据：`breparg_improvements/training_stability.py` 和 `plans/stable_vqvae_retraining_execplan.md`。
2. **AR 2048 推理尺寸回归**：旧 `generate_validate.py` 用默认 1024 构造模型，加载 2048 checkpoint 时 `transformer.wpe.weight` 尺寸不匹配；当前代码已从 checkpoint config 读取 `max_seq_len`。依据：`breparg_improvements/generate_validate.py:89-93` 与用户提供错误日志。
3. **服务器 AR 发散/污染回归**：用户日志显示 epoch 115 后 train CE 从约 0.33 持续上升，随后 OOM；后续从 best 恢复的分支又在 epoch 123-125 爆炸并在 epoch 130/140 记录 `train_CE=0.0000, val_CE=nan`。checkpoint 检查确认 `ar_latest.pt` 的 embedding nonfinite，而 `ar_best.pt` finite。当前 `_train_ar` 对 nonfinite train loss 只跳过 batch，若全 epoch 无 finite batch会得到 `0/max(1,0)=0`，validation 也没有 finite hard-fail，仍会保存 latest。依据：`breparg_improvements/train.py:718-759` 与用户终端记录。
4. **磁盘/重启回归**：BrepARG 短 VQ 在 D 盘写满时中断，长 AR 在 epoch 128 遇到电脑重启；两者均通过 best/periodic checkpoint 恢复，但改变了 run lineage。依据：`plans/breparg_long_baseline_and_v13_diagnosis_20260720_execplan.md`。
5. **依赖回归**：初始 sequence smoke 因 `ModuleNotFoundError: chamferdist` 失败；后来对小规模重建使用 CPU `torch.cdist` 兼容路径，但这不是服务器完整环境的证明。依据：`plans/ar_training_v13_execplan.md` 和用户终端记录。

### 0.4 Git、多人协作和问题引入区间

- 外层 Git 当前 `HEAD=16cf19b`，提交日期 2026-07-01，之后大量实验代码未提交：审计时约 13 个 modified、118 个 untracked、0 deleted；没有可见 remote。Git 历史无法重建 7 月中下旬的真实运行版本。
- `BrepARG/` 是被外层忽略的嵌套 Git 仓库，不是 submodule；其本地 `HEAD=07970a4`，有 5 个修改文件和未跟踪论文。自动 pull 或替换会丢失 baseline 实验实现。
- **多人无意修改：证据不足。** 当前没有作者映射、远端 PR 或提交历史证明是多人修改；只能确认“工作树修改缺乏提交边界”。
- 问题不是在单一时间点引入：数值稳定问题至少在 06-27 前存在；1024 context 与简单拓扑问题贯穿 06-30 至 07-05；复杂曲面根因到 07-15 才被隔离；parent-CAD 泄露到 07-31 才被明确发现。

结论：整理或继续训练前，必须先冻结外层与嵌套仓库的源码 provenance；否则下一次“相同参数复现”仍可能使用不同实现。

---

## 一、实验顶层目标设定

### 1.1 研究目的与当前实际输出

根据代码、计划和论文目录，当前研究目标可表述为：在 BrepARG 的整体 token 序列框架上，引入 FSQ 几何离散化、RCM/GNN 面排序和长 context AR，生成可读取、拓扑合法、水密、非退化、具有复杂曲面和足够多样性的 B-rep CAD，并与原始 BrepARG 在相同数据和评测协议下比较。

当前已经输出：VQ/AR checkpoint、FSQ+RCM sequence、STEP/STL/PNG、patch MSE/Chamfer、teacher-forcing CE 和本地质量门。当前尚未输出：严格协议下可用于论文主结论的生成指标和稳定的复杂 CAD 样例。

### 1.2 应明确的成功标准

本报告建议把成功标准分为三层，不能继续用单一 validation loss 代替：

| 层级 | 必须回答的问题 | 建议最低门槛 | 当前状态 |
| --- | --- | --- | --- |
| 数据/协议 | test 是否真正独立，指标是否与 baseline 同定义 | parent-CAD overlap 为 0；所有比例记录 attempts 分母；相同数据与 caps | **未通过** |
| 组件诊断 | 真实复杂几何不经自由 AR 能否稳定重建 | 固定 complex-curved holdout；true-token STEP ≥90%，BRep-valid ≥70%；surface Chamfer p95 相对当前至少下降 20% | **未通过，9/50 valid** |
| 自由生成 | 是否生成复杂、有效、多样几何 | 内部迭代至少 100 attempts；最终论文按官方规模/重复数；同时报告 Valid、复杂率、拓扑集中度、几何唯一性和人工盲审 | **未通过** |

这些数值是下一轮工程 promotion gate，不是声称来自 BrepARG 论文的官方阈值。最终论文对比仍需复现官方协议：3,000 生成、1,000 references、每模型 2,000 点、10 次独立运行，以及 COV/MMD/JSD/Novelty/Uniqueness/Valid。

### 1.3 baseline 标准

- 官方 BrepARG README 示例：VQ-VAE `3000` epochs、AR `500` epochs。依据：`BrepARG/README.md:24-36`。
- 当前 same-data baseline：train/val/test 为 10,000/1,000/1,000；短 run VQ best epoch 70、AR best epoch 77；长 run VQ 完成 400、AR 完成 300。
- **官方 baseline 尚未独立复现。** 官方权重与本地 vocab 不兼容，same-data baseline 又使用本地筛选、sampling 和验证工具，因此只能叫“same-data self-trained BrepARG baseline”。
- 下一次公平比较必须固定：parent-CAD split、过滤规则、augmentation、point normalization、max face/edge 语义、训练预算、context、生成 attempts、top-p/temperature 和 metric implementation。

### 1.4 约束条件

- 本地：Windows 11、i9-14900KF（32 logical processors）、约 64 GiB RAM、RTX 3060 12 GiB。
- 历史服务器：RTX 5090 32 GiB，用户日志显示 Python 3.14 环境与 CUDA 13 驱动栈；完整软件锁定缺失。
- 数据表示：当前 sequence 实际最大 50 faces、150 global edges；序列最大观测长度 2353。
- 存储：D 为健康 NTFS；E 为 exFAT 且 `Full Repair Needed`，不能作为唯一权威副本。
- 算法要求：若论文主张创新来自 FSQ/RCM/GNN，则 baseline 不能共享这些改动；若比较生成框架，则数据和评测必须共享。

---

## 二、工程文件体系构成与文件夹整理

### 2.1 当前根目录说明

审计时 `D:\luolin\V13` 约 20,018 files、747 directories、195.063 GiB logical；已知 hardlink 去重后的物理量约 188.475 GiB。根目录各条目职责如下：

| 路径 | 功能 | 治理判断 |
| --- | --- | --- |
| `.git/` | 外层代码历史 | 必须保留；当前历史落后于工作树 |
| `.agents/` | 空的 agent 占位目录 | 待确认删除，零空间收益 |
| `.pytest_cache/` | pytest 可重建缓存 | 待确认删除 |
| `ABC/` | parsed archives、训练输出和历史数据 | 核心数据区；约 171.034 GiB |
| `BrepARG/` | 上游/基线代码及嵌套 Git | 第三方依赖加本地 patch；必须先保存 provenance |
| `breparg_improvements/` | V13 FSQ、RCM/GNN、训练与生成实现 | 当前方法主代码 |
| `dist/` | 旧服务器打包产物 | 归档候选；不代表当前工作树 |
| `docs/` | 操作指南、清理记录、本报告与审计 JSON | canonical 文档区 |
| `local_reports/` | 历史运行分析、状态卡、服务器交接报告 | 应归档并建立索引，不能直接删除 |
| `local_runs/` | 本地 AR、重建、root-cause 和 baseline 产物 | 约 24.012 GiB；需按 run manifest 治理 |
| `papers/` | AAAI 稿件与图表候选 | 保留；当前只能支持诊断性表述 |
| `plans/` | ExecPlan 与实验决策历史 | 保留；需标 active/completed/superseded |
| `processed_local/` | 当前空目录 | 待确认删除 |
| `tests/` | 工具、训练稳定性和审计回归测试 | 保留并纳入 Git |
| `tools/` | 数据、训练、验证、迁移、治理脚本 | 自研工具区；需按用途分组 |
| `AGENTS.md`、`PLANS.md` | agent 与 ExecPlan 规范 | 保留 |
| `environment.server.yml` | 极简服务器环境声明 | 保留但必须扩充 lock |
| `local_training_config.json` | 旧本地 pipeline 配置 | 保留历史；路径仍指向 E，不能作为当前默认 |
| `PROJECT_INDEX.md`、`README.md` | 项目入口 | 内容已漂移：仍称 C 盘 parsed shards 为 authority |
| `e_drive_drop_*`、`recovery_status_*` | E 盘与恢复事件记录 | 移入 `docs/history/recovery/` 的归档候选 |

`D:\V13_rootcause_recovery_20260717` 约 11.288 GiB，是恢复与 baseline 证据根，不是临时目录。它包含 short/long BrepARG、staged same-data、DFS/RCM 评估和迁移日志；在建立 manifest 前不得整体删除。

### 2.2 配置与硬编码风险

1. `breparg_improvements/train.py:92-126` 通过大量 `NS_*` 环境变量配置，但 run 目录不一定保存完整环境快照；用户 shell 命令因此成为唯一参数来源。
2. `NS_AR_GRAD_CLIP` 并未被读取；当前 AR 梯度裁剪硬编码为 `1.0`。依据：`breparg_improvements/train.py:727`。在 shell 中 export 该变量不会改变行为。
3. V13 随机种子固定为 0，但没有 `torch.cuda.manual_seed_all`、worker seed、`torch.use_deterministic_algorithms`；同时允许 TF32。依据：`breparg_improvements/train.py:134-138`。
4. `local_training_config.json` 仍将 raw/parsed/train output 指向 E 盘，且 AR context 为 1024；与当前 D 盘 authority 和后期 2048 运行不一致。
5. `PROJECT_INDEX.md` 声称 `C:\V13_abc_parsed_shards` 是 authoritative root，但该路径当前不存在；实际观察到的健康重建源是 `ABC/processed/abc_parsed_full_archives`。
6. `max_edge=150` 的含义不统一：本地 sequence 统计体现 global edge cap 150，而 `BrepARG/utils.py` help 写“maximum number of edges per face”；论文协议又描述 per-face 30 edge cap。必须改名为 `max_global_edges` / `max_edges_per_face`，禁止复用同一变量名。

### 2.3 checkpoint、缓存、日志与输出规则

- V13 canonical 本地 VQ：`ABC/processed/train_outputs/newscheme_full_vqvae_epoch100/fsq_vqvae_best.pt`。
- V13 canonical 本地 AR：`local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/ar_best.pt`。
- 服务器回传三件套：`ABC/processed/train_outputs/ubuntu/{fsq_vqvae_best.pt,ar_best.pt,sequences_fsq_rcm.pkl}`；目录名无法表达训练 lineage，应补 manifest。
- BrepARG 长 baseline 分散在两个根：VQ/sequence/生成在 `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720`，完成的 resumed AR 在 `local_runs/breparg_long_ar_resume_best_20260724`。
- 84 个 checkpoint 共约 14.266 GiB，只有一对 SHA-256 完全相同；不能根据 `best/latest/final/epoch` 名称相似批量删。
- 六个 1.5 GiB V13 sequence 路径实际上是同一个 NTFS hardlink；删除别名几乎不释放物理空间。
- 日志中包含训练 stdout/stderr、TensorBoard、migration 和 timeout 记录。大日志可压缩归档，但 failure tail、命令、PID、start/end、checkpoint hash 必须进入 manifest。

### 2.4 第三方与自研工具边界

- `BrepARG/`：上游 baseline 代码，但已本地修改，必须作为“vendor + patch”管理。
- `breparg_improvements/`：V13 方法代码。
- `tools/`：目前混合了数据构建、训练 launcher、评测、磁盘迁移和一次性恢复脚本。建议拆为 `tools/data/`、`tools/train/`、`tools/eval/`、`tools/governance/`，但这是后续重构，不应在当前 dirty tree 直接机械移动。
- `tests/`：应与上述模块一一对应；当前大量 untracked test 必须先提交或 bundle。

### 2.5 命名与版本堆积问题

现有名称大量使用 `safe`、`stable`、`smoke`、`best`、`latest`、`final`、LR 后缀和日期，但无法唯一确定数据 hash、源码、resume 起点、context、seed 与指标协议。`ubuntu` 尤其不能作为长期 run ID。

建议 run ID：

    exp_YYYYMMDD_<method>_<dataset-split-id>_<representation>_<ordering>_<context>_<change>_s<seed>

示例：

    exp_20260802_v13_parentcad-v1_fsq8192_dfs_ctx2048_curvedloss_s0

每个 run 必须含 `manifest.json`：

- outer Git commit、dirty diff SHA-256、nested BrepARG commit/patch SHA-256；
- Python/package/CUDA/GPU；
- 数据、split、sequence、VQ、AR checkpoint SHA-256；
- 完整命令和所有环境变量；
- seed、context、caps 与 augmentation；
- best/latest/final 对应 epoch 和 finite 检查；
- attempts 分母、评测脚本版本与输出路径。

### 2.6 建议目标目录

```text
D:\
├─ luolin\V13\                           # 代码和轻量文档，Git 管理
└─ V13_store\
   ├─ dependencies\BrepARG\07970a4\      # clean clone + local.patch + manifest
   ├─ datasets\abc\
   │  ├─ raw-or-archives\
   │  ├─ parsed-shards\
   │  └─ splits\parentcad-v1\
   ├─ artifacts\v13\<run-id>\
   │  ├─ manifest.json
   │  ├─ checkpoints\
   │  ├─ sequences\
   │  ├─ eval\
   │  └─ generated\{step,stl,png}
   ├─ artifacts\breparg\<run-id>\...
   ├─ reports\{canonical,history,recovery}\
   ├─ archive\{runs,logs,packages}\
   └─ scratch\{cache,tmp,render-work}\
```

### 2.7 整理执行方案

1. **冻结 provenance**：外层工作树创建 bundle/commit；记录 118 个 untracked 文件 hash；嵌套 BrepARG 导出 binary-capable patch 和 untracked manifest。
2. **建立 authority index**：为 parsed ZIP、V13 VQ/AR/sequence、BrepARG 长 baseline、root-cause suite 生成 SHA-256 manifest。
3. **恢复健康冗余**：先把 E 盘唯一内容只读复制到健康 NTFS，再 hash 比对；不要先运行会写盘的 `chkdsk E: /f`。
4. **按 run ID 归档**：先在健康 NTFS 建新目录并复制/硬链接验证，更新 manifest 后再考虑清理旧路径。
5. **最后请求删除批准**：任何删除按精确路径、hash、预计回收空间逐项确认，不使用通配符批量删实验树。

### 2.8 待用户确认的删除候选，本次未执行

下面只把**完整路径已经确定**的项目列入可审核清单。即使属于可重建缓存，也不以模糊通配符请求批准。

| 精确候选路径 | 预计物理回收 | 删除前条件 |
| --- | ---: | --- |
| `D:\luolin\V13\.pytest_cache` | 约 0.001 MiB | 可重建；仍需用户明确确认 |
| `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\data_staged` | 约 7.447 GiB | 保留 staging manifest；确认 local-suite canonical source tree hash；修复仍引用 staged path 的 manifest |
| `D:\luolin\V13\ABC\processed\train_outputs\newscheme_vqvae_weighted_cap_local_20260707\fsq_vqvae_best_epoch87_snapshot_20260707.pt` | 约 218.178 MiB | 记录其与 sibling `fsq_vqvae_best.pt` SHA-256 均为 `CAA4512028D924F9F673A1142D38ED23F998DABA7B050CEF5ED689750EC2543B` |
| `D:\luolin\V13\local_runs\complex_curved_rootcause_suite_20260715\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_dfs_medium_safe_20260715\sequences_fsq_rcm.pkl` | 约 41.913 MiB | 保留 hash 相同的 `sequence_rebuild_medium\sequences_fsq_dfs.pkl`；SHA-256 `A62A27B8CBA6C505FA903DD57A8A5A48CD14C9AE71AB88D9387B3E800EF3C0FC` |
| `D:\luolin\V13\local_runs\complex_curved_rootcause_suite_20260715\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_rcm_medium_safe_20260715\sequences_fsq_rcm.pkl` | 约 41.913 MiB | 保留 hash 相同的 `sequence_rebuild_medium\sequences_fsq_rcm.pkl`；SHA-256 `46A047DA5907850C5D55C5ACF6E33F2736777A906F9634C0D232580546565E90` |
| `D:\luolin\V13\local_runs\complex_curved_rootcause_suite_20260715\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_dfs_medium_smoke_20260715\sequences_fsq_rcm.pkl` | 约 0.355 MiB | 保留 hash 相同的 `ar_train_smoke_medium\dfs_subset\sequences_fsq_rcm.pkl`；SHA-256 `B4C656EBB23B9503E5CD3C192EBCEF63D768767384AA6033A0E71175412C60A6` |
| `D:\luolin\V13\local_runs\complex_curved_rootcause_suite_20260715\experiments\02_dfs_rcm_ordering\ar_train_outputs\ar_rcm_medium_smoke_20260715\sequences_fsq_rcm.pkl` | 约 0.355 MiB | 保留 hash 相同的 `ar_train_smoke_medium\rcm_subset\sequences_fsq_rcm.pkl`；SHA-256 `57369553FE28FA42F58417E9DDE9AFFA30C8137CA3CB467F7577F374D13E7C71` |

上述大项合计约 `7.743 GiB`，但当前仍**未获批准、未删除**。以下五个路径是 canonical `D:\luolin\V13\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm.pkl` 的 NTFS hardlink 别名；删除这些别名不会释放约 5 倍空间，预计物理回收仍约 0 GiB：

- `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar\sequences_fsq_rcm.pkl`
- `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr1e5\sequences_fsq_rcm.pkl`
- `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr2e5\sequences_fsq_rcm.pkl`
- `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e5\sequences_fsq_rcm.pkl`
- `D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar_lr5e6\sequences_fsq_rcm.pkl`

各 `__pycache__`、generation `_work`、小 cache test、重复报告 alias、空目录及 42 个 AppleDouble 文件目前只是**分类候选，不是可审批清单**；必须先生成逐文件路径、大小和 hash manifest，才能请求删除。绝不能现在删除：100 个 parsed ZIP、任一 `.git`、dirty/untracked 源码、canonical VQ/AR/sequence、BrepARG 长 baseline best/final、100 个 STEP/PNG、root-cause 评估，以及任何仅在 E 盘存在的内容。

---

## 三、数据链路全流程分析

### 3.1 当前数据链路

```text
ABC STEP / 压缩 archive
  -> BrepARG/process_data/process_brep.py 解析、归一化、采样 surface/edge
  -> parsed record / parsed ZIP / parsed shard (.pkl.zst)
  -> VQ patch shard（surface 32x32，edge 重排/铺成 32x32）
  -> FSQ-VQ-VAE 训练与 patch token
  -> BrepARG/2sequence.py + V13 RCM/GNN ordering
  -> sequences_fsq_rcm.pkl（train/val/test）
  -> 按 max_seq_len 过滤后训练 GPT2 AR
  -> 生成或真实 token 重建
  -> OCC joint optimization / STEP writer
  -> STEP 读取、BRep-valid、实体/闭合/复杂度、PNG 人工检查
```

关键实现：`BrepARG/process_data/process_brep.py`、`breparg_improvements/vqvae_sampling.py`、`BrepARG/2sequence.py`、`breparg_improvements/train.py`、`tools/evaluate_reconstruction_v13.py`。

### 3.2 当前数据 authority 与完整性

- `ABC/processed/abc_parsed_full_archives` 有 100 个 ZIP（abc_0000 至 abc_0099），约 162.399 GiB，是当前观察到的唯一健康完整重建源。
- `PROJECT_INDEX.md` 所称 `C:\V13_abc_parsed_shards` 当前不存在。
- 100 ZIP 的 manifest 记录了大小/数量但缺少完整 SHA-256；在生成 hash、测试 ZIP、恢复至少两份健康副本之前不得删除。
- `ABC/processed/abc_parsed_full` 为空，不代表数据缺失；实际数据处于 archives 形式。
- sequence package 中 425,120 条序列全部 grammar-valid，说明当前序列结构不是大规模损坏；但 grammar-valid 不等于可重建 BRep。
- 当前 `ABC/processed/train_outputs/ubuntu/sequences_fsq_rcm.pkl` 的 split 是 `382,903/21,214/21,003`，而本地 1024-context AR 的历史 preflight 记录为 `382,720/21,124/21,276`，两者总数同为 425,120。总体 1024 长度保留数也同为 322,546，但没有 artifact 证明二者 record-for-record、split-for-split 完全一致。依据：`docs/audits/v13_ar_distribution_coverage_20260731.json`、`local_runs/ar_training/train_outputs/newscheme_full_v13_ar/ar_preflight_report.json`。

因此下文分布和 parent-CAD 泄露数字描述的是**当前 ubuntu package**；不能静默替代为每一个历史模型实际看到的 split。历史训练 snapshot 需要按其原始 package/hash 单独复审。

### 3.3 train/val/test 分布

| split | N | 长度 median | faces median | edges median | complex 比例 | 证据 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| train | 382,903 | 542 | 14 | 32 | 67.97% | `docs/audits/v13_ar_distribution_coverage_20260731.json` |
| val | 21,214 | 563 | 14 | 33 | 68.41% | 同上 |
| test | 21,003 | 552 | 14 | 32 | 68.16% | 同上 |

验证集不比训练集更短、更少面或更少边。“val CE 总比 train CE 低是因为 val 简单”已被数据统计排除。

当前 faces 范围 2-50，edges 范围 2-150。数据中确实有低面数 CAD，不能通过生成时强制最小面数来“修复”模型；正确做法是报告分桶表现，并在论文 baseline 对齐时明确是否采用论文的最少 10 faces 过滤。

### 3.4 长度过滤偏差

| context | 全部序列保留 | complex 序列保留 | 判断 |
| --- | ---: | ---: | --- |
| 1024 | 75.87% | 64.52% | 系统性排除长复杂样本 |
| 1536 | 92.48% | 88.94% | 成本和覆盖的中间点 |
| 2048 | 98.84% | 98.30% | 当前数据最合理的主 context |

依据：`docs/audits/v13_ar_distribution_coverage_20260731.json`。早期本地 AR120 的 1024 context 能收敛但训练分布偏向简单样本；服务器 2048 方向正确，却因训练稳定性和上游重建瓶颈没有转化为高质量生成。

### 3.5 数据泄露与污染

V13 exact source path、canonical path 和 basename 跨 split 重复均为 0，但 parent-CAD 仍重叠：

- train-val 共享 9,410 个 parent CAD；
- train-test 共享 9,437 个 parent CAD；
- val 中 12,038/21,214（56.75%）记录与其他 split 共享 parent；
- test 中 12,008/21,003（57.17%）共享 parent；
- 状态 `LEAKAGE_DETECTED`。

依据：`docs/audits/v13_sequence_split_integrity_20260731.json`。

same-data BrepARG 也有 parent-CAD 泄露：train `33.83%`、val `464/1000`（46.4%）、test `482/1000`（48.2%）记录共享 parent。依据：`docs/audits/breparg_same_data_split_integrity_20260731.json`。

影响：验证 loss 可被同 CAD 家族的其他 part 降低，不能用于独立泛化结论。该泄露可以解释部分 train/val gap，但不能单独解释复杂曲面 true-token 重建失败或自由生成简单坍塌。

### 3.6 加载、多线程和增强

- V13 AR 不是标准 DataLoader；`breparg_improvements/train.py:_ar_batches` 把所有序列载入内存、Python 排序/组 batch，再传 GPU。CPU worker 数不会提升这段训练，用户此前提高 sequence workers 只影响预处理阶段。
- V13 sequence shard 并行进程会同时用同一 GPU 做 VQ encode；workers 太高会竞争 GPU，CPU 利用率不能直接代表吞吐瓶颈。
- BrepARG VQ DataLoader 默认 workers 4；AR Windows/单机分支约 2，GPU 分支有限制到 8。依据：`BrepARG/trainer.py:59-107,689-733`。
- BrepARG VQ surface/edge 在训练时约 50% 概率随机旋转；sequence train 还可存 90/180/270 度 augmented，val/test 只用 original。依据：`BrepARG/dataset.py:43-98`、`BrepARG/2sequence.py:453-478`。
- V13 `stage_sequence` 当前构造 `args(..., aug=False)`，因此不能默认认为与官方 baseline 使用了相同 augmentation。依据：`breparg_improvements/train.py:611`。

### 3.7 归一化、缺失与异常样本

- parsed geometry 全局归一化到约 `[-1,1]`，surface/edge 还存在 NCS/WCS 转换；bbox token 再映射到离散区间。依据：`BrepARG/process_data/process_brep.py`、`BrepARG/utils.py:502-522`。
- FSQ patch loss 是归一化坐标上的 MSE；低 MSE 不保证边界在 OCC tolerance 下闭合，也不保证曲面局部 Chamfer tail 小。
- complex-curved suite 显示重建失败组 shape-level Chamfer p95 mean `0.32888`，BRep-valid 组仅 `0.07156`，证明几何 tail 与 BRep 失败显著相关，但尚不能区分 quantization、decoder 与 assembly 各自贡献。
- 历史 patch sampling 有大量 source 因 caps 被跳过；caps 改变了复杂度分布。必须把 `source_records_skipped_by_cap` 和每个 bucket 的保留率写入 future manifest。
- 数据标签本质为几何/token 自监督，不存在传统分类标签错误，但 source path、parent grouping、edge orientation、face loop 和 sequence ordering 就是等价的“结构标签”，需要单独完整性检查。

---

## 四、运行环境与硬件层排查

### 4.1 当前本机环境

审计时实际可运行 Python 为：

| 项目 | 当前值 |
| --- | --- |
| OS | Windows 11 Pro，build 26200 |
| CPU | Intel i9-14900KF，32 logical processors |
| RAM | 63.82 GiB，总空闲约 47.73 GiB |
| GPU | NVIDIA RTX 3060 12 GiB，WDDM |
| Driver / reported CUDA | 591.86 / 13.1 |
| Python | 3.9.23，`brepgen_env` |
| PyTorch | 2.2.2+cu118，runtime CUDA 11.8，cuDNN 8700 |
| NumPy / SciPy | 1.26.4 / 1.11.4 |
| Transformers / Diffusers | 4.38.2 / 0.27.0 |

GPU driver 报告支持 CUDA 13.1，不代表 PyTorch 使用 CUDA 13.1；当前 PyTorch wheel 实际是 cu118。两者可以兼容运行，但记录实验环境时必须分别保存 driver、runtime 和 framework build。

系统默认 `python` 不是项目权威环境，历史审计曾发现其没有 torch。所有命令必须显式使用环境 Python，不能只写 `python`。

### 4.2 历史服务器环境

用户终端记录显示服务器为 RTX 5090 32 GiB、driver 580.105.08、nvidia-smi 报告 CUDA 13.0，Python 路径位于 `.../conda_envs/breparg/bin/python`，traceback 显示 Python 3.14。`environment.server.yml` 却仅声明 Python 3.10、NumPy/SciPy/tqdm/zstandard，未锁定 torch、transformers、diffusers、OCC、chamferdist、CUDA 或 cuDNN。

因此：

- **已确认**：服务器产物不能从当前 `environment.server.yml` 严格复现。
- **证据不足**：高学习率发散时加载的服务器源码与当前 D 盘源码是否逐字相同。
- 下一次服务器运行必须在启动时写 `pip freeze`、`conda list --explicit`、`torch.__config__.show()`、GPU/driver 和源码 diff hash。

### 4.3 GPU、显存和训练稳定性

- 服务器 AR `max_seq_len=2048, bs=8` 时 GPU 利用率约 28%；改为 bs=32 后约 96%、显存 24.1 GiB，说明原先主要是 batch 太小而非 CPU worker 不足。
- 后续日志在 epoch 115 前后先出现 CE 持续上升，再在 HF `ForCausalLMLoss` 的 `logits.float()` 尝试额外申请 2.34 GiB 时 OOM。OOM 不是唯一异常，模型在 OOM 前已经数值发散。
- 改为 bs=24/32 不能只按平均显存决定安全性；2048 长序列的 logits、padding bucket 和 float32 loss buffer会造成峰值。稳定 batch 应通过最长 bucket 的多步 smoke 确定，并保留 15-20% VRAM headroom。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 只能缓解碎片，不能解决真实峰值显存不足或 loss 爆炸。
- 当前 V13 AR 没有 gradient accumulation。若需要有效 batch 32，优先物理 batch 16/24 加 accumulation，而不是再次冒险物理 bs=64。

### 4.4 CPU 与 DataLoader

- 审计时没有 V13/BrepARG 训练或生成进程；唯一 Python PID 36112 是 `python -m http.server 8765 --directory D:\luolin\LLM-CAD\...`。RTX 3060 利用率约 0-1%。
- 因此当前“GPU 低、CPU 高”不是本项目训练状态。
- 历史 sequence generation 的 workers 负责解压/解析和 GPU VQ encode；20 workers 会竞争同一 GPU，不能根据 25/32 CPU 核简单线性增加。
- V13 AR 自定义 `_ar_batches` 不读取 `num_workers`，提高 shell 中的 worker 数对 AR 训练无效。

### 4.5 依赖和运行警告

1. `ModuleNotFoundError: chamferdist` 直接导致早期 sequence smoke 失败；后续 CPU evaluation 有 `torch.cdist` fallback，但服务器生成/优化仍应显式检查扩展是否支持目标 GPU。
2. `loss_type=None was set ... default ForCausalLMLoss` 是 Transformers 配置警告，通常不是发散根因；真正风险是该 loss 会把 logits 转 float32，增加 2048 context 的峰值显存。
3. pythonocc/OCC 不是 VQ 训练依赖，却是 STEP 重建与有效性检查的硬依赖；环境文件应拆成 `train` 和 `cad-eval` extras。
4. zstandard 是 `.pkl.zst` shard 读取的硬依赖；缺失时 sequence 构建直接失败。
5. 本地 BrepARG `joint_optimize` 历史上硬编码 `.cuda()`，而安装的 chamferdist 未编译 GPU 支持；小评估中的 CPU patch 不代表原始生产路径已修复。

### 4.6 磁盘与文件系统

| 盘 | 文件系统 | 健康状态 | 空闲 |
| --- | --- | --- | ---: |
| C | NTFS | Healthy / OK | 148.92 GiB |
| D | NTFS | Healthy / OK | 62.01 GiB |
| E | exFAT | Warning / Full Repair Needed | 3,624.31 GiB |

E 的大容量不能抵消文件系统健康告警。正确顺序是先把 E 上唯一内容只读抢救到健康 NTFS并做 SHA-256，然后才考虑修复 E；不能因为“文件名在 E 上看到了”就删除 D 盘权威副本。

---

## 五、算法代码内核剖析

### 5.1 V13 模型结构

#### FSQ-VQ-VAE

`breparg_improvements/train.py:172-179` 构建 Diffusers `VQModel`：

- input/output 3 channels；
- 5 层 encoder/decoder blocks；
- channels `[32,64,128,256,512]`；
- latent channels 128，VQ embed dim 64；
- baseline FSQ levels `(8,8,8,16)`，隐式 codebook `8192`；
- `FSQQuantiser` 先用 1x1 conv 把 64 channels 投影到 FSQ 维数 4，再投回 64 channels；
- FSQ 本身没有 learned embedding/commitment loss，quantizer 返回的 loss 为 0。

这解释了为什么“只提高 levels”不是完整的容量实验：它改变每个量化轴的分辨率/总组合数，但不改变 encoder/decoder、patch 分辨率、4 维量化瓶颈、边界连续性或 BRep assembly。现有 `(16,16,8,8)` 候选只验证了一个 level allocation。

#### AR

`BrepARG/model.py:6-50` 使用 GPT-2 causal LM：

- vocab `10294`；
- d_model 256；
- 8 layers、8 heads、FFN 1024；
- dropout 0.1；
- absolute positional embedding 长度等于 `max_seq_len`。

该模型规模较小，适合本地单卡，但对最长 2,353-token、包含几何/拓扑/bbox 的序列是否足够，当前没有 matched scaling experiment。不能仅凭 validation CE 断言容量足够。

#### 排序

`breparg_improvements/gnn_ordering.py` 的 GNN 使用 RCM 顺序作为监督，通过 pairwise rank loss 学习。因此所谓“几何感知 GNN”并不是独立以 AR likelihood 或几何重建质量优化；它首先在模仿 RCM。若 RCM 对 token 局部性不是最佳，GNN 也可能继承这一限制。

### 5.2 VQ 损失、采样与 checkpoint 选择

- 训练目标是归一化 patch coordinate MSE；complex/curved 样本可通过采样和 per-patch weight 提权。依据：`breparg_improvements/train.py:271,395-415`。
- validation 调用 `weighted_reconstruction_loss(recon, xb)` 而不传 weights，因此仍是全局未分桶 MSE。best checkpoint 也按该全局 val 选取。复杂/曲面训练加权与 checkpoint promotion 指标不一致。
- FSQ patch MSE 平均会被大量简单 patch 主导；complex surface p95 可恶化而 overall val 继续下降。
- 当前 VQ gradient clip 固定 1.0，AdamW weight decay `1e-6`，支持 nonfinite skip 和 early stop；这些稳定性保护优于旧版，但不能代替 bucket-specific validation。
- 当前 resume checkpoint 只保存 `model_state_dict` 和 `fsq_levels`，没有 optimizer/scaler。VQ resume 实际是“加载权重后用新 optimizer 继续”，不是严格训练状态恢复。用户此前从 best resume 的 epoch 编号也不一定等于 checkpoint 权重真实 epoch。

### 5.3 AR 损失、优化和训练循环

- AdamW，weight decay `1e-4`，无显式 learning-rate scheduler；resume 后强制把 optimizer LR 改成 shell 指定值。
- gradient clip 固定 1.0；`NS_AR_GRAD_CLIP` 环境变量无效。
- 训练和验证均使用 teacher forcing causal CE，padding label 设为 `-100`。
- V13 epoch CE 是“每个 batch 的 mean CE 再等权平均”，而 batch 的有效 token 数不同；BrepARG 当前 trainer 使用有效 token 数加权。两套 CE 不能直接横向比较。
- train 使用 dropout，validation 在 `model.eval()` 下关闭 dropout，这是 val CE 低于 train CE 的正常贡献之一。
- validation 不检查 loss finite；训练 nonfinite batch 仅跳过。若整轮都 nonfinite，`tr_ce=0/max(1,0)=0`；latest/periodic 仍会保存。该逻辑与用户看到的 `train_CE=0.0000, val_CE=nan` 完全一致。
- 没有 AR early stop；`NS_AR_EPOCHS` 是目标总 epoch。resume from epoch 20 且设 120 会运行 21-120，不是额外再跑 120。

必须修复为：任一参数/梯度 nonfinite 或连续有限 batch 数不足即 hard fail；禁止覆盖 clean latest；写 `ar_quarantined_nonfinite_epoch*.pt` 仅供取证；从最近 finite best/periodic 恢复。

### 5.4 checkpoint 加载与静默失配

#### V13

V13 VQ、AR、generation 和 reconstruction 的模型权重加载默认 strict；key/shape 不匹配会抛错。依据：`breparg_improvements/train.py:536-537,685-686`、`breparg_improvements/generate_validate.py:93`、`tools/evaluate_reconstruction_v13.py:259,320`。2048 checkpoint 的旧推理 mismatch 已通过读取 config 修复。

#### 原始 BrepARG

这里发现一个高风险但需要准确解释的问题：

- `ARModel` 同时把 GPT-2 注册为 `self.model`，又把 `self.transformer` / `self.lm_head` 指向其子模块，checkpoint 因而可能含 202 keys：101 个 raw keys 和 101 个 `model.*` duplicate keys。依据：`BrepARG/model.py:37-39`。
- 长 baseline epoch-127 resume 的 fresh inspection 结果为 202 keys、missing 0、unexpected 101；101 对 raw/prefixed tensor 全部相等。实际 learned parameter 没有缺失，但 trainer 的 `strict=False` 丢弃 incompatibility lists，日志仍只打印“Model state loaded”。依据：`BrepARG/trainer.py:1045-1048`、`local_runs/breparg_long_ar_resume_best_20260724/ar_epoch127_best_to300/logs/train_ar_resume_best.stdout.log`。
- exception fallback 只取名称/shape 匹配参数，但对当前双注册 `ARModel` 路径本身并不可靠；checkpoint 完全加载失败时外层还可能继续“Training from scratch”。
- 正常 AR constructor 在创建 GradScaler 之前调用 checkpoint loader，因此该路径不能恢复 scaler state。依据：`BrepARG/trainer.py:745-751,1072`。
- VQ loader 会报告 missing/unexpected，长 VQ resume 日志显示完整加载；generation loader也会打印 key warning，已检查的 same-data best 归一化后 missing/unexpected 为 0。

结论：这个 duplicate-key 问题没有证据表明长 BrepARG run 丢了参数，不能用它解释生成质量差；但 permissive loader 会掩盖未来真实失配，必须改为 normalize key 后 strict load，并对 vocab/token semantic 做额外校验。

### 5.5 训练与推理不一致

1. train dropout=0.1，inference/eval 关闭 dropout：行为合理，但解释 train/val gap 时必须计入。
2. 原始 BrepARG generation 构造 dropout=0.0，随后立即 `.eval()`；对当前推理输出无实质影响，但 config 应从 checkpoint 完整恢复。
3. V13 旧 `generate_validate.py` 曾固定 1024，现已读取 checkpoint `max_seq_len`；服务器部署必须使用修复后文件。
4. sequence vocab、special token offsets 存在于外部 package/config；strict state-dict 只能检查 shape，不能检查“同样大小但 token 含义不同”。checkpoint manifest 必须保存 vocab hash。
5. 训练使用 max sequence length 2048，不意味着生成必须 `max_new_tokens=2048`。后者是新生成 token 上限；过小会无 END 截断，过大增加曝光偏差与计算成本。
6. constrained decoding 保证的是 token grammar 的局部规则，不等于几何闭合、曲面连续或 OCC BRep-valid。

### 5.6 报错、警告和异常中断解读

| 现象 | 实际含义 | 是否已解决 |
| --- | --- | --- |
| `No module named chamferdist` | 当前环境缺少构建/运行依赖 | 小评估有 fallback；完整环境仍需 preflight |
| 2048 `wpe.weight` vs 1024 mismatch | 推理模型 context 与 checkpoint 不一致 | 当前代码已修 |
| `loss_type=None ... ForCausalLMLoss` | Transformers 使用默认 causal loss | 警告本身无害；float32 logits 增显存 |
| epoch 115 CE 连续爆炸 | 优化已经失稳，不是普通 batch 波动 | 需 hard nonfinite/gradient guard |
| CUDA OOM 2.34 GiB | 2048 + 大物理 batch 的峰值不足 | 降物理 batch/accumulation；不要 bs64 |
| `train_CE=0, val_CE=nan` | 全部 train batch 被跳过且模型已污染 | 当前代码仍可能产生；必须修 |
| 电脑重启/磁盘写满 | 外部中断，checkpoint lineage 被切断 | 已恢复，但 run manifest 不完整 |
| 生成卡在 OCC/joint optimization | 单个病态 candidate 可无限拖延 | batch controller timeout 已提高鲁棒性，不改变模型质量 |

### 5.7 评价指标实现校验

这是当前实验最容易造成“假成功/假失败”的层面。

#### 官方与本地分母

- 官方 BrepARG README：`Valid = generated/attempts * watertight/generated = valid/attempts`。依据：`BrepARG/README.md:46-49`。
- 本地短/长 baseline 的 `75/92`、`86/100` 是已写出 STEP 幸存者中的 BRep-valid，不是总 attempts 的 Valid。
- 短 baseline 最后可见进度已经到 280 attempts、只保留 92 STEP，且任务未达到 100 STEP 目标；因此 writable success 至多 `92/280=32.86%`，若以 75 个 BRep-valid 为分子，官方式 Valid 上界至多 `26.79%`。依据：`D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\generated_3060_safe_len1536_bs4_20260717_d\generate_serial_fixed.err.log`。
- 长 baseline 的 retained STEP 为 100，但两个 timeout batch 没有终止 attempt summary，前 8 个文件又早于 batch-state logging；完整 aggregate attempt 数不可恢复。现有日志只能确认至少 179 attempts，因此 writable success 上界为 `100/179=55.87%`，以 86 个 BRep-valid 为分子时官方式 Valid 上界为 `48.04%`，不能报告成精确成功率。依据：`D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\generated_3060_long_breparg_resume_best_20260726\generation_manifest.json` 及其 `batch_logs`。

#### 指标语义冲突

- `summarize_generated_quality.py` 把 `brep_valid=true` 称作 strict-valid；`audit_breparg_baseline_outputs.py` 的 strict accepted 则是 grammar、STEP、复杂度、caps、BRep、closed、非 primitive 等完整 gate。两者不能共用“strict”。
- `solid_closed_no_open_shell` 只是 STEP 文本中同时存在 `MANIFOLD_SOLID_BREP`、`CLOSED_SHELL` 且没有 `OPEN_SHELL` 的正则实体检查，不是 CAD kernel 水密证明。依据：`tools/validate_step_quality_once.py:27-83`。
- 当前 BRep-valid 是自定义 one-solid、修复后 wire order/self-intersection、shell bad edge 和 free edge 检查；导入的 `BRepCheck_Analyzer` 没有使用。依据：`BrepARG/utils.py:122-222`。
- wire 先经过 `ShapeFix_Wire.Perform()` 再检查，可能把原始缺陷修复后视为有效；报告应称“post-repair custom BRep-valid”。
- timeout/not-evaluated 常被记为 false 而不是 unknown，可能造成假失败。

#### complexity 与 primitive 冲突

- complex 默认：`faces>=12 OR edges>=20`。
- primitive-like 又定义为 `faces<=12 AND edges<=24` 或命中特定 bucket。
- 因此 12F/20E、12F/24E 会同时是 complex 和 primitive-like。当前 strict gate 对真实、合理的少面 CAD 也可能误判。

必须改为互斥或多标签报告：`topology_size_bucket`、`primitive_classifier`、`geometric_complexity` 分开，不用单一 hard-coded rectangle 代替语义判断。

#### MSE、Chamfer 和 CE

- 本地 MSE：归一化 patch 上所有 `3x32x32` 坐标的 mean squared error。
- 本地 Chamfer：patch 点云下采样后，两方向 unsquared Euclidean nearest-neighbor mean 的和；不除以 2。分母是 patches，不是 shapes。
- 它不是论文 full-shape MMD；不能把 patch Chamfer p95 写成生成 MMD。
- teacher-forcing `token_weighted_ce` 是按真实 target token 数加权；`mean` 是每 shape 等权；V13 epoch CE 又是 batch 等权。表格必须写清 aggregation。

#### `VERIFIED` 假成功

- `tools/evaluate_reconstruction_v13.py:721` 在不写 STEP 时只要有 rows 即 VERIFIED；写 STEP 时只需至少一个 STEP saved，不要求 BRep-valid。
- `tools/complex_curved_diagnostics.py:827` 只要选满样本就 VERIFIED，即使 reconstruction skipped。
- 因而 VERIFIED 只表示“脚本按配置完成”，不能作为几何质量通过。

#### 尚未实现的官方指标

当前没有协议匹配的 COV、MMD、JSD、Novelty、Uniqueness 和总 attempts Valid。STEP SHA-256 uniqueness 只是字节唯一，不是几何唯一。没有这些指标前，不能宣称 V13 优于官方 BrepARG，也不能宣称官方 BrepARG 失败。

---

## 六、多轮实验结果纵向对比

### 6.1 VQ-VAE 训练与 checkpoint 选择

| run | 主要改动 | 训练结果 | 能支持的结论 | 不能支持的结论 |
| --- | --- | --- | --- | --- |
| `newscheme_full_local` | 初始全量 FSQ | best val 约 `0.00082`，末期 val inf | 旧训练有数值稳定性缺陷 | 不能用末期 report 评价复杂曲面 |
| `newscheme_full_vqvae_stable` | finite guard、关闭/控制 AMP、history | 40 epochs，best `0.00056`，全 finite | 稳定性修复有效 | 不能证明 paper-quality reconstruction |
| `newscheme_full_vqvae_epoch100` | 从 stable best continuation、低 LR | early stop epoch 85，best epoch 73、约 `0.00037` | overall val MSE 继续改善 | best 权重不等于 exact epoch-40 resume；不证明 curved tail |
| 5090 resume branch | 从旧 full epoch100 best 继续 | 用户日志 best val 约 `0.00005`，early stop epoch 164 | 在该 validation 上继续训练有效 | 缺完整本地 history/optimizer provenance |
| 5090 scratch branch | 从零到 epoch 107，再继续至 early stop 440 | continuation best epoch 340、日志 best val 约 `0.00005` | scratch 也能达到相近 aggregate val | 当前没有保留的同一 50k 独立评估证明其超过 resume |
| levels `(16,16,8,8)` | 增大 FSQ levels/codebook | overall Chamfer p95 比 baseline 差 `4.19%`；surface p95 好 `26.91%`；edge p95 差 `2.48%` | 容量信号 mixed/inconclusive | 不能得出“提高 levels 可解决曲面” |

用户曾提供独立 50k patch MSE 比较：

| candidate | mean | median | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| old full epoch100 | 0.00042419 | 0.00002229 | 0.00075367 | 0.34659 |
| 5090 resume best | **0.00037250** | **0.00000914** | **0.00064665** | 0.43964 |
| 当时 scratch best | 0.00045153 | 0.00003321 | 0.00104564 | **0.32611** |

该 JSON 后来按用户要求删除，因此表格依据是用户提供终端记录，不是当前仓库文件。它支持当时选择 resume 作为下游候选；它不包含 scratch continuation 到 epoch 340 后的 matched 重评。`ABC/processed/train_outputs/ubuntu/fsq_vqvae_best.pt` 当前 finite，levels `[8,8,8,16]`，但 checkpoint 本身没有 epoch/optimizer/config，无法独立恢复其完整 lineage。

### 6.2 V13 AR 训练

#### 本地 1024 context

- 多个 LR continuation 最终到 epoch 120，best val CE `0.2949333`，model/optimizer/scaler 可加载。
- 训练曲线整体收敛，说明 token prediction pipeline 可学习。
- 但 1024 只覆盖约 64.5% complex rows，且 split 有 parent leakage；该 CE 不能作为复杂 CAD 泛化证明。

依据：`plans/ar_training_v13_execplan.md`、`local_runs/ar_training/train_outputs/newscheme_full_v13_ar_lr5e6/ar_history.jsonl`。

#### 服务器 2048 context

当前回传 `ABC/processed/train_outputs/ubuntu/ar_best.pt` 的 CPU 只读检查：

- epoch `122`；
- train CE `0.3160388`；
- val/best CE `0.2983596`；
- physical batch `32`；
- LR `2e-5`；
- max_seq_len `2048`；
- 101 个 floating tensors 全部 finite。

它是可用生成候选，不是历史 nonfinite latest。由于数据/context/代码和 CE aggregation 可能与本地 1024 run 不同，`0.29836` 与 `0.29493` 不能直接判断谁更好。

历史服务器 run 的 `ar_latest.pt` 曾被确认 `transformer.wte.weight` nonfinite；任何 `val_CE=nan` 的 latest 均不得作为 resume 或生成权重。

### 6.3 V13 自由生成

本地 G100（epoch-120 AR、temperature 0.9、top_p 0.92、max_new 320）：

- attempts 100；grammar-valid/STEP 87；当前 custom BRep-valid 78；
- 13 个失败均为到 321 tokens 仍无 END；
- 62/87 落在 6F/12E 或 4F/6E；
- 87/87 STEP 文件 hash 唯一；
- complex strict-valid 0；paper gate `hold_for_failure_analysis`。

后续服务器 2048-context 权重又留下了三种**不可混为一张成功率表**的协议：

| 协议 | attempts | STEP | custom BRep-valid | complex | strict accepted | 可解释范围 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 标准自由生成 | 103 | 100 | 94 | 未审计 | 未审计 | 只能证明多数 attempt 可写 STEP，不能推断复杂度 |
| 同一 V13 权重 + BrepARG 原始 sampler | 124 | 103 | 94 | 11 | 2 | 换 sampler 后仍以简单拓扑为主 |
| rejection quality gate | 2,405 | 1,630 | 1,242 | gate 内含 | 100 accepted | 100 是筛选幸存者，不是 100 attempts 的模型成功率 |

依据：`local_runs/ubuntu_generation_eval_20260713/generated100_step_png_best/generation_report.json`、`local_runs/breparg_logic_compare_20260715/breparg_original_logic_100_20260715/breparg_logic_report.json`、`local_runs/breparg_logic_compare_20260715/breparg_original_logic_100_20260715/breparg_baseline_quality_summary.json`、`local_runs/ubuntu_generation_eval_20260713/quality_gated_100_20260713/quality_gated_report.json`。

这些实验共同说明失败不是字节级复制，而是语义/拓扑分布坍塌；调 sampler 或拒绝坏样本只能改变幸存者构成。当前模型可以生成可读样本，但不能作为高质量复杂 CAD 生成结果。

依据：`local_reports/generated100_lr5e6_epoch120_best_20260705.md`、`local_reports/v13_generated_quality_gate_20260705.md`。

### 6.4 复杂曲面纯 FSQ 与真实 token 重建

固定 50 shape、3,399 patches：

| 指标 | 数值 |
| --- | ---: |
| patch Chamfer mean / median / p95 | 0.04662 / 0.01526 / 0.15012 |
| surface Chamfer p95 | 0.41238 |
| edge Chamfer p95 | 0.10169 |
| AR token-weighted teacher CE | 0.74674 |
| true-token STEP saved | 27/50 |
| true-token custom BRep-valid | 9/50 |

shape-level FSQ/BRep 相关性：

- BRep-valid 组 Chamfer-p95 mean `0.07156`、median `0.05650`；
- invalid/failed 组 mean `0.24405`；
- reconstruct-failed 组 mean `0.32888`、median `0.12003`。

这是“AR 采样不是唯一根因”的直接证据。需要注意，true-token reconstruction 同时经过 FSQ decode 和 BRep assembly，所以它仍不能单独证明量化器而不是 assembly 是唯一问题。

### 6.5 faces/edges/length 分桶

50-shape cohort 的 true-token BRep-valid：

| bucket | valid / attempted |
| --- | ---: |
| faces 0-11 | 1/13 |
| faces 12-19 | 5/18 |
| faces 20-29 | 2/9 |
| faces 30-50 | 1/10 |
| length 0-512 | 4/20 |
| length 513-1024 | 3/18 |
| length 1025-1536 | 2/8 |
| length 1537-2048 | 0/4 |

所有 face/length 桶都很差；最长桶没有有效样本，但 N=4 太小，不能证明单调因果。patch Chamfer 甚至在高 face bucket 更低，说明 patch 加权统计受到 shape composition 和 patch 数影响。下一次应至少固定 200-500 shapes，shape 等权报告，并对 source surface type 分层。

依据：`local_runs/complex_curved_rootcause_suite_20260715/experiments/01_teacher_forcing_true_token_reconstruction/bucket_summary.json`。

### 6.6 DFS 与 RCM

同一 50-shape recovered cohort：

| order | target tokens | token-weighted CE | shape-mean CE |
| --- | ---: | ---: | ---: |
| DFS | 38,324 | **1.25869** | **1.40148** |
| RCM | 38,324 | 1.31302 | 1.44099 |

DFS 相对 RCM token-weighted CE 约低 4.14%。这是有价值的 ordering signal，但不是完整公平训练：保存的最终 DFS/RCM checkpoint 已不在原 D 路径，当前只保留 finite-check 和评估产物；也没有 parent-isolated、同 epoch、多 seed 自由生成对照。

### 6.7 BrepARG 短/长 baseline

| 指标 | 短 baseline | 长 baseline | 说明 |
| --- | ---: | ---: | --- |
| train/val/test | 10k/1k/1k | 同一 split | 但 parent leakage |
| VQ 训练 | configured 160；best 70，约 epoch73 磁盘中断 | 完成 400；best 269 | 长 VQ best val recon 约 0.0002106；epoch 400 recon/total 约 0.000219/0.000262 |
| AR 训练 | 完成 80；best 77，val CE 0.871925 | 完成 300；best 254，val CE 0.765318；epoch 300 CE 0.767252 | 重启后从 epoch127 best 恢复 |
| STEP survivors | 92 | 100 | 不是 attempts 分母 |
| custom BRep-valid | 75/92 | 86/100 | survivor-conditioned |
| complex | 5/92 | 13/100 | 有改善，仍低 |
| strict accepted | 0/92 | 6/100 | 有改善，仍不合格 |
| face/edge median | 6/12 | 6/12 | 主分布未改变 |

短 baseline 在最后可见的 280 attempts 后只保留 92 STEP，生成任务未达到目标；长 baseline 保留 100 STEP，但完整 attempts 因 timeout 和早期日志缺口无法恢复。表内 valid/complex/strict 都是 retained-STEP 条件分母，不是官方 `valid/attempts`。

长训练证明 epoch 不足是一个次要因素，但不支持“只要继续加 epoch 就会解决”。官方 README 的 3000/500 示例说明当前 VQ400/AR300 仍比示例短，但数据规模、硬件和 early-stop 都不同；更重要的是曲面/assembly 组件 gate 尚未通过，盲目堆 epoch 的信息增益低。

### 6.8 为什么 validation CE 长期低于 train CE

**已排除：验证集更简单。** 三个 split 的长度、faces、edges、complex 比例近似相同。

更可信的组合解释：

1. 训练时 dropout=0.1，validation 关闭 dropout；train CE 自然偏高。
2. parent-CAD 泄露让 validation 含训练 CAD 家族的其他 part，降低泛化难度。
3. train 和 val 的 batch 顺序/长度组成不同，又使用 batch 等权 mean，进一步制造 gap。
4. 若训练含 augmentation 而 validation 不含，baseline gap 会扩大；V13 当前 original-only 需单独确认每个 run manifest。
5. train CE 是参数持续更新时的 online average，validation 是 epoch 末固定模型的 average；这也可造成 validation 较低。

因此该 gap 本身不是 NaN 或过拟合证据。真正异常是 val 持续改善而自由生成/复杂 subset 不改善，以及任何 CE 爆炸、0 或 NaN。

### 6.9 核心现象归纳

- **不是传统单一过拟合。** 全局 val CE 低且下降，但复杂几何重建和生成仍差，属于目标/分布/指标错配。
- **不是简单欠拟合。** 模型对常见简单序列有很低 CE，继续训练主要强化头部模式；复杂 tail 仍弱。
- **存在数值发散事件。** 高 LR/大 batch 2048 run 在 epoch 115 后爆炸并污染 latest。
- **存在模式坍塌。** 自由生成集中于少数简单拓扑，byte hash 唯一不能否定语义坍塌。
- **存在上游重尾误差。** surface Chamfer p95 和 true-token BRep 失败说明平均 MSE 掩盖 tail。
- **存在评测污染。** parent leakage、survivor denominator 和 strict 语义冲突会同时制造假成功。

### 6.10 复现性

- V13 在若干工具中固定 seed 0，生成工具可指定 seed；但没有完整 CUDA deterministic 配置。
- 没有同一最终协议的 3-seed 训练；现有不同 seed 多为生成小样本，不是训练方差。
- 没有保存每个历史 run 的源码 hash 与完整环境，因此“同参数”不一定是同实现。
- 官方协议要求 10 次独立生成运行；当前没有完成。

结论：现有差结果的跨实验一致性说明问题更像系统性而非单次随机波动，但定量方差仍是证据不足。

---

## 七、易遗漏自查清单

| 核查项 | 当前状态 | 风险与动作 |
| --- | --- | --- |
| 参数初始化 | GPT2 initializer 0.02；VQ 使用库默认 | 保存完整 config 和权重初始化 seed |
| Python/NumPy/Torch seed | 部分固定为 0 | 加 `manual_seed_all`、worker seed |
| CUDA 确定性 | 未完整开启，TF32 开启 | 复现实验使用 deterministic profile；性能 run 单独标记 |
| train/val/test parent 隔离 | **失败** | 先建 parent-CAD split，再做最终训练 |
| exact path overlap | 通过 | 不能替代 parent 检查 |
| 训练/验证分布 | 已审计，近似一致 | 保持自动审计为 preflight |
| VQ checkpoint save best vs last | 有 best/final，但 resume 缺 optimizer | 记录真实 epoch；严格区分 weight continuation 与 state resume |
| AR save best vs latest | 有，但 latest 可被 NaN 污染 | nonfinite hard fail；原子写；finite 校验后更新 alias |
| early stop | VQ 有；AR 无 | AR 加 patience 前先用 complex subset metric |
| LR scheduler | V13 AR 无 | 不应直接加 scheduler；先做稳定、matched LR ablation |
| gradient clip | 实际硬编码 1.0 | 暴露受测 config，记录 pre/post clip norm |
| AMP/GradScaler resume | V13 可恢复；BrepARG constructor 顺序有缺陷 | loader 后创建 scaler 或二次 restore；写测试 |
| dropout/eval mode | train 0.1、eval 关闭 | 行为正确；解释 train/val gap |
| BN eval | 当前 VQModel 主要 GroupNorm，无典型 BN 风险 | 仍统一 `.eval()` 评测 |
| strict checkpoint load | V13 strict；BrepARG permissive | normalize keys 后 strict；禁止 silent scratch fallback |
| vocab/token semantic | 只靠 size 不够 | 保存 vocab JSON/hash、special token offsets |
| max_seq_len | 早期 1024 排除复杂样本 | 以 2048 为主，1536 作成本对照 |
| END/truncation | G100 有 13/100 无 END | 报告 truncation；不要只加 max_new 掩盖 |
| augmentation 对齐 | V13 与 BrepARG 可能不同 | manifest 固定并做 same-protocol 对照 |
| normalization/tolerance | normalized MSE 与 OCC tolerance 未闭环 | 同时报 WCS boundary gap、loop closure error |
| face/edge cap 语义 | global/per-face 名称混乱 | 拆成三个明确参数并单测 |
| 低面数真实数据 | 存在 2-9 face 样本 | 不用生成硬门替代数据协议；分桶报告 |
| 50-face 边界 | 在协议内且最难 | 失败率高可以预期，但不能归为 out-of-scope |
| patch/shape aggregation | 当前多为 patch-weighted | 增加 shape-equal、surface-type 分层 |
| codebook 使用率 | 局部有 perplexity，未纳入最终诊断 | 报 occupancy、dead/rare codes、curved code entropy |
| teacher-forced argmax | **未做** | 真实 prefix 逐 token argmax 后重建 |
| continuous-latent bypass | **未做** | 隔离 quantization penalty |
| ground-truth assembly oracle | **未做** | 隔离 OCC/loop assembly 错误 |
| BRep validator | 自定义 post-repair，未用 Analyzer | 同时报告 raw、repaired、BRepCheck、free-edge |
| timeout 语义 | false/unknown 混用 | 三态 `pass/fail/not_evaluated` |
| attempts 分母 | baseline 不完整 | generation controller 必须原子计数所有 attempts |
| geometric uniqueness | 只有 SHA-256 | 加 canonical geometry/point-cloud distance 去重 |
| official metrics | 未复现 | 最终才跑 3k x 10；内部 gate 不冒充官方 |
| 日志与源码 hash | 历史缺失 | run manifest 强制化 |
| 磁盘空间预警 | 历史写满 | 启动前空间预算；checkpoint retention policy |

### 7.1 已排除或不能再作为“唯一解释”的假设

1. **“验证集更简单，所以 val CE 低”**：分布统计不支持。
2. **“只是 temperature 太低”**：多温度实验没有恢复复杂 valid 分布。
3. **“关闭 bbox monotonic 就好”**：复杂度与 invalid 同时上升，根因仍在。
4. **“强制最小 faces/edges 就好”**：会产生碎片、自交和非闭合，且会误伤真实低面 CAD。
5. **“生成后筛掉坏样本即可”**：约 2,405 attempts 才收集 100 accepted 的历史实验只改变幸存者，不改变模型分布。
6. **“换成 BrepARG 原始生成逻辑即可”**：同权重对照仍以简单拓扑为主。
7. **“所有 STEP hash 唯一，说明多样性足够”**：G100 87/87 hash 唯一但 top-two topology 71.26%。
8. **“AR 是唯一问题”**：真实 token 直接重建只有 9/50 BRep-valid。
9. **“只提高 FSQ levels 即可”**：`(16,16,8,8)` overall p95 反而差 4.19%。
10. **“只增加训练 epoch 即可”**：BrepARG AR80 到 AR300 有改善，但 median topology 不变。
11. **“50-face 失败说明超出数据上限”**：50 正是当前表示上限，属于 in-protocol 边界；只能单独报告难度，不能排除。
12. **“官方 BrepARG 也已经证明效果不好”**：未完成兼容官方权重/协议复现，不能下此结论。

### 7.2 尚未排除的假设

- FSQ 4 维量化瓶颈不足，但需要 continuous bypass 和等 codebook-size dimensionality control。
- decoder 或 patch parameterization 对高曲率 surface 不足。
- face/edge boundary 独立 decode 后的连续性误差被 OCC 放大。
- BRep assembly/loop orientation 是与 FSQ 并列的主瓶颈。
- RCM 破坏局部 token transition，但需 full matched multi-seed 验证。
- AR d256/L8 容量不足或训练分布不平衡，但必须在上游重建 gate 通过后验证。
- parent-CAD 泄露使所有 validation-based model selection 过度乐观。

---

## 八、归因总结与分级整改计划

### 8.1 两类根因必须分开排序

#### A. 科学结论可信度风险

| 优先级 | 根因 | 置信度 | 判断依据 |
| --- | --- | --- | --- |
| P0 | parent-CAD split 泄露 | 已确认 | val 56.75%、test 57.17% 记录共享 parent |
| P0 | metric/denominator 与官方协议不一致 | 已确认 | survivor-conditioned validity、strict 多义、官方指标缺失 |
| P1 | 运行源码/环境/manifest 不完整 | 已确认 | Git HEAD 落后、dirty tree、server freeze 缺失 |
| P1 | baseline 非官方权重复现且 protocol 不完全同条件 | 已确认 | vocab 7222 vs 10294；augmentation/caps/sampling 差异 |
| P2 | 随机种子和 CUDA 确定性不足 | 已确认 | 无完整 3-seed / deterministic 设置 |

#### B. 生成质量技术根因

| 优先级 | 根因 | 置信度 | 判断依据 |
| --- | --- | --- | --- |
| P0 | 复杂 surface/edge patch 的重尾几何误差 | 强证据 | surface Chamfer p95 0.41238；失败组 p95 显著高 |
| P0 | BRep reconstruction/assembly 对边界与 loop 错误脆弱 | 强证据 | true-token 仅 27/50 STEP、9/50 valid；所有桶均差 |
| P1 | AR 对复杂长序列建模与自由运行误差累积 | 强证据 | 1024 coverage 偏差、complex CE、简单拓扑/截断、历史发散 |
| P2 | RCM/GNN ordering 增加建模难度 | 中等证据 | DFS CE 约优 4.14%，未完成 full matched |
| P2 | 数据/损失/selection 被简单 patch 主导 | 强推断 | overall val 与 curved tail/生成质量脱钩 |
| P3 | 生成约束和质量门诱发幸存者偏差 | 已确认 | 大量 attempts 换少量 accepted，不提升模型分布 |

不能说“VQ、AR、排序都同等失败”。准确表述是：VQ/decoder/assembly 有直接失败证据；AR 在其上增加额外失败；ordering 只有次级对照信号。

### 8.2 一级修复：1-2 轮即可判定方向

#### 一级 A：先修实验协议，不训练

1. 按 parent CAD UUID 重建 `train/val/test`，固定并 hash。
2. 冻结一个独立 `complex_curved_test_v1`，建议至少 200 shapes，覆盖 surface type、faces、edges、length，禁止从 train 选择。
3. 统一 metric schema：`attempted`、`generated`、`step_readable`、`raw_brep_valid`、`repaired_brep_valid`、`closed_kernel`、`complex`、`geometric_unique`，每项有分母和 unknown。
4. 冻结当前 V13、same-data BrepARG 和 evaluator 的源码/environment/vocab hash。

**通过条件**：split audit `NO_LEAKAGE`；所有 metric JSON schema test 通过；任意比例可还原 numerator/denominator。

#### 一级 B：四阶组件 oracle + free-running 参照，隔离真正主因

在同一 complex-curved cohort、同一 shape 顺序上跑：

1. **Ground-truth assembly oracle**：从 parsed 的真实 surface/edge/loop 数据直接走 assembly，不经 VQ/AR。
2. **Continuous autoencoder**：encoder -> continuous latent -> decoder -> assembly，绕过 FSQ rounding。
3. **FSQ oracle**：encoder -> FSQ -> decoder -> assembly，真实 topology/token。
4. **AR teacher-forced argmax**：每个位置用真实前缀预测 argmax token，拼成预测序列后走 grammar/assembly。
5. **Free-running AR**：不属于 oracle，作为最终参照层。

这比笼统“再测一次 FSQ”更能定位：

- 1 已失败：首查 parsed/assembly/OCC；
- 1 通过、2 失败：decoder/parameterization；
- 2 通过、3 失败：FSQ quantization；
- 3 通过、4 失败：AR conditional prediction；
- 4 通过、5 失败：exposure bias/sampling。

**建议通过门槛**：ground-truth assembly ≥95% raw valid；continuous ≥90%；FSQ true-token ≥70%；teacher-forced argmax 显著高于 free-running且 grammar ≥90%。门槛是工程决策线，必须在独立 test 上报告置信区间。

#### 一级 C：修训练安全性

1. AR 任一参数/梯度 nonfinite 立即停止，禁止更新 latest。
2. validation loss nonfinite 立即失败，不允许 `VERIFIED`。
3. 物理 bs 从 16/24 起，用 gradient accumulation 到 effective 32；2048 最长 bucket 做 100-step smoke。
4. checkpoint 写临时文件、CPU finite+key+vocab 检查通过后原子替换 alias。
5. BrepARG loader normalize duplicate namespace 后 strict load；checkpoint 无 model state 或路径不存在必须 fatal，除非显式 `--allow-scratch`。

### 8.3 二级优化：一次只改一个变量

前提：一级 oracle 已定位瓶颈，所有实验使用相同 parent split、cohort、seed 和训练预算。

#### 实验 E1：FSQ 量化维数，而非只扩 levels

- baseline `(8,8,8,16)`：4 quantized dims，8192 combinations。
- control `(8,8,8,4,4)`：5 quantized dims，同为 8192 combinations。
- encoder/decoder、采样、loss、epoch 全固定。

目的：在总离散容量不变时检查更多量化轴是否改善曲面。主指标：shape-equal surface Chamfer p95、boundary gap p95、true-token raw/repaired valid。

#### 实验 E2：curved sampling

只把 curved patch sampling fraction 提高，loss weight 保持 1；记录每个 source shape 的采样概率，避免单一复杂 shape 贡献过多 patches。

#### 实验 E3：surface/curved loss

sampling 与模型固定，只改变 curved/surface loss weight。checkpoint 按 complex-curved validation promotion，不按 overall MSE。

#### 实验 E4：decoder capacity / boundary objective

先只扩 decoder channels或 residual blocks，不改 FSQ；若 oracle 显示 boundary 为主，再加 edge-surface boundary consistency loss。不要一次同时改 decoder、levels、sampling、loss。

每个 E1-E4 先 1 seed 做 stop/go；候选至少满足：surface Chamfer p95 相对 baseline 改善 ≥15-20%，edge p95不恶化 >5%，true-token valid绝对提升 ≥15 percentage points。再做 3 seeds。

#### 实验 E5：DFS/RCM full matched

- 固定 promoted VQ、parent split、sequence caps、augmentation、AR architecture、context 2048、training tokens 和 seed；
- 只改 order；
- 同时报 token-weighted CE、teacher-forced argmax reconstruction、free-running Valid/complex 和 transition entropy；
- 3 seeds 后才决定主线。

若 DFS 持续显著领先，V13 主线暂切 DFS/hybrid，RCM/GNN 降为 ablation；不要为保留创新点而忽略负结果。

#### 实验 E6：AR 复杂度与 context

在 promoted VQ/order 后依次：

1. 2048 uniform baseline；
2. 2048 shape-equal complexity-balanced sampler；
3. 1024 -> 1536 -> 2048 curriculum；
4. 如仍欠拟合，再单独扩大 d_model/layers。

每次只改一项。记录全局与 complex subset 的 token-weighted CE、EOS rate、grammar、teacher-forced argmax 和 free-running结果。validation best 不得只看泄露 split 的全局 CE。

### 8.4 三级重构备选

仅当一级 oracle 表明当前表示/assembly 无法达到门槛时启动：

1. surface 与 edge 使用独立容量和损失，不再共享同一 patch treatment。
2. 显式建模 face-loop-edge incidence、orientation 和 boundary endpoints，减少 OCC 事后猜测。
3. decoder 增加 boundary-conditioned surface/edge reconstruction和 loop closure objective。
4. 将 topology generation 与 geometry refinement 分成两阶段；先生成合法拓扑，再条件生成/优化几何。
5. 使用 CAD kernel-in-the-loop repair 作为可测模块，但同时报告 repair 前后 Valid，避免把 repair 当成生成质量。
6. 若无条件 AR 仍坍塌，先做条件生成/前缀 completion 作为诊断和应用分支，不把它冒充无条件 baseline。

### 8.5 推荐执行顺序与停止规则

```text
冻结源码/环境/parent split
  -> ground-truth / continuous / FSQ / teacher-argmax 四阶组件 oracle
  -> 若 ground-truth assembly 不过：先修数据与 assembly
  -> 若 continuous 过而 FSQ 不过：做 FSQ dim/sampling/loss 单变量
  -> 若 FSQ 过而 teacher-argmax 不过：做 ordering/AR
  -> 若 teacher-argmax 过而 free-running 不过：做 curriculum/exposure/sampling
  -> 组件 gate 全过后，才跑 100/500 内部生成
  -> 最后跑官方规模、多 seed、共同 metric 的 V13 vs BrepARG
```

停止规则：任一 VQ candidate 在固定 complex-curved test 上不能优于 baseline，就不重建全量 sequence；任一 ordering candidate teacher-forced argmax 不优于 baseline，就不训练长 AR；任一 AR branch 出现 nonfinite，立即回滚并隔离，不继续“看它会不会恢复”。

### 8.6 当前最小可落地下一步

1. 先不继续盲目延长现有 AR/VQ epoch。
2. 保存当前 `ubuntu/ar_best.pt` epoch 122 和 VQ checkpoint 的 SHA-256/manifest；隔离历史 nonfinite latest。
3. 实现并运行四阶组件 oracle 的前 3 阶：ground-truth assembly、continuous bypass、FSQ true-token；先用现有 50 shapes做工具验证，再扩大到 parent-isolated 200 shapes。
4. 同时修 metric schema 和 parent-CAD split；这两个是任何后续论文比较的前置条件。
5. oracle 确认 FSQ 是主要增量损失后，优先跑等 codebook-size的 4-dim vs 5-dim实验，而不是再次单纯扩大 levels。
6. VQ 通过后再做 full DFS/RCM，最后才做 AR 2048 和生成质量门。

### 8.7 当前项目状态

- V13/BrepARG 训练：无运行任务。
- 生成：已结束；已有 STEP/PNG 证据，但视觉质量一般，与量化审计一致。
- 当前可用 V13 AR：`ABC/processed/train_outputs/ubuntu/ar_best.pt`，epoch 122，finite。
- 当前文件整理：仅完成只读盘点和建议；本报告创建前后均未删除、移动、修复或覆盖数据/权重/结果。
- 待用户决定：是否批准 2.8 节的精确删除候选；未批准前保持原样。

---

## 证据索引

### 核心代码

- V13 训练：`breparg_improvements/train.py`
- FSQ：`breparg_improvements/fsq_quantise.py`
- RCM/GNN：`breparg_improvements/gnn_ordering.py`
- AR model：`BrepARG/model.py`
- BrepARG trainer：`BrepARG/trainer.py`
- sequence：`BrepARG/2sequence.py`
- V13 reconstruction：`tools/evaluate_reconstruction_v13.py`
- complex diagnostics：`tools/complex_curved_diagnostics.py`
- split audit：`tools/audit_split_integrity.py`
- quality gate：`tools/generation_quality_gate.py`、`tools/summarize_generated_quality.py`
- STEP/BRep validation：`tools/validate_step_quality_once.py`、`BrepARG/utils.py`

### 核心审计结果

- `docs/audits/v13_sequence_split_integrity_20260731.json`
- `docs/audits/breparg_same_data_split_integrity_20260731.json`
- `docs/audits/v13_ar_distribution_coverage_20260731.json`
- `local_runs/complex_curved_rootcause_suite_20260715/fsq_capacity_comparison.json`
- `local_runs/complex_curved_rootcause_suite_20260715/experiments/00_fsq_only_patch_metrics/complex_curved_diagnostics_report.json`
- `local_runs/complex_curved_rootcause_suite_20260715/experiments/01_teacher_forcing_true_token_reconstruction/complex_curved_diagnostics_report.json`
- `local_runs/complex_curved_rootcause_suite_20260715/experiments/01_teacher_forcing_true_token_reconstruction/reconstruction_fsq_correlation.json`
- `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval`
- `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback\breparg_same_data_quality_summary.json`
- `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\breparg_same_data_resume_best_quality_summary_20260726.json`

### 历史决策与视觉证据

- `plans/stable_vqvae_retraining_execplan.md`
- `plans/vqvae_epoch100_continuation_execplan.md`
- `plans/ar_training_v13_execplan.md`
- `plans/complex_curved_fsq_ar_diagnostics_execplan.md`
- `plans/breparg_long_baseline_and_v13_diagnosis_20260720_execplan.md`
- `local_reports/v13_generation_quality_root_cause_20260705.md`
- `local_reports/v13_generated_quality_gate_20260705.md`
- `local_reports/breparg_official_weight_probe_20260717.md`
- `local_runs/reconstruction_eval/eval_generated100_lr5e6_epoch120_best_temp09_topp92_max320_random_cpu_20260705_005342/renders/contact_sheet.png`
- `D:\V13_rootcause_recovery_20260717\breparg_same_data_fallback_long_20260720\generated_3060_long_breparg_resume_best_20260726`

## 审计声明

本轮新增了报告与只读统计审计文件，并增强了 sequence 分布统计测试。没有删除或移动任何既有文件；没有启动训练；没有将 E 盘视为健康 authority；没有把同名 checkpoint 判作重复；没有将用户已证明无效的生成调参重新列为主修复方案。
