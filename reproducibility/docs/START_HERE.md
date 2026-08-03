# V13 / BrepARG 可复现实验包

这是一个面向 Linux x86_64 NVIDIA GPU 服务器的轻量级源码包。它包含当前可运行源码、干净 Git 参考层、实验描述、环境配置、完整复盘、实验记录、事故记录和外部制品身份契约；不包含模型权重、序列包、解析数据、STEP、PNG 或其他大型输出。

## 先读结论

项目已经完成端到端训练与 STEP 生成，但当前复杂 CAD 生成质量仍不合格。现有证据将首要瓶颈定位在复杂曲面 FSQ/VQ 表征、解码与 BRep 装配链路：复杂曲面子集的 surface Chamfer p95 约为 `0.41238`；绕过自由运行 AR、直接重建真实 token 时，50 个样本只有 27 个写出 STEP，9 个通过当前 BRep-valid 检查。AR 还有独立退化，表现为复杂子集 CE 较差和自由生成向简单拓扑坍塌。DFS 比 RCM 的 teacher-forcing CE 略好，但排序差异不足以解释主要失败。

历史 V13 与 same-data BrepARG 划分都存在 parent-CAD 跨 split 重叠，因此历史 validation CE 不能当作独立泛化证据。当前 BrepARG 对照是同数据自训 baseline，不是官方权重复现；官方权重词表大小 `7222`，本地协议词表大小 `10294`，不可直接互换。更长的 BrepARG VQ400/AR300 训练改善了复杂和严格有效样本数量，但没有消除简单拓扑坍塌。

当前表示协议允许 2 至 50 faces、2 至 150 global edges。50-face 样本是协议内高难度边界；失败可以预期，但不能标为 out-of-scope。数据中本来就有低面数 CAD，生成时强制最小面数只能改变幸存者集合，不能修复模型分布。

完整论证见 `project_history/01_full_postmortem/full_experiment_postmortem_20260731.md`，浓缩结论见 `reports/current_conclusions.md`。

## 包内结构

- `source/current/`：打包时的当前脏工作树，默认可运行源码快照。
- `source/clean_head_16cf19b/`：外层仓库提交 `16cf19b` 的干净参考层。
- `provenance/`：当前/干净源码清单、diff、未跟踪文件决策和 BrepARG commit/provenance。
- `experiments/`：`recommended`、`baselines`、`diagnostics`、`historical_failed` 四类实验描述。
- `artifact_specs/`：每个 external artifact 的大小、哈希、兼容性、路径变量和恢复说明。
- `environments/`：Linux GPU 环境、固定 pip 版本、CUDA/OCC 探针和可选 CAD 层安装器。
- `project_history/`：完整复盘、时间线、实验账本、计划、事故、协议和原始文本证据。
- `launchers/`：不依赖 PyTorch 的控制面和 BrepARG 生成包装器。
- `reproduce.sh`：唯一公共命令入口。

## 路径一：理解项目过程

按以下顺序阅读：

1. `reports/current_conclusions.md`
2. `project_history/00_READ_ME_FIRST.md`
3. `project_history/01_full_postmortem/full_experiment_postmortem_20260731.md`
4. `project_history/02_timeline/project_timeline.md`
5. `project_history/03_experiment_ledger/experiment_ledger.md`
6. `project_history/06_failure_incidents/incident_register.md`
7. `project_history/08_evidence_index/README.md`

## 路径二：配置服务器

要求：Linux x86_64、NVIDIA 驱动、支持 CUDA 12.8 的 GPU、Conda/Mamba/Micromamba、Bash 和足够的外部数据空间。推荐 RTX 5090 或同等级 32 GB GPU；不同实验的实际显存要求见实验描述。

```bash
unzip v13_repro_source_20260802.zip
cd v13_repro_source_20260802
bash reproduce.sh list
bash reproduce.sh preflight
bash reproduce.sh bootstrap
```

`bootstrap` 创建 `v13-repro-cu128` 环境，安装 PyTorch `2.8.0+cu128` 和固定核心依赖，并运行 CUDA 探针。需要 STEP 生成或几何有效性检查时，再安装可选 CAD 层：

```bash
V13_INSTALL_OCC=1 bash reproduce.sh bootstrap
```

该步骤通过 conda-forge 安装 `pythonocc-core=7.9.3`，并从固定 commit 安装 `occwl`，随后执行真实 box STEP 写入、读取和 BRep-valid 往返测试。`chamferdist` 不是核心依赖；当前复杂曲面诊断使用 `torch.cdist`，避免其编译兼容问题。

复制路径模板并填入服务器上的真实位置：

```bash
cp configs/paths.env.example configs/paths.env
editor configs/paths.env
bash reproduce.sh preflight
```

不要修改 `artifact_specs/` 来绕过哈希。制品不匹配意味着数据、checkpoint 或词表身份不同，应先定位来源。

## 路径三：检查和运行实验

```bash
bash reproduce.sh list
bash reproduce.sh explain v13_complex_curved_true_token
bash reproduce.sh smoke v13_complex_curved_true_token
bash reproduce.sh run v13_complex_curved_true_token
bash reproduce.sh status v13_complex_curved_true_token
bash reproduce.sh verify v13_complex_curved_true_token
```

推荐先做组件隔离，不要直接重训数百 epoch：

1. `v13_complex_curved_fsq_only`：不接 AR，只测复杂曲面 FSQ patch MSE/Chamfer。
2. `v13_complex_curved_true_token`：真实前缀 CE 加真实 token 重建，隔离 AR 与装配。
3. `v13_parent_cad_split_audit`：确认当前数据泄露程度。
4. `v13_ar_length_coverage`：确认 1024/1536/2048 context 覆盖率。
5. `v13_fsq_capacity_candidate_eval`：复查已有高容量 FSQ 候选，不默认晋级。
6. `v13_dfs_medium_teacher_forcing` 与 `v13_rcm_medium_teacher_forcing`：排序局部对照。

`historical_failed` 中的命令用于审计旧失败，不是推荐路线。必须显式确认风险：

```bash
bash reproduce.sh run historical_ar_bs32_lr5e4_oom --allow-historical-failed
```

## 外部制品

大型文件不在 ZIP 中。`configs/paths.env` 中的每个路径必须对应一个 `artifact_specs/*.json` 契约。核心已知制品包括：

- selected V13 FSQ VQ-VAE checkpoint；
- selected finite V13 AR best checkpoint；
- V13 RCM sequence package；
- parsed ABC archive root；
- FSQ capacity 候选；
- DFS/RCM medium 对照；
- BrepARG same-data 短训和长训制品。

已知 nonfinite 的历史 `ar_latest.pt` 不是推荐制品。`ar_best.pt` 与后来恢复训练的 lineage 必须按契约区分，不能用文件名相似代替身份校验。

## 结果解释

- `process exit code = 0` 只表示命令完成，不等于科学结果通过。
- Valid 必须以全部 generation attempts 为分母，不能只在幸存 STEP 上计算。
- STEP 可读、BRep-valid、修复后 valid、closed/watertight、complex、unique 是不同指标，必须分别报告。
- teacher-forcing CE、真实 token 重建和自由运行生成回答不同问题，不能互相替代。
- train loss 高于 validation loss 可由训练态 dropout、batch 等权聚合和 parent-CAD 泄露共同造成，不能单独证明 validation 更容易。
- 质量门和约束解码是评估/筛选工具，不是上游模型修复。

## 尚未完成

以下是下一阶段 paper-grade 证据缺口，不应被当前包隐藏：

- parent-CAD 隔离后的 train/val/test 与从零重训；
- ground-truth parsed geometry 直接装配 oracle；
- continuous latent 绕过 FSQ 的 decoder/assembly oracle；
- teacher-forced one-step argmax token 重建；
- 完整同数据、同预算 DFS/RCM 对照；
- 官方 BrepARG 完整词表、配置和评测协议下的正式复现；
- 多随机种子训练与生成方差。

Windows 构建主机只验证源码、归档、JSON、控制面和环境文件。Linux CUDA kernel、OCC STEP 往返及真实 external artifact smoke 必须在目标服务器执行。
