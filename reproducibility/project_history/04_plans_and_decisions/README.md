# 计划与决策索引

## 核心计划

构建器会把仓库 `plans/` 和显式恢复根中的小型文本复制到本目录下的 `plans/` 或 `05_original_records/`。优先阅读：

- `stable_vqvae_retraining_execplan.md`：finite VQ 基线和早停防护。
- `vqvae_epoch100_continuation_execplan.md`：VQ continuation 与 best checkpoint。
- `ar_training_v13_execplan.md`：AR 多 LR 训练、恢复和生成。
- `v13_generation_quality_recovery_execplan.md`：复杂度分桶、质量门与排除假设。
- `complex_curved_fsq_ar_diagnostics_execplan.md`：FSQ-only、teacher forcing、capacity 与 DFS/RCM 组件实验。
- `breparg_long_baseline_and_v13_diagnosis_20260720_execplan.md`：same-data BrepARG 短/长基线。
- `full_experiment_postmortem_and_workspace_governance_20260731_execplan.md`：零至八全链路复盘与工程治理。
- `docs/superpowers/specs/2026-08-02-v13-repro-source-package-design.md`：当前包设计。
- `docs/superpowers/plans/2026-08-02-v13-repro-source-package.md`：当前打包 ExecPlan。

## 已接受决策

### D001：轻量源码包

不复制权重、sequence、parsed archives、STEP、PNG、环境目录或旧 ZIP。所有大型对象用 artifact contract 表示，包含哈希、大小、兼容性和路径变量。

### D002：双源码层

`source/current` 是默认运行层，保留打包时 dirty working tree；`source/clean_head_16cf19b` 是外层干净参考。两者不能互相替代。嵌套 BrepARG 保留 commit `07970a4` 和本地 patch provenance，不复制 `.git`。

### D003：Linux GPU 唯一目标

Windows 构建主机只验收控制面和归档。运行目标固定 Linux x86_64 NVIDIA GPU，环境为 Python 3.11 + PyTorch cu128。CAD/OCC 作为可选层单独安装和验收。

### D004：实验分四类

实验描述分 `recommended`、`baselines`、`diagnostics` 和 `historical_failed`。已知失败命令默认禁止，必须使用 `--allow-historical-failed`。

### D005：先隔离组件再长训

在 oracle、parent-CAD split 和 matched protocol 完成前，不以继续同配置数百 epoch 作为默认修复。训练预算由组件证据决定。

### D006：同协议 baseline

V13 与 BrepARG 论文比较必须共享 parent-CAD split、预处理、caps、context、生成 attempts、seed 和指标实现。官方权重协议不兼容时标为 unavailable；same-data 自训不能冒充官方复现。

### D007：生成 Valid 的分母

Valid 使用全部 attempts 作为分母。质量门筛出的 survivors 必须同时保留 attempts、reject reasons 和未筛选分布，禁止只在幸存者上报告成功率。

### D008：制品身份优先于文件名

`best/latest/final/epoch` 只是标签。恢复或生成前必须核对 SHA-256、size、vocab、model config、max sequence length、lineage 和 finite tensors。

### D009：不在打包中做破坏性治理

打包过程不删除、移动或清理项目数据。工程清理必须使用精确路径、预估物理回收量、hash/manifest 验证和用户逐项确认。

## 被拒绝或降级的路线

- 仅调 temperature/top-p；
- 仅关闭 bbox monotonic 或 face uniqueness；
- 仅提高生成最小 faces/edges；
- 只保留 water-tight/valid survivors；
- 只替换为 BrepARG 原始 sampler；
- 只增加 batch 或 epoch；
- 依据低 validation loss 宣称泛化；
- 依据 `latest` 文件名恢复；
- 将 50-face 边界失败排除为协议外；
- 将 same-data BrepARG 自训写成官方权重结果。

## 下一批需要作出的决策

1. ground-truth assembly oracle 的统一装配入口和 raw/repaired validity 定义；
2. continuous latent bypass 的 API 与是否需要单独 decoder checkpoint；
3. parent-CAD split 的 UUID 提取规则、固定 manifest 与 train/val/test 比例；
4. shape 等权与 patch 等权 Chamfer 的主次报告口径；
5. matched full DFS/RCM 所需预算；
6. 官方 BrepARG 是走官方数据协议复现，还是在论文中仅引用公开报告并声明 unavailable。
