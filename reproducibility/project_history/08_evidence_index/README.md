# 证据索引

构建后，本目录还会包含 `history_inventory.json`，列出自动收集的每个原始文本记录、原始标签、包内路径、大小、SHA-256、复制/摘要模式和实验分类。大日志只保留 SHA-256、head/tail 和诊断事件行；二进制由 `artifact_specs/` 表示。

## 核心证据 ID

| ID | 类型 | 主要内容 | 包内或构建时来源 | 限制 |
| --- | --- | --- | --- | --- |
| E001 | execution_plan_record | 旧 VQ 非有限与 stable 40 epoch 修复 | `plans/stable_vqvae_retraining_execplan.md` | 历史训练原始 tensor 不全在包内 |
| E002 | execution_plan_record | VQ continuation、weighted/cap 与难桶判断 | `plans/vqvae_epoch100_continuation_execplan.md`, `plans/v13_generation_quality_recovery_execplan.md` | 多轮摘要需结合原始 histories |
| E003 | source_audit | V13 length/face/edge/complex 分布 | `docs/audits/v13_ar_distribution_coverage_20260731.{json,md}` | 基于历史泄露 split |
| E004 | source_audit | V13 parent-CAD split 泄露 | `docs/audits/v13_sequence_split_integrity_20260731.json` | UUID 规则基于 source path 命名 |
| E005 | source_audit | same-data BrepARG parent-CAD 泄露 | `docs/audits/breparg_same_data_split_integrity_20260731.json` | 同上 |
| E006 | original_machine_record | complex-curved FSQ-only Chamfer | `local_runs/complex_curved_rootcause_suite_20260715/experiments/00_fsq_only_patch_metrics/complex_curved_diagnostics_report.json` | patch 等权为主，shape 等权不足 |
| E007 | original_machine_record | true-token reconstruction 27 STEP/9 valid | `.../01_teacher_forcing_true_token_reconstruction/complex_curved_diagnostics_report.json` | 当前 custom BRep validity，不等同所有官方指标 |
| E008 | original_machine_record | complex-subset teacher-forcing CE 与分桶 | 同 E007 下 `ar_teacher_forcing.json` / report | 真实前缀，不是 free-running |
| E009 | original_machine_record | 原始 sampler 与 quality gate attempts | `local_runs/breparg_logic_compare_20260715/comparison_summary.json` | survivor 指标不能替代 attempts |
| E010 | original_machine_record | DFS/RCM medium CE | 恢复根 `ar_complex_curved_eval/{dfs,rcm}_teacher_forcing/ar_teacher_forcing.json` | medium control，不是 full matched retrain |
| E011 | execution_plan_record | V13 AR 训练到 epoch 120 | `plans/ar_training_v13_execplan.md` | 历史 split 泄露 |
| E012 | original/report record | 100 generation 与质量观察 | `local_reports/generated100_lr5e6_epoch120_best_20260705.md`, generation reports | 历史 validity 与复杂门定义需分开 |
| E013 | original_machine_record | BrepARG short quality summary | 恢复根 `breparg_same_data_fallback/breparg_same_data_quality_summary.json` | 自训，不是官方权重 |
| E014 | original_machine_record | BrepARG long VQ400/AR300 summary | 恢复根 `breparg_same_data_fallback_long_20260720/breparg_same_data_resume_best_quality_summary_20260726.json` | 同一泄露 split |
| E015 | source_audit | dropout、seed、TF32、clip、metric implementation | `breparg_improvements/train.py`, `BrepARG/model.py`, full postmortem | 源码行为不证明每个历史 run 的环境一致 |
| E016 | original_machine_record | FSQ capacity comparison | `local_runs/complex_curved_rootcause_suite_20260715/fsq_capacity_comparison.json` | 单候选，容量结论不充分 |
| E017 | conversation_derived_audit_record | 服务器 resume/scratch 训练与 50k 对比 | `06_failure_incidents/conversation_derived_incidents.md` | 若原 JSON 不在包内，依赖用户终端转录 |
| E018 | conversation_derived_audit_record | zst sequence、chamferdist、全量 pipeline | 同上 + sharded sequence plans | 部分服务器原日志可能不在本机 |
| E019 | conversation_derived_audit_record | AR loss 爆炸与 OOM trace | 同上 | 关键行来自用户粘贴 |
| E020 | conversation_derived/source audit | finite best vs nonfinite latest、ctx mismatch | 同上 + current checkpoint contracts/source | selected best hash 可验证，latest 不在推荐契约 |
| E021 | report/source audit | 官方 BrepARG vocab incompatibility | `local_reports/breparg_official_weight_probe_20260717.md`, `BrepARG/README.md` | 未完成官方协议生成 |
| E022 | execution_plan_record | BrepARG short training config | `plans/breparg_long_baseline_and_v13_diagnosis_20260720_execplan.md` | 同数据短训 budget 小于官方示例 |
| E023 | original/execution record | long AR resume lineage、best epoch/CE | same plan + `local_runs/breparg_long_ar_resume_best_20260724` | epoch-127 source与完成 checkpoint 必须区分 |
| E024 | recovery record | E drive 与跨盘恢复事故 | `e_drive_drop_20260717_0613.md`, `recovery_status_*.md`, recovery logs | 物理空间口径依赖文件系统 |
| E025 | source_audit | workspace size、hardlinks、cleanup governance | `docs/full_experiment_postmortem_20260731.md` | 审计时间点后可能变化 |
| E026 | provenance | dirty outer source、clean commit、nested BrepARG | `provenance/*` | 当前源码不是 clean commit 本身 |
| E027 | catalog/decision record | 推荐、诊断、阻塞与 historical failed 实验 | `experiments/*`, `PACKAGE_MANIFEST.json` | blocked 项没有伪造命令/结果 |

## 结论追踪规则

1. 每个强结论至少引用一个 original machine record 或 source audit。
2. conversation-derived 事实可以解释事故，但不单独支撑论文指标。
3. 自训 baseline 不支撑“官方模型效果差”的结论。
4. 不存在的原始证据必须保留为 unavailable，不从目录中静默消失。
5. artifact contract 的 `verification_strength` 必须区分 content SHA-256、name/size inventory 和 unresolved。
