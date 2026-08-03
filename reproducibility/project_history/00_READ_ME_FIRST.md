# 项目历史阅读说明

本目录保留 2026-06 至 2026-08 的 V13 / BrepARG 研发、训练、故障、恢复、诊断与工程治理过程。它面向没有原对话上下文的研究者。

## 证据等级

- `original_machine_record`：程序直接写出的 JSON/JSONL/log/checkpoint history。
- `source_audit`：对源码、数据包或制品进行的结构化审计。
- `execution_plan_record`：当时持续更新的 ExecPlan，包含命令、观察与决策。
- `conversation_derived_audit_record`：仅从用户提供终端输出恢复的事实；不伪装成原始日志。
- `inference`：由多份证据推导的解释，必须与原始事实分开。
- `insufficient_evidence`：缺少可靠制品、协议或对照，不能下结论。

## 阅读顺序

1. `01_full_postmortem/`：零至八板块的完整全链路复盘。
2. `02_timeline/`：按日期追踪关键改动与问题引入区间。
3. `03_experiment_ledger/`：人读和机器读实验账本。
4. `04_plans_and_decisions/`：设计、执行计划和关键决策索引。
5. `05_original_records/`：构建时自动收集的小型原始文本；大日志保留哈希与诊断摘录。
6. `06_failure_incidents/`：OOM、NaN、路径、依赖、存储和协议事故。
7. `07_data_and_protocol/`：数据链路、上限、split 和指标口径。
8. `08_evidence_index/`：证据 ID 到包内/原始记录的映射。

## 不应混淆的对象

- selected finite `ar_best` 与历史 nonfinite `ar_latest`；
- BrepARG 长训 epoch-127 来源 checkpoint 与恢复后完成到 epoch 300 的 best；
- same-data BrepARG 自训与官方 BrepARG 权重复现；
- patch reconstruction MSE、patch Chamfer、true-token BRep 重建和 free generation；
- STEP 写出、STEP 可读、BRep-valid、repair-valid、closed/watertight 和 strict complex-valid；
- 记录级随机 split 与 parent-CAD 隔离 split；
- `max_global_edges=150` 与论文/帮助文本中的 per-face edge cap。

## 当前状态

当前没有证据支持继续靠相同配置延长训练来解决主要问题。推荐先完成三个 oracle、parent-CAD 隔离重划分和同协议基线，再决定改 FSQ、decoder/assembly 或 AR。历史 failed 实验仍被保留，但默认执行被阻止。

`05_original_records/` 和 `08_evidence_index/history_inventory.json` 由构建器生成；它们会覆盖仓库内 `plans/`、`local_reports/`、`local_runs/` 以及显式提供的恢复证据根目录。大型二进制仅由 `artifact_specs/` 描述，不会复制进包。
