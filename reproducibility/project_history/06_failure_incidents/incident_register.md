# 故障事故登记

| ID | 严重度 | 事故 | 直接影响 | 根因/状态 | 防复发措施 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| I001 | 高 | 旧 VQ run `val=inf` 仍被标记成功 | 非有限 checkpoint 可能被晋级 | 早期 success gate 只看流程完成；已修复 | finite batch/epoch、NaN 不更新 best、连续非有限停止 | E001, E002 |
| I002 | 高 | AR bs32/lr5e-4/ctx2048 在 epoch 115 后爆炸并 OOM | 长时间训练丢失，后续权重污染 | LR/batch/长上下文组合过激；OOM 前 loss 已持续上升 | 监控趋势、较小 LR/BS、立即停止、回滚 finite best | E019 |
| I003 | 严重 | `ar_latest.pt` epoch 142 embedding nonfinite | 若自动选 latest，生成/恢复全部失效 | OOM/发散后仍写 latest；文件名掩盖健康状态 | checkpoint tensor finite 扫描、内容哈希、known-bad 契约 | E020 |
| I004 | 高 | 用户设置 `NS_AR_GRAD_CLIP=1.0` 但代码不读取该变量 | 误以为策略变化已生效 | V13 AR/VQ clip 都硬编码 `1.0` | 配置审计；未来接线前测试 env 行为；运行 manifest 记录 effective config | E015 |
| I005 | 中 | 2048 AR checkpoint 被按 1024 positional embedding 实例化 | `wpe.weight` size mismatch，生成无法启动 | 推理代码最初未从 checkpoint config 读取 max sequence length | 由 checkpoint config 构建模型，strict load，预检 wpe shape | E020 |
| I006 | 中 | sequence smoke 缺 `chamferdist` | 上游 shard 失败后 audit/preflight 连续 FileNotFound | 原始 BrepARG utils 顶层硬依赖可选 CUDA extension，shell 未阻止后续步骤 | `set -euo pipefail`；diagnostic 改用 `torch.cdist`；依赖分层 | E018 |
| I007 | 高 | 只构建 0-49 sequence 后准备训练 AR | 数据协议不完整，可能形成不可比较模型 | smoke 与正式训练边界不清 | 半量只允许 smoke；正式 AR 必须绑定全量 sequence identity | E018, E027 |
| I008 | 高 | train/val/test exact path 不重复但 parent CAD 泄露 | validation CE 被高估，baseline 比较失真 | 记录级随机划分未按 parent UUID 分组 | parent-CAD manifest；split audit 成为训练前硬门 | E004, E005 |
| I009 | 高 | quality gate 只看 survivors | 可能虚高 Valid/复杂率 | 分母从 attempts 偷换为 retained | attempts 为固定分母；保留 reject reasons 与全部计数 | E009, E012 |
| I010 | 中 | 增加最小 faces/edges 排除真实低拓扑 CAD | 模型分布与数据分布错位 | 把筛选当训练修复 | 按 topology 桶报告；只在论文协议要求时过滤并公开规则 | E003, E012 |
| I011 | 高 | official BrepARG checkpoint vocab 7222 vs local 10294 | 权重无法公平加载，官方结果不可复现 | 数据/词表协议不同 | 官方协议独立复现或明确 unavailable；禁止 resize 后声称官方 | E021 |
| I012 | 中 | VQ 全局 MSE 很低但复杂曲面/STEP 很差 | 错误 checkpoint promotion | 单一 patch MSE 掩盖 surface tail 和 assembly | complex-curved Chamfer、shape-equal buckets、true-token gate | E006, E007 |
| I013 | 中 | `max_edge` 语义混用 global 150 与 per-face cap | baseline 数据协议可能不一致 | 参数命名和论文/代码帮助文本漂移 | 改用 `max_global_edges` / `max_edges_per_face`；manifest 固化 | E003, E015 |
| I014 | 中 | seed 只固定 Python/NumPy/CPU torch，TF32 开启 | 相同命令仍可能跨 GPU/worker 波动 | 缺 CUDA seed、worker seed 和 deterministic algorithm policy | 每 run 记录 determinism profile；paper run 多 seed | E015 |
| I015 | 高 | E 盘文件系统异常与跨盘恢复 | 权威副本不清，重启后训练中断 | exFAT 健康问题、恢复路径分散 | 健康 NTFS authority、manifest/hash、只读复制后验证 | E024, E025 |
| I016 | 中 | 复制约 10 GB 却删除约 58 GB 的空间口径疑问 | 用户难以确认数据是否安全 | logical size、physical allocation、hardlink、压缩和目录选择口径混合 | 删除前列 exact paths、logical/physical size、link count 和 hash | E024 |
| I017 | 中 | 20 sequence workers CPU 利用率仍低或 GPU 争用 | 误判 `num_workers` 越高越快 | 每个进程同时抢单 GPU，瓶颈不只 CPU | 分阶段 profile；CPU parse 与 GPU encode 分离；不盲增 workers | E018 |
| I018 | 中 | validation CE 一直低于 train CE | 被误判为验证集更简单或模型异常 | dropout、batch 等权 CE 和 parent leakage 共同作用 | token-weighted CE、eval mode audit、parent-isolated test | E003, E004, E015 |
| I019 | 中 | Python 3.14/依赖漂移与失效 `OCC` pip pin | chamfer/OCC/Transformers 行为不一致 | 缺完整环境锁，上游 requirements 含陈旧包 | Python 3.11 + cu128 固定线；OCC 单独 conda；真实探针 | E015, E021 |
| I020 | 中 | 大量 `best/latest/final`、重复 run 和 hardlink | 清理风险高，物理空间估计错误 | 缺制品 authority index 和 lineage manifest | artifact contracts、run manifest、先 hash 后归档/删除 | E025, E026 |

## 事故响应规则

1. loss 连续明显上升、出现非有限或 OOM 时立即停止；不要等待 epoch 结束继续污染 optimizer/model。
2. 从最近 finite `best` 恢复，不从 crash 后 `latest` 恢复；恢复前扫描全部 floating tensors。
3. 修改 batch size 时说明 optimizer state 是否保留；修改 context 时确认 positional embeddings 与数据覆盖。
4. 上游步骤失败后不执行下游 audit/train。Bash 使用 `set -euo pipefail`，Python launcher保留 return code 和 stderr。
5. 每个结果记录 attempts、成功阶段和 failure taxonomy；不能把部分输出等同 pipeline success。
6. 磁盘恢复与训练解耦。先建立健康副本、hash 和 authority，再启动写密集任务。
