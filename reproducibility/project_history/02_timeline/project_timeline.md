# V13 / BrepARG 项目时间线

本时间线将代码提交、ExecPlan、程序 JSON、用户终端输出和后续审计合并。日期以 Asia/Shanghai 为主；无法从原始文件确认的时间标为“约”或“证据不足”。

| 日期 | 阶段 | 行动与配置 | 观察结果 | 当时/当前判断 | 证据 |
| --- | --- | --- | --- | --- | --- |
| 2026-06-24 至 06-27 | 初始 V13 VQ | FSQ-VQ-VAE 全链路训练与 finite 防护 | 旧 run 末期曾出现 `val=inf` 仍被标记成功；修复后 40 epochs 全部 finite，best val reconstruction 约 `0.00056` | 建立 finite gate、best/last 区分和 early stop | E001, E002 |
| 2026-06-27 | VQ continuation | 从稳定 checkpoint 继续至绝对 epoch 85 | best epoch 73，best val 约 `0.00037`，正常 early stop | 全局 MSE 改善，但尚未证明复杂曲面质量 | E002 |
| 2026-06-30 至 07-05 | 本地 V13 AR | 多学习率 continuation，context 主要为 1024 | 最终到 epoch 120，best val CE `0.2949333`；100 attempts 写出 87 STEP，78 个通过当时 BRep-valid | 训练可收敛，但 valid 未衡量复杂性与拓扑坍塌 | E011, E012 |
| 2026-07-05 至 07-07 | VQ slice/complexity 诊断 | 按 shortest/random/most-faces/longest 分桶，对比 VQ100、e6 continuation、weighted/cap 分支 | 简单/平面改善较多，复杂/曲面 heavy tail 持续；部分 continuation 在难桶退化 | 停止把单一全局 validation MSE 当 promotion gate | E002, E006 |
| 2026-07-07 | 复杂采样与 loss 加权 | 引入 source cap 50 faces/150 global edges、复杂/曲面采样和 weighted loss | 训练稳定，局部难桶均值有改善但远未达到目标；surface 桶并非一致改善 | 基础设施可用，科学门仍关闭 | E002, E003 |
| 2026-07-10 约 | 服务器 VQ resume vs scratch | resume 分支 bs256/lr5e-5；scratch bs128/lr1e-4，之后继续训练 | 独立 50k patch MSE 中 resume overall mean/median/p95 最优；scratch 继续到 early stop epoch 440，best epoch 340 | 选择 resume-best 作为当时 VQ 候选；长训 scratch 未证明更优 | E017 |
| 2026-07-11 | 全量 sequence | 从 100 个 parsed chunks 构造 FSQ+RCM sequence；初次 smoke 缺 `chamferdist` | 修复环境后完成 train/val/test `382903/21214/21003`，vocab `10294`，max observed length `2353` | sequence 可训练，但依赖和来源路径须预检 | E003, E018 |
| 2026-07-11 至 07-12 | 服务器 AR 长 context | `max_seq_len=2048`，bs 从 8 提到 32，lr `5e-4` | GPU 利用率提高；epoch 115 后 loss 爆炸并 OOM，后续恢复分支出现 NaN/nonfinite latest | 大 batch + 高 LR 不稳定；必须回滚 finite best、降低 LR/BS并检查 tensor finite | E019, E020 |
| 2026-07-12 | AR checkpoint 审计 | 对 `ar_best.pt` 与 `ar_latest.pt` 做 finite tensor 检查 | best epoch 117 finite、val CE `0.2990635`；latest epoch 142 的 embedding nonfinite、val CE NaN | 禁止按 `latest` 名称自动选择；selected AR 必须内容哈希和 finite gate | E020 |
| 2026-07-12 至 07-13 | AR 恢复与生成 | 从 finite best 以较小 LR/BS 恢复；修复生成时 positional embedding 长度不匹配 | 获得 finite selected AR；可生成 STEP，但输出偏简单 | 训练稳定不等于生成质量合格 | E011, E012, E020 |
| 2026-07-13 | 生成参数深度对比 | 温度、top-p、bbox 单调约束、最小 faces、quality gate、STEP/PNG | 参数可改变复杂度/invalid 比例；quality gate 用 2405 attempts 才保留 100，不能修复分布 | 排除“只调 sampler/约束”的主修复假设 | E009, E012 |
| 2026-07-15 至 07-16 | complex-curved 组件隔离 | 固定 50 shapes，3,399 patches；FSQ-only、teacher forcing、真实 token reconstruction | all Chamfer p95 `0.15012`，surface p95 `0.41238`；真实 token 27/50 STEP、9/50 BRep-valid | FSQ/decoder/assembly 成为最强瓶颈证据；AR 仍有独立问题 | E006, E007, E008 |
| 2026-07-15 至 07-17 | capacity 与排序对照 | levels `(16,16,8,8)` 候选；DFS/RCM medium matched controls | capacity overall p95 恶化 `4.19%`，surface p95 改善 `26.91%`；DFS CE `1.25869` vs RCM `1.31302` | capacity 信号不一致；排序为次级因素 | E010, E016 |
| 2026-07-15 | BrepARG 原始 sampler 对照 | 当前 V13 权重改用原始 BrepARG generation logic | 124 attempts 保留 100，103 STEP；复杂仅 11/100，faces median 6 | sampler 不是简单拓扑坍塌的主因 | E009 |
| 2026-07-17 | 官方 BrepARG 权重 probe | 尝试加载官方/下载权重并比对本地 sequence vocab | 官方 embedding vocab `7222`，本地 `10294`，协议不兼容 | 不能声称已经复现官方权重质量 | E021 |
| 2026-07-17 至 07-20 | same-data BrepARG 短基线 | 10k/1k/1k，自训 VQ 与 AR，context 1536 | VQ best epoch 70；AR best epoch 77/80，val CE `0.871925`；92 个结果中 complex 5、strict 0 | baseline 可运行但质量差；不是官方复现 | E013, E022 |
| 2026-07-20 至 07-26 | BrepARG 长基线 | VQ 400；AR 从 epoch-127 lineage 恢复到 300 | VQ best epoch 269；AR best epoch 254、val CE `0.765318`；100 结果中 complex 13、strict 6 | 长训有真实改善但仍简单坍塌，不能只靠更多 epoch | E014, E023 |
| 2026-07-17 至 07-24 | 存储/恢复 | E 盘健康异常，部分内容经 C/D 过渡恢复；训练因重启续训 | 复制与删除量因 hardlink/压缩/目录口径不同曾造成误解；恢复后完成 long AR | 数据 authority 与物理空间必须由 manifest/hash 定义 | E024, E025 |
| 2026-07-31 | 全链路复盘 | 扫描源码、制品、日志、数据分布、split 与指标实现 | V13 val/test 约 57% 记录跨 split 共享 parent；same-data baseline 亦泄露 | 历史 validation 仅作开发曲线，不是独立泛化证据 | E003, E004, E005, E015 |
| 2026-08-02 | 可复现包 | 冻结 dirty current source、clean `16cf19b`、nested BrepARG `07970a4`；整理实验/制品/历史 | 形成轻量源码包和统一 `reproduce.sh` 控制面 | 后续实验以制品身份、parent-CAD split 和组件 oracle 为先 | E026, E027 |

## 问题引入区间

1. **数值稳定性回归**最迟在 2026-06-24 至 06-27 的旧 VQ run 已出现；finite 防护后可稳定训练。
2. **AR 训练爆炸**明确出现在 2026-07-12 的 bs32/lr5e-4/ctx2048 continuation，随后污染 `ar_latest`；finite best 未被污染。
3. **复杂几何质量问题不是一次代码回归。** 即使真实 token、原始 sampler、较长训练和两种排序，复杂曲面/装配与简单拓扑问题仍存在，更符合系统性表征与协议问题。
4. **parent-CAD 泄露**来自历史记录级划分方式，不是训练后才引入；V13 和 same-data BrepARG 都受影响。
5. **环境漂移**贯穿 Windows Python/Conda、AutoDL Python 3.14、CUDA 13 driver 与不同包版本；本包重新定义 Python 3.11 + PyTorch cu128 目标环境。

## 版本层

- 当前默认：打包时 dirty working tree，文件级 SHA-256 见 `provenance/current_source_manifest.json`。
- 外层干净参考：Git commit `16cf19b`，只用于差异审计。
- 嵌套 BrepARG 基线：commit `07970a4` 加本地修改，patch 与 manifest 位于 `provenance/`。
- 多人无意修改：没有足够证据确认具体人员；dirty/untracked 范围很大，因此不能排除手工脚本漂移，现由 source manifest 和 diff 固化。
