# Protocol V2 parent-CAD 隔离设计

## 目标

本轮先修复会让所有模型比较失真的数据协议问题，再做小规模、可归因的 FSQ 实验。完成后，V13 自有代码能够从 parsed CAD 文件生成一份统一协议清单，按 parent CAD 而不是单条记录划分 train、validation 和 test，并保证 VQ-VAE 的 validation patch 只来自 validation CAD。上游 `BrepARG/` 和尚未开始的 `papers/` 不作任何修改。

本轮不以提高 epoch 数作为修复手段。只有协议审计通过且小规模 reconstruction 指标可解释，才允许后续扩大训练规模。

## 现状与根因

`breparg_improvements/train.py` 当前先随机打乱 parsed 文件，再按记录做 90/5/5 切分。同一 parent CAD 的不同 STEP part 因而可能跨 split。历史审计已经证实 V13 validation 和 test 中约 57% 的记录与其他 split 共享 parent CAD。

同一文件的 VQ-VAE 阶段还有第二次泄露：它只从 train CAD 收集 patch，再把 patch 数组前 95% 和后 5% 当作 train/validation。同一 CAD 的 surface 或 edge patch 会同时进入两侧。patch shard 和 sample cache 路径甚至不携带可信 split 身份，不能证明隔离。

历史过滤只稳定表达 faces 和 global edges 的上限。论文协议要求剔除少于 10 faces 的简单 CAD，并将每面最多 30 edges 与 global 最多 150 edges 分开表达。`faceEdge_adj` 的缺失、长度错误和越界 edge index 也必须成为显式拒绝原因，而不能在后续 sequence 阶段静默失败。

## 推荐方案

采用“统一协议清单 + fail-closed 训练入口”的方案。

第一种备选是只改 `train.py` 中的两个切片位置。它改动最少，但 sequence、VQ patch、审计仍可能各自使用不同过滤规则，且旧 shard/cache 会继续绕过隔离，因此不采用。

第二种备选是直接强化历史 `prepare_breparg_same_data_inputs.py::valid_record`。它会无意改变旧 same-data fallback 和相关测试的语义，也会触及为上游对照准备数据的路径，因此不采用。

第三种方案是在 `breparg_improvements/cad_protocol.py` 建立 V13 自己的、仅主动依赖 Python 标准库的协议核心，并由 CLI、训练入口和审计共同使用。这是本轮采用的方案。核心逻辑不需要 PyTorch、CUDA 或 OpenCascade；但是 ABC parsed pickle 内部通常序列化了 NumPy ndarray，因此读取真实 archive 时仍必须使用包含 NumPy 的数据环境（本机为 `brepgen_env`）。纯 Python fixtures 和 manifest 操作仍可在轻量环境运行。

当前健康数据不是解包目录，而是 `ABC/processed/abc_parsed_full_archives` 下的 100 个 ZIP。旧 split 中约 68 万条路径指向已不存在的解包位置。因此 CLI 同时支持普通 parsed 目录和 ZIP archive root；archive 模式直接流式读取成员，不解压约 604.8 GiB 的完整内容。协议清单中的 `source_path` 使用稳定的 `archive-file.zip!/abc_XXXX/member.pkl` 身份，只有 smoke 或训练实际选中的 eligible rows 才物化到外部实验目录。

## 协议模型

每个扫描到的 parsed CAD 都生成一条 manifest row。无论接受还是拒绝都保留，这样过滤前后数量和拒绝原因能够复现。每条至少包含：

- `source_path`：本次读取的 parsed 文件路径；
- `source_key`：大小写和路径分隔符规范化后的稳定来源键；
- `parent_id`：从文件名提取的 24 至 32 位十六进制 parent CAD 标识；
- `num_faces`：`surf_ncs` 的第一维长度；
- `global_edges`：`edge_ncs` 的第一维长度；
- `max_edges_per_face`：`faceEdge_adj` 中最长一项的长度；
- `protocol_eligible`：是否满足 Protocol V2；
- `reject_reason`：接受时为 `null`，拒绝时为一个稳定原因码；
- `split`：只有通过过滤并完成 parent 分组切分后才为 `train`、`val` 或 `test`。

Protocol V2 的边界为 `10 <= num_faces <= 50`、`global_edges <= 150`、`max_edges_per_face <= 30`。`surf_ncs`、`edge_ncs` 和 `faceEdge_adj` 必须存在；`faceEdge_adj` 的长度必须等于 faces；每个 edge index 必须是整数且位于 `[0, global_edges)`。无法可靠提取 parent ID 的记录不能进入 split，因为把未知来源当作彼此独立会制造假安全。

拒绝原因按数据可解释性排序并保持确定：加载失败、缺字段、无法提取 parent、faces 下限、faces 上限、global edges 上限、adjacency 长度、per-face edges 上限、edge index 类型和 edge index 越界。汇总报告记录每个原因的数量。

## Parent 分组切分

所有 eligible rows 先按 `parent_id` 分组，再以固定 seed 做确定性分配。输入文件顺序变化不能改变 assignment。同一 parent 的所有 STEP part 和能被 parent 规则识别的旋转变体必须落入同一个 split。

目标比例为 8:1:1，平衡对象优先采用 record 数而不是仅 parent 数。实现按 parent group 大小从大到小放置，每次选择相对目标缺口最大的 split；哈希值只用于稳定打破平局。超大 parent group 永不拆分，因此极小数据集可能无法精确达到 8:1:1，summary 必须报告目标与实际数量。

`NS_N` 的语义改成 eligible record 上限。选取上限时也按完整 parent group 操作，不能为了精确凑数切开 parent。正式运行扫描全部候选；小规模 smoke 可以显式设置扫描上限，但报告必须标为 smoke，不得外推为全量分布。

## VQ-VAE 数据流

parsed-file 路线改成两次独立采样：

    train CAD paths -> train patch inventory -> split 内去重 -> train samples
    val CAD paths   -> val patch inventory   -> split 内去重 -> val samples

不再对一个混合 patch 数组做 95/5 切片。训练 cap 与 validation cap 独立记录，validation 不从 train cap 中扣除。

旧 `NS_VQ_PATCH_SHARD_ROOT` 和 `NS_VQ_SAMPLE_CACHE` 不包含足以证明 parent/split 隔离的 provenance。Protocol V2 下它们必须 fail closed，而不是自动继续。未来若使用 shard/cache，需要分别提供 train 和 val 资产，并在资产 manifest 中记录 protocol hash、split 和 parent identity；这属于后续独立里程碑，不能用目录名代替身份验证。

## Patch 去重与监控

本轮采用 exact content hash 去重作为默认安全措施。hash 输入包含 patch kind、shape 和规范化为 little-endian contiguous float32 的内容，避免 surface 与 edge 因像素恰好相同被错误合并。去重发生在 parent split 之后、每个 split 内部，且保留重复来源计数。

rounded hash 只做审计：同时报告 exact duplicate rate 和 round-to-4-decimals duplicate rate，但在看到真实数据统计前不让 rounded hash 改变训练集。跨 split exact hash overlap 单独报告；它不一定等同 parent 泄露，但会提示数据重复或派生泄露。

FSQ validation 记录全 validation token histogram，再一次性计算 unique bins、coverage、entropy perplexity。不能平均 batch perplexity。reconstruction 至少分为 `surface_planar_like`、`surface_curved_proxy` 和 `edge` 三桶；其中 bbox/span 或点集非共面程度只称 proxy，不宣称是真实曲率。

## 小规模实验

实验分两层。

协议 smoke 从现有 parsed 数据中确定性扫描一小批 parent/CAD，输出过滤前后 record 数、parent 数、拒绝原因、faces/global edges/per-face edges 分布、实际 split 数量和 parent overlap。硬验收是 eligible rows 的 faces 范围为 10 至 50、global edges 不超过 150、per-face edges 不超过 30，且三组 pairwise parent overlap 全为 0。

本机第一轮使用 `abc_0000_parsed.zip`，该 archive 有 5,943 条 parsed CAD，扫描前 2,000 条足以形成真实但明确受限的 protocol smoke。原始 ZIP 只读，raw 输出写到空间充足的 E 盘；Git 只接收汇总 JSON/Markdown。

模型 smoke 使用相同 protocol hash 的小训练集和独立 validation CAD，固定 seed，只训练少量 epoch。第一轮不同时改变 ordering、augmentation、FSQ levels、loss weighting 和 AR 参数。它验证数据加载、前向/反向、checkpoint、全 validation FSQ usage 和分桶 reconstruction 指标能完整产出；它不是最终质量结论。

若资源允许，第二个小实验只比较 FSQ capacity，例如当前 4 维 levels 与 6 维 4096 levels，其余 cohort、split、seed、batch、epoch 和 loss 完全一致。只有 validation curved-proxy reconstruction 改善且 usage 没有恶化，才支持继续全量容量实验。

## 验证与发布

所有新行为先有失败测试，再实现。协议单测覆盖边界、缺字段、adjacency 错误、parent 解析、确定性 split 和输入顺序不变性。集成测试证明 VQ train 与 val 的 source 和 parent 不相交，并证明旧无 provenance shard/cache 在 Protocol V2 下被拒绝。

分支为 `experiment/protocol-v2-parent-isolated`。只提交 V13 自有代码、测试、设计/计划、聚合实验 JSON/Markdown 和必要的轻量 TensorBoard 日志；不提交 parsed 数据、pickle、checkpoint、CAD、图片、`BrepARG/` 或 `papers/`。完成验证后推送该分支，不直接合并 `main`。

## 明确延期

AR context 2048、gradient accumulation、warmup、token-weighted CE、sequence augmentation、自由生成抽检以及 generation/evaluation 口径仍然重要，但它们依赖可信 split 和晋级后的 VQ checkpoint。本轮只在整理文档中保留这些后续措施，不把它们和数据协议修复同时投入同一个小实验，避免无法归因。
