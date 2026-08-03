# 数据链路与评测协议

## 1. 数据链路

```text
ABC STEP archives
  -> OpenCascade/occwl parse
  -> per-solid parsed geometry records
  -> deduplication and protocol filters
  -> parsed archives or parsed shards
  -> VQ patch shards (surface and edge patches)
  -> FSQ VQ-VAE encode/decode
  -> face ordering (RCM/GNN or DFS control)
  -> holistic token sequences
  -> train/val/test sequence package
  -> AR teacher forcing or free generation
  -> token grammar parsing
  -> surface/edge decode
  -> BRep assembly/sewing/repair
  -> STEP/STL/PNG
  -> validity, closure, complexity, uniqueness and geometry metrics
```

每一箭头都可能失败。评测必须记录最早失败阶段，不能只记“没有 STEP”。

## 2. 当前数据身份

- canonical V13 sequence：train `382,903`、validation `21,214`、test `21,003`，vocab `10,294`。
- sequence length：最小 49，最大 2,353；validation median 563，p95 1,717。
- topology：faces 2-50，global edges 2-150。
- parsed archive root：101 files，总约 174.37 GB；包内使用排序文件名+大小 inventory SHA-256，不把它冒充内容哈希。
- selected VQ、finite AR best 和 sequence 的内容 SHA-256 位于 `artifact_specs/`。

## 3. 面/边上限

当前表示和 sequence 代码实例化 `max_face=50`、`max_edge=150`。VQ patch 构建使用 `max_source_faces=50` 和 `max_source_edges=150` 丢弃超限来源。

- `faces > 50`：当前协议外。
- `faces == 50`：协议内高难度边界，不是越界；失败应保留在总指标并单列 30-50 桶。
- `edges > 150`：按当前本地实现视为 global-edge 协议外。
- 低面/低边 CAD：数据真实组成，不能靠生成时强制最小值删掉后宣称模型修复。
- `max_edge` 在历史帮助文本中有 per-face/global 混用风险。新实验 manifest 必须显式写 `max_global_edges` 与 `max_edges_per_face`，不能只写 `max_edge`。

## 4. Split 协议与泄露

历史 sequence exact source path 在 split 间不重复，但不同 part 可能来自同一 parent CAD UUID。

V13 历史 sequence：

- validation 中 `12,038/21,214 = 56.7455%` records 与其他 split 共享 parent；
- test 中 `12,008/21,003 = 57.1728%`；
- train 中 `110,539/382,903 = 28.8687%`。

same-data BrepARG 10k/1k/1k：

- validation `464/1,000 = 46.4%`；
- test `482/1,000 = 48.2%`；
- train `3,383/10,000 = 33.83%`。

论文级新 split 必须：

1. 从 source path 提取 parent CAD UUID；
2. 所有 parts 绑定同一 split；
3. 固定 sorted parent list 和 seed；
4. 输出 parent manifest、record manifest、hash 和比例；
5. 审计 exact/canonical/basename/parent overlap 全为 0；
6. V13 与 BrepARG 使用同一 parent split。

历史 val CE 仍可用于同一 run 内 checkpoint selection，但不得作为独立 test 泛化主结论。

## 5. VQ/FSQ 评测

必须同时报告：

- unweighted reconstruction MSE，保持训练曲线可比；
- patch-equal Chamfer mean/median/p90/p95/max；
- shape-equal Chamfer，避免多 patch shape 主导；
- surface 与 edge 分开；
- surface type 分层（plane/cylinder/cone/sphere/torus/B-spline 等）；
- faces、global edges、sequence length 分桶；
- complex-curved 固定 cohort 的 source identity。

当前核心 diagnostic 用 `torch.cdist` 计算 Chamfer，避免 `chamferdist` CUDA extension 成为环境阻塞。若切换实现，必须用小 fixture 验证数值定义、平方/非平方距离、双向聚合和归一化完全一致。

## 6. AR 评测

三种实验不可混用：

1. **teacher-forcing CE**：真实 token 前缀，衡量条件预测；应报告 token-weighted CE，同时可给 shape-equal CE。
2. **teacher-forced argmax reconstruction**：每一步输入真实前缀、收集 argmax 预测，再重建；当前尚未实现。
3. **free-running generation**：模型以自身历史继续，包含 exposure、sampling 和 termination。

Context coverage：

- 1024、1536、2048 都是 AR 最大位置长度/训练序列截断选择，不是几何 faces 上限。
- 当前 train allowed 分别约 290,706、354,149、378,496；2048 仍不能覆盖全部 382,903。
- checkpoint 的 positional embedding shape 必须与实例化 `max_seq_len` 一致。

训练与验证 CE 的实现需审计：历史 V13 以 batch 等权聚合，padding/token 数不同会改变解释。后续主指标应使用全有效 token 的 loss sum / token count。

## 7. BRep 与生成评测

逐级报告：

1. attempts；
2. sequence terminated；
3. grammar valid；
4. geometry decoded；
5. assembly returned shape；
6. STEP written；
7. STEP read back；
8. raw BRep-valid；
9. repaired BRep-valid；
10. solid/closed/no open shell；
11. complex；
12. unique；
13. PNG rendered。

推荐核心比例：

```text
success_rate = STEP_written / attempts
watertight_rate = watertight / STEP_written
Valid = watertight / attempts
complex_valid_rate = complex_and_valid / attempts
```

必须同时报告 numerator/denominator。若保留 survivors，另报 acceptance rate；禁止以 target retained count 作为 attempts。

当前内部 complex 参考为 `faces >= 12 or global_edges >= 20`，strict gate 还会考虑 primitive-like、STEP entity complexity、BRep validity 和 closure。该阈值是项目诊断口径，不自动等同 BrepARG 官方论文指标。

## 8. Baseline 公平性

同数据 V13/BrepARG 比较必须固定：

- parent-CAD split 和源文件集合；
- max faces/global edges/per-face edges；
- normalization、surface/edge sampling 和 augmentation；
- VQ 与 AR 训练 token/sample 预算；
- context 和截断策略；
- random seeds；
- generation attempts、temperature、top-p、termination；
- OCC/repair 流程；
- Valid、complexity、uniqueness、Chamfer 等指标实现。

官方 BrepARG README 的 VQ `3000` / AR `500` 是示例命令，不证明公开权重的真实训练过程。官方 vocab 7222 与本地 10294 不兼容时，必须使用官方协议独立复现，或明确写“官方权重评测 unavailable”。

## 9. 随机性与复现

历史 V13 固定 `torch.manual_seed(0)`、NumPy 和 Python seed，但未完整固定 CUDA/worker；TF32 开启。因此：

- 每个 run 保存 Python/NumPy/torch/CUDA seed；
- DataLoader worker 使用确定 seed；
- 记录 TF32、AMP、deterministic algorithms 和 cuDNN benchmark；
- 组件诊断固定 cohort identity；
- paper 结果至少 3 个训练/生成 seed，报告均值、标准差和完整 attempts；
- deterministic 模式可能降低性能，应作为显式 profile，而不是隐式默认。

## 10. Promotion gate

在进入昂贵 AR 重训前，建议最低门：

- parent-CAD overlap = 0；
- 复杂曲面固定 holdout identity 完整；
- ground-truth assembly oracle 达到高成功率；
- true-token STEP >= 90%，BRep-valid >= 70%；
- surface Chamfer p95 相对当前 `0.41238` 至少下降 20%；
- checkpoint 全 finite，config/vocab/context identity 匹配；
- free generation 至少 100 attempts，完整失败分布与 PNG/STEP 留存。

这些是项目下一轮内部 promotion gate，不是声称来自官方论文的阈值。
