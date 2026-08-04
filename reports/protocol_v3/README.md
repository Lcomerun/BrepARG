# Protocol V3 双随机种子 FSQ 工程实验

本目录记录从干净、不可变提交 `df194b1e481662c132d452da618e1d511d49ec3a` 运行的三臂双随机种子实验。它验证了修正后的全源扫描、exact 去重、parent/source 平衡采样、parent-CAD 隔离验证、旋转不变曲面分桶、checkpoint 绑定和晋级门控。它是 15 epoch 工程 cohort，不是 100 epoch 容量结论，也不包含序列、AR、自由生成或 OCC Valid 评测。

## 结论

程序结论为 `NO_PROMOTED_ARM`，`winner=null`。三种配置在两个 seed 上均未同时达到 perplexity `>=800`、曲面 parent-cluster MSE `<=5e-5`、parent coverage `>=90%` 且无非有限验证样本的绝对门槛，因此不能进入长训、序列重生成或 AR。

4096/6D 在两个 seed 的 15 epoch checkpoint MSE 都最低，均值为 `0.018478755`；这是早期工程排名，不是容量胜者。它的 checkpoint perplexity 范围仅为 `310.25–548.04`，曲面 parent-cluster MSE 均值为 `0.0252675`，仍明显未达晋级门。

| 配置 | checkpoint MSE 均值 | perplexity 范围 | code coverage 均值 | 曲面 parent-cluster MSE 均值 | 通过 seed 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192/4D `[8,8,8,16]` | 0.037100185 | 171.10–309.79 | 17.01% | 0.0506760 | 0/2 |
| 4096/6D `[4,4,4,4,4,4]` | 0.018478755 | 310.25–548.04 | 31.37% | 0.0252675 | 0/2 |
| 8192/6D `[4,4,4,4,4,8]` | 0.027943020 | 252.48–462.13 | 17.77% | 0.0358591 | 0/2 |

## 数据与控制

- train/validation cap 为 `12000/4637`，batch size 为 `128`，learning rate 为 `3e-4`，每臂 15 epoch，seed 为 0 和 1。
- 两个 seed 均扫描全部 `795/100` 个 train/validation source，加载失败为 0；最终 inventory 为 322/100 个 parent、source/parent/exact overlap 全部为 0。
- 训练采样的 post-filter parent coverage 为 `95.98%`，validation 为 `100%`；两个 cap 均精确满足。
- 三臂使用匹配的公共网络初始化和重置后的训练 RNG；曲面/复杂过采样关闭，所有 loss weight 为 1。
- protocol SHA-256 为 `43d0c5b36375cc78f3386a78a020a9baacc5a314372380f29e2eedb446345e6f`，split SHA-256 为 `df72b5757c3aabc89c707fd351c086ca8914cd96a49868decb8d15c104b17357`。

## 证据

`fsq_abc_15epoch_two_seed_20260804.json` 是 E029 结构化证据，包含六条完整 15-row history、采样/重叠审计、checkpoint-bound promotion 结果、运行 commit、协议/切分哈希和六个 event 的 SHA-256。JSON 自身 SHA-256 为 `4533aa250ac2cfb8e345a89b97c6bde17515412dd4dd0e041b2359ae61fa5d42`。

六个轻量 TensorBoard event 位于 `reports/tensorboard/protocol_v3_fsq_abc_15epoch_two_seed_20260804/`。每个 event 都有相同的 13 个 scalar tag 和 epoch 0–14 共 15 步；构建审计已逐 tag、逐 step 与 JSON 中的 history 数值核对。可在仓库根目录运行：

    tensorboard --logdir reports/tensorboard/protocol_v3_fsq_abc_15epoch_two_seed_20260804

原始数据、checkpoint、模型权重、完整 `local_runs/`、无效的中断/dirty 实验和机器绝对路径均不纳入 Git。

## 下一步

不要从本 cohort 进入 AR，也不要仅因 4096/6D 的 MSE 较低就直接做正式容量训练。下一轮优先继续拆分 FSQ、连续 decoder 与 assembly 的贡献；如果仍要扩展 VQ cohort，应先定义比 `5e-5` 更贴合当前归一化 MSE 口径、且能与 CAD/OCC 质量关联的可校准门槛，再用更多 parent 或全量 ABC、更多 seed 和更长但受控的训练验证。
