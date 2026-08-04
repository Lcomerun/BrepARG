# Protocol V2 修改与十轮 FSQ 小实验总结

## 本分支完成的修改

本分支只修改 V13 自有代码，不修改或提交 `BrepARG/` 与 `papers/`。数据侧新增 ABC Protocol V2：只接收 10–50 faces、全局 edges 不超过 150、每个 face 的 edges 不超过 30、`faceEdge_adj` 完整且 parent CAD 可解析的记录；再按 parent 分组进行确定性 8:1:1 划分。真实小样本扫描得到 795/100/99 条 train/val/test 记录，三对 parent overlap 均为 0。

VQ/FSQ 侧不再从同一 patch 数组切 train/validation。训练 patch 只来自 Protocol train，validation patch 只来自 Protocol val；每个 split 内按 kind、shape 和标准化 float32 bytes 做 exact 去重，跨 split 的 exact 重复只从 train 删除。tensor 化前后都检查 source、parent 和 exact hash overlap，身份不完整、请求 source 加载失败或经过 source cap 后没有可用 patch 时 fail closed。

训练监控改为严格 sample-weighted validation MSE，并记录平面近似 surface、曲面代理 surface、edge 三桶重建误差。FSQ usage 由整个 validation 集合的编码计数计算 unique bins、coverage 和 entropy perplexity。任一 validation sample 为 NaN/Inf 时整轮 validation 失效，不能覆盖 best checkpoint。

## 数据与实验控制

两组实验均从 commit `a9a562ad3fe8cf28a5d0d3c508e9535faabdf413` 全新开始，使用同一 Protocol hash `43d0c5b36375cc78f3386a78a020a9baacc5a314372380f29e2eedb446345e6f` 和同一 split hash `df72b5757c3aabc89c707fd351c086ca8914cd96a49868decb8d15c104b17357`。两组最终都使用 951 个 train patch 和 370 个 validation patch；source、parent 和 exact hash overlap 均为 0。去重后的 patch 实际来自 30/7 个 train/validation source key、27/7 个 parent CAD；因此 370 个 validation patch 是 7 个 validation parent 内的聚类观测，不能解释成 370 个独立 CAD 样本。

共同控制为 seed 0、10 epochs、batch 128、learning rate `3e-4`、AMP、无 resume、无曲面或复杂样本过采样、loss weight 均为 1。唯一改变的科学超参数是由 `NS_LEVELS` 指定的 FSQ levels 配置；`NS_OUT` 与 `NS_VQ_TB_LOG_DIR` 只用于隔离两组输出：

| 配置 | `NS_LEVELS` | FSQ 维度 | 理论 bins |
| --- | --- | ---: | ---: |
| 8192/4D | `8,8,8,16` | 4 | 8192 |
| 4096/6D | `4,4,4,4,4,4` | 6 | 4096 |

因此本实验比较的是两个完整 FSQ 配置，不能把差异单独归因于“码本从 8192 降为 4096”；FSQ 维度也同时从 4 增至 6。

## 十轮结果

| 指标 | 8192/4D | 4096/6D | 4096 相对变化 |
| --- | ---: | ---: | ---: |
| best validation MSE | 0.18834483（epoch 6） | 0.16261402（epoch 9） | 低 13.66% |
| epoch-9 validation MSE | 0.19929265 | 0.16261402 | 低 18.40% |
| epoch-9 平面近似 surface MSE | 0.15026901 | 0.11809767 | 低 21.41% |
| epoch-9 曲面代理 surface MSE | 0.19944016 | 0.17206810 | 低 13.72% |
| epoch-9 edge MSE | 0.20898938 | 0.16733651 | 低 19.93% |
| epoch-9 unique bins | 29 | 87 | 3.0 倍 |
| epoch-9 coverage | 0.3540% | 2.1240% | — |
| epoch-9 entropy perplexity | 13.4513 | 29.7591 | 2.21 倍 |

两组 80 个 train batches、30 个 validation batches 全部有限，没有跳过 train batch，也没有 nonfinite validation sample。两份 curated TensorBoard event 的九个关键 scalar tag 均有 10 个 step。

## 判断

在这次单 seed、微型 cohort 的严格配对实验中，4096/6D 在全局重建、三个重建桶和最终 usage 上都优于 8192/4D，适合作为下一轮受控实验的优先候选。8192/4D 的 usage 在 epoch 4 达到 65 bins/perplexity 26.89 后回落到 29 bins/perplexity 13.45，说明只看曾经达到的峰值会高估稳定利用率。

本次预先定义的相对晋级门是“曲面代理桶改善且 aggregate usage 不恶化”。4096/6D 的曲面代理 MSE 更低、unique bins 和 perplexity 更高，因此通过该门，支持进入更大 cohort、约 100 epoch 的受控 VQ 实验。末轮 perplexity 13.45/29.76 与曲面代理 MSE 0.199/0.172 距离 800–1500 和 `5e-5` 仍很远，但这两组绝对值是后续训练的启发式健康参考，不是本次微实验的验收条件。

十轮 patch 重建实验不包含自由生成、STEP 装配或 OCC Valid，且 validation 只覆盖 7 个 parent CAD，因此 18.40% 的末轮差异需要在更大 parent cohort 上复现，不能与论文 Valid 67.54 对比，也不能证明最终 CAD 可用。因此 E028 只晋级下一阶段 VQ 实验，不晋级 AR 或正式全量训练。

## 下一步措施

不在当前 951/370 patch 微型 cohort 上简单续训，也不进入 AR。下一步按原计划在更大的 Protocol V2 cohort 上运行约 100 epoch 的 4096/6D 受控 VQ 实验，并继续逐 epoch 监控曲面桶与 aggregate usage；条件允许时增加配对 arm 或一次只改变一个因素的消融，以分离 bins 数和 FSQ 维度。只有下一阶段同时建立稳定 usage、曲面重建和多 seed 证据，才晋级正式全量 VQVAE 与后续 AR。完整逐 epoch数字、hash、控制变量和限制见 `fsq_micro_comparison_20260804.json`。
