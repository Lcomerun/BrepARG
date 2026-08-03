# 实验账本

机器可读完整记录位于 `experiment_ledger.json`。本表只保留决策所需字段；未知值没有补猜。

| 实验 | 数据/唯一变量 | 关键结果 | 状态与结论 | 证据 |
| --- | --- | --- | --- | --- |
| Stable VQ 40 | 修复 finite 防护后的 FSQ VQ | best val MSE 约 `5.6e-4`，batch finite | 完成；建立稳定基线 | E001, E002 |
| VQ continuation 85 | 同 cohort 延长训练 | best epoch 73，约 `3.7e-4` | 完成；全局 MSE 改善，复杂曲面未知 | E002 |
| Resume/scratch 50k | 三个 VQ checkpoint，同 50k patch | resume mean/median/p95 `3.725e-4/9.14e-6/6.4665e-4` 最优 | 完成；选择 resume-best | E017 |
| Full RCM sequence | ABC 0000-0099；50 faces/150 global edges | train/val/test `382903/21214/21003`，vocab 10294 | 完成；后来确认 parent 泄露 | E003, E004 |
| AR bs32/lr5e-4/ctx2048 | 增大 batch 与高 LR continuation | epoch 115 后 loss 爆炸、OOM、latest nonfinite | **失败**；禁止恢复污染 latest | E019, E020 |
| AR finite best | health audit | epoch 117，val CE `0.2990635`，101 float tensors finite | 完成；作为恢复/选择依据 | E020 |
| Original BrepARG sampler | 只换采样逻辑，沿用 V13 权重 | 124 attempts，100 retained，complex 11，faces median 6 | 完成；sampler 不是主因 | E009 |
| Generation quality gate | 只加筛选/约束 | 2405 attempts 才保留 100 | 完成；筛选不修复分布 | E009, E012 |
| FSQ-only complex-curved | 不接 AR/assembly；50 shapes/3399 patches | all p95 `0.15012`，surface p95 `0.41238` | 完成；复杂曲面 heavy tail | E006 |
| True-token reconstruction | 绕过自由 AR | 50 grammar-valid，27 STEP，9 BRep-valid | 完成；最强上游瓶颈证据 | E007, E008 |
| FSQ capacity 16,16,8,8 | 只提高离散容量候选 | overall p95 `+4.19%`，surface p95 `-26.91%` | 完成；混合信号，不晋级 | E016 |
| DFS vs RCM medium | matched medium ordering control | token-weighted CE `1.25869` vs `1.31302` | 完成；DFS 小幅更好，排序次级 | E010 |
| Official BrepARG probe | 官方权重 vs 本地 vocab | 7222 vs 10294 | **阻塞**；不是协议兼容复现 | E021 |
| BrepARG short same-data | 10k/1k/1k，VQ best70，AR best77/80 | val CE `0.871925`；complex 5/92，strict 0/92 | 完成；质量差且 split 泄露 | E005, E013 |
| BrepARG long same-data | 同 split，VQ400/AR300 | val CE `0.765318`；complex 13/100，strict 6/100 | 完成；长训改善但不根治 | E005, E014 |
| V13 parent split audit | 只审计身份 | val/test parent-shared records `56.75%/57.17%` | 完成；历史泛化结论降级 | E004 |
| Ground-truth assembly oracle | 未来：完全绕过 VQ/AR | 未执行 | **阻塞**；缺 package-safe CLI | E027 |
| Continuous latent oracle | 未来：绕过 FSQ quantization | 未执行 | **阻塞**；缺 verified bypass | E027 |
| Teacher-forced argmax | 未来：真实前缀下一步 argmax 重建 | 未执行 | **阻塞**；现有实验只算 CE/重建 target | E027 |

## 统一解释规则

1. `validation reconstruction MSE` 不是 surface Chamfer，也不是 BRep-valid。
2. teacher-forcing CE 衡量真实前缀下条件预测；true-token reconstruction 衡量 VQ/decoder/assembly；free generation 还包含 exposure 和 sampling。
3. 所有生成比例必须保存全部 attempts 分母。`retained` 或 `STEP saved` 不能替代 attempts。
4. 历史 split 有 parent-CAD 泄露，因此其 val CE 仅可用于同 run 内的开发和 checkpoint 选择。
5. same-data BrepARG 结果只能标为自训 baseline；官方权重结果仍 unavailable。
6. 只有 finite best checkpoint 可作为恢复或生成输入；文件名中的 `latest`、`final`、`best` 不是身份保证。
