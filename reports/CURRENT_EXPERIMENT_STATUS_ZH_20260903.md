# 最近实验总体状态（2026-09-03）

仓库：`Lcomerun/BrepARG`；分支：`experiment/protocol-v5-scaling-ladder`

## 一页结论

当前阶段尚未达到“可生成可用 CAD”的最终目标，但根因范围已经明显收窄：

1. 表示层容量对照已经结束。固定 100-CAD、相同未修复装配链下，continuous bypass 为 `70/100` strict-valid，VQ-8192/64D 为 `69/100`，两者只差 `1 pp`；RVQ-2x4096 为 `65/100`，FSQ@300k 为 `49/100`。因此正式表示主臂是 `VQ-8192/64D`，RVQ 和 FSQ 不晋级，旧训练不需要恢复。
2. 当前主要阻塞是 CAD 装配链。failure-triggered selector 的正式结果为 STEP-readable `97/100`、OCC native-valid `90/100`、project strict-valid `91/100`、both-valid `88/100`；84 个历史有效控制全部保持有效，回退为 0。发布门仍要求 strict-valid `>=95/100`，因此还缺至少 4 个安全恢复案例。
3. 最新 construction-stage periodic-pcurve census 已经完成并得出有效负结果。固定 5 个 residual CAD 的 5 个独立 OCC worker 全部完成，共覆盖 `134/134` 个 face，观测到 6 个 strict-style bad face；6 个均为 U/V 两向都非周期的 `Geom_BSplineSurface`。worker、协议、source binding 和 measurement failure 均为 0。因此预注册判决为 `CLOSE_PERIODIC_PCURVE_ROUTE`。
4. 这个判决只关闭固定五 CAD 上的“按曲面周期平移 pcurve branch”方案，不代表装配问题已经解决，也没有把 91/100 提高。下一方向应针对已经观测到的二维 trim/wire 相交，以及 shell/connectivity 失败族；不得为了提高数字放宽 schema-v2 topology/geometry gate。
5. 在装配 strict-valid 达到 `>=95/100` 前，boundary-consistency loss、全量 VQ、序列重生成、AR 和最终 OCC 生成评测继续阻塞。当前 GPU 空闲是正常状态，不应重启旧训练。

## 当前正式数字

| 层次/实验 | 正式结果 | 当前判定 |
| --- | ---: | --- |
| Continuous bypass @ 60k | `70/100` strict-valid | 连续表示参考 |
| VQ-8192/64D @ 60k | `69/100` strict-valid | 主臂；量化税约 `1 pp` |
| RVQ-2x4096/64D @ 60k | `65/100` strict-valid | 不晋级 |
| FSQ @ 300k | `49/100` strict-valid | 退役 |
| 当前 assembly selector | `91/100` strict-valid | 未过 `>=95/100` 门 |
| 历史有效控制 | `84/84` | 零回退 |
| 最新 periodic census | `5/5` completed，`0` periodic bad face | 关闭该固定 cohort 路线 |

## 最新判决实验的证据完整性

第一次 formal attempt 使用 commit `ad6f385`，在创建 run manifest 时因为 Python 3.9 不支持 `Path.write_text(newline=...)` 而失败。该次尝试发生在任何 CAD worker 启动之前，只留下 writer lock，没有 case row，属于基础设施失败，不进入科学结论。

修复提交 `0c1ce51` 改用显式打开、flush、`fsync` 和原子替换，并补了正常替换、拒绝 non-finite JSON、替换失败清理临时文件的回归测试。之后从 clean worktree 在新的 immutable `_v2` root 正式运行，结果为：

```text
cases                         5
completed                     5
all faces observed            134 / 134
bad faces                     6
periodic bad faces            0
repairable periodic faces     0
worker/protocol failures      0
decision                      CLOSE_PERIODIC_PCURVE_ROUTE
```

正式 run signature：

```text
eb3520b167ea912c8f7d7d291a99072940b132764eb61d72eeba0e2719b4226b
```

三份原始核心结果已经逐字节归档并通过路径去敏及 SHA-256 复验：

| 文件 | SHA-256 |
| --- | --- |
| `periodic_pcurve_cases.jsonl` | `efc0bce0900dadd1ae3b88c11e202b0e509dde0b9c53fb047ebbc57f25c6078a` |
| `periodic_pcurve_summary.json` | `a505e396ff84bc6f876e2cef01689be6711da2ac88cea6c0fcc7c11ea6993cbf` |
| `periodic_pcurve_run.json` | `28edafc097c9bf27c81f64641914171d68fec18d109a6059db80892f7697cd12` |

## 下一步工作边界

优先级仍然是装配链，而不是训练：

1. 从 9 个 selector residual 中按已经观测到的失败机制拆分二维 trim/wire 相交与 shell/connectivity 家族。
2. 新 repair 必须先做 exact CAD/face/wire 小范围 pilot，并继续通过 STEP-readable、native、strict、both-valid、schema-v2 topology/incidence、3D curve preservation 和 geometry preservation 全部门。
3. 只有出现 net-new gate-accepted CAD，才值得扩大到 invalid subset；接近 95 后才重跑完整固定 100-CAD selector。
4. 最终放行条件保持不变：strict-valid `>=95/100`、历史控制 `84/84`、regressions `0`、worker/protocol failures `0`。
5. 装配门通过后，才重新测 repaired-chain 的 VQ-8192 与 bypass 差距；之后才讨论全量 VQ、词表/序列、AR 和最终生成质量。

## 建议阅读顺序

1. `reports/periodic_pcurve_applicability_census_20260903/README.md`：最新五 CAD 判决及逐 face 摘要。
2. `reports/periodic_pcurve_applicability_census_20260903/archive_validation.json`：路径、哈希、行数及失败计数复核。
3. `reports/assembly_selector_main_100cad_20260818/README.md`：当前 91/100 selector 正式结果。
4. `reports/capacity_ab_posthardening_assembly_measurement_20260818/capacity_ab_assembly_measurement.md`：VQ-8192/RVQ/bypass 的表示层选择依据。
5. `plans/periodic_pcurve_applicability_census_20260903.md`：完整执行过程、异常分类和决策边界。

Git 中只保存代码、测试、文档、JSON/JSONL 和哈希。STEP、source pickle、worker 原始日志、NumPy 数组、模型 checkpoint、上游 `BrepARG/` 源码和 `papers/` 均未上传。
