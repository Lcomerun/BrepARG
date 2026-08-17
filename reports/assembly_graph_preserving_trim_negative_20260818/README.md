# Graph-preserving trim diagnosis: no candidate promoted

## 结论

本轮没有找到可安全晋级的 graph-preserving trim 候选，正式生产构造器保持不变，schema-v2 geometry/topology gate 也没有放宽。

- 00032101 的 OCC native/strict valid 依赖合法 source vertex 被合并。历史路径从 18 个 source vertex 变成 16 个；最小 ShapeFix 路径进一步变成 15 个。历史路径的最近空间对应组为 {3,7} 和 {6,9}，最小路径为 {3,7,8} 和 {6,9}。
- 00076198 的历史路径只有删除 source edge/vertex 后才得到 OCC valid：106 edges / 68 vertices / 212 face-edge occurrences 变成 104 / 65 / 208。最小路径可以保住 106 edges 和 212 occurrences，但仍只有 65 vertices、保留 1 个 wire self-intersection，并且 bbox relative delta 为 0.04960，超过 gate 的 0.02。
- 00076198 唯一自交定位到 candidate face ordinal 10、wire ordinal 0。几何最近匹配指向 source face 28，完整 source edge 集为 [86,91,92,93]；二边自交子环对应 source edges 86 与 91。
- sewing tolerance 1e-3 / 1e-4 / 1e-5 的结论完全相同，收紧 tolerance 不能解决问题。

因此正式判定为：

    00032101: REJECT — OCC-valid depends on source-vertex merging.
    00076198: REJECT — graph-preserving path remains strict-invalid;
              OCC-valid path deletes source edges/vertices.

## 首个明确破坏点

曲线构造没有丢边：两个 CAD 分别完整构造了 28/28 与 106/106 edges。阶段诊断显示第一个明确的 topology change 发生在历史 ShapeFix_Face：

| CAD | Face | Before | After |
| --- | ---: | --- | --- |
| 00032101 | 4 | 4 edges / 4 vertices | 4 edges / 3 vertices |
| 00076198 | 28 | 4 / 4 | 2 / 2 |
| 00076198 | 29 | 4 / 4 | 2 / 2 |
| 00076198 | 36 | 4 / 4 | 4 / 3 |

这说明故障不是曲线拟合漏 edge，也不是简单缺少显式共享 vertex；当前拟合曲面/曲线与 source graph 在 OCC face repair、sewing 和 STEP roundtrip 下不兼容。

## 安全复跑

安全复跑共 8 次：2 CAD × historical + minimal-no-topology 的 3 档 sewing tolerance。结果为：

- worker/protocol failure：0
- schema-v2 selector eligible：0
- native crash：0

独立 probe 只暴露 historical 和 minimal-no-topology 两种已完成无 crash 复跑的策略。它在每个 worker 内临时替换 fix_face/sewing 行为，并在 finally 恢复；生产构造器没有新增实验参数。

## 被隔离的不安全路径

以下 12 次尝试均在独立 worker 中以 Windows 0xC0000005（十进制 3221225477）退出：

- 全局 skip face fix：4 次；
- orientation-only：4 次；
- BRepBuilderAPI_Copy(face) 后执行 topology-guarded face fix：4 次。

这些路径已从可复跑 probe 和生产接口中移除。显式复用 source vertices 虽然没有 native crash，但令 00032101 的 candidate vertices 降到 14、00076198 降到 64，同样明确拒绝。

## 证据与边界

完整紧凑数据见 result.json。报告只保留 source 文件 SHA-256/大小、拓扑计数、gate 原因、索引映射和数值残差。STEP、pickle、NumPy、原始 CAD、重建几何和 checkpoint 都没有进入 Git。

source vertex merge group 与 source edge 对应来自最近空间/曲线采样匹配，是解释性定位证据；schema-v2 的离散计数、incidence、validity 与几何 gate 仍是正式判决依据。

后续不应继续在这两个 CAD 上放宽 tolerance 或 topology gate。为了达到 GT 95/100，应转向剩余失败族，例如 single-edge curve fallback、multi-edge closure 和 single-shell/connectivity，同时继续要求原 84/84 control 零退化。
