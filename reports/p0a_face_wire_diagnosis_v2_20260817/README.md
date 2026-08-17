# P0-A face/wire crossing diagnosis v2

This Git-safe report extends, but does not replace, the v1 face/wire report.
It binds the same 16 stage-aware P0-A baseline cases and records independent
OCC crossing modes with one-based edge positions. No STEP, pickle,
reconstruction, model, local path, or upstream-source bytes are archived.

- Frozen cases: `16`
- Saved STEP cases with direct OCC diagnosis: `11`
- Pre-STEP cases explicitly marked unavailable: `5`
- Edge-position basis: `occ_1_based`
- Aggregate self-intersecting wires: `16`
- Wires with at least one classified occurrence: `16`

## Occurrence taxonomy

| Kind | Occurrences | CADs |
| --- | ---: | ---: |
| `adjacent` | 11 | 6 |
| `closure` | 5 | 4 |
| `non_adjacent` | 4 | 2 |
| `self_only` | 0 | 0 |
| `pcurve_gap` | 7 | 2 |
| `seam` | 0 | 0 |
| `disconnected` | 0 | 0 |
| `unavailable` | 5 | 5 |

`closure` is only the cyclic `(n, 1)` pair. `adjacent` is only `(i-1, i)`
for positions 2 through n. `non_adjacent` contains only pairs whose cyclic
distance exceeds one. `self_only`, `pcurve_gap`, `seam`, and `disconnected`
are independent evidence and may coexist on one wire. The `status` field
distinguishes detected geometry from unavailable pcurves and wrapped OCC
failures; missing evidence is never interpreted as a clean check.

The five no-STEP CADs remain pre-STEP investigations. This report localizes
evidence only and does not claim that an assembly repair has been implemented.
