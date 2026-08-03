# Complex Curved Diagnostics

- Status: `VERIFIED`
- Selected records: `50`
- Output dir: `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_teacher_forcing`
- FSQ patch count: `3265`
- FSQ MSE mean: `0.003927467227110702`
- FSQ Chamfer mean: `0.045526498192094075`
- AR token-weighted teacher CE: `1.258693912499957`
- True-token reconstruction success: `skipped`
- True-token STEP saved: `skipped`
- True-token BRep valid: `skipped`

## Selection

- split: `val`
- requested: `50`
- selected: `50`
- max_scan: `5000`
- scanned: `5000`
- grammar_ok: `4559`
- complex_candidates: `3095`
- parsed_loaded: `3095`
- parsed_failed: `0`
- curved_threshold: `0.02`
- complex_min_faces: `12`
- complex_min_edges: `20`
- max_source_faces: `50`
- max_source_edges: `150`
- curvature_rank_key: `p95`
- failures: `{'not_complex': 1464, 'not_curved': 302, 'too_long': 441}`

## Interpretation Hooks

- If FSQ MSE and Chamfer are high on these real complex curved patches, prioritize FSQ capacity or loss changes before more AR sampling work.
- If FSQ patch metrics are acceptable but AR teacher CE is high in long or high-face buckets, prioritize AR training distribution, context length, or ordering.
- If true-token reconstruction fails while patch metrics are acceptable, inspect topology/OCC reconstruction and token-to-BRep assembly.
