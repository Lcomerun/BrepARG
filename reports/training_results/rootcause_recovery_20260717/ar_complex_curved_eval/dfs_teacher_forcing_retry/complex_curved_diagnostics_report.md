# Complex Curved Diagnostics

- Status: `VERIFIED`
- Selected records: `3`
- Output dir: `D:\V13_rootcause_recovery_20260717\ar_complex_curved_eval\dfs_teacher_forcing_retry`
- FSQ patch count: `107`
- FSQ MSE mean: `0.0005380724702214376`
- FSQ Chamfer mean: `0.018820279395364434`
- AR token-weighted teacher CE: `0.65511443997794`
- True-token reconstruction success: `skipped`
- True-token STEP saved: `skipped`
- True-token BRep valid: `skipped`

## Selection

- split: `val`
- requested: `3`
- selected: `3`
- max_scan: `500`
- scanned: `500`
- grammar_ok: `494`
- complex_candidates: `348`
- parsed_loaded: `348`
- parsed_failed: `0`
- curved_threshold: `0.02`
- complex_min_faces: `12`
- complex_min_edges: `20`
- max_source_faces: `50`
- max_source_edges: `150`
- curvature_rank_key: `p95`
- failures: `{'not_complex': 146, 'not_curved': 21, 'too_long': 6}`

## Interpretation Hooks

- If FSQ MSE and Chamfer are high on these real complex curved patches, prioritize FSQ capacity or loss changes before more AR sampling work.
- If FSQ patch metrics are acceptable but AR teacher CE is high in long or high-face buckets, prioritize AR training distribution, context length, or ordering.
- If true-token reconstruction fails while patch metrics are acceptable, inspect topology/OCC reconstruction and token-to-BRep assembly.
