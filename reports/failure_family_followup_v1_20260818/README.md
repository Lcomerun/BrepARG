# Failure-family follow-up: curve fit and local pcurve continuity

This is a Git-safe CPU-only probe on the frozen 16 historical-invalid CADs.
It ran two profiles with isolated workers and the schema-v2 selector geometry
gate enabled:

| Profile | Attempts | STEP-readable | Native | Strict/both | Gate accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| `directed_trim_curve_fit` | 16 | 14 | 6 | 0 | 0 |
| `directed_trim_local_pcurve_continuity` | 16 | 13 | 7 | 2 | 0 |

The two strict-valid local-pcurve candidates were rejected by the gate. CAD
`00016845...` changed edge/face-boundary and vertex-incidence topology. CAD
`00032004...` returned a geometry-measurement error, so it is not eligible for
promotion. The curve-fit profile had no strict-valid candidate. There were
zero worker or protocol failures and no historical-control regression was
measured because this was the invalid-only diagnostic cohort.

The complete path-free per-attempt evidence is in
`geometry_gate_attempts.jsonl`; aggregate counts and rejection reasons are in
`geometry_gate_summary.json`. STEP, pickle, log, and model bytes remain outside
the repository. This probe does not authorize a 100-CAD run, boundary loss,
sequence regeneration, or AR training.
