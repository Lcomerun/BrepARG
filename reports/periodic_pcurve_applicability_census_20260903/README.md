# Periodic-pcurve construction-stage applicability census

This is the Git-safe snapshot of the signed, read-only five-CAD census. Every
CAD ran in an isolated Open CASCADE worker using the same
`directed_trim_curve_fit` construction profile. Faces were observed after the
baseline wire fix and pcurve attachment and before any optional strategy
repair. The census did not mutate a face and did not write STEP candidates.

## Result

- Decision: `CLOSE_PERIODIC_PCURVE_ROUTE`
- Cases completed: `5/5`
- Worker, protocol, binding, or measurement failures: `0`
- Fully observed bad faces: `6`
- Periodic bad faces: `0`
- Repairable periodic bad faces: `0`
- Assembly selector remains: `91/100` strict-valid; release gate: `>=95/100`

All six diagnosed bad faces are non-periodic fitted B-spline surfaces:

- `00032101_674d8fea687f4d9bbca6599b_step_000` face `3`: `Geom_BSplineSurface`, U/V periodic = `false/false`, occurrences = `non_adjacent, pcurve_gap`, decision reason = `surface_not_periodic`.
- `00032101_674d8fea687f4d9bbca6599b_step_000` face `4`: `Geom_BSplineSurface`, U/V periodic = `false/false`, occurrences = `non_adjacent, pcurve_gap`, decision reason = `surface_not_periodic`.
- `00076198_7fde7438ca5d3ccb8a1dd1f4_step_000` face `28`: `Geom_BSplineSurface`, U/V periodic = `false/false`, occurrences = `adjacent, closure`, decision reason = `surface_not_periodic`.
- `00076198_7fde7438ca5d3ccb8a1dd1f4_step_000` face `29`: `Geom_BSplineSurface`, U/V periodic = `false/false`, occurrences = `adjacent, closure`, decision reason = `surface_not_periodic`.
- `00051602_7f1947595ae247e0a4a32f43_step_000` face `2`: `Geom_BSplineSurface`, U/V periodic = `false/false`, occurrences = `adjacent`, decision reason = `surface_not_periodic`.
- `00051602_7f1947595ae247e0a4a32f43_step_000` face `3`: `Geom_BSplineSurface`, U/V periodic = `false/false`, occurrences = `non_adjacent`, decision reason = `surface_not_periodic`.

The conclusive result closes the periodic-pcurve branch-translation route only
for this frozen five-CAD cohort. It does not claim that the overall assembly
problem is solved and does not authorize boundary-consistency training,
full-scale VQ training, sequence regeneration, AR training, or a full 100-CAD
rerun. The next assembly investigation should target the observed two-dimensional
trim/wire intersection and shell/connectivity failure families without relaxing
the schema-v2 topology and geometry gates.

`periodic_pcurve_cases.jsonl`, `periodic_pcurve_summary.json`, and
`periodic_pcurve_run.json` are byte-identical to the completed local run and are
bound by SHA-256. No STEP, source pickle, worker log, NumPy array, checkpoint,
upstream source, or machine-local absolute path is archived.
