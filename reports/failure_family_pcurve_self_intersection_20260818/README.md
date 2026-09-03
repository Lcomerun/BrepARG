# Failure-family probe: pcurve self-intersection

This is a Git-safe CPU/OCC probe on the frozen 16 historical-invalid CADs. It
ran `directed_trim_pcurve` with isolated workers and the schema-v2 selector
geometry/topology gate enabled.

| Profile | Attempts | STEP-readable | Native | Strict/both | Gate accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| `directed_trim_pcurve` | 16 | 13 | 7 | 2 | 0 |

The two strict-valid candidates were both rejected by the gate. CAD
`00016845...` changed edge count, face-edge occurrences/incidence, edge-face
incidence, and vertex-edge incidence. CAD `00032004...` could not complete the
gate measurement (`ValueError`), so it is not eligible for promotion. There
were zero worker or protocol failures. This profile adds no topology-preserving
restoration and does not authorize a 100-CAD rerun, boundary-consistency loss,
sequence regeneration, or AR training.

Per-attempt path-free evidence is in `geometry_gate_attempts.jsonl`; aggregate
counts and rejection reasons are in `geometry_gate_summary.json`. STEP, pickle,
log, array, and checkpoint bytes remain outside Git.
