# Failure-family follow-up: local intersection topology and single solid

This is a Git-safe CPU-only probe on the frozen 16 historical-invalid CADs.
It ran two repair profiles in isolated workers with the schema-v2 selector
geometry/topology gate enabled:

| Profile | Attempts | STEP-readable | Native | Strict/both | Gate accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| `directed_trim_local_intersection_topology` | 16 | 13 | 7 | 3 | 0 |
| `single_solid` | 16 | 11 | 5 | 1 | 1 |

The local-intersection profile produced three strict-valid candidates, but the
gate rejected every one: two changed source topology and one could not complete
the geometry measurement, which is a fail-closed rejection. The single-solid
profile produced one gate-accepted candidate, CAD `00000444...`, but that CAD
was already recovered by the production selector. The net number of new,
topology-preserving restorations is therefore zero. There were zero worker or
protocol failures.

Per-attempt path-free evidence is in `geometry_gate_attempts.jsonl`; aggregate
counts and rejection reasons are in `geometry_gate_summary.json`. STEP, pickle,
log, array, and checkpoint bytes remain outside Git. This negative diagnostic
does not authorize a full 100-CAD rerun, boundary-consistency loss, sequence
regeneration, or AR training.
