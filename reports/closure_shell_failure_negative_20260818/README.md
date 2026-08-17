# Closure and face-boundary repair diagnosis (negative result)

This report records the bounded, one-CAD-at-a-time diagnosis of three
remaining assembly failures in the frozen 100-CAD cohort. It is a negative
result: no tested repair is eligible for the unchanged schema-v2 selector.
The report is intentionally Git-safe. It contains compact counts, failure
categories, source bindings, run signatures, and hashes only. STEP files,
source pickles, reconstructed arrays, checkpoints, and machine-local paths
remain outside Git.

## Scope and gate

Each Open CASCADE operation ran in a child process with a finite timeout. A
candidate could be accepted only when STEP was readable, native OCC validity
and project strict validity were both true, the result had one solid and no
self-intersecting wires, and the existing schema-v2 geometry/topology gate
accepted unchanged face/edge/vertex counts, incidence, sampled 3-D curves,
and bounds. No tolerance or gate was relaxed.

The authoritative six-attempt run is `closure-shell-stage-probe-v5`:
three CADs crossed with `historical` and `directed_interpolate`. It completed
6/6 attempts with 0 worker/protocol failures and 0 both-valid candidates.

## Stage diagnosis

| CAD | Historical path | Interpolation path | First usable defect | 3-D loop endpoint gap at bad faces | Result |
| --- | --- | --- | --- | ---: | --- |
| `00061931` | curve fit fails at edge 0 | `face_raw`, faces 0 and 23 self-intersect | 2-D face/pcurve boundary | 0.0 | reject |
| `00087341` | curve fit fails at edge 11 | `face_raw`, faces 0, 2, 12, 16-26, 29, 30 self-intersect | 2-D face/pcurve boundary | 0.0 | reject |
| `00095733` | `face_raw`, faces 0 and 26 self-intersect | same | 2-D face/pcurve boundary | 0.0 | reject |

The zero 3-D endpoint gap is measured from the oriented WCS edge polylines
before face construction. It rules out a simple adjacent 3-D endpoint closure
defect for these cases; it does not imply that the 2-D pcurves on the surface
are valid.

For `00061931`, the bad faces contain closure, adjacent, and many
non-adjacent crossing reports and remain invalid after sewing. For `00087341`,
the defect is distributed over many faces rather than a single local edge.
For `00095733`, face 0 has a non-adjacent crossing at edge positions 8-10 and
face 26 has a closure crossing at positions 19-1. The historical path reaches
one in-memory solid, but STEP roundtrip produces zero solids and one
self-intersecting wire.

## Candidate families

* Periodic pcurve branch translation was not applicable: both target surfaces
  reported `surface_not_periodic`.
* Local pcurve continuity was rejected. Face 0 changed edge count 20 to 19
  and retained a self-intersection. Face 26 preserved the measured geometry
  and topology but retained its self-intersection, so it was not accepted.
* Post-sewing pcurve reprojection was incomplete for the full sewn shell at
  precisions `1e-2` through `1e-6`. On isolated face 26, topology checks passed
  but the strict 3-D curve-preservation gate rejected every precision with
  `three_dimensional_curve_changed`; no candidate was promoted.
* Endpoint-constrained curve rescue exactly matched both endpoints for the
  rescued edge in the `00051602` probe, but the final result remained native
  invalid, strict invalid, and both invalid with one self-intersecting wire.
* Graph-preserving trim alternatives for `00032101` and `00076198` are
  archived separately on `protocol-v5-graph-trim`; OCC-valid alternatives
  depended on source edge/vertex deletion or merge, while the no-topology path
  remained strict invalid.

## Decision

No repair switch is promoted. The production assembly implementation and the
schema-v2 gate remain unchanged. Do not start an invalid16 or 100-CAD matrix
from this branch. The next repair effort should target a different diagnosed
failure family (for example single-edge curve fallback, multi-edge closure,
or shell/connectivity) and must first pass a one-CAD gate while retaining the
84/84 original strict-valid controls.

See `stage_summary.jsonl`, `repair_candidate_summary.json`,
`source_bindings.json`, `run_manifest.json`, and `artifact_manifest.json` for
the machine-readable evidence contract.
