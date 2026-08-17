# ADR-0004: Hold face/pcurve repair candidates at the fail-closed gate

## Status

Accepted as a diagnostic result. No candidate is promoted to the production
assembly overlay.

## Date

2026-08-18

## Context

The best frozen-cohort assembly selector reaches 91 strict-valid CADs out of
100, retains all 84 original strict-valid controls, and therefore remains
below the predeclared 95/100 release gate. Three remaining failures were
suspected to be closure or shell problems. A stage probe was added so that
curve fitting, face construction, sewing, solid construction, and STEP
roundtrip could be distinguished without putting malformed OCC shapes in the
parent process.

The authoritative six-attempt run covers three CADs and two construction
variants. The historical paths fail curve fitting for `00061931` edge 0 and
`00087341` edge 11. Interpolated paths reach face construction but produce
self-intersecting pcurve boundaries. `00095733` produces two raw-stage bad
faces (0 and 26). The maximum oriented 3-D endpoint gap for every reported bad
face is zero, so changing endpoint closure alone is not a supported diagnosis.

## Decision

Keep the stage observer and isolated diagnostic probe available on the feature
branch, but promote no face/pcurve repair. Reject any candidate that changes
face/edge/vertex counts or incidence, changes sampled 3-D curves, leaves a
self-intersection, produces more than one shell/zero solids after roundtrip,
or fails project strict validity. Keep the existing schema-v2 geometry/topology
gate and the 95/100 release gate unchanged.

## Alternatives considered

### Periodic pcurve branch translation

Rejected because both target faces report `surface_not_periodic`; applying an
integer period shift would be an unsupported mutation.

### Local pcurve continuity repair

Rejected because face 0 changed edge count and still self-intersected, while
face 26 retained its self-intersection. A geometry-preserving but invalid face
is not a recovery.

### Post-sewing pcurve reprojection

Rejected because full-shell edge mapping was incomplete. The isolated face 26
operation preserved the topology signature but failed the strict 3-D curve
preservation gate at every tested precision.

### Broad ShapeFix or tolerance relaxation

Rejected because it can silently remove or merge source topology. The graph
preserving trim study separately showed OCC-valid results that failed the
source incidence gate, and the tolerance scan did not change the decision.

## Consequences

The production assembler remains unchanged and no invalid16 or 100-CAD matrix
is authorized from this branch. Future work must target a different diagnosed
failure family and pass a one-CAD promotion gate before broader evaluation.
Boundary-consistency loss, sequence regeneration, and AR remain blocked until
the assembly gate is met.
