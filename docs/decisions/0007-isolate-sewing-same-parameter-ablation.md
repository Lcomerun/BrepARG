# ADR-0007: Isolate and reject sewing SameParameter as the 63055 repair

## Status

Rejected after development ablation

## Date

2026-09-03

## Context

The fixed selector is strict-valid for 91 of 100 CADs and must reach at least
95 without regressing any of the 84 historical controls. The representation
decision is already closed in favor of VQ-8192/64D, so assembly is the active
release blocker.

A clean, signed four-stage lineage run completed two residual CADs with zero
mapping, observation, coverage, source-binding, worker, or protocol failures.
`00047472...` is already bad immediately after pcurve construction.
`00063055...`, however, is clean after pcurve construction and after optional
face repair; source face 5 first acquires one closure and one adjacent report
for the same reversed source edge pair during sewing. The defect persists after
STEP roundtrip.

Prior experiments changed sewing tolerance, reconciled near vertices, or
applied face repair before sewing. None isolates OCC's SameParameter processing,
which is enabled by default in `BRepBuilderAPI_Sewing`.

## Decision

The single-variable development ablation was run before any production switch
was added. It held joint optimization at 200 iterations, the ordered 18 copied
faces and sewing tolerance `1e-3` fixed, and compared only
`SameParameterMode(True)` with `SameParameterMode(False)`. Face mode remained
true, floating-edge mode false, and non-manifold mode false.

Both arms produced the same result: source face 5 was strict-clean before
sewing and acquired the same closure and adjacent defect for source edges 9
and 23 after sewing. Both STEP roundtrips were native-invalid, strict-invalid,
and both-invalid, with 18 faces, zero free edges, 50 contiguous edges, and zero
multiple edges. Disabling SameParameter therefore has no observed benefit for
the exact failure it was meant to address.

Reject this route. Do not add a `sewing_no_same_parameter` profile, do not
change the assembler API, and do not expand the ablation to the 16- or 100-CAD
cohorts. The next `00063055...` work is a diagnostic-only, graph-preserving
post-sewing feasibility spike. It must prove unchanged face/edge/vertex counts,
incidence, shared-edge correspondence, and sampled 3D curves before it can
become a default-off repair candidate.

## Alternatives Considered

### Change sewing tolerance again

Rejected. Tolerances `1e-4`, `1e-3`, and `1e-2` already leave the target
invalid, so another tolerance point is unlikely to identify the responsible
operation and would confound stitching distance with parameter repair.

### Apply local face repair before sewing

Rejected for this target. Both observed pre-sewing stages are strict-clean;
mutating them cannot be attributed to the first failure and prior local
pcurve/topology profiles did not recover this CAD.

### Apply a broad post-sewing ShapeFix

Rejected. Earlier broad repairs sometimes achieved apparent validity by
changing topology. Any next post-sewing experiment must first demonstrate that
it preserves the source graph and 3D geometry.

### Repair `00047472...` in the same change

Rejected. It fails at a different stage, and one additional source face becomes
bad only after STEP. Combining the routes would destroy causal attribution.

## Consequences

The assembler gains no keyword argument and the repair matrix gains no
SameParameter profile. The negative result is retained because it prevents a
future contributor from repeating a plausible but ineffective experiment.
The selector stays unchanged at 91/100 strict-valid.

This decision does not lower strict validity or geometry thresholds and does
not authorize full VQ training, sequence regeneration, boundary loss, or AR.
