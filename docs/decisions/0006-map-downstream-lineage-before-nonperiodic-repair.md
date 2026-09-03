# ADR-0006: Map downstream topology lineage before non-periodic repair

## Status

Accepted

## Date

2026-09-03

## Context

The fixed 100-CAD assembly selector is strict-valid for 91 CADs and must reach
at least 95 without regressing any of the 84 historically valid controls. The
capacity comparison is already closed in favor of VQ-8192/64D; no representation
training is currently required. Nine CADs remain strict-invalid after all five
selector profiles.

Two residuals, `00047472...` and `00063055...`, are the smallest plausible
non-periodic boundary-repair family. Both already form one shell and one solid.
The first is OCC native-valid but has three strict wire self-intersections; the
second has one strict self-intersection. Earlier construction observations did
not see their final bad wires, while STEP-roundtrip diagnostics did. Those
observations used different profiles and their face indices are not proven to
refer to the same source faces.

Open CASCADE preserves usable identity and modification history during face
construction and sewing, but STEP serialization destroys `IsSame` and
`IsPartner` identity. Existing diagnostics report one-based wire edge positions
but do not map them to source `faceEdge_adj` IDs. Implementing another repair
from matching integer indices would risk acting on the wrong topology and
would repeat the broad ShapeFix experiments that produced invalid apparent
recoveries by deleting or merging source entities.

## Decision

Add a separate, default-off multi-stage observation API to the local directed
assembler. Preserve the existing periodic-census hook unchanged. The new API
will expose explicit events after pcurve construction, after optional face
repair, after sewing, and after solid creation. A signed two-CAD runner will
write and reimport STEP to add the final roundtrip stage.

Every bad wire occurrence must map to a source face and source edge by proof:

- At construction, use unique `IsSame` correspondence to fitted source edges.
- Across optional face repair, prefer identity and otherwise require a unique
  sampled 3D-curve assignment constrained to the source face.
- Across sewing, use `BRepBuilderAPI_Sewing.ModifiedSubShape` and reject a
  split, disappearance, or merge involving distinct source edge IDs.
- Across STEP roundtrip, use source-face incidence plus orientation-invariant
  curve fingerprints and accept only a unique perfect assignment. Explorer
  order is never evidence of identity.

The implementation fixes the STEP curve comparison at normalized tolerance
`1e-4`. It samples open curves in both directions and closed curves in both
directions across cyclic phase shifts. Face compatibility uses the incident
3D boundary multiset, surface type and periodicity, and wire pattern; trimmed
area and centroid are not hard correspondence gates because the defective
pcurves themselves can change those derived quantities. The matcher requires
a unique global face assignment, a unique edge-occurrence assignment within
every face, and consistent shared-edge incidence across faces. Zero matches,
multiple matches, split source edges, merged distinct source edges, or any
non-finite measurement remain inconclusive.

During construction, ShapeFix may copy and reorder topology, so identity is a
preferred proof rather than an assumed invariant. The observer can instead
record `exact_face_local_geometry` after a unique face-local assignment. At
sewing, `exact_sewing_history` is reserved for the case where
`ModifiedSubShape` and `Modified` both return one agreeing result. If those
history APIs are incomplete but an independent unique boundary and face-local
edge assignment succeeds, the distinct status
`exact_sewing_face_local_geometry` is used and failed history attempts are
retained only as diagnostic notes.

The first cohort is exactly `00047472...` and `00063055...`, reconstructed with
200 joint-optimization iterations and the selector's current primary profile,
`directed_trim_local_intersection_topology`. All native work remains in
one-CAD child processes. Ambiguity, missing evidence, a changed input binding,
or incomplete stage coverage is inconclusive and cannot authorize mutation.

No repair switch will be registered until the read-only report identifies an
exact first-defective stage and a topology/3D-curve-preserving local action.
Any later candidate remains copy-only, default-off, and subject to the complete
schema-v2 gate and the unchanged 95/100, 84/84, zero-regression release gate.

## Alternatives Considered

### Register the existing periodic pcurve branch helper

Rejected. The formal construction-stage census observed six bad faces and all
were non-periodic B-splines in both parameter directions. Inventing a period
would violate the geometry-preservation contract.

### Apply `FixGap2d` directly to the previously reported STEP face indices

Rejected. The STEP indices are not source identities, and the two targets were
observed under a different earlier construction profile. The apparent match
could be an ordinal coincidence after ShapeFix, sewing, or STEP transfer.

### Add another broad ShapeFix profile

Rejected. Broad pcurve, intersection, graph-trim, and post-sewing reprojection
profiles have produced no net-new schema-v2-accepted CAD. Some OCC-valid
candidates silently changed edge or vertex incidence.

### Match STEP entities by nearest curve only

Rejected. Repeated and symmetric edges can have equal or nearly equal geometry.
A nearest neighbor can force a plausible but non-unique assignment. The probe
must require a unique perfect matching or fail closed.

### Resume capacity or AR training first

Rejected. VQ-8192/64D is already within one percentage point of the continuous
bypass on the same unrepaired 60k chain. The active blocker is the 91/100
assembly gate; downstream training would preserve that known failure mode.

## Consequences

The local assembler gains a diagnostic API, but its normal path and every
existing repair profile remain unchanged when the callback is absent. The
periodic census keeps its frozen one-phase callback contract.

The lineage runner is more conservative than ordinal or nearest-neighbor
matching. Some cases may become inconclusive, particularly after STEP transfer
on repeated geometry. That is intentional: an inconclusive result triggers a
stronger observer or a closed route, not an unsafe repair.

The resulting stage map will distinguish whether the defect originates in
pcurve construction, optional face repair, sewing, or STEP roundtrip. This
turns a future local repair into an attributable experiment and records the
repair-to-recovered-case evidence needed for the project report and eventual
paper motivation.

This decision does not change the current selector score. Until a later full
100-CAD run proves at least 95 strict-valid CADs, 84/84 historical controls,
zero regressions, and zero worker/protocol failures, full VQ training, sequence
regeneration, and AR remain blocked.
