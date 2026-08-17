# ADR-0003: Hold production assembly hardening below the 95 of 100 gate

## Status

Accepted as a reproducible evidence result. Rejected for promotion into the
shared production `BrepARG` checkout.

## Date

2026-08-17

## Context

The upstream production assembler is an ignored junction-backed checkout, not
source tracked by this repository. Its historical implementation silently
remapped face IDs, appended same-face edges twice, divided by closed-edge
endpoint spans, used fragile face-local loop traversal, accepted unfinished
OCC builders, and converted sewing output to a solid without proving a
single-shell/single-solid result.

A clean isolated upstream worktree was patched with explicit face-ID
preservation, fail-closed adjacency construction, closed-edge bbox scaling,
guarded directed trim loops including one-edge closed loops, bounded curve-fit
fallback, copied-face self-intersection repair with geometry preservation
checks, and OCC builder/shell/solid checks. Every production attempt ran in a
separate worker process. The frozen original-control cohort was then rerun
with the exact 100-CAD manifest.

The first signed production run recovered three historical failures, preserved
all 84 historically strict-valid controls, and produced STEP for all 100 CADs.
Project-strict validity was 87 of 100, eight below the predeclared 95 of 100
promotion gate.

The separately proven `single_solid` near-vertex reconciliation was then
composed with the isolated production backend in the matrix runner, not
applied to the shared upstream source. It remaps only same-face, one-to-one
mutually nearest endpoint-id pairs within `2e-4`, and moves only endpoints in
accepted pairs to their shared representative before production construction.
The signed composition at commit `aeec941` wrote 100 of 100 STEP files,
recovered the disjoint `00000444_4ed4c78d6d754aac90876fc2_step_003`, retained
the three prior recoveries, preserved every original strict-valid control, and
reached 88 strict-valid, 90 native-valid, and 86 both-valid CADs. The
Git-safe report verifies that its CAD set and per-CAD parent/historical-strict
map equal the prior production 100-CAD report despite a reconstructed sorted
input order. This is the best production result so far, but remains seven
below the promotion gate. The remaining twelve strict failures retain
diagnosed wire self-intersections, free-edge, or native-solid defects.

## Decision

Keep the production patch and its tests as a reviewable overlay in this
repository, together with the Git-safe 100-CAD evidence snapshot. Do not apply
the patch to the shared `D:\luolin\V13\BrepARG` source and do not treat the
result as authorization to begin boundary-consistency, sequence regeneration,
or autoregressive work.

The repair runner keeps production evaluation opt-in through
`--assembly-backend production` and requires `--isolate-cad-workers`. This
allows future candidate fixes to be tested against the exact production
implementation without mutating the training checkout.

## Alternatives Considered

### Apply the patch because it has zero regressions

Rejected. Zero regression is necessary but not sufficient. The fixed release
gate is at least 95 strict-valid CADs out of 100, and the best result reaches
88.

### Lower the gate after observing 87 or 88 of 100

Rejected. The gate was set before these implementations and remains the
control against accepting repairs that only move a small subset of the failure
family.

### Retry broad ShapeFix or pcurve mutation globally

Rejected. The prior isolated pcurve-continuity matrix reached only 86 of 100
and added no recovery beyond guarded trim. Broad topology mutation also risks
changing geometry outside the diagnosed face.

## Consequences

The next assembly increment must target a remaining diagnosed failure family
and first show a new strict-valid recovery on the frozen invalid subset while
retaining all four existing recoveries. A new candidate still needs a signed
100-CAD matrix with at least 95 strict-valid CADs, all original 84 retained,
and zero regressions before the shared production source can change.
