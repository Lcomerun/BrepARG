# ADR-0003: Hold production assembly hardening at 87 of 100

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

The signed run recovered three historical failures, preserved all 84
historically strict-valid controls, and produced STEP for all 100 CADs.
Project-strict validity was 87 of 100, eight below the predeclared 95 of 100
promotion gate. The remaining thirteen strict failures retain diagnosed wire
self-intersections or invalid native BReps. Earlier bounded pcurve-continuity
and selective curve-rescue probes did not add recoveries beyond this result.

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
gate is at least 95 strict-valid CADs out of 100, and this result reaches 87.

### Lower the gate after observing 87 of 100

Rejected. The gate was set before this implementation and remains the control
against accepting repairs that only move a small subset of the failure family.

### Retry broad ShapeFix or pcurve mutation globally

Rejected. The prior isolated pcurve-continuity matrix reached only 86 of 100
and added no recovery beyond guarded trim. Broad topology mutation also risks
changing geometry outside the diagnosed face.

## Consequences

The next assembly increment must target a remaining diagnosed failure family
and first show a new strict-valid recovery on the frozen invalid subset while
retaining the three existing recoveries. A new candidate still needs a signed
100-CAD matrix with at least 95 strict-valid CADs, all original 84 retained,
and zero regressions before the shared production source can change.
