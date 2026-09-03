# ADR-0005: Census periodic pcurve applicability at the construction hook

## Status

Accepted for the periodic-pcurve diagnostic. Mutation remains conditional on a
conclusive positive census.

## Date

2026-09-03

## Context

The current no-regression assembly selector reaches 91 of 100 project-strict
valid CADs, below the fixed release gate of 95. Several residual failures contain
self-intersecting two-dimensional face boundaries, so translating equivalent
pcurve branches by integer surface periods is a possible local repair. The
existing helper can search bounded U/V-period offsets while preserving the
three-dimensional edge curves, but a periodic branch repair is meaningful only
when the face's actual Open CASCADE surface reports a finite, positive period.

Previously inspected STEP-roundtripped residual faces are fitted
`Geom_BSplineSurface` instances with `IsUPeriodic=False` and
`IsVPeriodic=False`. That is useful negative evidence, but STEP serialization may
not preserve the exact construction-stage representation. Conversely, observing
a face before its pcurves are attached cannot answer whether its oriented
two-dimensional branches have a repairable gap. Observing it after a local or
global face repair would confound applicability with mutation.

The diagnostic therefore needs an authoritative construction-stage observation
point, an explicit separation between measurement and repair, and a decision
rule that cannot silently turn missing or malformed evidence into permission to
mutate geometry or topology.

## Decision

Observe each constructed face after the baseline `fix_wires(face)` step and
immediately after `add_pcurves_to_edges(face)` succeeds, but before any optional
strategy-specific local or global face-repair operation. This
post-baseline-fix-wires/post-add-pcurves/pre-optional-repair hook is the
authoritative phase for the periodic-pcurve applicability census. The default
hook is inert so normal assembly behavior is unchanged.

Run the first census read-only. It may inspect surface type and periodicity,
strict bad-wire indices, seam positions, pcurve availability and endpoints,
bounded integer-period offset plans, and predicted before/after UV gaps. It must
not call the mutating periodic repair helper, write a candidate STEP file, or
replace a face. A conclusive positive result authorizes only a separately gated,
targeted repair pilot for the exact identified CAD, face, and wire.

Treat surface periodicity as OCC-owned evidence. A fitted B-spline for which OCC
reports both U and V as non-periodic remains non-periodic; the census and repair
code must not infer, synthesize, or assign a period from bounds, endpoint
proximity, apparent closure, or another representation of the source geometry.
Non-periodic surfaces are explicit negative observations, not missing data.

The census has exactly three terminal decisions:

- `PROMOTE_TARGETED_REPAIR_PROBE`: every required case completed without a
  worker, protocol, source-binding, or measurement failure, and at least one
  face has a genuine OCC-reported period and every strict-bad wire diagnosed on
  that face has a bounded non-seam integer-period plan that changes an offset
  and reduces its UV closure gap from greater than `1e-7` to at most `1e-7`.
  A face on which only some bad wires close is recorded as `partial_only` and
  does not authorize a repair probe.
- `CLOSE_PERIODIC_PCURVE_ROUTE`: every required case completed conclusively and
  no repairable periodic branch gap exists. This closes this repair family for
  the frozen cohort; it does not mean the overall assembly task is complete.
- `INCONCLUSIVE_REQUIRES_RERUN`: any required row is absent or incomplete, or a
  worker timeout/crash, malformed sentinel, source/hash mismatch, missing
  authoritative observation, OCC exception, or protocol failure prevents a
  complete determination. This state grants no repair permission.

All applicability checks fail closed. Invalid or non-finite periods, missing or
malformed pcurves, invalid wire indices, seam-only proposed changes, offsets
outside the registered search range, unavailable strict diagnosis, and
unresolved UV gaps are non-candidates. Infrastructure and evidence-integrity
failures make the overall decision inconclusive rather than negative. Every
attempt remains in the denominator, and Open CASCADE work runs in isolated
one-CAD workers.

## Alternatives Considered

### Infer a period for apparently closed fitted B-spline surfaces

Rejected. Equivalent-looking geometry does not establish a valid parametric
period. Inventing one can move trim curves to a branch that the fitted surface
does not support and would violate the geometry-preservation contract.

### Inspect only STEP-roundtripped faces

Rejected as the authoritative census. STEP evidence is a useful consistency
check, but serialization can alter surface representation, pcurve attachment,
or trimming semantics. The construction hook observes the exact face on which a
repair would operate.

### Invoke repair while collecting applicability data

Rejected. Mutation would make it impossible to distinguish whether a periodic
branch gap existed before repair and could accidentally promote a topology or
geometry change from an exploratory diagnostic.

### Treat worker or measurement failures as no candidate

Rejected. That would convert missing evidence into a false negative and could
close a viable repair route. Such failures produce
`INCONCLUSIVE_REQUIRES_RERUN` and retain their attempts in the signed record.

## Consequences

The directed-trim construction API gains an optional observation boundary, but
its normal path remains behaviorally unchanged when no observer is supplied.
The census produces signed, Git-safe evidence tied to the frozen cohort, source
bytes, implementation hashes, and isolated worker results.

A positive census does not itself change the selector result of 91/100. The
identified candidate must still pass a targeted mutation pilot, STEP-readable,
native-valid, project-strict, both-valid, topology/incidence, sampled 3-D curve,
and geometry-preservation gates. Only a new safe recovery may justify an
invalid-subset test and then a complete 100-CAD selector rerun. Final promotion
still requires at least 95/100 strict-valid, preservation of all historical
84/84 controls, zero regressions, and zero worker or protocol failures.

A conclusive negative result closes the periodic-pcurve family for the frozen
targets and redirects work to another diagnosed assembly failure mechanism. An
inconclusive result requires a new signed rerun after the evidence or runtime
failure is corrected. Boundary-consistency training, full-scale representation
training, sequence regeneration, and autoregressive work remain blocked by the
independent assembly release gate.
