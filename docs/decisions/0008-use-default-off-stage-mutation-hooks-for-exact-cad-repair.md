# ADR-0008: Use default-off stage mutation hooks for exact-CAD repair

## Status

Accepted for diagnostic feasibility experiments; not accepted as a production
selector profile.

## Date

2026-09-04

## Context

The signed lineage probe found two residual assembly defects at different
stages. CAD 47472 first becomes bad immediately after pcurve construction on
two faces, while CAD 63055 first becomes bad after sewing. The existing
`construct_brep_directed` API exposed observation callbacks at those stages,
but an observer cannot replace a face or sewn shape. Running only a face-local
helper would therefore not test the complete CAD, STEP roundtrip, strict
validity, or schema-v2 geometry/topology gate.

Changing the default assembly profile before an exact-CAD experiment would
confound the established 91/100 selector result. Hard-coding CAD or face IDs in
production repair code would also turn diagnostic evidence into a brittle
dataset exception.

## Decision

Add two keyword-only, default-`None` mutation boundaries to
`tools/directed_trim_assembly.py`:

- `post_pcurve_face_mutator`, invoked after pcurves are attached and before the
  optional face repair;
- `post_sewing_shape_mutator`, invoked after sewing and before solid creation.

A hook can replace its input only when it returns mapping diagnostics with
`accepted is True`. A rejected hook leaves the current shape unchanged. The
constructor independently rebuilds or retains exact source mapping needed by
later stages. Native work behind these hooks must run in an isolated one-CAD
process.

The post-sewing hook deliberately receives the sewn shell before MakeSolid.
When a candidate needs final-solid native or project-strict validation, it may
build a disposable single-solid wrapper solely for that validation. The shape
returned to the constructor remains the graph-preserved shell, and acceptance
still requires unchanged topology and incidence evidence on that shell.

Face-local diagnostics never authorize whole-CAD defect claims. A candidate
whose failure can change during STEP roundtrip must collect a complete
post-sewing source-face census and re-establish a unique source-to-STEP
geometry/incidence mapping. Whole-CAD target removal, absence of new defects,
and shared-edge preservation are true only when the final source-indexed STEP
diagnosis is complete and clean; ambiguous mapping, source-edge split or merge,
or any non-target occurrence fails closed.

The exact-CAD runner owns the diagnostic IDs and expected edge pairs. The two
low-level repair modules remain ID-agnostic and diagnosis-driven. No new
production repair switch or selector fallback is registered by this decision.

## Alternatives Considered

### Modify the assembly profile directly

Rejected because it would change the current control path before the candidate
has passed exact-CAD, residual-family, invalid-16, and fixed-100 gates.

### Treat a face-local helper result as experimental success

Rejected because a locally clean face can still fail sewing or STEP roundtrip,
or change source topology. The whole-CAD runner must independently measure the
saved STEP and apply the unchanged schema-v2 gate.

### Copy and fork the complete constructor in each experiment

Rejected because two large constructor copies would drift from the control and
make causal comparison harder. Small explicit stage boundaries preserve one
construction implementation and one historical default.

## Consequences

- Existing callers and profiles retain the historical behavior because both
  hooks default to `None` and sewing tolerance remains `1e-3`.
- The exact pilot can exercise two stage-specific repairs through the same
  constructor used by the control.
- Any accepted candidate still authorizes only a relevant residual-family
  expansion. Production adoption requires 95/100 strict-valid, 84/84 controls,
  zero regressions, zero worker/protocol failures, and schema-v2 acceptance.
- The extra hook surface must remain narrow. Future code must not add CAD IDs or
  expected diagnostic pairs to `construct_brep_directed` or either low-level
  repair module.
- A helper can report local mutation success while the whole-CAD adapter still
  rejects the candidate. This separation is intentional: local topology and
  curve preservation are necessary, while final STEP lineage and validity are
  the authoritative whole-CAD evidence.
