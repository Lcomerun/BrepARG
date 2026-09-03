# Close or promote the periodic pcurve repair route

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must stay current
while the work proceeds. This plan follows `PLANS.md` in the repository root.

## Purpose / Big Picture

The fixed 100-CAD assembly selector is strict-valid for 91 CADs and must reach
at least 95 while preserving all 84 historically valid controls. Representation
capacity is no longer the active uncertainty: VQ-8192 reaches 69/100 on the
unrepaired chain versus 70/100 for the continuous bypass. The remaining release
blocker is the assembly chain.

This experiment answers one narrow question before another repair is added:
do five near-success residual CADs contain a self-intersecting face whose
two-dimensional trim curves lie on different integer branches of a genuinely
periodic fitted surface? The observable outcome is a signed, path-free census
that either identifies exact faces and edges eligible for the existing
topology-preserving periodic branch helper or closes that repair route. The
census never writes a STEP candidate and cannot by itself authorize a full
100-CAD run.

## Progress

- [x] (2026-09-03 15:25 +08:00) Reconcile the capacity and assembly evidence:
  VQ-8192 is selected, the selector remains 91/100, and no training job needs
  continuation.
- [x] (2026-09-03 15:35 +08:00) Audit the periodic helper, its unit tests, the
  remote closure-shell evidence, 72 local run directories, and all nine
  residual CADs.
- [x] (2026-09-03 15:45 +08:00) Verify that all five source pickles exist,
  deserialize successfully, and bind to the frozen 100-CAD manifest; identify
  the common `directed_trim_curve_fit` construction profile.
- [x] (2026-09-03 16:45 +08:00) Add a default-off observation hook after the
  baseline wire fix and pcurve attachment, but before any optional repair
  strategy is applied to each constructed face.
- [x] (2026-09-03 16:59 +08:00) Implement, test, and commit an isolated
  five-CAD applicability census with strict 100-to-9-to-5 cohort selection,
  all-face coverage, any-versus-all applicability, exact loaded-byte bindings,
  strict sentinel parsing, and fail-closed contracts. Commit `ad6f385` passed
  90 focused tests before the first formal attempt.
- [x] (2026-09-03 17:12 +08:00) Commit the atomic-manifest compatibility fix so
  the rerun binds a clean source revision. Replaced the unsupported
  `Path.write_text(newline=...)` call with an explicitly opened, flushed, and
  fsynced file, add success, non-finite-value, and failed-replace regressions,
  and committed the correction as `0c1ce51`; 93 focused tests passed and the
  formal rerun started from a clean worktree.
- [x] (2026-09-03 17:15 +08:00) Run the five-CAD census from commit `0c1ce51`
  in the immutable `_v2` root. All five workers completed, all 134 source faces
  were observed, six strict-style bad faces were measured, and there were zero
  worker, protocol, binding, or measurement failures. Every bad face was a
  non-periodic `Geom_BSplineSurface`, so the preregistered decision is
  `CLOSE_PERIODIC_PCURVE_ROUTE`.
- [x] (2026-09-03 17:30 +08:00) Add a fail-closed Git-safe snapshot tool and
  five archive contract tests; archive the byte-identical cases, summary, and
  completed run manifest plus README, validation, and hashes. The combined
  focused suite passes 98 tests.
- [x] (2026-09-03 17:36 +08:00) Push the implementation, compatibility fix,
  living plan, ADR, Git-safe result, and cross-device Chinese status page to
  `experiment/protocol-v5-scaling-ladder`. Result commit `7aadcb3` and its two
  prerequisite commits were accepted by GitHub.

## Surprises & Discoveries

- Observation: The periodic helper is implemented and unit-tested but is not a
  registered production repair profile.
  Evidence: `tools/local_wire_topology_repair.py` exports
  `repair_face_periodic_pcurve_branches`, while
  `tools/assembly_repair.py::REPAIR_SWITCHES` has no periodic branch switch.

- Observation: The only previous construction-stage periodic probe covered
  `00095733`; its two bad faces were non-periodic.
  Evidence: remote branch `protocol-v5-closure-shell-probe`, report
  `reports/closure_shell_failure_negative_20260818/`.

- Observation: A read-only scan of existing selector STEP outputs found only
  `Geom_BSplineSurface` bad faces with `IsUPeriodic=False` and
  `IsVPeriodic=False`, but STEP roundtrip is not authoritative for the face
  immediately after pcurve construction.
  Evidence: face indices and topology may change during ShapeFix, sewing, and
  STEP serialization, so the formal census must observe the construction hook.

- Observation: `periodic_pcurve_continuity_state` aggregates a non-periodic
  wire's `before_max_gap=None` as a top-level zero.
  Evidence: `tools/local_wire_topology_repair.py` uses `value or 0.0` for the
  aggregate. Therefore `max_gap == 0` alone must never be interpreted as a
  closed periodic wire.

- Observation: The first implementation could label a CAD complete after only
  one observed face if a later face failed construction.
  Evidence: the focused contract now requires observed face indices to equal
  the complete contiguous range derived from `len(faceEdge_adj)`; partial
  coverage becomes `measurement_incomplete` and can never close the route.

- Observation: The calibration manifest alone did not prove that the five
  registered targets remained among the current selector failures.
  Evidence: the census now binds the completed selector run and matrix, proves
  100 calibration originals with 84 historical controls, 91 selector-valid
  CADs with zero historical regressions, exactly nine residuals, and the
  ordered five targets as a subset of those residuals.

- Observation: The first formal attempt could not create its run manifest on
  the installed Python version because `Path.write_text()` does not accept the
  `newline` keyword there.
  Evidence: commit `ad6f385` raised `TypeError: write_text() got an unexpected
  keyword argument 'newline'` inside `atomic_json` before any CAD worker was
  launched. The original output root contains only its one-byte writer lock;
  there are no case rows and therefore no scientific result.

- Observation: Construction-stage evidence agrees with the earlier indirect
  STEP observations but is now authoritative for the registered cohort.
  Evidence: the corrected run observed 134/134 faces across all five CADs. It
  found six bad faces in three CADs; all six were `Geom_BSplineSurface` with
  both OCC U-periodic and V-periodic flags false. The other two CADs had no
  strict-style bad face at the observation phase.

## Decision Log

- Decision: Use the same bounded sanitized fallback profile,
  `directed_trim_curve_fit`, for all five CADs.
  Rationale: It reaches face and STEP construction for every target, including
  `00051602`, whose historical curve fitter stops at edge 4. A common profile
  removes a construction-policy confound.
  Date/Author: 2026-09-03 / Codex.

- Decision: Observe the face after baseline `fix_wires(face)` and
  `add_pcurves_to_edges(face)`, but before optional local or global face repair.
  Rationale: earlier faces may not have pcurves, while later faces may already
  have topology or geometry changed by a strategy-specific repair. This is
  strict-style evidence, not an untouched raw-face observation.
  Date/Author: 2026-09-03 / Codex.

- Decision: Keep Open CASCADE entirely inside one-CAD child processes.
  Rationale: malformed topology can terminate a native process. A timeout,
  crash, missing sentinel, malformed result, or source binding change must
  remain an explicit failed row rather than disappearing from the denominator.
  Date/Author: 2026-09-03 / Codex.

- Decision: Do not call the mutating periodic repair during the census.
  Rationale: the first question is whether a valid periodic branch gap exists.
  Candidate mutation and schema-v2 validation are authorized only if the
  read-only result identifies an exact face on which every diagnosed bad wire
  is repairable. A `partial_only` face grants no mutation.
  Date/Author: 2026-09-03 / Codex.

- Decision: Distinguish an `any` periodic-gap candidate from a fully applicable
  face profile.
  Rationale: one repairable bad wire cannot make a face eligible when another
  diagnosed bad wire remains unresolved. The report retains partial evidence,
  but only all-target closure may authorize a mutation pilot.
  Date/Author: 2026-09-03 / Codex.

- Decision: Preserve the failed first output root and run the corrected code in
  `periodic_pcurve_applicability_census_20260903_v2`.
  Rationale: the run-root immutability rule prevents an infrastructure failure
  from being overwritten or confused with the later scientific result. The
  original attempt never entered a worker, so it cannot close the periodic
  route or alter the 91/100 selector result.
  Date/Author: 2026-09-03 / Codex.

- Decision: Close periodic pcurve branch translation for the frozen five-CAD
  cohort and do not implement a mutation pilot.
  Rationale: all five isolated workers completed with full face coverage and
  zero evidence-integrity failures, yet none of the six diagnosed bad faces is
  periodic in either parameter direction. Inventing a period would violate the
  geometry-preservation contract, and the preregistered conclusive-negative
  rule therefore applies.
  Date/Author: 2026-09-03 / Codex.

## Outcomes & Retrospective

The census completed conclusively from clean commit `0c1ce51`. All five CADs
completed, all 134 source faces were observed, and the run had zero worker,
protocol, source-binding, or measurement failures. Six bad faces were localized
across three CADs, but every one was a non-periodic fitted B-spline in both U
and V. There were zero periodic bad faces and zero repairable faces, so the
registered outcome is `CLOSE_PERIODIC_PCURVE_ROUTE` for this five-CAD cohort.

The first attempt remains preserved as a pre-worker infrastructure failure and
does not enter the scientific result. The signed `_v2` run and path-free report
bind the cases, summary, code revision, upstream runtime hash, and source pickle
hashes without archiving source bytes. The result prevents an unsupported
periodic mutation and redirects assembly work toward the measured planar trim
intersection and shell/connectivity families. It does not improve or release
the current selector: the assembly gate remains 91/100 strict-valid versus the
required 95/100, and downstream boundary, full-training, sequence, and AR work
remain blocked.

## Context and Orientation

`tools/directed_trim_assembly.py::construct_brep_directed` constructs surfaces,
edge curves, wires, faces, a sewn shell, and a solid from reconstructed grids.
After each raw face is built it calls runtime-only upstream helpers
`fix_wires(face)` and `add_pcurves_to_edges(face)`, then applies the selected
repair strategy. A pcurve is a two-dimensional curve in the parameter domain
of a surface; Open CASCADE needs it to trim a three-dimensional surface into a
face. On a truly periodic surface, such as a cylinder, two equivalent pcurves
may differ by an integer surface period. The existing helper can translate only
that two-dimensional branch while proving that the three-dimensional edge
curve and topology remain unchanged.

`tools/local_wire_topology_repair.py` contains four relevant functions.
`wire_self_intersection_state` identifies bad wire indices using the project
strict semantics. `periodic_pcurve_continuity_state` extracts surface periods,
ordered pcurve endpoints, seam status, and the integer-branch optimization
plan. `select_periodic_pcurve_branches` is the pure optimizer.
`repair_face_periodic_pcurve_branches` is the mutating candidate path and is
not invoked in this census.

The frozen manifest is local-only at
`D:/luolin/V13/local_runs/assembly_calibration_100cad_v1_20260809/calibration_manifest.jsonl`.
It contains exactly 100 original rows, 84 historically strict-valid. Its
SHA-256 is
`426809e39cf2f4ee13c2e86b542c76a5d1c80ce6abfa4ac4a2e84135f580f4ef`.
The five target CADs are `00047472...`, `00063055...`, `00032101...`,
`00076198...`, and `00051602...`. The census verifies the complete CAD IDs and
source pickle byte hashes from the manifest instead of embedding machine paths
in archived evidence.

The selector has four other residual CADs outside this registered near-success
census. `00067160...` has no current self-intersecting wire and is blocked by
`solid_count=0`. `00095733...` already has a construction-stage probe whose
two target faces were OCC-nonperiodic. `00061931...` and `00087341...` began as
curve-fit failures and later exposed multiple free-edge, shell, and
self-intersection failures; `00051602...` is the preregistered representative
of that curve-fit-to-boundary family. Their STEP-roundtrip evidence is negative
but is not treated as authoritative construction evidence. Consequently, a
negative census closes this route only for the registered five-CAD cohort; it
does not prove every possible residual face is nonperiodic and does not finish
the assembly project.

## Milestones

Milestone 1 freezes the observer contract. `construct_brep_directed` gains a
default-off hook after baseline `fix_wires` and pcurve attachment and before
optional repair strategies. Running
`python -m pytest -q tests/test_directed_trim_assembly.py` must prove signature,
guard, metadata, and phase stability while existing callers remain compatible.

Milestone 2 completes the signed isolated runner. It proves the frozen
100-CAD/84-control calibration, current 91-valid/nine-residual selector,
ordered five targets, exact loaded pickle bytes, all-face coverage, strict
sentinel protocol, and any-versus-all applicability. Run the four focused test
modules below, commit implementation, tests, this plan, and ADR-0005, then
require `git status --porcelain` to be empty; formal execution rejects a dirty
worktree.

Milestone 3 runs five isolated OCC workers from that clean commit. A valid run
contains five complete rows. Any crash, timeout, malformed sentinel, hash
drift, incomplete face set, or OCC measurement error yields
`INCONCLUSIVE_REQUIRES_RERUN`, never a false negative.

Milestone 4 archives path-free JSON/JSONL, README, hashes, and the retrospective
without STEP, pickle, checkpoint, array, or local absolute paths. A second
commit and push publishes that evidence. A positive result permits only the
exact targeted mutation pilot; a negative result redirects investigation while
the 91/100 release score remains unchanged.

## Plan of Work

First, extend `construct_brep_directed` with a callback whose default is
`None`. Compute the oriented three-dimensional endpoint gap for every face
loop only when the callback is present. Invoke it after baseline `fix_wires`
and upstream pcurve attachment but before any optional repair strategy.
Document that the callback is observation-only; its return value is ignored.
No default profile or repair switch changes.

Second, add `tools/probe_periodic_pcurve_applicability.py`. Its parent mode
loads only JSON and source byte bindings, creates an immutable run signature,
and launches one child per CAD. Its worker mode verifies the exact pickle bytes
before and after deserialization, performs 200 CPU joint-optimization
iterations, and rebuilds the CAD with `directed_trim_curve_fit`. The callback
copies each face, diagnoses strict bad wires, records exact crossing categories,
surface type and periodicity, seam immovability, branch offsets, before/after
UV gaps, topology incidence, and three-dimensional endpoint gaps. It records
only scalar and discrete evidence, never raw geometry.

The parent also requires a clean Git worktree before it creates a formal run
manifest. This makes the recorded commit and source hashes a reproducible code
identity rather than a commit plus an unarchived local patch.

Third, add focused tests for the pure applicability decision, summary gate,
worker sentinel parser, source/run binding validation, and callback phase. The
test for the callback substitutes harmless runtime helper functions and raises
after observation, proving the callback occurs after pcurve addition without
requiring the remainder of a solid to be accepted.

Fourth, commit the code, tests, plan, and ADR. Run the formal census from that
clean commit into a new directory under `D:/luolin/V13/local_runs/`. If all five
CADs finish and no face has every diagnosed bad wire covered by a genuine
periodic branch plan that changes at least one non-seam edge per wire, starts
above `1e-7`, and ends at or below `1e-7`, close the route for this registered
cohort. If at least one face satisfies that all-bad-wires condition, add a
separate repair profile in a later commit and run only those CADs in isolated
workers with the unchanged schema-v2 gate. A partially repairable face does not
promote.

Finally, snapshot path-free JSON/JSONL, a concise README, hashes, and an artifact
manifest beneath `reports/periodic_pcurve_applicability_census_20260903/`.
Update this plan's retrospective; ADR-0005 is already part of the clean source
commit. Do not copy STEP, pickle, NumPy, checkpoint, or upstream `BrepARG/`
bytes into Git.

## Concrete Steps

Work from `D:/luolin/BrepARG2` with the `brepgen_env` interpreter.

Run focused tests while developing:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe -m pytest -q tests/test_probe_periodic_pcurve_applicability.py tests/test_directed_trim_assembly.py tests/test_local_wire_topology_repair.py

After committing the implementation, run the signed census:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe tools/probe_periodic_pcurve_applicability.py --calibration-manifest D:/luolin/V13/local_runs/assembly_calibration_100cad_v1_20260809/calibration_manifest.jsonl --selector-matrix D:/luolin/V13/local_runs/assembly_selector_main_100cad_20260818/assembly_repair_matrix.jsonl --selector-run D:/luolin/V13/local_runs/assembly_selector_main_100cad_20260818/assembly_repair_run.json --breparg-root D:/luolin/V13/BrepARG --output-dir D:/luolin/V13/local_runs/periodic_pcurve_applicability_census_20260903_v2 --joint-iterations 200 --worker-timeout-seconds 600

Expect exactly five result rows. A negative, conclusive summary must report
zero worker/protocol failures, zero repairable bad faces, and decision
`CLOSE_PERIODIC_PCURVE_ROUTE`. A positive summary must name at least one exact
CAD/face and prove every diagnosed bad wire on that face is repairable before
reporting `PROMOTE_TARGETED_REPAIR_PROBE`; it still must not claim the 95/100
assembly gate has passed.

After an inconclusive result that requires a code, input, or runtime correction,
retain the original output and rerun from a clean corrective commit into a new
immutable root such as
`periodic_pcurve_applicability_census_20260903_v2`. Change only
`--output-dir`; never delete or overwrite the earlier failure evidence.

Before each commit and push, run:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe -m pytest -q tests/test_probe_periodic_pcurve_applicability.py tests/test_directed_trim_assembly.py tests/test_local_wire_topology_repair.py tests/test_run_assembly_repair_matrix.py
    git diff --check
    git status --short --branch

## Validation and Acceptance

The observation interface is accepted when its focused test proves that it is
called after pcurves are added, receives the construction face index and loop
endpoint metadata, and leaves the default `None` path unchanged at the API
boundary.

The census is accepted when the run manifest binds the ordered five CAD IDs,
the frozen manifest hash, each source pickle size/SHA-256, the common profile,
joint iteration count, worker timeout, Git commit/dirty state, repository source
hashes, and upstream `utils.py` hash. Each child row must bind the same run
signature and exact source bytes. Five unique complete rows and zero native,
timeout, sentinel, or binding failures are required for a conclusive decision.

A bad wire is repairable only if the fitted surface reports an actual positive
U or V period, pcurve collection succeeds, the diagnosed wire index exists,
at least one non-seam edge receives a nonzero integer offset, the pre-repair
maximum UV gap exceeds `1e-7`, and the optimized gap is at most `1e-7`.
Missing pcurves, OCC exceptions, invalid indices, seam-only changes, and
non-periodic surfaces are explicit fail-closed negatives.

The overall project remains incomplete after a negative census. A positive
census authorizes only a targeted repair pilot. A full-cohort promotion still
requires at least 95/100 strict-valid, all historical controls at 84/84, zero
regressions, zero worker/protocol failures, and accepted schema-v2 gates for
every fallback.

## Idempotence and Recovery

The parent creates or validates an immutable run signature. Repeating the same
command resumes only missing CADs; changing CAD order, source bytes, code,
runtime hash, profile, iteration count, or timeout requires a new output root.
Rows are appended only after sentinel and binding validation. A native crash or
timeout creates a permanent explicit failure row for that run and makes the
decision inconclusive; it is never silently retried under the same signature.

No command deletes source data. All local source pickle and possible runtime
logs remain under explicitly named local directories. The Git snapshot is
regenerated from the signed local run and may be safely replaced only before it
is committed.

## Artifacts and Notes

Starting selector evidence:

    STEP-readable: 97/100
    native-valid: 90/100
    strict-valid: 91/100
    both-valid: 88/100
    historical controls: 84/84
    regressions: 0
    release target: at least 95/100 strict-valid

Starting capacity decision:

    bypass@60k strict: 70/100
    VQ-8192@60k strict: 69/100
    RVQ-2x4096@60k strict: 65/100
    selected representation: VQ-8192/64D

## Interfaces and Dependencies

In `tools/directed_trim_assembly.py`, extend
`construct_brep_directed(..., post_pcurve_face_observer=None)` so an observer is
called as:

    observer(face_index, face, metadata)

`metadata` contains `loop_count`, `outer_loop_index`,
`loop_3d_endpoint_max_gaps`, and `face_3d_endpoint_max_gap`. The observer return
value is ignored and exceptions fail the current isolated worker.

In `tools/probe_periodic_pcurve_applicability.py`, provide pure functions for
applicability classification, worker sentinel parsing, row validation, and
summary generation. Parent mode must not import Open CASCADE. Worker-only
functions may import Open CASCADE, NumPy, the joint optimizer, and the assembly
modules lazily.

The runtime dependencies are Python, NumPy, PyTorch for the unchanged CPU joint
optimizer, pythonocc-core, and `D:/luolin/V13/BrepARG/utils.py` imported at
runtime only. No upstream source file is modified or committed.

Revision note (2026-09-03): Created this plan after the capacity experiment was
closed and a requirement-by-requirement assembly audit identified the periodic
pcurve applicability census as the remaining lowest-cost diagnostic. The plan
formalizes the construction phase, fail-closed decision, and Git-safe evidence
needed to avoid another indirect STEP-only conclusion.

Revision note (2026-09-03 16:45 +08:00): Updated the living plan after contract
review exposed two false-negative risks: partial face observation and an
unbound selector residual set. The implementation and command now bind the
completed selector and require exact all-face coverage before a conclusive
decision.

Revision note (2026-09-03 16:50 +08:00): Reconciled promotion with the
all-bad-wires implementation, documented baseline `fix_wires` semantics, added
independently verifiable milestones and five-of-nine scope, and defined
immutable recovery directories.

Revision note (2026-09-03 17:10 +08:00): Recorded the first formal attempt's
pre-worker `Path.write_text(newline=...)` compatibility failure, the absence of
case evidence, the atomic-write regression fix, and the mandatory `_v2` rerun
root so infrastructure failure cannot be misreported as a scientific negative.

Revision note (2026-09-03 17:30 +08:00): Completed the signed `_v2` census and
recorded its conclusive negative result: 5/5 CADs, 134/134 faces, six bad faces,
zero periodic or repairable bad faces, and zero worker/protocol failures. Added
the Git-safe snapshot outcome and closed only the registered periodic-pcurve
route while retaining the overall 91/100 assembly gate.

Revision note (2026-09-03 17:36 +08:00): Recorded successful publication of
commits `ad6f385`, `0c1ce51`, and result commit `7aadcb3` to the shared
experiment branch. This closes the census ExecPlan; the separate assembly
release objective remains open at 91/100 strict-valid.
