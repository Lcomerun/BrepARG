# Trace downstream bad wires to their source topology before another repair

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current
while the work proceeds. This plan follows `PLANS.md` in the repository root.

## Purpose / Big Picture

The fixed 100-CAD failure-triggered selector is strict-valid for 91 CADs. It
must reach at least 95 while preserving all 84 historically valid controls and
passing the unchanged schema-v2 geometry/topology gate. The representation
decision is already closed: VQ-8192/64D reaches 69/100 on the unrepaired chain,
only one percentage point below the continuous bypass, so no old VQ, RVQ, or
FSQ training should be resumed. The remaining release blocker is CAD assembly.

This plan makes the next assembly experiment attributable. For the two
near-success residuals `00047472...` and `00063055...`, a user can see whether
each strict wire defect already exists immediately after pcurve construction,
is introduced or exposed by the optional face repair, appears during sewing,
or appears only after STEP serialization and reimport. Every reported bad wire
must be linked back to an exact source face and source edge occurrence, or be
explicitly marked ambiguous. No repair is allowed merely because two explorer
indices happen to be equal.

The first observable outcome is a signed, Git-safe two-CAD stage-lineage report
with no worker, input-binding, coverage, or mapping failures. Only if that
report identifies a local, graph-preserving intervention point will a separate,
default-off repair switch be implemented. The experiment does not lower strict
validity, does not weaken schema-v2, and cannot by itself release the 91/100
selector.

## Progress

- [x] (2026-09-03 18:05 +08:00) Revalidated the repository, remote, runtime,
  process, and GPU state. The branch is clean and synchronized at `c7ab18d`;
  no training or OCC experiment is running, and the RTX 3060 is idle in P8.
- [x] (2026-09-03 18:10 +08:00) Audited all nine current selector residuals,
  all five candidate profiles per residual (`45/45` rows), the P0-A tolerance
  and joint-optimization matrix, and the later graph, closure/shell, endpoint,
  local-pcurve, and periodic probes.
- [x] (2026-09-03 18:12 +08:00) Selected `00047472...` and `00063055...` as
  the first lineage cohort. Both have one shell and one solid; the former is
  native-valid with three strict self-intersections, while the latter has one
  strict self-intersection. Neither has a proven source-topology mismatch.
- [x] (2026-09-03 18:18 +08:00) Verified both frozen source pickle bindings,
  the selector profile, the 100-CAD manifest binding, and the upstream runtime
  hash. Confirmed that no existing helper maps a STEP wire occurrence back to
  a source edge.
- [x] (2026-09-03 18:22 +08:00) Audited OCCT lineage capabilities. `IsSame`
  can bind face-build occurrences to the fitted source edges;
  `BRepBuilderAPI_Sewing.ModifiedSubShape` can retain sewing provenance; STEP
  reimport deliberately destroys object identity and therefore requires a
  unique geometry-plus-incidence assignment.
- [x] (2026-09-03 20:47 +08:00) Added and tested a default-off multi-stage
  observer to `tools/directed_trim_assembly.py` without changing any current
  repair route. Face and edge lineage is accepted only through identity,
  mutually agreeing sewing history, or a unique geometry assignment.
- [x] (2026-09-03 20:47 +08:00) Implemented a signed, isolated two-CAD
  lineage runner with complete-stage, source-byte, source-revision, finite,
  unique geometry-plus-incidence, split/merge, and JSON-safety gates. The full
  focused suite passes 205 tests; the two `pythonocc` warnings are deprecation
  notices only.
- [x] (2026-09-03 21:03 +08:00) Ran the exact two-CAD pilot from clean commit
  `9bfda7b2634e4477cac540a2167f26fc0de63fdd` in the immutable local output
  directory `downstream_bad_wire_lineage_47472_63055_20260903_v2`. Both cases
  completed conclusively and every coverage, observation, mapping,
  source-binding, worker, and protocol failure count is zero.
- [x] (2026-09-03 21:17 +08:00) Generated and independently validated the
  six-file Git-safe snapshot under
  `reports/downstream_bad_wire_lineage_47472_63055_20260903/`. The archive is
  path-free and contains no STEP, pickle, checkpoint, native handle, worker
  log, or upstream-source payload.
- [ ] If and only if the lineage is conclusive, implement the smallest
  copy-only repair candidate and test it on exact CAD/face/wire targets before
  expanding to the nine residuals.
- [ ] Commit and push the validated snapshot, snapshot tool, tests, and updated
  living documentation to the current GitHub branch.

## Surprises & Discoveries

- Observation: The prior construction-stage periodic census and the selector
  do not use the same repair profile.
  Evidence: the census uses `directed_trim_curve_fit`, whereas both selected
  target rows use `directed_trim_local_intersection_topology`. Absence of a bad
  face in the earlier census cannot prove that the selector profile is clean at
  the same stage.

- Observation: STEP face and edge ordinals are not source identities.
  Evidence: `00047472...` is reported bad at STEP faces 3, 12, and 43 and
  `00063055...` at STEP face 3, but ShapeFix, sewing, and STEP transfer may
  reorder or replace topology. The current diagnostics contain no entity
  correspondence proof.

- Observation: identity is useful before STEP but is intentionally lost by a
  STEP roundtrip.
  Evidence: an OCCT 7.7.2 box probe maps construction edges by `IsSame` and
  sewing subshapes by `ModifiedSubShape`; none of the twelve pre-STEP edges is
  `IsSame` or `IsPartner` to its reimported counterpart.

- Observation: the two target inputs are still byte-identical to the signed
  selector inputs.
  Evidence: their SHA-256 values are
  `68343d8203dc640d75cd3f4b2a7fea119e637b4f23f3c3ca26fc5c739196eab1`
  and
  `39f56ba42e0f25044d2ef9e3b2f6d14e5d092cf0be887bb221b68103f037cf96`.

- Observation: ShapeFix may reorder a wire even when all source occurrences
  remain geometrically identifiable.
  Evidence: the real OCC regression changed the four edge occurrences from
  source order to `[3, 0, 1, 2]`. Mapping diagnostic `edge_positions` back by
  list position would therefore be wrong; the implementation reads the actual
  post-ShapeFix `WireData` edges and requires a unique proof.

- Observation: sewing history is useful evidence but is not complete enough
  to be the only proof on the two real targets.
  Evidence: `ModifiedSubShape` and `Modified` do not both return a unique,
  agreeing edge result for every occurrence. A unique full-boundary face
  assignment followed by a unique face-local edge assignment maps all 18
  faces of `00063055...` without allowing a history failure to masquerade as
  either success or a fatal failure.

- Observation: a boundary-only STEP assignment can recover exact source
  lineage even when bad pcurves change trimmed area and centroid.
  Evidence: at normalized curve tolerance `1e-4`, all 44 faces of
  `00047472...` and all 18 faces of `00063055...` have a globally unique
  geometry-plus-incidence assignment. In `00063055...`, STEP face ordinal 3
  maps to source face 5, directly disproving ordinal identity. Ambiguity first
  appears near `3.122968e-4`, leaving approximately 3.12 times margin at the
  signed tolerance.

- Observation: the clean-commit formal run localizes the two defects without a
  mapping, coverage, observation, source-binding, worker, or protocol failure.
  Evidence: `00047472...` is first bad immediately after pcurve construction
  at source face/edge pairs `10:[20,13]` and `43:[16,24]`; source face 1 pair
  `[10,12]` is clean in memory and appears only after STEP roundtrip.
  `00063055...` is clean through optional face repair and first bad after
  sewing, at source face 5 with closure `[9,23]` and adjacent `[23,9]`.

- Observation: the signed STEP correspondence has measurable ambiguity
  margin rather than merely returning one assignment.
  Evidence: every face and edge maps uniquely at normalized curve tolerance
  `1e-4`; the first observed ambiguity is approximately `3.122968e-4`, a 3.12
  times margin. The archive recomputes the summary and validates the signed
  rows and run hashes before accepting the result.

## Decision Log

- Decision: Diagnose `00047472...` and `00063055...` before attempting
  `00032101...` or `00076198...`.
  Rationale: the first two have one shell and one solid and no current evidence
  that validity requires deleting source topology. Every known OCC-valid route
  for the latter two merges or deletes legitimate source vertices or edges and
  is correctly rejected by schema-v2.
  Date/Author: 2026-09-03 / Codex.

- Decision: Introduce a new generic stage observer and retain the existing
  `post_pcurve_face_observer` unchanged.
  Rationale: the existing hook is a frozen contract used by the signed
  periodic census. Reusing it at multiple phases would silently change its
  one-call-per-source-face meaning. A separate default-off API preserves old
  evidence while making later stages explicit.
  Date/Author: 2026-09-03 / Codex.

- Decision: Treat entity mapping as a proof obligation, not a nearest-neighbor
  convenience.
  Rationale: repeated and symmetric CAD edges can have identical local
  geometry. Before STEP, accept only a unique OCC identity or sewing-history
  mapping. After STEP, require a unique perfect assignment using source-face
  incidence and orientation-invariant 3D curve fingerprints; ambiguity,
  missing curves, split entities, merged distinct source IDs, or incomplete
  sampling makes the case inconclusive.
  Date/Author: 2026-09-03 / Codex.

- Decision: Keep every OCC operation in a one-CAD child process.
  Rationale: malformed topology can terminate the native process. A crash,
  timeout, malformed sentinel, or missing measurement must remain a counted
  protocol failure rather than disappearing from the denominator.
  Date/Author: 2026-09-03 / Codex.

- Decision: Do not register a repair switch until the read-only pilot names an
  exact downstream stage and exact source occurrences.
  Rationale: the existing broad ShapeFix profiles already produced apparent
  validity by changing topology. Another mutation before lineage is known
  would be neither attributable nor safe.
  Date/Author: 2026-09-03 / Codex.

- Decision: Use normalized curve tolerance `1e-4` and boundary/topology-only
  face compatibility for STEP roundtrip correspondence.
  Rationale: this setting gives unique assignments for every face and edge in
  both targets with measured margin to the first ambiguity. Trimmed area and
  centroid are intentionally excluded as hard gates because the pcurve defect
  under diagnosis can change both without changing the source 3D boundary.
  Date/Author: 2026-09-03 / Codex.

- Decision: Treat incomplete sewing history as an annotated failed proof
  attempt, while permitting an independently unique geometry proof.
  Rationale: declaring incomplete history exact is unsafe, but declaring it
  fatal after a separate unique perfect assignment would discard valid
  evidence. `exact_sewing_history` requires both OCC history APIs to agree;
  `exact_sewing_face_local_geometry` names the independent proof explicitly.
  Date/Author: 2026-09-03 / Codex.

- Decision: Promote two separate, stage-local feasibility routes rather than
  one shared wire repair.
  Rationale: `00047472...` is already defective before optional repair and
  needs an exact-pair non-periodic pcurve reconstruction, whereas
  `00063055...` is clean until sewing and needs a graph-preserving post-sewing
  feasibility spike. Combining them would destroy causal attribution. Source
  face 1 in `00047472...` remains an explicit STEP-roundtrip regression gate.
  Date/Author: 2026-09-03 / Codex.

## Outcomes & Retrospective

The evidence audit, implementation, clean-commit formal experiment, and
Git-safe snapshot are complete. The run signature is
`38ac843ee80611615351db47f38540f7ff27a19dc1f7f1f28883e2c915069271`.
Its two case rows, summary, and run payload are bound by SHA-256 values
`f7a7da3969097f6afc2419c9c4407b4f8c07e1f53a1ddda8d3da451472ca1847`,
`4e6d5daec0f096b9a3e4dc6ce12449be76353f17d2c180c56f0878bc3c57d0df`,
and `ad9d887960192738d5f3b6a4026c5d23befa875bf27228328ed6e49d7e830e4b`.

The result is conclusive: `00047472...` is first bad after pcurve construction
on source faces 10 and 43, while `00063055...` is first bad after sewing on
source face 5. Exact source occurrence lineage survives every observed phase
and STEP roundtrip. This closes the diagnostic milestone and promotes two
separate feasibility probes; it does not itself improve the selector. The
authoritative selector remains 91/100 strict-valid with 84/84 historical
controls, zero regressions, and zero worker/protocol failures.

## Context and Orientation

`tools/directed_trim_assembly.py::construct_brep_directed` fits one surface per
source face and one three-dimensional curve per source edge. It groups the
source `faceEdge_adj` row into oriented loops, creates OCC wires and a face,
calls the upstream `fix_wires` and `add_pcurves_to_edges` helpers, applies one
optional repair strategy, sews all faces, and builds a solid. A pcurve is the
two-dimensional representation of a three-dimensional edge on a surface; the
strict validator uses pcurves when checking wire order and self-intersection.

The existing keyword-only `post_pcurve_face_observer` is called once per
source face after pcurve attachment and before optional repair. It was added
for `tools/probe_periodic_pcurve_applicability.py` and must keep its exact
phase and cardinality. The new observer will be a separate keyword-only
callback and will receive a shape plus path-free metadata. With no callback,
the construction path must remain behaviorally unchanged.

`tools/diagnose_assembly_face_wires.py::_wire_row_v2` locates strict-style wire
defects and reports one-based OCC edge positions, including adjacent,
non-adjacent, closure, and pcurve-gap categories. It does not know source edge
IDs. `tools/local_wire_topology_repair.py` already supplies face topology,
geometry, sampled 3D-curve, and strict wire helpers that the runner can reuse
on disposable copies.

The frozen input manifest is local-only at
`D:/luolin/V13/local_runs/assembly_calibration_100cad_v1_20260809/calibration_manifest.jsonl`.
It contains exactly 100 original rows and 84 historically strict-valid rows.
Its SHA-256 is
`426809e39cf2f4ee13c2e86b542c76a5d1c80ce6abfa4ac4a2e84135f580f4ef`.
The runtime-only upstream tree is `D:/luolin/V13/BrepARG`; it must never be
modified, staged, or copied into this repository.

The registered target CAD IDs are
`00047472_197769bbdd814278b715d88a_step_000` and
`00063055_e309c689b9b44f0686f47966_step_000`. Both must run with the current
selector primary profile, `directed_trim_local_intersection_topology`, after
200 CPU joint-optimization iterations. A different profile would confound the
location of the defects that keep the selector at 91/100.

## Milestones

Milestone 1 adds the observation contract. `construct_brep_directed` gains a
default-off `assembly_stage_observer` independent of the frozen periodic hook.
It emits source-bound face events after pcurve attachment and after optional
face repair, followed by shape events after sewing and after solid creation.
The source-bound metadata includes the exact source face index, ordered loop
indices, global source edge IDs, reversal flags, outer-loop index, and 3D
endpoint gaps. Focused AST and injected-runtime tests prove phase ordering,
metadata completeness, and a no-op default path.

Milestone 2 implements the signed runner. The coordinator selects exactly the
two registered targets from the complete frozen 100-CAD manifest, verifies
that they remain among the current nine selector residuals, binds the exact
source bytes, repository revision, relevant source hashes, runtime `utils.py`,
profile, joint iterations, and worker timeout, and launches one isolated child
per CAD. The child diagnoses every source face at both face stages and observes
the sewn and solid shapes. It writes a temporary STEP locally, reimports it,
and diagnoses the same shape after roundtrip.

The runner maps occurrences in layers. At face construction, an OCC occurrence
must be `IsSame` to exactly one fitted source edge. For optional face repair,
unchanged identity is preferred and a sampled 3D-curve match is permitted only
when it gives a unique assignment within that source face. During sewing,
`ModifiedSubShape` and face provenance constrain the candidate set; merging
occurrences from distinct source edge IDs is rejected. After STEP reimport,
the runner builds a bipartite graph from source-face incidence and
orientation-invariant 3D curve samples. It accepts correspondence only when a
unique perfect matching exists. The report records `mapped`, `ambiguous`,
`unmapped`, and `measurement_failed` explicitly.

Milestone 3 runs the exact cohort from a clean committed revision into a new,
immutable directory under `D:/luolin/V13/local_runs/`. A conclusive result has
two unique completed case rows; exact source hashes; all source faces observed
at both per-face phases; all four required stages present; no native worker,
sentinel, source-binding, finite-value, or mapping failure; and an exact source
mapping for every bad-wire edge occurrence. Any missing proof yields
`INCONCLUSIVE_REQUIRES_RERUN`, never a guessed repair target.

Milestone 4 makes the repair decision. If a local stage first introduces a
small number of bad source occurrences and those occurrences retain the source
graph and 3D curves, a later commit may add one copy-only, diagnosis-gated
repair switch for those exact semantics. If defects exist before the proposed
operation, appear only after an ambiguous STEP mapping, require merging source
entities, or cannot be uniquely localized, the candidate route is closed or
the observer is strengthened before any mutation. A successful exact pilot is
then expanded to the relevant residual family, followed by the invalid subset
and fixed 100-CAD selector. The immutable release gate remains 95/100 strict,
84/84 controls, zero regressions, and zero worker/protocol failures.

## Plan of Work

First, edit `tools/directed_trim_assembly.py`. Add a keyword-only
`assembly_stage_observer` without changing any existing default. Create one
small local emitter that invokes it only when provided. Keep the old
`post_pcurve_face_observer` call exactly where it is. Emit a face event at that
same point, another after the branch at which `face` has its final pre-sewing
value, a sewn-shape event immediately after `SewedShape()`, and a solid event
after `maker.Solid()`. Metadata must be composed only when observation is
enabled so normal training/evaluation does not pay diagnostic cost.

Second, add `tools/probe_downstream_bad_wire_lineage.py`. Reuse the frozen
cohort, input-binding, isolated-worker, strict-sentinel, atomic-manifest, and
Git-clean patterns of `tools/probe_periodic_pcurve_applicability.py`, but do not
reuse its periodic scientific decision. Implement pure helpers for target
selection, stage completeness, assignment uniqueness, path-free compaction,
and the final conclusive/inconclusive decision so they can be tested without
OCC. Put OCC imports inside the one-CAD worker and observation functions.

Third, add `tests/test_probe_downstream_bad_wire_lineage.py` and extend
`tests/test_directed_trim_assembly.py`. Unit tests must reject duplicate or
missing target IDs, source hash drift, partial face coverage, a missing phase,
ambiguous assignments, non-finite fingerprints, source-edge merges, malformed
worker sentinels, and a positive summary with any mapping failure. The callback
contract test must prove the original periodic observer is still called once
at its original phase.

Fourth, commit code, tests, this plan, and ADR-0006. Run the signed two-CAD
probe only from that clean revision. Snapshot a whitelist of path-free
artifacts beneath a new report directory. Do not copy STEP, pickle, NumPy,
checkpoint, raw logs with absolute paths, or upstream source. Update this plan
and ADR with the measured first-defective stages and the repair decision, then
commit and push the evidence.

## Concrete Steps

Work from `D:/luolin/BrepARG2` with the environment interpreter:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe

During implementation, run:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe -m pytest -q tests/test_directed_trim_assembly.py tests/test_probe_downstream_bad_wire_lineage.py tests/test_diagnose_assembly_face_wires.py tests/test_local_wire_topology_repair.py
    git diff --check
    git status --short --branch

After committing the implementation, run the formal probe into a new output
root whose contents do not already exist:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe tools/probe_downstream_bad_wire_lineage.py --calibration-manifest D:/luolin/V13/local_runs/assembly_calibration_100cad_v1_20260809/calibration_manifest.jsonl --selector-matrix D:/luolin/V13/local_runs/assembly_selector_main_100cad_20260818/assembly_repair_matrix.jsonl --selector-run D:/luolin/V13/local_runs/assembly_selector_main_100cad_20260818/assembly_repair_run.json --breparg-root D:/luolin/V13/BrepARG --output-dir D:/luolin/V13/local_runs/downstream_bad_wire_lineage_47472_63055_20260903 --joint-iterations 200 --worker-timeout-seconds 600

Expect exactly two case rows. A conclusive summary must report zero worker,
protocol, source-binding, coverage, observation, and mapping failures. It must
name the first bad phase for each mapped source face/edge occurrence. An
inconclusive summary must state `INCONCLUSIVE_REQUIRES_RERUN`; it must not name
a repair target based on an ordinal-only association.

Before each source or evidence commit, run the focused suite above plus:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe -m pytest -q tests/test_run_assembly_repair_matrix.py tests/test_run_assembly_repair_selector.py tests/test_assembly_selector_geometry.py
    git diff --check

## Validation and Acceptance

The observer API is accepted when tests prove it is keyword-only, defaults to
`None`, emits the documented phase sequence only when provided, preserves the
existing periodic hook's phase and call count, and carries an exact ordered
source-edge-use record for every constructed face loop.

The runner is accepted when its non-OCC tests prove that a case cannot be
called complete with a missing source face, missing phase, duplicate stage,
non-finite metric, ambiguous entity mapping, mismatched source binding,
unregistered target, malformed worker result, or input mutation. Native OCC
work must remain in child processes, and every child must return one strict
sentinel row bound to the coordinator's run signature.

The scientific pilot is accepted only when both registered CADs complete and
every strict bad-wire occurrence at every observed stage has a proven source
mapping. The result must distinguish `clean`, `bad`, `ambiguous`, and
`unavailable`; absence of a mapped row is not cleanliness. A promote decision
must identify a stage and exact source occurrences whose topology and sampled
3D curves remain unchanged. Otherwise the decision is close or rerun.

A later repair is accepted only if an isolated exact-CAD pilot becomes
STEP-readable, OCC native-valid, project strict-valid, and both-valid; passes
all 23 schema-v2 checks; preserves source counts and incidence; and leaves the
original face unchanged on a rejected candidate. No result counts toward
91-to-95 until the fixed 100-CAD selector also preserves 84/84 controls with
zero regressions and zero worker/protocol failures.

## Idempotence and Recovery

All observation callbacks are read-only and default-off. Formal output roots
are immutable. A worker crash or coordinator interruption may be resumed only
when the signed payload is byte-for-byte compatible; a torn final JSONL line
may be discarded, but a completed row or artifact is never overwritten. A
code, runtime, manifest, selector, or input change requires a new output root.

If an OCC worker exits natively, the parent records its return code, timeout,
and protocol status without importing OCC. If correspondence is ambiguous,
retain all candidate IDs and mark the stage inconclusive; never resolve it by
explorer order. If a proposed mutation fails a local invariant, return the
untouched face and record the rejection. Git recovery uses a new commit; do not
reset or overwrite user changes.

## Artifacts and Notes

Current authoritative controls are:

    selector strict-valid:       91/100
    selector native-valid:       90/100
    selector both-valid:         88/100
    historical controls:         84/84
    selector regressions:        0
    worker/protocol failures:    0
    assembly release threshold:  >=95/100 strict-valid

The formal selector run signature is
`a4f1208d4a74026be313a6dfff6b6a1dc92ce0c79c154f5ea9dc9bf113b55cf1`.
The final selector matrix SHA-256 is
`d3cb1ba56fbc67cdb4db3828cc1ba3036e800ccd32b20b47071144c081b65fe8`.

The prior periodic census remains a separate, conclusive negative experiment:
it observed 134/134 construction faces, six bad faces, zero periodic bad faces,
and zero repairable periodic faces. This plan must not reinterpret that result
or invent a period on a non-periodic B-spline.

## Interfaces and Dependencies

In `tools/directed_trim_assembly.py`, retain:

    post_pcurve_face_observer: Callable[[int, Any, Mapping[str, Any]], None] | None

and add a distinct keyword-only interface equivalent to:

    assembly_stage_observer: Callable[[Any, Mapping[str, Any]], None] | None

The callback receives a disposable observation target and metadata containing
`phase`, `entity_kind`, and phase-specific lineage. Per-face events contain
`source_face_index`, `source_loop_edge_uses`, `outer_loop_index`, and endpoint
gaps. Shape events contain their phase and the expected source face/edge
population. Callback return values are ignored.

`tools/probe_downstream_bad_wire_lineage.py` must expose pure selection,
mapping-decision, population-validation, and summary functions alongside its
CLI. Its parent process uses only the standard library and NumPy; OCC imports
must be delayed until worker execution. It uses
`tools.diagnose_assembly_face_wires`,
`tools.local_wire_topology_repair`,
`tools.run_assembly_calibration_oracle.cpu_joint_optimize`, and
`tools.run_assembly_repair_matrix.profile_kwargs` rather than modifying the
runtime-only upstream tree.

Revision note 2026-09-03 18:25 +08:00: Created the plan after the complete
nine-residual/five-profile audit. It freezes the two-CAD cohort, separates the
new multi-stage observer from the signed periodic hook, and makes unique source
entity correspondence a prerequisite for any non-periodic repair.

Revision note 2026-09-03 22:05 +08:00: Recorded the completed immutable run,
its cryptographic bindings and Git-safe archive, corrected source face 1 of
`00047472...` to a STEP-only regression, and split the promoted work into an
exact-pair pcurve route and a post-sewing graph-preserving feasibility route.
