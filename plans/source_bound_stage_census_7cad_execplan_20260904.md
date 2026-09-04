# Locate the first source-bound failure stage in the seven remaining CADs

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current
while the work proceeds. This plan follows `PLANS.md` in the repository root
and must be maintained in accordance with that file.

## Purpose / Big Picture

The fixed 100-CAD failure-triggered assembly selector is strict-valid for 91
CADs. The release gate is at least 95/100, all 84 historically strict-valid
controls preserved, zero regressions, zero worker or protocol failures, zero
non-finite measurements, and acceptance by the unchanged schema-v2 geometry
and topology gate. Representation capacity is no longer the active unknown:
VQ-8192/64D is strict-valid for 69/100 on the unchanged chain and the continuous
bypass is 70/100, so their assembly gap is one percentage point. Full VQ
training, sequence regeneration, and autoregressive training remain blocked by
the assembly gate.

Nine CADs remain strict-invalid in the signed selector. Two of those, CADs
47472 and 63055, have already received a conclusive four-cell exact experiment.
Both controls reproduced, both registered `FixRemovePCurve` followed by
`FixAddPCurve` candidates were invoked and rejected, and there were no worker,
protocol, or non-finite failures. This plan therefore excludes those two exact
negative candidates and measures the seven remaining CADs without mutating
them. After this change, a user can read one signed seven-stage report and see
the earliest construction boundary at which each source-bound CAD becomes
invalid, loses an exact mapping, changes topology, or fails to advance. The
result will choose the next smallest causal experiment instead of launching
another broad repair sweep.

The observable outcome is exactly ten denominator rows: seven unchanged
primary-control rows plus three read-only curve-interpolation bridge rows for
the three controls that currently stop during curve construction. Every row is
run in a fresh child process, binds the exact source pickle bytes through six
comparisons, and reports all reached stages from S1 through S7. A crash,
timeout, malformed worker result, missing stage, ambiguous source mapping, or
non-finite value remains an explicit false or inconclusive row; it never
disappears from the denominator. This census cannot itself add a selector
profile, claim a recovered CAD, increase 91/100, authorize training, or weaken
schema-v2.

## Progress

- [x] (2026-09-04 16:56 +08:00) Revalidated the signed selector evidence: the
  frozen cohort contains 100 CADs, the current selector is strict-valid for 91,
  and exactly nine rows remain strict-invalid.
- [x] (2026-09-04 16:56 +08:00) Bound the exclusion decision to the Git-safe
  exact-CAD archive at commit `afafeb81e1674078aa4e08c2987f4343d4734808`.
  Its four-row run signature is
  `1d4f68839aadc8b3f8fb38eea642a1f7ea4f6d8d51b61152f943c725832ffcad`,
  terminal row hash is
  `f158f2ca7f9bf2adceb7a56434ca4925bed99e34d5791e8867ca476f32d70a34`,
  and summary hash is
  `545802b4e783a3f3f76039d70e983fba1ab5eb29af0748e5db032e731c925f60`.
- [x] (2026-09-04 16:56 +08:00) Pre-registered the seven-CAD cohort, ten-row
  task order, seven observation stages, six-comparison source binding, and the
  fail-closed rules in this plan and ADR 0009 before the formal experiment.
- [x] (2026-09-04 18:29 +08:00) Added the default-off, read-only
  `assembly_stage_observer` to `tools/directed_trim_assembly.py` without moving
  construction work out of the historical loops. S1/S2 now observe the exact
  distributed order `fit(i) -> S1(i) -> MakeEdge(i) -> S2(i)`, and S3/S4
  observe `build face(i) -> S3(i) -> optional repair(i) -> S4(i)`. Focused
  tests cover default-off behavior, ordering, return-value isolation, and
  fail-closed callback errors.
- [x] (2026-09-04 18:29 +08:00) Added the OCC-independent normalization and
  first-bad inference core in `tools/assembly_stage_lineage.py`. It recognizes
  a proved distributed prefix separately from a proved local terminal failure,
  validates canonical face-edge and edge-face relations, and treats topology
  drift as scientific evidence rather than a worker/protocol failure.
- [x] (2026-09-04 18:29 +08:00) Proved on calibration CAD
  `00066307_28655d45c4dc4e378db24d63_step_000` (30 faces, 86 edges) that
  enabling the observer preserves the completed construction path and emits
  the expected interleaved events: S1=86, S2=86, S3=30, S4=30, S5=1, S6=1.
  A direct A/B against the constructor at `afafeb81...` also matched the native
  result and all compared construction diagnostics. Both direct analyzers were
  native-invalid, so this is evidence of path equivalence and event ordering,
  not evidence of a recovered CAD or the formal seven-CAD census.
- [x] (2026-09-04 19:18 +08:00) Added a first implementation of global
  source-vertex proofs for S2-S6 and a STEP-roundtrip proof that jointly requires
  one unique face, edge, and vertex assignment. S7 derives optimized source
  vertex positions from `edge_wcs` endpoint samples, not stale pre-optimization
  `corner_unique`, and uses the same STEP bounding-box scale for its frozen
  normalized tolerance. A later real-OCC smoke found that the first S2-S4
  identity contract was too strong; the correction is tracked below rather than
  treating the earlier passing fixtures as final evidence.
- [x] (2026-09-04 20:20 +08:00) Re-ran the real-OCC smoke on calibration CAD
  `00066307_28655d45c4dc4e378db24d63_step_000` through STEP roundtrip. It proved
  observer-off/on path equivalence and exposed two different phenomena that the
  protocol must not conflate. Before sewing, 86 independently built edges have
  172 native endpoint handles for 58 source vertex labels, and face construction
  may legally copy those handles. After sewing S5/S6 recover exactly 30 faces,
  86 edges, and 58 vertices. After STEP roundtrip, however, the shape changes to
  30 faces, 90 edges, and 60 vertices; source faces 1, 6, and 28 have real edge
  splits. This smoke is protocol discovery, not a formal denominator row and
  not a selector-score change.
- [x] (2026-09-04 23:05 +08:00) Closed the final 66307 smoke as a conclusive
  protocol check. It completed in about 6.84 seconds with zero non-finite and
  zero protocol failures. The exact prefix and S5/S6 global-proof path localize
  the first bad boundary to S6 because construction-native validity is false.
  STEP reimport is later native-valid and strict-valid, but changes the source
  86 edges, 58 vertices, and 172 face-edge occurrences into 90, 60, and 180.
  The assessment is conclusive with `first_bad_stage=S6`, reason
  `construction_native_invalid`, and `valid_chain=false`; downstream validity
  recovery cannot erase the earlier bad boundary. This remains a development
  smoke, not one of the ten formal denominator rows.
- [x] (2026-09-04 23:34 +08:00) Finished the immutable ten-task runner and
  Git-safe snapshotter, including the corrected S2-S4 scope, registered
  non-exact S5/S6 proof states, S7 implementation-exception escalation,
  selector/input gates, persistent STEP verification, strict JSON and manifest
  resume validation, six source-byte comparisons, and terminal hash checks.
- [x] (2026-09-04 23:34 +08:00) Isolated each formal worker with
  `python -I -c` plus a controlled `runpy` bootstrap. The actual worker measures
  its representative ABI sentinel before input selection, source-byte access,
  deserialization, or scientific imports and must exactly match the signed and
  frozen sentinel. Scientific rows retain that path-free proof; worker and
  protocol failure rows must retain `null`.
- [x] (2026-09-04 23:36 +08:00) Passed the current combined regression suite:
  `456 passed, 2 warnings`. The warnings are the pre-existing pythonocc
  `topods_Wire` deprecation notices. Focused runner/snapshot coverage reports
  `148 passed`; `py_compile` and `git diff --check` also pass.
- [ ] Validate the Git-safe snapshotter against the completed signed formal
  run. Its implementation and fixtures pass, but no formal snapshot exists yet.
- [ ] Freeze and push one clean implementation commit, verify local HEAD,
  upstream, and the live remote tip are identical, then run exactly ten formal
  cells into a new unused local output root and independently validate every
  row, hash, source binding, stage sequence, worker sentinel, and summary.
- [ ] From one clean implementation commit, run exactly ten formal cells into
  a new unused local output root and independently validate every row, hash,
  source binding, stage sequence, and terminal summary.
- [ ] Use a conclusive stage census only to pre-register one new exact-CAD
  control/candidate experiment. Do not expand to a residual family, invalid-16,
  or fixed 100 CADs until that exact candidate independently passes.

## Surprises & Discoveries

- Observation: the remaining problem is not a generic nine-CAD cohort after
  the latest exact experiment.
  Evidence: the selector has nine strict-invalid rows, but the signed four-cell
  archive conclusively rejects the registered candidates for 47472 and 63055.
  The next non-duplicative measurement cohort is therefore exactly seven CADs.

- Observation: three of the seven primary controls do not produce STEP and
  cannot expose all downstream boundaries under the unchanged fit path.
  Evidence: selector rows 51602, 61931, and 87341 end with `assembly_error` and
  no saved STEP. A separate `directed_trim + curve_interpolate` row can serve as
  a reachability bridge, but it is a changed input path and cannot be called a
  repair or compared as though it were the primary control.

- Observation: construction validity and STEP-reimport validity are distinct
  measurements and can move in either direction.
  Evidence: existing evidence records 32101 as construction-native false and
  STEP-reimport native false. It records 76198 as construction-native false,
  STEP-reimport native true, while project strict validity remains false. The
  census must retain both values instead of inferring one from the other.

- Observation: a parent coordinator importing pythonocc can make a worker
  crash contaminate the whole experiment even when CAD work is nominally
  isolated.
  Evidence: OCC performs native work at import and call boundaries. The parent
  must therefore restrict itself to JSON, hashing, signatures, subprocesses,
  files, and pure validators; each task child alone may import OCC-dependent
  assembly code.

- Observation: an apparently cleaner global S1/S2 snapshot changed the program
  being measured.
  Evidence: the first observer draft fitted every curve before constructing any
  edge, changing the historical order from `fit(0) -> MakeEdge(0) -> fit(1)` to
  `fit(all) -> MakeEdge(all)`. A failure in `MakeEdge(0)` would therefore allow
  later curve fits that the unchanged constructor never executes. That draft
  was rejected as confounded; the accepted observer remains inside the original
  per-edge loop.

- Observation: S1-S4 are distributed event censuses, not four moments at which
  the whole CAD exists in a uniform stage.
  Evidence: the accepted real-OCC smoke emitted the sequence S1(edge 0),
  S2(edge 0), S1(edge 1), S2(edge 1), and later S3(face 0), S4(face 0),
  S3(face 1), S4(face 1). The runner must aggregate these source-bound events
  without describing them as global snapshots.

- Observation: a passed prefix and a local terminal have different causal
  meanings.
  Evidence: when `MakeEdge(i)` fails, S1 has proved that curve `i` crossed its
  boundary while S2 has the terminal event. The normalized S1 status is an
  exact prefix pass and cannot itself be first-bad; the normalized S2 status is
  `local_exact_failure` and may be first-bad when all preceding proof holds.

- Observation: shape counts alone are insufficient once endpoint topology is
  part of the release gate.
  Evidence: prior apparent assembly recoveries changed vertex or incidence
  topology. The census therefore has to bind each edge to its two source vertex
  IDs from S2 through S6 and include vertices in the unique global S7 assignment,
  rather than merely comparing face and edge totals.

- Observation: native handle identity before sewing is not the same thing as
  source topology identity.
  Evidence: the real-OCC 66307 smoke built 86 standalone edges with 172 pairwise
  distinct endpoint handles even though their endpoint labels describe only 58
  source vertices. Face/wire construction retained only 38 of 172 endpoint
  occurrences as `IsSame` to the standalone-edge handles, yet sewing produced
  the exact source population of 30 faces, 86 edges, and 58 vertices. Requiring
  cross-edge or cross-builder `IsSame` at S2-S4 therefore turns legal OCC object
  copying into a false protocol ambiguity.

- Observation: OCC validity after STEP does not prove source-exact topology.
  Evidence: the same smoke is native-valid and project-strict-valid after STEP,
  but STEP reimport has 90 edges, 60 vertices, and 180 face-edge occurrences
  instead of the source 86, 58, and 172. Source faces 1, 6, and 28 preserve area
  and bounding boxes to about `1e-13` normalized error while their edge counts
  increase from 4/9/4 to 6/13/6. The unchanged S7 exact topology gate must report
  that transfer split rather than coercing the row to exact.

- Observation: the runner's formal evidence contract had three independent
  readiness gaps even though its focused fixtures passed.
  Evidence: selector source selection accepted a row mutated to
  `status=worker_timeout`; ordinary exceptions after exact S4 or exact S5 were
  not localizable to S5/S6; and S7 deleted `roundtrip.step` unconditionally even
  though this plan requires an attempt-unique local STEP whose size and SHA-256
  are checked during resume. These are formal-run blockers, not post-run cleanup.

- Observation: Python's default JSON decoder and equality rules are too loose
  for signed evidence.
  Evidence: default decoding accepts duplicate object members, `NaN`,
  `Infinity`, and finite-looking exponents that overflow; ordinary equality
  also treats an integer as equal to the same-valued float. The runner now
  rejects those forms at every trust boundary, compares JSON values with exact
  types, validates status-specific manifest key sets, and recomputes the stored
  payload signature before accepting a resume.

- Observation: an isolated probe grandchild does not prove which runtime the
  actual OCC worker used.
  Evidence: environment paths, user site packages, or customization modules
  could affect a normally launched worker while leaving a separate `-I` probe
  unchanged. Formal workers now start with `python -I -c` and a controlled
  `runpy` bootstrap. The same process measures its representative ABI sentinel
  before input selection or source/scientific work and includes it only in a
  successfully completed scientific row.

- Observation: the current runtime evidence is deliberately representative,
  not a complete inventory of every lazily loaded OCC module.
  Evidence: it binds the interpreter, NumPy version, pythonocc `_Standard.pyd`,
  loaded `TKernel.dll`, PE versions, binary sizes and hashes, plus Python
  isolation flags. Its explicit scope is
  `representative_abi_sentinel_not_complete_module_inventory`; unenumerated
  lazy modules remain outside the claim.

- Observation: expected non-matching geometry and an internal matching failure
  have different scientific meanings.
  Evidence: zero or multiple perfect S7 assignments are registered non-exact
  observations. An exception inside the matching implementation is now raised
  as `StepGeometryIncidenceMatchingError` and retained as a worker/protocol
  failure instead of being mislabeled as ordinary geometry `unavailable`.

## Decision Log

- Decision: derive the cohort as frozen 100, then current nine strict-invalid,
  then remove only 47472 and 63055, yielding seven ordered CADs.
  Rationale: this preserves the authoritative selector denominator while
  avoiding immediate repetition of two conclusive negative exact candidates.
  Date/Author: 2026-09-04 / Codex.

- Decision: register seven primary controls and three curve-interpolation
  reachability bridges as exactly ten denominator cells.
  Rationale: all seven need the unchanged control, while only 51602, 61931, and
  87341 require a bridge to observe past their early construction error. A
  bridge is evidence about reachability, not evidence of recovery.
  Date/Author: 2026-09-04 / Codex.

- Decision: observe six in-memory constructor stages and create the seventh
  stage only after an independent STEP write and reimport.
  Rationale: separating construction-native validity from reimport-native and
  project-strict validity identifies serialization changes and prevents an
  in-memory shape from standing in for the actual downstream artifact.
  Date/Author: 2026-09-04 / Codex.

- Decision: keep the coordinator OCC-free and launch one fresh subprocess for
  every `(CAD, arm)` cell.
  Rationale: an OCC abort must produce one retained denominator row, not end or
  corrupt the other nine attempts.
  Date/Author: 2026-09-04 / Codex.

- Decision: require unique source-bound correspondence and fail closed on any
  missing, split, merged, duplicated, non-finite, or ambiguous mapping.
  Rationale: explorer order and nearest-neighbor guesses can assign a defect to
  the wrong source entity. An inconclusive result is scientifically safer than
  a fabricated first-bad stage.
  Date/Author: 2026-09-04 / Codex.

- Decision: forbid `FixRemovePCurve`, both exact experiment mutators, and all
  other shape mutation in this census.
  Rationale: this is a measurement experiment. Reusing the just-rejected route
  or combining diagnosis with repair would destroy causal attribution and
  overwrite the meaning of the negative archive.
  Date/Author: 2026-09-04 / Codex.

- Decision: retain the complete schema-v2 gate unchanged even though no census
  row can be promoted.
  Rationale: topology and geometry thresholds are release invariants. A later
  candidate must be evaluated under the same gate, not a threshold selected
  after seeing its output.
  Date/Author: 2026-09-04 / Codex.

- Decision: define S1/S2 as source-edge distributed events and S3/S4 as
  source-face distributed events, preserving the historical interleaving.
  Rationale: moving the callbacks to global all-curves or all-faces boundaries
  changes which later native operations execute after an early failure. The
  census must observe the existing constructor, not a reordered surrogate.
  Date/Author: 2026-09-04 / Codex.

- Decision: distinguish `exact_prefix_pass` from `local_exact_failure` in
  causal inference. The implementation may accept the legacy serialized alias
  `exact_prefix`, but reports describe its meaning as an exact prefix pass.
  Rationale: a boundary that succeeded for the canonical prefix before its
  paired next boundary failed is positive preceding evidence, not the defect.
  Only a uniquely ordered terminal at the current boundary can be first-bad.
  Date/Author: 2026-09-04 / Codex.

- Decision: accept a terminal only when it is explicit at an instrumented
  failure site or is the unique next S3-S6 event implied by a canonical event
  prefix and an ordinary construction exception.
  Rationale: S1/S2 have instrumented curve-fit and MakeEdge failure sites. A
  face-loop exception may occur between S3/S4 callbacks, but its next boundary
  is still uniquely determined by the alternating prefix. Likewise, complete
  exact S4 followed by no S5 localizes S5, and one exact S5 followed by no S6
  localizes S6. Observer failures, malformed order, duplicate events, ambiguous
  or missing prerequisite lineage, and exception text cannot be used to
  synthesize a scientific terminal.
  Date/Author: 2026-09-04 / Codex.

- Decision: require vertex lineage from S2 onward and a full vertex-aware
  geometry-and-incidence proof at S7.
  Rationale: exact face/edge counts can conceal endpoint merges, splits, or
  rewiring. The same canonical `edge_vertex_source_ids` relation and incidence
  sums must survive every in-memory stage, and STEP reimport must recover it by
  one unique global assignment rather than explorer order or nearest matching.
  Date/Author: 2026-09-04 / Codex.

- Decision: derive S7 source vertex positions from the optimized edge endpoint
  population and freeze `solid_topology_repair=False` for both census arms.
  Rationale: `cpu_joint_optimize` may move endpoints away from the source
  pickle's `corner_unique`, while topology reconciliation could change endpoint
  labels. The proof must compare STEP against the geometry and adjacency that
  the unchanged constructor actually consumed. All samples for one source
  vertex must have pairwise diameter at most `1e-4 * STEP shape scale` before
  they can define that vertex's WCS point.
  Date/Author: 2026-09-04 / Codex.

- Decision: separate exact source-entity binding from OCC native-handle
  population at S2-S4.
  Rationale: each source edge is authoritatively bound by its construction
  position at S2, and each S3/S4 face and edge occurrence must still have one
  unique identity/history/geometry assignment. Independently built edges and
  face/wire copies are not required to share `IsSame` endpoint handles before
  sewing. The stage records the current handle-population relation separately;
  it may not claim a global source-vertex bijection until a stage, such as S5 or
  S6, actually supplies one. This removes a false ambiguity without weakening
  source binding or the S7 topology gate.
  Date/Author: 2026-09-04 / Codex.

- Decision: represent inferred S5/S6 ordinary construction failures as explicit
  whole-shape boundary terminals rather than invented source-face terminals.
  Rationale: S5 and S6 are single CAD-wide boundaries. Full exact S4 followed by
  an ordinary exception before S5, or exactly one exact S5 followed by an
  ordinary exception before S6, uniquely identifies the missing boundary but
  does not identify a face or edge. A distinct whole-shape terminal scope keeps
  that causal fact exact without fabricating an entity ID.
  Date/Author: 2026-09-04 / Codex.

- Decision: retain every S7 STEP artifact in the machine-local run root under an
  attempt-unique logical name and verify its bytes and SHA-256 on resume and
  terminal validation; never copy STEP bytes into the Git-safe snapshot.
  Rationale: a hash recorded for a file immediately deleted cannot be
  independently replayed after the run. Local persistence closes the formal
  evidence chain while preserving the repository's no-STEP policy.
  Date/Author: 2026-09-04 / Codex.

- Decision: fail closed on selector worker/protocol-health drift and on any
  formal input SHA that differs from the three pre-registered digests.
  Rationale: the seven-CAD derivation is valid only from the completed,
  zero-protocol-failure selector and the exact calibration/selector files
  audited before implementation. Structural similarity is insufficient for a
  signed census.
  Date/Author: 2026-09-04 / Codex.

- Decision: decode signed, resumable, worker, and terminal JSON with duplicate-
  key, non-finite, and overflow rejection; compare recursive JSON values with
  exact types; and rehash the stored manifest payload under an exact status-
  specific key schema before resume.
  Rationale: JSON syntax or Python numeric coercion must not let altered
  evidence retain an old signature or pass an identity comparison.
  Date/Author: 2026-09-04 / Codex.

- Decision: call the runtime proof a representative ABI sentinel with schema
  `source-bound-runtime-abi-sentinel-v1` and scope
  `representative_abi_sentinel_not_complete_module_inventory`.
  Rationale: interpreter and selected Python/OCC ABI binaries are sufficient to
  detect the registered environment drift, but they do not inventory every
  lazily loaded `.pyd` or `.dll`; the claim must not exceed the evidence.
  Date/Author: 2026-09-04 / Codex.

- Decision: launch each worker as `python -I -c <bootstrap> <repo>` and require
  that same worker to measure its sentinel before source selection, source-byte
  access, deserialization, or scientific imports. A scientific row must contain
  an exact signed/frozen sentinel; every worker/protocol failure row must keep
  the field `null`.
  Rationale: a separate probe process cannot prove the runtime of the process
  that read the CAD and executed OCC, while isolated bootstrap blocks
  `PYTHONPATH`, user site, `sitecustomize`, and `usercustomize` interference.
  Date/Author: 2026-09-04 / Codex.

- Decision: retain registered zero/non-unique S5-S7 assignments as scientific
  non-exact evidence, but promote an exception inside S7 matching to a worker/
  protocol failure via `StepGeometryIncidenceMatchingError`.
  Rationale: a well-formed proof that finds no unique assignment is a result;
  a broken implementation is not geometry evidence and must make the formal
  census fail closed.
  Date/Author: 2026-09-04 / Codex.

## Outcomes & Retrospective

The cohort, protocol, and decision boundary are pre-registered, and the
diagnostic implementation now exists in the working tree. The first major
mid-implementation correction rejected a globalized observer that changed the
historical fit/MakeEdge order. The second came from a full real-OCC roundtrip:
pre-sewing native handle sharing is not a valid source-topology oracle, whereas
the 86-to-90 edge change after STEP is a real transfer-stage split. The accepted
distributed observer is path-equivalent on one real CAD, but no smoke constitutes
one of the ten formal denominator rows. Corrected lineage, S5/S6 closure,
selector/input gates, persistent STEP verification, strict transport and
resume rules, isolated same-process runtime proof, and the combined regression
suite are complete. A clean implementation commit and push, live remote-tip
verification, the signed ten-row run, and its Git-safe archive still remain.

No new CAD has been repaired, the selector remains 91/100, and no training
stage is authorized. Completion still requires the formal report to assign an
exact first-bad stage only where the source-edge, source-face, and source-vertex
evidence is unique and to say inconclusive everywhere else. The resulting map
can pre-register one exact-CAD candidate; only later gated expansion experiments
can change the assembly score.

## Context and Orientation

The maintained Git repository is `D:\luolin\BrepARG2`. Machine-local input
pickles, STEP artifacts, native logs, and formal run directories live under
`D:\luolin\V13\local_runs` and must not be committed. The current branch is
`experiment/protocol-v5-scaling-ladder`.

`reports/assembly_selector_main_100cad_20260818/` is the Git-safe snapshot of
the current fixed selector. Its signed run has 100 final rows, 91 strict-valid
CADs, all 84 historical controls preserved, and zero worker or protocol
failures. The nine strict-invalid CAD identifiers are:

    00051602_7f1947595ae247e0a4a32f43_step_000
    00061931_dcdd8a95feac4121adfd341f_step_000
    00067160_2a27016aa44f42c69c1079f7_step_000
    00063055_e309c689b9b44f0686f47966_step_000
    00047472_197769bbdd814278b715d88a_step_000
    00087341_6a73c5e821934d3fe4d0d555_step_000
    00076198_7fde7438ca5d3ccb8a1dd1f4_step_000
    00095733_8b325d2fcb27ec9e79388602_step_000
    00032101_674d8fea687f4d9bbca6599b_step_000

`reports/exact_cad_repair_feasibility_20260904/` is the Git-safe, conclusive
negative archive for 47472 and 63055. Its source commit is `e870bb25...`, and
the archive itself is present at commit `afafeb81...`. It closes only the two
registered `FixRemovePCurve`-then-`FixAddPCurve` implementations. It does not
prove that every possible pcurve or wire mechanism is impossible, but it does
mean this census must not call those implementations again.

The ordered seven-CAD census cohort is the nine-CAD list with those two exact
negative CADs removed:

    00051602_7f1947595ae247e0a4a32f43_step_000
    00061931_dcdd8a95feac4121adfd341f_step_000
    00067160_2a27016aa44f42c69c1079f7_step_000
    00087341_6a73c5e821934d3fe4d0d555_step_000
    00076198_7fde7438ca5d3ccb8a1dd1f4_step_000
    00095733_8b325d2fcb27ec9e79388602_step_000
    00032101_674d8fea687f4d9bbca6599b_step_000

A denominator row is one requested task even if its child times out, crashes,
does not reach STEP, or returns malformed data. The seven `primary_control`
tasks use the unchanged `directed_trim_local_intersection_topology` constructor
settings. Three additional `curve_interpolate_bridge` tasks are registered for
51602, 61931, and 87341. A bridge changes only the existing
`curve_interpolate` switch in the directed-trim path so the observer may reach
later stages. It is not `curve_fit_rescue`; it must not be selected, scored as
a recovery, accepted by schema-v2 as a new profile, or included in a 91/100
numerator.

The seven ordered stage names identify causal boundary types, not seven repair
algorithms and, for S1-S4, not simultaneous whole-CAD snapshots:

1. `S1_post_surface_curve_fit_pre_edge_build` is emitted once per source edge
   after that edge's curve fit and immediately before that edge's MakeEdge
   operation. All surfaces have already been fitted, but later curves have not
   necessarily been fitted. The source-edge event order starts S1(0), S2(0),
   S1(1), S2(1).
2. `S2_post_edge_build_pre_face_build` is emitted once per source edge after
   that edge was built and before the next source curve is fitted. It binds the
   built edge to the effective unordered pair of source vertex IDs; a self-loop
   retains the same ID twice. Each independently built source edge is its own
   pre-sewing scope, so its endpoint handles form stage-local identity classes;
   no cross-edge native-handle sharing is required.
3. `S3_post_add_pcurves_pre_optional_face_repair` is emitted once per source
   face after that face and its pcurves have been built, immediately before the
   optional repair for that same face.
4. `S4_post_optional_face_repair_pre_sewing` is emitted once per source face
   after that face's optional repair, before construction advances to the next
   source face. Thus the source-face event order is S3(0), S4(0), S3(1), S4(1).
5. `S5_post_sewing_pre_solid` is immediately after `SewedShape()` and before a
   solid is constructed or a post-sewing mutator could run.
6. `S6_post_solid_pre_step` is after `maker.Solid()` and before STEP export;
   it records `construction_native_valid` for the in-memory result.
7. `S7_post_step_roundtrip_strict` is created by the child runner only after
   writing and independently re-reading STEP; it records
   `reimport_native_valid`, project `strict_valid`, and the full strict
   component diagnosis.

“Source-bound” means every observed face, edge occurrence, edge endpoint, and
vertex is related to the source arrays from the exact pickle by identity,
authoritative OCC modification history, or a stage-appropriate geometry-and-
incidence assignment. At S3/S4, each source face is a separate scope: source
face and edge-occurrence mapping remains complete and unique, while endpoint-
class topology needs at least one valid label bijection rather than a unique
global vertex-label permutation. Explorer ordinal alone is never
correspondence. “First-bad stage” is the earliest causal boundary with either a proved local terminal failure or
complete exact evidence that is invalid or topologically inconsistent. A
distributed `exact_prefix_pass` says only that the canonical prefix crossed
that boundary before a paired later boundary stopped traversal; it is positive
preceding evidence and is not itself bad. If a row begins bad at S1, skips a
required event, has more than one possible mapping, or cannot prove the
transition, its reason must say so; the runner must not guess a later stage.

`tools/directed_trim_assembly.py` contains `construct_brep_directed`. It gains
one keyword-only callback named `assembly_stage_observer` with default `None`.
The callback is read-only and has the shape:

    assembly_stage_observer(observation_target, metadata) -> None

The return value is ignored. `metadata` contains at least `stage`, `phase`, and
`entity_kind`, plus stage-specific source indices, mapping evidence, counts,
or construction status. The callback may inspect but never replace the target.
When it raises, the constructor must fail closed with a stage-labeled error and
preserve the original exception as its cause. Existing
`assembly_stage_face_observer`, `post_pcurve_face_mutator`, and
`post_sewing_shape_mutator` contracts stay unchanged; the new census passes no
mutation callback.

`tools/assembly_stage_lineage.py` is an OCC-independent module. Its public
surface contains `STAGE_ORDER`, `normalize_stage_record`,
`validate_stage_sequence`, `assess_stage_lineage`, and
`infer_first_bad_stage`. These functions validate finite, JSON-safe evidence;
distinguish missing, invalid, ambiguous, and exact mappings; identify topology
drift without approving it; preserve the S6/S7 validity distinction; and
derive an answer only from ordered evidence.

`tools/probe_source_bound_stage_census.py` is the parent/child coordinator.
The parent does not import pythonocc, `tools.directed_trim_assembly`, the local
BrepARG runtime, or any module that imports OCC transitively. It reads JSON,
selects and hashes source files, binds code and Git state, launches children,
validates their final sentinel, persists denominator rows, and derives the
summary using the pure lineage module. A child alone loads one pickle, imports
the local runtime and OCC-dependent assembly modules, constructs one arm,
captures stages, writes STEP when reachable, reimports it, and emits exactly
one final JSON sentinel.

`tools/assembly_selector_geometry.py` defines
`assembly-selector-geometry-gate-v2`. This is the “schema-v2” gate. Its current
continuous thresholds remain `0.02` maximum relative bounding-box delta,
`0.05` maximum relative total edge-length delta, `0.01` maximum normalized
edge-sample RMS, and `0.05` maximum normalized edge-sample maximum, together
with exact topology, incidence, finiteness, and sample-accounting checks. The
census measures rather than promotes candidates, so it may report a gate as
not applicable; it must never alter these constants or call an invalid bridge
accepted.

## Plan of Work

First, extend `construct_brep_directed` with the default-inert observer without
reordering any historical construction operation. In the existing edge loop,
emit S1(i) after `fit(i)` and before `MakeEdge(i)`, then S2(i) after the edge is
built and before `fit(i+1)`. In the existing face loop, emit S3(i) after the
face's pcurves are present and before its optional repair, then S4(i) after that
repair and before work on face i+1. Emit S5 once for the sewn shape before
either existing post-sewing mutation or solid creation, and S6 once for the
final in-memory solid. Do not describe the aggregated S1-S4 records as global
snapshots. Keep observation separate from mutation: the observer cannot return
a replacement, and this census must pass both existing mutation arguments as
`None`. Add tests that prove no callback means identical output and diagnostics,
the edge order is exactly S1(0), S2(0), S1(1), S2(1), the face order is exactly
S3(0), S4(0), S3(1), S4(1), complete traversals cover every source entity, the
return value is ignored, and a callback exception is labeled with its precise
stage.

Second, implement the pure lineage normalizer. A normalized stage record must
name its task, CAD, arm, stage, whether the stage was reached, whether coverage
is complete, whether every numeric value is finite, mapping status, topology
and incidence counts where meaningful, validity measurements, and a bounded
path-free reason. The mapping vocabulary must at least distinguish `exact`,
`ambiguous`, `missing`, `split`, and `merged`; only `exact` can support a first-
bad claim on a complete stage. A validated `exact_prefix_pass` may support the
next boundary's causal proof but may not be called bad; a validated
`local_exact_failure` may be first-bad without pretending that its partial
prefix is a complete topology census. S1/S2 terminal events are emitted
explicitly at instrumented curve-fit and MakeEdge failures. S3-S6 may synthesize
the unique next terminal only from an otherwise canonical prefix and an
ordinary construction exception: the alternating face stream identifies S3 or
S4, full exact S4 coverage followed by no S5 identifies S5, and one exact S5
followed by no S6 identifies S6. Never synthesize from exception text, an
observer exception, a duplicate or reordered stream, missing prerequisite
evidence, or an ambiguous earlier stage. S6 uses only
`construction_native_valid`. S7 uses only
`reimport_native_valid` and `strict_valid`. Tests must include the corrected
32101 false-to-false construction/reimport case and 76198 false-to-true native
transition with strict false, proving that the inference does not rewrite the
observations into a monotonic validity sequence.

Third, implement the coordinator contract before invoking OCC. Load exactly
100 `arm == original` rows from the calibration manifest and prove 84 have the
historical strict flag. Validate the completed selector run and matrix, prove
they contain the same 100 CAD identities, strict-valid count 91, historical
84/84 preservation, zero selector worker/protocol failures, and exactly nine
strict-invalid rows. Require those nine IDs to equal the pre-registered set in
this plan. Remove exactly the constant
`EXCLUDED_EXACT_NEGATIVE_CAD_IDS={47472,63055}` and require the remainder to
equal the ordered seven-CAD constant. The code comment and run payload must
cite archive commit `afafeb81...` plus its run, row, and summary hashes; do not
add the local four-cell files as fragile runtime inputs.

Register exactly these ordered tasks:

    01  51602  primary_control
    02  61931  primary_control
    03  67160  primary_control
    04  87341  primary_control
    05  76198  primary_control
    06  95733  primary_control
    07  32101  primary_control
    08  51602  curve_interpolate_bridge
    09  61931  curve_interpolate_bridge
    10  87341  curve_interpolate_bridge

Do not expose `--max-cads`, a generic arm selector, or manifest-prefix behavior
for the formal path. Bind the exact ordered tasks, calibration and selector
file hashes, all seven source byte identities, joint iterations, timeout,
runtime `utils.py`, Python/runtime versions, repository commit, relevant code
hashes, stage schema, and unchanged schema-v2 identity in one canonical run
signature. Refuse a formal run unless `git status --porcelain` is empty and the
bound commit is a real clean commit. A dirty-tree development smoke may use a
separate explicit test-only mode, but its output can never become formal or be
resumed into the formal directory.

Fourth, implement the six-comparison source byte chain. The parent signs the
expected `bytes` and `sha256`; the child hashes the path before load; hashes the
exact bytes passed to `pickle.loads`; hashes the path again after load; and
hashes it after all measurement. Store these as
`source_binding_expected`, `source_binding_before_load`,
`source_binding_loaded_bytes`, `source_binding_after_load`, and
`source_binding_after_measurement`. After each child returns, the parent
rehashes the source into `source_binding_parent_after_child`. All six must be
identical. Terminal reopen additionally rehashes the current seven source files
as a run-level audit without inventing a seventh row field. Any mismatch is a
retained failed row or fail-closed terminal audit and makes the run
inconclusive.

Fifth, isolate all native execution. Start one fresh child per registered task
with `python -I -c` and the registered `runpy` bootstrap, capture its stdout and
stderr only in the local output root, enforce one final sentinel and a finite
timeout, and record spawn failure, timeout, native exit, malformed sentinel,
wrong task identity, stage observer error, internal matching exception, source
mismatch, and STEP implementation failure as distinct worker/protocol-failure
rows. Registered no-match, non-unique, or lineage ambiguity remains a
scientific-inconclusive row. Both classes stay in the denominator. The parent
must not parse a plausible partial line as success. Every task gets one logical
row, even when no stage is reached. Promoted local STEP and raw log names must
be attempt-unique so retry cannot overwrite an orphan; logical task IDs remain
stable so resume cannot add an eleventh denominator row.

Sixth, validate each stage and derive the census. Exact source binding requires
complete coverage of every source population that exists at that boundary, one
unique source-to-observation assignment, mutually inverse canonical
`face_edge_source_ids` and `edge_face_source_ids` once faces exist, exact
canonical endpoint labels from S2 onward, consistent incidence sums, and finite
measurements. At S2, each source edge is authoritatively bound to the just-built
standalone OCC edge and its unordered pair of local endpoint handles; a
self-loop keeps the repeated source endpoint label twice. Independent edges are
not required to share native vertex handles before sewing. At S3 and S4, every
face and edge occurrence must have one exact identity, OCC-history, or
face-local geometry assignment, but a legitimate face/wire copy is not required
to be `IsSame` to the corresponding S2 standalone edge endpoint. S2-S4 record a
separate stage-local handle census and endpoint-label coverage so copying is
visible and malformed local endpoints still fail closed; they do not claim a
global source-vertex bijection that OCC has not yet created. S5 and S6 may
claim `exact_identity` only with one unique global source-to-observed vertex
proof over the post-sewing target. Copied handles are legal, but splits, merges,
missing vertices, reconnects, or nonunique assignments are registered
scientific non-exact states. A well-formed non-exact proof retains its bounded
failure code and count instead of becoming a protocol failure. It cannot
establish exactness or first-bad by itself, but a downstream non-exact
observation also cannot erase an earlier directly localized bad boundary.

STEP reimport has no object identity, so an S7 exact claim must solve one
unique global face, edge, and vertex geometry-and-incidence assignment. Face and edge assignment
must handle both curve directions and, for closed curves, cyclic phase. Vertex
candidates derive only from source edge endpoint labels and the already mapped
STEP edges; compatibility requires the exact multiset of incident mapped source
edge IDs plus distance within the frozen normalized tolerance times shape scale.
The source WCS point for one label is derived from every optimized `edge_wcs`
first/last point carrying that ordered `edge_vertex_adj` label; all such samples
must agree within that same normalized tolerance. Raw `corner_unique` is not an
S7 correspondence oracle.
Exactly one perfect vertex assignment is required for an exact claim, and every mapped STEP edge's
unordered endpoint labels must then equal its source labels, including self-loop
multiplicity. A zero-match or multi-match assignment is `ambiguous` or
`missing`, never coerced by explorer index or nearest distance. It is scientific
non-exact evidence; an unexpected exception while computing the match is
instead promoted through `StepGeometryIncidenceMatchingError` to a worker/
protocol failure. The summary
reports per-task stage reachability, first-bad
stage and reason, native/strict transitions, mapping failures, topology drift,
source-binding failures, worker/protocol failures, and non-finite counts. It
reports primary and bridge observations separately and must contain explicit
booleans showing that it authorizes no repair, expansion, selector score
change, schema relaxation, training, sequence generation, or AR.

Seventh, finish the formal run from a clean commit. The first preferred local
root is
`D:\luolin\V13\local_runs\source_bound_stage_census_7cad_20260904_v1`.
If it already contains any artifact from another signature, do not delete or
reuse it; create the corresponding `_v2` root. Resume is allowed only when the
canonical signed JSON data model is exact-type equal. Strict parsing rejects
duplicate members, non-finite or overflowing numbers, extra status-specific
keys, and integer-to-float substitution, and the stored payload must rehash to
its stored and current signature. Validate all ten rows, each six-comparison
binding chain, local STEP size and hash where STEP exists, JSONL terminal
hash, recomputed summary, and run status before interpreting the result.

Finally, archive only compact evidence. Add a snapshot tool and report
directory only after the local run validates. The report contains a README,
path-free compact ten-row ledger, compact run contract, derived summary,
artifact manifest, and archive-validation JSON. It may contain CAD IDs,
relative logical names, schemas, booleans, counts, bounded diagnostics,
cryptographic hashes, and the clean commit. It must not contain STEP or pickle
bytes, checkpoints, NumPy arrays, raw stdout/stderr, absolute paths, upstream
source payload, or serialized OCC/native handles. Only after this archive is
independently reproducible may the stage map select one new mechanism for a new
ExecPlan and exact-CAD control/candidate experiment.

## Concrete Steps

Work from `D:\luolin\BrepARG2` in PowerShell. Before editing or running native
work, inspect the branch and all shared-agent changes rather than overwriting
them:

    git status --short --branch
    git diff --check

Use the project environment explicitly; the system Python on this machine does
not contain NumPy. Run pure and constructor-focused tests as their
implementations land:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest -q tests\test_directed_trim_assembly_stage_observer.py
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest -q tests\test_assembly_stage_lineage.py
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest -q tests\test_probe_source_bound_stage_census.py
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest -q tests\test_snapshot_source_bound_stage_census.py
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m py_compile tools\directed_trim_assembly.py tools\assembly_stage_lineage.py tools\probe_source_bound_stage_census.py tools\snapshot_source_bound_stage_census.py
    git diff --check

The focused tests must pass before a formal commit. Run the broader assembly
regression suite selected by the existing test layout and expect no regression
in historical observer, mutator, selector, schema-v2, exact-CAD, and lineage
contracts:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest -q tests\test_directed_trim_assembly.py tests\test_directed_trim_assembly_stage_observer.py tests\test_assembly_stage_lineage.py tests\test_probe_source_bound_stage_census.py tests\test_snapshot_source_bound_stage_census.py tests\test_run_exact_cad_repair_feasibility.py tests\test_assembly_selector_geometry.py

Commit the implementation only after those checks pass. Do not use a dirty
worktree or an uncommitted patch as the formal code identity. Then run the
formal census with the frozen local inputs:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools\probe_source_bound_stage_census.py --calibration-manifest D:\luolin\V13\local_runs\assembly_calibration_100cad_v1_20260809\calibration_manifest.jsonl --selector-matrix D:\luolin\V13\local_runs\assembly_selector_main_100cad_20260818\assembly_repair_matrix.jsonl --selector-run D:\luolin\V13\local_runs\assembly_selector_main_100cad_20260818\assembly_repair_run.json --breparg-root D:\luolin\V13\BrepARG --output-dir D:\luolin\V13\local_runs\source_bound_stage_census_7cad_20260904_v1 --joint-iterations 200 --worker-timeout-seconds 600

The frozen selector binds the local runtime `utils.py` to SHA-256
`e2509a844db0a9e0f8eaf670fffb9d4ad9e240af755155d25891d37b4468d521`.
If `D:\luolin\V13\BrepARG\utils.py` does not have that identity, supply the
exact local runtime root that does. The coordinator must reject a mismatch
instead of silently using another checkout. A successful terminal transcript
should have the following shape; hashes and individual stage results are
discovered values:

    status=COMPLETED
    denominator_rows=10/10
    primary_controls=7/7
    curve_interpolate_bridges=3/3
    source_binding_failures=0
    worker_or_protocol_failures=0
    nonfinite_count=0
    summary_recomputed_equal=true
    authorizes_exact_candidate_design=true|false
    authorizes_residual_expansion=false
    authorizes_full_100cad=false
    authorizes_training_or_ar=false

Re-running the identical command may resume missing tasks or verify an already
terminal run. Any changed source byte, manifest, selector ledger, code, commit,
runtime, task order, joint iteration count, timeout, stage schema, or bridge
configuration must be rejected and requires a new unused output root.

After terminal validation, run the snapshotter and its focused test using the
names implemented in the same milestone. The snapshot command must take the
local run root as input and a new repository-relative
`reports/source_bound_stage_census_7cad_20260904/` directory as output. Before
commit, search that directory for drive-letter paths, `.step`, `.pkl`, raw log
names, and native-handle representations, then run `git diff --check`. Commit
and push the compact evidence to the current branch only after archive
validation reports `valid=true`.

## Validation and Acceptance

Implementation acceptance requires tests that prove all of the following
observable behaviors. With no new observer, existing constructor output and
diagnostics remain unchanged. With an observer, the edge event stream is
exactly S1(0), S2(0), S1(1), S2(1), and the face event stream is exactly S3(0),
S4(0), S3(1), S4(1), with complete source coverage whenever construction
finishes. S5 and S6 each occur once. The observer cannot replace a target, and
its exception fails closed with the exact stage. The pure normalizer rejects
reordered, duplicated, missing, non-finite, and ambiguous stage evidence. It
recognizes a proved exact prefix as preceding evidence, a proved local terminal
as the current first-bad candidate, and never treats a partial prefix as a full
topology census. The parent process remains able to import and validate the
runner in an environment where importing OCC-dependent modules is made to fail,
proving that parent orchestration is OCC-free.

Failure-path tests must prove that a curve-fit failure emits one explicit S1
terminal and a MakeEdge failure emits one explicit S2 terminal, both as the
last event for that task. They must prove S3/S4 terminal synthesis only from a
canonical alternating prefix plus an ordinary constructor error, and reject
synthesis after an observer exception, malformed sequence, ambiguous S1/S2
lineage, duplicate event, or reordered event. They must likewise prove that S5
can be synthesized only after full exact S4 coverage and that S6 can be
synthesized only after one exact S5; duplicate S5/S6 events or an ambiguous
prerequisite are inconclusive. A passed S1 or S3 prefix cannot be named
first-bad when S2 or S4 owns the terminal.

Lineage tests must prove S2's two local endpoint occurrences belong to the
authoritatively constructed source edge, including the repeated source label of
a self-loop. They must prove that two independently built edges sharing a source
vertex can have distinct endpoint handles without losing exact source-edge
binding; that S3/S4 can retain one unique source-edge occurrence mapping even
when face construction copies edge or vertex handles; and that malformed local
endpoints or an ambiguous face-local mapping still fail closed. An S5/S6 exact
claim must have exactly one global source-to-observed vertex matching, and an
S7 exact claim must have exactly one global face, edge, and vertex assignment
with every endpoint relation restored. Registered missing, ambiguous, or
nonunique S5/S6 proof states must remain archivable scientific non-exact
evidence, and a downstream exact-claim tamper must still be rejected. A genuine
split, merge, reconnect, missing entity, or nonunique assignment at a stage
whose topology is globally materialized must remain scientifically non-exact
without being mislabeled as a worker/protocol failure.

Protocol tests must prove exact derivation `100 -> 9 -> exclude 2 -> 7`, exact
ten-task order, no user option that changes the formal denominator, and strict
separation of primary controls from the three bridges. They must prove that a
bridge uses `directed_trim + curve_interpolate`, not `curve_fit_rescue`, and
that even a native/strict-valid bridge cannot increment restored counts or
authorize a selector change. They must also prove timeout, crash, spawn error,
malformed/multiple sentinel, stage callback error, missing STEP, source-binding
drift, and torn final JSONL each retain or recover a single logical denominator
row without being interpreted as success. Registered mapping ambiguity must
remain a scientific-inconclusive denominator row, while an internal S7
matching exception must be upgraded to a worker/protocol-failure row.

Transport and runtime tests must reject duplicate JSON members, `NaN`,
`Infinity`, overflowing exponents, extra manifest keys, and equal-valued
integer/float substitution. Resume must reject a stored payload whose self-
signature no longer verifies. The worker command must have the exact
`python -I -c` bootstrap prefix, and importing the parent coordinator must not
load NumPy or OCC. In the worker, same-process runtime measurement must precede
input selection and all source/scientific work; drift or measurement failure
must yield a worker failure with a null row sentinel. Scientific rows with a
missing, altered, or extra sentinel field must be rejected. Tests must also
separate a registered S7 no-match from an internal matching exception.

The formal evidence is conclusive only when there are exactly ten valid rows
for the registered tasks, all six source-binding comparisons match in every
scientific row, code and input hashes match the signed clean commit, all numeric
evidence is finite, every scientific row's worker sentinel is exact-type equal
to the signed and frozen representative ABI sentinel, every worker/protocol
failure row keeps that field null, and the terminal summary is exactly
reproducible from the ledger. A primary row can support a named
first-bad stage only when every required preceding stage is complete and exact
or is a validated distributed exact prefix pass, the terminal or invalid
transition is directly observed under the rules above, and all required source
face, edge, occurrence, endpoint, and vertex correspondence is unique.
A bridge can reveal later reachability but cannot establish that the primary
path was repaired. Incomplete evidence remains explicitly inconclusive.

No result from this census passes the assembly release gate. The only positive
authorization it may issue is `authorizes_exact_candidate_design=true`, and
only when at least one row exposes an exact, localizable transition that admits
a topology- and 3D-geometry-preserving hypothesis. The next experiment must be
one new exact-CAD control/candidate pair with immutable gates and a separate
ExecPlan. It must not call `FixRemovePCurve`,
`post_pcurve_face_mutator`, or `post_sewing_shape_mutator` from the closed
experiment. An exact candidate must independently save and re-read STEP, be
native true, project strict true, both-valid true, retain exact source mapping,
and pass all existing schema-v2 checks before any residual-family expansion.

Only after that exact candidate passes may it expand in order to its relevant
residual family, then the frozen invalid-16 cohort, then the fixed 100 CADs.
Release still requires at least 95/100 strict-valid, historical controls 84/84,
zero regressions, zero worker/protocol failures, zero non-finite measurements,
and every selected repair accepted by unchanged schema-v2. Training remains
blocked until those assembly conditions and the repaired-chain representation
comparison pass.

## Idempotence and Recovery

The parent holds a nonblocking writer lock for a local output root. The first
formal invocation creates its signed manifest only in an otherwise empty root.
Each JSONL append is flushed and fsynced. Resume may remove only an unterminated
torn final line; it validates every complete row and any local STEP size/hash
before skipping a task. It never overwrites a completed logical task or changes
the ten-row order. Attempt-unique physical names prevent a retry from replacing
an orphan left between child exit and ledger append.

Only a `RUNNING` ledger may recover one final JSONL fragment that lacks its
terminating newline. A terminal manifest and terminal ledger are immutable and
must never use torn-tail recovery. Every manifest is strictly parsed, checked
against its exact status-specific key set, and accepted only when the stored
payload's canonical self-hash, stored signature, and current signature agree
under exact JSON types.

Do not delete an unexpected artifact or alter an old run to make validation
pass. Preserve it for diagnosis and choose `_v2` for a different signed run.
Timeout and native crash are data, not reasons to restart the whole cohort.
Re-run only the missing identical task through signature-compatible resume.
Never recursively clean `D:\luolin\V13\local_runs` or any repository root.

The formal parent is safe to retry because it performs no OCC work and no shape
mutation. The observer is default-off, so removing the census callback restores
the exact historical constructor path without a migration. Git snapshotting is
also idempotent only for byte-identical input; a different local run must use a
new report directory or be rejected.

## Artifacts and Notes

The run payload must retain the exact negative evidence identity without adding
the local negative run as a new input:

    exact_negative_archive_commit = afafeb81e1674078aa4e08c2987f4343d4734808
    exact_negative_run_signature = 1d4f68839aadc8b3f8fb38eea642a1f7ea4f6d8d51b61152f943c725832ffcad
    exact_negative_rows_sha256 = f158f2ca7f9bf2adceb7a56434ca4925bed99e34d5791e8867ca476f32d70a34
    exact_negative_summary_sha256 = 545802b4e783a3f3f76039d70e983fba1ab5eb29af0748e5db032e731c925f60
    excluded_cads = [47472, 63055]

The formal local output should contain a signed run manifest, a ten-row JSONL
ledger, a derived summary, attempt-unique local worker logs, and STEP files only
for cells that reached S7. Exact filenames are implementation constants and
must be recorded here when finalized. The Git-safe archive must replace local
paths with logical task or artifact names while keeping source/STEP identities
as hashes and byte counts.

The following historical observations are orientation only, not acceptance
expectations: 51602, 61931, and 87341 stop as assembly errors under the primary
path; 67160 has no final solid; 32101 is construction-native false and reimport-
native false; 76198 is construction-native false but reimport-native true and
strict false; 95733 remains strict-invalid. The formal census may correct or
refine a causal explanation, but it must never rewrite these values merely to
fit a monotonic model.

## Interfaces and Dependencies

In `tools/directed_trim_assembly.py`, the final constructor interface includes:

    construct_brep_directed(
        ...,
        *,
        assembly_stage_observer: Callable[[Any, Mapping[str, Any]], None] | None = None,
        ...,
    )

The observer is called as `assembly_stage_observer(observation_target,
metadata)`. `metadata["stage"]` uses only the six in-memory names listed in
this plan. S7 is forbidden inside the constructor and is added by the child
after STEP reimport. The observer's return value is ignored. Neither existing
mutation hook is passed by the census.

In `tools/assembly_stage_lineage.py`, define these pure interfaces:

    STAGE_ORDER: tuple[str, ...]
    normalize_stage_record(record: Mapping[str, Any], ...) -> dict[str, Any]
    validate_stage_sequence(records: Sequence[Mapping[str, Any]], ...) -> dict[str, Any]
    assess_stage_lineage(records: Sequence[Mapping[str, Any]], ...) -> dict[str, Any]
    infer_first_bad_stage(records: Sequence[Mapping[str, Any]], ...) -> dict[str, Any]

Exact optional parameters may be refined during implementation, but every
function must remain OCC-independent, deterministic, JSON-safe, and fail-
closed. It cannot turn ambiguous correspondence into `exact` or infer a missing
validity value.

In `tools/probe_source_bound_stage_census.py`, define immutable task and schema
constants plus public helpers for building the run payload, binding or resuming
the manifest, running one child, validating one row, deriving the summary, and
validating terminal hashes. Follow the proven runner pattern in
`tools/probe_downstream_bad_wire_lineage.py` and
`tools/run_exact_cad_repair_feasibility.py`, but do not import either module if
doing so would transitively import OCC in the parent. Reuse pure JSON, hashing,
writer-lock, source-binding, and schema-v2 helpers where their import graph is
safe.

The only external native dependency is the already installed pythonocc/OCC
runtime used by the local BrepARG assembly chain. Do not add a package or
network dependency. Cryptographic identities use SHA-256. JSON serialization
must set `allow_nan=False`; parsing must reject duplicate members, non-finite
constants, finite-looking overflow, unexpected keys, and type-substitution.
The runtime proof is the representative ABI sentinel named above, not an
inventory of every lazy-loaded `.pyd` and `.dll`. Formal workers use the
isolated bootstrap and include same-process proof in each scientific row. The
parent uses Python's subprocess and filesystem APIs and remains portable enough
to turn a Windows access violation into an ordinary failed denominator row.

Revision note 2026-09-04 16:56 +08:00: created after the formal exact-CAD
four-cell experiment conclusively rejected the registered 47472 and 63055
`FixRemovePCurve` candidates. This revision freezes the non-overlapping
seven-CAD cohort, seven-control plus three-bridge denominator, read-only seven-
stage contract, six-comparison source binding, clean-commit execution rule, and
exact-CAD-before-expansion gate so the next repair is chosen from causal
evidence rather than another broad mutation sweep.

Revision note 2026-09-04 18:29 +08:00: updated during implementation after a
protocol audit rejected the first globalized observer as a causal confound. The
plan now defines S1/S2 and S3/S4 as historical-order distributed edge and face
events, separates an exact prefix pass from a local exact failure, constrains
explicit and synthesized terminal evidence, requires vertex lineage from S2
through S7, records the real-OCC and baseline/current A/B smokes, and states the
remaining test and formal-run work without claiming that the census has run.

Revision note 2026-09-04 19:18 +08:00: strengthened the vertex contract after
the endpoint-lineage audit showed that per-edge checks cannot exclude a shared
vertex split or distinct-vertex merge. The plan now requires a unique global
proof for every nonempty completed S2-S4 population, derives S7 WCS vertices
from optimized edge endpoints at the STEP scale, and freezes topology repair
off so source endpoint labels remain the actual constructor labels. The S2-S4
global-proof portion of this historical revision is superseded by the 20:20
revision below; it is retained only to document why the design changed.

Revision note 2026-09-04 20:20 +08:00: corrected the pre-sewing vertex contract
after the full real-OCC 66307 roundtrip showed that standalone `MakeEdge`
objects and face/wire construction legally use copied native vertex handles.
The plan now separates exact source-edge/face binding from stage-local handle
population at S2-S4, retains the unique global vertex gate at S5-S7, records the
real STEP split from 86/58 to 90/60 edges/vertices, requires a whole-shape
terminal for S5/S6 inferred exceptions, and closes selector-health,
frozen-input-SHA, and persistent-local-STEP evidence gaps before the formal run.

Revision note 2026-09-04 23:40 +08:00: closed the implementation protocol
before the formal run. This revision records strict JSON and manifest self-
signature validation, status-specific non-exact S5/S6 proof schemas, escalation
of internal S7 matching exceptions, six source-byte comparisons, the honest
representative ABI sentinel boundary, `python -I -c` worker bootstrap, and
same-process runtime proof before any source or scientific work. It also records
the conclusive 66307 development smoke (`first_bad_stage=S6`, later STEP native
and strict validity with 86/58/172 to 90/60/180 topology drift) and the current
`456 passed, 2 warnings` combined regression result. The signed formal census
remains unrun at this revision.
