# Diagnose closure and single-shell assembly failures

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current while work proceeds. It follows the checked-in root `PLANS.md`. `AGENTS.md` names `.agent/PLANS.md`, but that path is absent in this worktree, so `PLANS.md` is the available authority.

## Purpose / Big Picture

The frozen 100-CAD assembly selector is currently strict-valid for 91 CADs. Three of the nine remaining failures are `00061931_dcdd8a95feac4121adfd341f_step_000`, `00087341_6a73c5e821934d3fe4d0d555_step_000`, and `00095733_8b325d2fcb27ec9e79388602_step_000`. The first two historically fail while fitting an edge curve before STEP export; fallback construction later reaches STEP but contains self-intersecting wires. The third reaches STEP but has one self-intersecting wire and no unit solid. This plan determines exactly whether each defect first appears during curve construction, face/wire construction, sewing, solid construction, or STEP roundtrip. A narrow repair may be retained only when it produces a STEP-readable, Open CASCADE native-valid, project strict-valid, both-valid result while preserving the frozen schema-v2 topology and geometry signature.

The observable outcome is a local, path-aware diagnostic run for all three CADs plus Git-safe compact JSON evidence. A successful repair prototype additionally has focused tests and a one-CAD transcript proving the unchanged geometry/topology gate. No 100-CAD run is permitted in this plan. A 16-invalid run is allowed only after at least one target CAD passes every one-CAD gate and the repair is fail-closed on non-applicable inputs.

## Progress

- [x] (2026-08-18 04:20 +08:00) Read `AGENTS.md` and the complete root `PLANS.md`, created the isolated `protocol-v5-closure-shell-probe` worktree at clean selector commit `0d49473754aa0a3f06a9e2290a76671e6e19bec2`, and verified that `D:\luolin\V13\BrepARG` remains outside the branch.
- [x] (2026-08-18 04:20 +08:00) Reconfirmed the frozen evidence: `00061931` and `00087341` fail at `curve_fit`; `00095733` saves STEP but reports native invalid, one wire self-intersection, and zero solids.
- [x] (2026-08-18 04:40 +08:00) Added a read-only stage probe that records post-joint topology, per-edge curve-fit outcome, per-face wire construction, pre-sewing compound diagnostics, post-sewing shell/solid diagnostics, and bounded child-process failures. The probe exposes a default-off observer and does not change the default assembler path.
- [x] (2026-08-18 05:05 +08:00) Ran independent workers for the three target CADs and classified the first defective stage with compact machine-readable evidence. `00061931` and `00087341` first fail at `face_raw` after curve interpolation; `00095733` first fails at `face_raw` on faces 0 and 26. All three have zero 3D adjacent-endpoint gaps, so the defect is a 2D pcurve/face-boundary self-intersection rather than loop endpoint closure.
- [x] (2026-08-18 05:30 +08:00) Tested periodic pcurve branches, local pcurve continuity, and post-sewing pcurve reprojection as isolated, fail-closed prototypes. Periodic branches are inapplicable because the target surfaces are not periodic. Local continuity either changed topology or left self-intersections. Reprojection remained incomplete on the full sewn shape and was rejected by the unchanged three-dimensional-curve gate on the isolated face-26 case.
- [x] (2026-08-18 05:45 +08:00) Tested graph-preserving trim alternatives on the two topology-changing cases in a separate worktree. Historical OCC-valid candidates depended on source vertex/edge deletion or merge; no topology-preserving candidate reached strict validity. Twelve unsafe OCC attempts crashed with Windows `0xC0000005` in isolated workers and are excluded from production.
- [x] (2026-08-18 06:00 +08:00) No narrow prototype passed STEP-readable, native, strict, both-valid, schema-v2, and roundtrip gates. The negative diagnosis is complete; no invalid16 or 100-CAD follow-up is authorized by this plan until a one-CAD candidate passes every gate.
- [x] (2026-08-18 05:00 +08:00) Archived the compact negative report under `reports/assembly_closure_shell_negative_20260818/`; it contains no STEP, pickle, array, checkpoint, or machine-local path.
- [x] (2026-08-18 05:05 +08:00) Added an explicit `--cad-id` worker binding and regression test after an arbitrary-CAD probe exposed a fail-closed input-selection omission. Focused tests pass; the branch is ready for a selective commit and push.
- [x] (2026-08-18 06:45 +08:00) Re-ran the final signed six-attempt matrix after making observation-only construction conditional on `stage_observer`. The run completed 6/6 rows with zero worker/protocol failures, zero both-valid candidates, and maximum oriented 3-D endpoint gap 0.0 on every bad raw face.
- [x] (2026-08-18 07:00 +08:00) Added the Git-safe negative report, source/run hash bindings, crossing taxonomy, repair checklist, and ADR-0004. No STEP, pickle, NumPy, checkpoint, raw geometry, or machine-local path is archived.
- [x] (2026-08-18 07:10 +08:00) Final verification passed: 80 focused tests, Python compilation, JSON/JSONL parsing, forbidden-extension/size/path scan, and scoped `git diff --check`. The report artifact manifest binds every archived evidence file.
- [ ] Commit and push only the explicitly scoped files; leave unrelated log line-ending noise and the parallel closure-shell draft unstaged.

## Surprises & Discoveries

- Observation: The two nominally "closure" cases `00061931` and `00087341` do not share the same historical first failure as `00095733`.
  Evidence: `reports/p0a_assembly_chain_evidence_20260817/assembly_chain_cases.jsonl` classifies the first two as `pre_step:curve_fit`, while `00095733` has a saved STEP with `wire_self_intersection` and `nonunit_solid_count`.

- Observation: Existing broad fallback profiles reach STEP for all three but do not establish a safe repair.
  Evidence: `reports/assembly_repair_pilot_switches_20260817/assembly_repair_attempts.jsonl` records two self-intersections for `00061931`, four for `00087341`, and one for `00095733` under `curve_fit_fallback`; none is strict-valid.

- Observation: `00061931` has stable self-intersections on faces 0 and 23, while `00087341` has multiple self-intersecting faces (0, 2, 12, 16-26, 29, and 30). Both survive sewing and have a single shell with no final solid.
  Evidence: stage probe summaries in `D:\luolin\V13\local_runs\closure_shell_stage_probe_20260818_v2` through `_v4` and crossing probe `closure_shell_crossing_probe_20260818_v1`.

- Observation: `00095733` has two raw-stage self-intersections before sewing: face 0 has a non-adjacent crossing at edge positions 8-10 and face 26 has a closure crossing at positions 19-1. All adjacent 3D endpoint gaps are exactly zero; sewing leaves one shell and no roundtripped solid.
  Evidence: `D:\luolin\V13\local_runs\closure_shell_crossing_probe_20260818_v1` and `closure_shell_reproject_95733_20260818_v1`.

- Observation: The candidate repair families are fail-closed by the existing gates. Periodic pcurve repair reported `surface_not_periodic`; local pcurve continuity either changed edge counts or left self-intersections; post-sewing reprojection was incomplete for the full shell and rejected by `three_dimensional_curve_changed` on face 26.
  Evidence: `D:\luolin\V13\local_runs\closure_shell_periodic_probe_20260818_v1`, `closure_shell_local_pcurve_probe_20260818_v1`, and `closure_shell_face_reproject_95733_20260818_v1`.

## Decision Log

- Decision: Diagnose each construction stage before changing any assembler behavior.
  Rationale: A curve-fit exception, a malformed wire, a multi-shell sewn shape, and a STEP-roundtrip defect require different repairs. Treating them as a single closure problem risks topology mutation.
  Date/Author: 2026-08-18 / Codex closure-shell probe.

- Decision: Use the existing frozen original-control manifest and existing schema-v2 gate implementation rather than constructing a new cohort or relaxing thresholds.
  Rationale: The current 91/100 result and the 84 historical controls are meaningful only when source bytes, topology semantics, geometry metrics, and strict-validity definitions remain identical.
  Date/Author: 2026-08-18 / Codex closure-shell probe.

- Decision: Keep every Open CASCADE operation inside a one-CAD child process with a timeout and structured sentinel result.
  Rationale: OCC may terminate the process for malformed topology. Isolation keeps the denominator and diagnostic evidence intact.
  Date/Author: 2026-08-18 / Codex closure-shell probe.

- Decision: Do not modify or stage anything under `D:\luolin\V13\BrepARG`.
  Rationale: That directory is upstream source explicitly excluded from this project branch. It may be imported at runtime only.
  Date/Author: 2026-08-18 / Codex closure-shell probe.

## Outcomes & Retrospective

The stage diagnosis is complete and reproducible, but no repair outcome is established. The final six-attempt run completed with zero worker failures and zero both-valid candidates. The three target defects are face-boundary/pcurve failures, and all tested repair families were rejected without changing the default production behavior or relaxing schema-v2. Compact evidence is archived in `reports/closure_shell_failure_negative_20260818/`; a 100-CAD run remains blocked until a one-CAD candidate passes all gates.

## Context and Orientation

`tools/run_assembly_repair_matrix.py` is the fixed-cohort coordinator. It hashes each source pickle, runs high-risk profiles through `run_one_isolated`, writes local STEP files beneath an untracked output directory, and records STEP readability plus native, strict, and both-valid outcomes. `tools/directed_trim_assembly.py` is the repository-owned experimental assembler; it consumes surface grids, edge grids, face-edge adjacency, and edge-vertex adjacency after CPU joint placement. `tools/run_assembly_calibration_oracle.py::cpu_joint_optimize` performs that placement. `tools/diagnose_step_validity_components.py::diagnose_step` applies the project strict diagnosis after STEP reimport. `tools/assembly_selector_geometry.py` computes the schema-v2 input and candidate signatures and applies the fixed `geometry_topology_gate`.

"Construction" means creation of 3D edge curves and trimmed faces. "Sewing" is the Open CASCADE operation that joins faces into shells. A "unit solid" means exactly one solid results. "STEP roundtrip" means writing a candidate to STEP, reading those bytes back with Open CASCADE, and re-running native and project strict validation. "Fail-closed" means an ambiguous, non-applicable, timed-out, crashed, topology-changing, or geometry-changing attempt returns no accepted candidate.

The frozen source rows come from `D:\luolin\V13\local_runs\assembly_calibration_100cad_v1_20260809\calibration_manifest.jsonl`. Their source pickle paths point into the local Protocol V5 materialized validation set. Source pickle bytes, reconstructed arrays, and STEP files are local-only. Compact JSON may record CAD ids, stage names, counts, scalar distances, booleans, SHA-256 hashes, byte sizes, and rejection codes; it must not archive machine paths, raw pickle contents, STEP bytes, arrays, checkpoints, or upstream source.

## Plan of Work

First, add `tools/probe_closure_shell_stages.py`. Its parent CLI selects exactly the three registered CAD ids from the frozen manifest and starts a fresh child process for each CAD. The child verifies the exact source-pickle hash, deserializes the CAD, applies the unchanged CPU joint optimizer, and builds the shape through repository-owned assembly functions while recording checkpoints before and after curve construction, face/wire construction, sewing, solid construction, STEP write, and STEP reimport. The child prints one JSON sentinel; the parent records timeout, exit-code, missing-sentinel, and malformed-result failures without interpreting a crash as a geometric result.

The probe must avoid raw in-process OCC objects in its report. For each checkpoint it records only counts and validity facts: faces, edges, vertices, wires, shells, solids, free edges, wire order failures, wire self-intersections, failed entity id, exception type, and a stable failure code. Where construction has not yet reached STEP, it may write a temporary local BREP or STEP only inside the untracked output directory for subprocess diagnosis; it records only size and SHA-256 in compact evidence.

Second, run the probe once per target CAD with `C:\Users\YU\.conda\envs\brepgen_env\python.exe`, `D:\luolin\V13\BrepARG` supplied as a runtime import root, and a new directory beneath `D:\luolin\V13\local_runs`. Compare the first failing checkpoint across the three cases. If no operation can preserve graph identity, record a negative result and do not invent a repair.

Third, only if stage evidence gives a narrow operation, add it behind an independent switch in repository-owned code. The switch must be applicable only to the diagnosed failure signature, copy topology before mutation, preserve every face/edge/vertex incidence multiset, preserve all sampled 3D curves, and use the existing schema-v2 gate after STEP reimport. Non-applicable and rejected repairs must preserve the original attempt rather than become worker failures.

Fourth, validate the target one-CAD result. Promotion requires STEP readability, native validity, strict validity, both-valid, zero wire self-intersections, exactly one solid, a positive unchanged schema-v2 gate, and no child-process/protocol failure. Only then run the same profile on the frozen invalid16 subset. Do not run the 100-CAD matrix in this worktree.

## Concrete Steps

Work from `D:\luolin\V13\.worktrees\closure-shell-probe-20260818`.

Inspect the frozen rows and existing stage evidence:

    Select-String -Path reports\p0a_assembly_chain_evidence_20260817\assembly_chain_cases.jsonl -Pattern '00061931|00087341|00095733'

Run pure-Python focused tests after adding the probe contract:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m pytest tests\test_probe_closure_shell_stages.py -q

Run the three-CAD parent probe. The parent, not the calling shell, starts each OCC child:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe tools\probe_closure_shell_stages.py --calibration-manifest D:\luolin\V13\local_runs\assembly_calibration_100cad_v1_20260809\calibration_manifest.jsonl --breparg-root D:\luolin\V13\BrepARG --output-dir D:\luolin\V13\local_runs\closure_shell_stage_probe_20260818 --cad-id 00061931_dcdd8a95feac4121adfd341f_step_000 --cad-id 00087341_6a73c5e821934d3fe4d0d555_step_000 --cad-id 00095733_8b325d2fcb27ec9e79388602_step_000 --worker-timeout-seconds 600

The expected report contains exactly three final rows, one source binding per row, zero duplicate CAD ids, and an explicit first-defective-stage classification. A crashed or timed-out child remains an explicit failed row.

After any prototype, run its focused tests and the one-CAD probe. Run invalid16 only after all target gates pass. Finish with:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m pytest tests\test_probe_closure_shell_stages.py tests\test_assembly_repair.py tests\test_run_assembly_repair_matrix.py -q
    git diff --check
    git status --short

## Validation and Acceptance

The diagnostic milestone passes when one parent invocation yields exactly one structured row for each target CAD, all source bindings match the bytes consumed by the child, every OCC invocation is a child process, and every case has an explicit first defective stage or an explicit bounded worker failure. Local STEP/BREP files may exist only under the untracked output root.

A repair prototype passes the one-CAD milestone only when the written STEP reimports, native validity is true, project strict validity is true, both-valid is true, wire self-intersection count is zero, solid count is exactly one, and the unchanged schema-v2 geometry/topology gate is true. A candidate that changes effective face, edge, or vertex counts, incidence multisets, edge curves, bounds beyond the existing threshold, boundary residuals beyond the existing threshold, or fails any sample is rejected even when Open CASCADE calls it valid.

An invalid16 pilot is allowed only after a one-CAD pass. It must complete all 16 rows with zero worker/protocol failures and zero regression against any already recovered candidate semantics. This plan explicitly does not authorize a 100-CAD run, a gate relaxation, boundary-consistency training, sequence generation, or AR training.

## Idempotence and Recovery

The probe uses a new output root or verifies an immutable run signature before resuming. It appends one complete JSON line only after validating a child sentinel and source binding. A torn final line may be removed, but prior complete rows are immutable. Repeating the exact command resumes missing CADs; changing a source hash, code hash, CAD list, timeout, or runtime identity requires a new output root. Child timeouts terminate only that child and remain in the denominator.

No cleanup command deletes source data. Temporary STEP/BREP output is local and recoverable by deleting only the explicitly named local run directory after its compact hashes have been captured. This plan never writes to `D:\luolin\V13\BrepARG`.

## Artifacts and Notes

Frozen starting evidence:

    00061931: baseline first failure curve_fit; fallback STEP has two SI wires
    00087341: baseline first failure curve_fit; fallback STEP has four SI wires
    00095733: baseline STEP exists; native false; SI wires 1; solids 0

Current assembly release state outside this worktree:

    strict valid: 91/100
    original strict controls preserved: 84/84
    release requirement: at least 95/100 and zero control regression

## Interfaces and Dependencies

`tools/probe_closure_shell_stages.py` must expose a parent `main(argv)` and a hidden `--worker-cad-id` mode. The parent launches the worker with `subprocess.run`, a finite timeout, captured stdout/stderr, and a unique expected JSON sentinel. The worker returns only JSON-safe values. Parent and worker both verify a `{bytes, sha256}` source binding before and after deserialization.

The probe may import repository-owned functions from `tools/run_assembly_repair_matrix.py`, `tools/directed_trim_assembly.py`, `tools/run_assembly_calibration_oracle.py`, `tools/diagnose_step_validity_components.py`, and `tools/assembly_selector_geometry.py`. It may add `D:\luolin\V13\BrepARG` to `sys.path` at runtime to use installed upstream utilities, but it must not patch or write that tree.

Any repair switch must be declared in `tools/assembly_repair.py`, implemented in repository-owned modules, routed through the existing isolated-worker predicate, and covered by a pure contract test plus a real one-CAD subprocess result. The existing `geometry_topology_gate` threshold and schema must remain byte-for-byte unchanged.

Revision note 2026-08-18 04:20 +08:00: Created the plan after isolating the worktree and reconfirming the three target failure signatures. It freezes stage-by-stage diagnosis, child-process containment, schema-v2 preservation, one-CAD-first promotion, local artifact policy, and the explicit prohibition on a 100-CAD run.

Revision note 2026-08-18 06:00 +08:00: Recorded the completed stage probe and the negative results for periodic pcurve, local continuity, post-sewing reprojection, and graph-preserving trim. The plan now explicitly states that all three targets fail in face/pcurve construction, that no candidate is promotion-eligible, and that the next contributor must archive evidence rather than start a broader matrix.

Revision note 2026-08-18 07:00 +08:00: Bound the conclusion to the final six-attempt signed run (`d560f963403dc9fc7f4b52449598858d6019b381a538892fd2bf7fa7f3e2fe02`), recorded the zero 3-D endpoint-gap measurement for every bad raw face, added ADR-0004 and the Git-safe report, and left only final tests plus push outstanding.

Revision note 2026-08-18 07:10 +08:00: Recorded 80 passing focused tests and the completed archive safety validation. Only a selective commit and push remain.
