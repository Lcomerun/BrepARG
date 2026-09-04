# Run an immutable exact-CAD repair feasibility matrix

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain
current while work proceeds. This document follows `PLANS.md` in the
repository root.

## Purpose / Big Picture

The current assembly selector is project-strict-valid for 91 of the fixed 100
CADs and must reach at least 95 without losing any of the 84 historical valid
controls. Signed lineage evidence identifies two concrete residual failures,
but neither a successful face-local operation nor a manually valid STEP is
enough to authorize a production repair. After this change, a contributor can
run one fixed four-attempt experiment that independently measures the unchanged
control and proposed repair for each of those two CADs. A native crash,
timeout, missing integration hook, malformed child result, or incomplete proof
is retained as a false denominator row instead of disappearing.

The observable result is an immutable local directory containing one signed
run manifest, exactly four JSONL rows, a derived summary, local STEP files,
and worker logs. The JSON results explicitly state whether each control
reproduced its known failure and whether each candidate was both-valid and
accepted by the unchanged schema-v2 geometry/topology gate. This exact pilot
can authorize only a follow-up on the relevant residual family; it never
authorizes the fixed 100-CAD cohort, model training, sequence generation, or
AR by itself.

## Progress

- [x] (2026-09-04 01:20 +08:00) Replaced the truncated 222-line draft with a
  complete importable coordinator in
  `tools/run_exact_cad_repair_feasibility.py`.
- [x] (2026-09-04 01:40 +08:00) Implemented exact two-CAD lineage and source
  binding, a fixed control/candidate 2x2 matrix, one fresh child per cell,
  strict sentinel parsing, and false denominator rows for every process and
  protocol failure.
- [x] (2026-09-04 02:00 +08:00) Added independent STEP validation, native and
  project-strict validity, schema-v2 geometry/topology validation, immutable
  run signatures, writer locking, artifact hashes, and exact row-level resume.
- [x] (2026-09-04 02:10 +08:00) Added focused protocol tests in
  `tests/test_run_exact_cad_repair_feasibility.py`; the initial suite reports
  `10 passed`.
- [x] (2026-09-04 02:45 +08:00) Integrated both low-level candidates through
  the new constructor mutation hooks. The focused runner, constructor,
  47472-helper, and 63055-helper suite reports `70 passed, 2 warnings`; the
  warnings are existing pythonocc `topods_Wire` deprecations.
- [x] (2026-09-04 15:35 +08:00) Added terminal ledger/summary SHA-256 and
  derived-content revalidation so a completed run cannot be resumed after its
  small evidence files are modified.
- [x] (2026-09-04 16:05 +08:00) Closed the two final integration audit gaps.
  The 63055 helper now validates a pre-MakeSolid shell through a disposable
  single-solid wrapper without changing the returned shell, and rejects
  repeated occurrence positions or non-finite pcurve removal evidence. The
  47472 adapter now derives whole-CAD defect, mapping, and shared-edge claims
  only from an exact post-STEP geometry/incidence match over the complete
  source-face census; the known STEP-only face-1 defect is therefore inside
  the acceptance denominator.
- [x] (2026-09-04 16:10 +08:00) Re-ran the final focused implementation suite:
  `171 passed, 2 warnings`; `py_compile` and `git diff --check` also pass. The
  warnings are existing pythonocc `topods_Wire` deprecations.
- [ ] From a clean implementation commit, run the four formal isolated
  attempts into a new unused directory under the machine-local experiment
  root outside this repository.
- [ ] If and only if an exact candidate passes, expand it to its related
  residual family, then frozen invalid16, then the fixed 100 CADs.

## Surprises & Discoveries

- Observation: the inherited runner draft was syntactically truncated inside
  `EXPECTED_LINEAGE` and could not be imported.
  Evidence: it ended at line 222 with `CAD_630-HERE: {}`. Replacing the draft,
  rather than attempting to preserve its partial structure, was required for
  a testable protocol.

- Observation: callback self-report is insufficient for acceptance even when
  it contains valid-looking booleans.
  Evidence: the runner deliberately ignores callback validity and geometry
  claims. It re-reads the callback STEP through
  `strict_validate_step`, builds a new candidate signature, and invokes
  `geometry_topology_gate`; a test proves that `candidate_accepted` with a
  rejected geometry gate is refused.

- Observation: a missing full-CAD adapter is scientifically different from a
  negative repair result.
  Evidence: `candidate_hook_missing` makes the four-cell summary inconclusive,
  while a completed callback whose independently measured candidate fails is
  `candidate_rejected` and can contribute to a conclusive negative result.

- Observation: promoted native artifacts need an attempt-unique name, not
  only a task-derived name.
  Evidence: interruption after STEP promotion but before JSONL append would
  otherwise leave an unreferenced file at the deterministic retry target. The
  runner now includes the fresh child directory token in local log and STEP
  names, so retry cannot overwrite an orphan.

- Observation: the 63055 post-sewing hook receives a shell because it runs
  before the constructor creates a solid, while the first helper draft applied
  a strict validator that required exactly one solid.
  Evidence: a real OCC box-shell smoke measured zero solids on the hook input;
  the validation-only wrapper measured one strict/native-valid solid while the
  original and returned candidate remained shells. Without this separation the
  formal 63055 candidate would have been rejected regardless of its repair.

- Observation: face-local success cannot support a whole-CAD 47472 defect
  claim, even when the coordinator independently applies strict and schema-v2
  validity later.
  Evidence: signed lineage already shows source face 1 is clean in memory but
  acquires edge pair `[10,12]` only after STEP roundtrip. The adapter now reuses
  the full STEP geometry/incidence matcher and accepts defect booleans only
  when the final source-indexed occurrence inventory is empty.

## Decision Log

- Decision: freeze exactly four ordered tasks instead of exposing a generic
  CAD or variant filter.
  Rationale: `--max-cads` or manifest-prefix selection can silently choose a
  different cohort. Each task id binds the exact CAD and arm, and both parent
  and child independently match it against the frozen selector evidence.
  Date/Author: 2026-09-04 / Codex.

- Decision: require one fresh child process per `(CAD, arm)` cell.
  Rationale: pythonocc and OCC may terminate the native process. A separate
  child confines that outcome to one explicit row and protects the remaining
  denominator.
  Date/Author: 2026-09-04 / Codex.

- Decision: treat candidate callbacks as construction adapters, not verdict
  providers.
  Rationale: a low-level helper may clean one face yet change topology or fail
  after STEP roundtrip. Only the coordinator's independent STEP, validity, and
  schema-v2 measurements can accept a whole-CAD candidate.
  Date/Author: 2026-09-04 / Codex.

- Decision: fail closed when a callback or any proof field is absent, and
  distinguish that state from a measured negative result.
  Rationale: recording `candidate_hook_missing` prevents unfinished plumbing
  from being mistaken for a failed algorithm or a successful pilot.
  Date/Author: 2026-09-04 / Codex.

- Decision: resume only an identical signed contract and revalidate saved STEP
  size/hash plus source bytes before terminal summary.
  Rationale: a partially completed output root is useful after interruption,
  but neither argument drift nor local artifact mutation can be allowed to
  change the meaning of existing rows.
  Date/Author: 2026-09-04 / Codex.

- Decision: keep promoted STEP and worker-log names attempt-unique while rows
  retain stable task ids.
  Rationale: stable task ids enforce one logical denominator row; unique
  physical names make crash recovery non-destructive. Orphaned local native
  artifacts can be inspected but can never be mistaken for the next attempt.
  Date/Author: 2026-09-04 / Codex.

- Decision: verify the terminal JSONL and summary hashes before accepting a
  resumed terminal run.
  Rationale: source and STEP binding alone cannot detect a subsequently edited
  summary. The run manifest already records both hashes, so checking them and
  recomputing the summary closes the evidence-integrity gap without affecting
  a first execution.
  Date/Author: 2026-09-04 / Codex.

- Decision: validate a pre-MakeSolid sewn shell through a disposable solid,
  but return and compare the graph-preserved shell itself.
  Rationale: native/project strict validation is defined on the final solid,
  whereas the mutation boundary intentionally precedes MakeSolid. A validation
  wrapper reconciles those stage contracts without changing face, edge,
  vertex, wire, shell, or shared-edge identity in the candidate passed back to
  the constructor.
  Date/Author: 2026-09-04 / Codex.

- Decision: reserve whole-CAD 47472 defect, mapping, and shared-edge booleans
  for exact post-STEP geometry/incidence evidence.
  Rationale: the face helper proves only a local mutation. Complete source-face
  census plus unique face/edge occurrence matching is required to detect the
  known STEP-only regression and to reject global source-edge split or merge.
  Date/Author: 2026-09-04 / Codex.

## Outcomes & Retrospective

The coordinator contract, whole-CAD adapters, and focused tests are complete. It is now
possible to prove that exactly four isolated attempts were requested, every
process outcome remains in the denominator, and no callback can promote itself
without whole-CAD evidence. The scientific experiment itself remains pending
until the code is committed cleanly and the four formal cells are executed.
Therefore the assembly selector remains at 91/100 and no
training stage is authorized by this milestone.

## Context and Orientation

`tools/run_exact_cad_repair_feasibility.py` is the coordinator. The parent
process reads the frozen calibration manifest and selector matrix, proves that
the exact two IDs are among the current nine residuals, binds the previous
lineage cases/run, signs source pickle bytes and code hashes, and launches the
four workers. A denominator row is one requested matrix cell regardless of
whether OCC returned a STEP.

`tools/run_assembly_repair_matrix.py` supplies the unchanged directed control,
STEP validity measurement, and source-byte binding helpers.
`tools/assembly_selector_geometry.py` supplies the existing schema-v2 gate;
this experiment does not change its thresholds.
`tools/targeted_nonperiodic_pcurve_repair.py` and
`tools/post_sewing_graph_repair.py` contain low-level candidates. Their
whole-CAD integration callbacks are deliberately late-bound as
`module:function`, so an absent adapter is a recorded protocol state rather
than an import-time crash.

The fixed CADs are
`00047472_197769bbdd814278b715d88a_step_000` and
`00063055_e309c689b9b44f0686f47966_step_000`. The former is first bad after
pcurves are added on source faces 10 and 43; the latter is first bad after
sewing on source face 5. These identifiers configure only this exact
diagnostic experiment. They must never become production repair eligibility.

## Plan of Work

First, keep the exact source-selection and lineage checks independent from OCC
so a stale manifest is rejected before native work. Bind the completed
100-CAD selector run, its matrix hash, lineage row/run hashes, the two source
pickle hashes, runtime `utils.py`, repository commit, and source file hashes
into one canonical run signature.

Second, launch every registered task in a fresh child. The parent captures
stdout and stderr locally, accepts exactly one final JSON sentinel, validates
the row against the requested task and source binding, promotes the STEP by
atomic rename, and appends one fsync'd JSONL row. Timeout, native exit, spawn
failure, malformed sentinel, and forged evidence become false rows.

Third, have the child hash the source before reading, hash the exact bytes
passed to `pickle.loads`, hash the path again after load and after the callback,
then resolve the callback. The callback constructs a whole CAD and emits a
STEP inside its private output. Independently re-read that STEP, calculate
native/project-strict validity and geometry/topology schema-v2, and accept only
if the callback application and defect-preservation proofs also pass.

Finally, summarize four ordered rows. A missing callback, worker/protocol
failure, or drifting control makes the pilot inconclusive. Completed negative
candidates yield a conclusive close. Any accepted candidate authorizes only
its relevant residual family, never the whole 100-CAD release.

## Concrete Steps

Work at the repository root in an activated environment that provides
pythonocc 7.7.2. Validate the protocol contract with:

    python -m pytest -q tests/test_run_exact_cad_repair_feasibility.py
    python -m py_compile tools/run_exact_cad_repair_feasibility.py tests/test_run_exact_cad_repair_feasibility.py
    git diff --check -- tools/run_exact_cad_repair_feasibility.py tests/test_run_exact_cad_repair_feasibility.py plans/exact_cad_repair_feasibility_execplan_20260904.md

After both adapters exist and all focused tests pass, commit the implementation
so the formal run starts clean. Then use the frozen local artifacts:

    python tools/run_exact_cad_repair_feasibility.py --calibration-manifest <local-runs>/assembly_calibration_100cad_v1_20260809/calibration_manifest.jsonl --selector-matrix <local-runs>/assembly_selector_main_100cad_20260818/assembly_repair_matrix.jsonl --selector-run <local-runs>/assembly_selector_main_100cad_20260818/assembly_repair_run.json --lineage-cases <local-runs>/downstream_bad_wire_lineage_47472_63055_20260903_v2/downstream_bad_wire_lineage_cases.jsonl --lineage-run <local-runs>/downstream_bad_wire_lineage_47472_63055_20260903_v2/downstream_bad_wire_lineage_run.json --breparg-root <local-upstream-runtime> --output-dir <local-runs>/exact_cad_repair_feasibility_<new_unique_suffix> --joint-iterations 200 --worker-timeout-seconds 600

Never reuse a directory from a different signed run. Re-running the exact same
command safely resumes missing rows; a terminal identical run is read-only.

## Validation and Acceptance

The protocol unit tests must prove exact 2x2 registration, strict final
sentinel parsing, false timeout/malformed/missing-hook denominator rows,
source/lineage binding, torn-tail recovery, immutable signature resume, and
rejection of an overclaimed candidate. `py_compile` and `git diff --check`
must also pass.

Formal exact-CAD acceptance requires four rows and two reproduced controls.
Each candidate must report an actually attempted and applied repair, zero
nonfinite evidence, removed target defects, no new non-target defects, exact
mapping, preserved source topology/shared-edge correspondence/3D curves/source
binding, a non-empty independently read STEP, native true, project strict true,
both-valid true, and a fully accepted schema-v2 gate. Worker and protocol
failures must be zero.

Even a successful pilot sets `authorizes_full_100cad=false`. The next stage is
the relevant residual family, followed by frozen invalid16 and fixed 100-CAD
experiments. The project release gate remains at least 95/100 strict,
historical controls 84/84, regressions zero, worker/protocol failures zero,
and every selected fallback schema-v2 accepted.

## Idempotence and Recovery

The parent holds a nonblocking OS writer lock. The first run creates a signed
manifest only in an otherwise empty directory. JSONL writes are flushed and
fsync'd; resume may remove only an unterminated torn final line. Completed rows
are validated rather than rerun, including their promoted STEP hashes. Any
changed code, source bytes, runtime, manifest, selector, lineage evidence,
timeout, or iteration count changes the signature and requires a new directory.

All STEP files, source pickle bytes, and raw worker logs remain under local
run storage and are not Git artifacts. A later snapshotter may archive only
path-free compact JSON evidence.

## Artifacts and Notes

The coordinator currently registers callback references:

    tools.run_exact_cad_repair_feasibility:run_47472_candidate_variant
    tools.run_exact_cad_repair_feasibility:run_63055_candidate_variant

The callback keyword interface is:

    callback(source, source_bytes, parsed, output_dir, breparg_root,
             joint_iterations, variant, expected_source_binding, run_signature)

The callback result must include a private `step_path` or portable
`step_relative_path`, plus a path-free `candidate_application` mapping and a
path-free `defect_gate` mapping. The exact required booleans are documented in
the coordinator module docstring and checked by `_normalize_defect_gate`.

## Interfaces and Dependencies

The public coordinator helpers are `validate_lineage_evidence`,
`build_run_payload`, `bind_run_manifest`, `run_worker`, `run_isolated`,
`validate_attempt_row`, and `summarize`. `VariantSpec` is immutable, and
`VARIANTS` defines the only accepted task order. No new dependency is added;
the implementation reuses Python's process, hashing, JSON, pickle, and file
APIs plus the repository's existing NumPy/pythonocc measurement stack.

Revision note 2026-09-04: created after the inherited feasibility runner was
found truncated. This revision freezes the fail-closed callback boundary,
four-row denominator, independent STEP verdict, and exact-resume semantics so
the two repair implementations can be integrated without weakening the
existing assembly release gates.

Revision note 2026-09-04 02:45 +08:00: integrated both full-CAD adapters after
`construct_brep_directed` gained default-inert post-pcurve face and post-sewing
shape mutation hooks. Added adapter contract tests that prove the exact faces,
edge pairs, complete post-sewing binding counts, and preregistered `1e-4`
tolerance are forwarded without changing the default assembler path.

Revision note 2026-09-04 15:35 +08:00: added terminal ledger/summary hash and
derived-summary validation after review identified that the hashes were written
but not checked on resume.

Revision note 2026-09-04 16:10 +08:00: incorporated the final integration
audit. The 63055 helper now reconciles the shell-stage mutation boundary with
solid-level validation without mutating the returned shell. The 47472 adapter
now makes whole-CAD claims only from a complete, exact post-STEP
geometry/incidence observation, including the known face-1 STEP-only failure.
The final focused suite is 171 passed with only two existing deprecation
warnings.
