# Repair stage-local non-periodic wire regressions without changing source topology

This ExecPlan is a living document. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current
while work proceeds. This document follows `PLANS.md` in the repository root.

## Purpose / Big Picture

The fixed 100-CAD selector is strict-valid for 91 CADs. It must reach at least
95 while preserving all 84 historically strict-valid controls, producing no
worker or protocol failure, and passing the existing schema-v2 geometry and
topology gate. A signed four-stage lineage experiment now proves that two
residual CADs fail at different places. This plan turns those proofs into two
small, independently selectable repairs instead of another broad ShapeFix
profile.

The first preregistered experiment isolated one documented Open CASCADE sewing
option for `00063055...`, because the input faces are strict-clean immediately
before sewing and the bad two-edge wire first appears immediately afterward.
That experiment is now a recorded negative result: disabling SameParameter did
not change the defect or final validity. The active `00063055...` route is
therefore a graph-preserving post-sewing feasibility spike, not a production
profile. The separate `00047472...` route may rebuild only the defective
non-periodic pcurves on copied faces after pcurve attachment and before optional
face repair. A pcurve is the two-dimensional representation of a
three-dimensional edge on a surface. No candidate may delete, merge, split, or
replace source topology merely to become valid.

## Progress

- [x] (2026-09-03 20:55 +08:00) Completed the signed two-CAD lineage run from
  clean commit `9bfda7b2634e4477cac540a2167f26fc0de63fdd` with 2/2 completed
  cases and zero coverage, observation, mapping, source-binding, worker, or
  protocol failures.
- [x] (2026-09-03 21:15 +08:00) Pre-registered the first single-variable
  sewing experiment and the later copied-pcurve route before observing either
  candidate's validity result.
- [x] (2026-09-03 21:45 +08:00) Ran a development single-variable smoke for
  `00063055...` at 200 joint iterations with the same 18 ordered copied faces,
  tolerance, and sewing modes. `SameParameterMode(False)` produced the same
  post-sewing defect and invalid STEP as the default and is closed without a
  production API or repair profile.
- [ ] Run a graph-preserving post-sewing feasibility spike for `00063055...`;
  do not register a selector candidate unless exact source mapping, discrete
  incidence, shared topology, and sampled 3D curves all remain unchanged.
- [ ] Design and implement the exact-pair non-periodic copied-pcurve route for
  `00047472...`; treat STEP-only source face 1 as a separate acceptance failure
  if it remains bad.
- [ ] Promote only a successful exact-CAD candidate to the relevant residuals,
  then the frozen 16 invalid CADs, and finally all 100 controls.
- [ ] Archive each experiment as path-free Git evidence, update the repair to
  recovered/regressed case map, and push each independent commit.

## Surprises & Discoveries

- Observation: the two similar final STEP failures have different first-bad
  stages.
  Evidence: the signed lineage run reports `00047472...` first bad at
  `post_add_pcurves_pre_repair`, while `00063055...` is clean through optional
  face repair and first bad at `post_sewing_pre_step`.

- Observation: the `00063055...` failure is one reversed pair reported twice,
  not two unrelated boundary defects.
  Evidence: source face 5 reports closure source edges `[9,23]` and adjacent
  source edges `[23,9]` after sewing and after STEP.

- Observation: changing sewing tolerance and disabling SameParameter are both
  closed routes for `00063055...`.
  Evidence: the P0-A tolerance scan at `1e-4`, `1e-3`, and `1e-2` leaves the
  target invalid. In the later isolated development smoke, both
  `SameParameterMode(True)` and `SameParameterMode(False)` leave source face 5
  clean before sewing, introduce the same reversed edge-pair defect after
  sewing, and finish native-, strict-, and both-invalid.

- Observation: repairing only the two pre-STEP bad faces of `00047472...`
  cannot by itself prove a strict-valid CAD.
  Evidence: faces 10 and 43 are bad from pcurve construction onward, but source
  face 1 becomes a third bad face only after STEP roundtrip. The candidate must
  pass the final STEP diagnostic rather than infer success from pre-sewing
  state.

## Decision Log

- Decision: run a `SameParameterMode(False)` sewing ablation before implementing
  another pcurve mutation.
  Rationale: it is a single variable at the exact first-defective stage for
  `00063055...`, costs no topology or AR token change, and has not been covered
  by prior tolerance or face-repair experiments. A negative result closes the
  route cheaply; a positive result is directly attributable.
  Date/Author: 2026-09-03 / Codex.

- Decision: freeze all other sewing settings explicitly for the ablation.
  Rationale: both arms use tolerance `1e-3`, face mode enabled, floating edges
  disabled, non-manifold mode disabled, and the same ordered copied faces. The
  sole intended difference is SameParameter true versus false.
  Date/Author: 2026-09-03 / Codex.

- Decision: keep the switch default-off and require isolated workers.
  Rationale: native OCC work can terminate the process, and a cohort-wide
  default would confound current selector evidence. The switch must first earn
  promotion through exact-CAD, invalid-subset, and full-cohort gates.
  Date/Author: 2026-09-03 / Codex.

- Decision: reject the SameParameter route without adding a production switch.
  Rationale: the preregistered single-variable smoke was exactly negative:
  both arms retained 18 faces, zero free edges, 50 contiguous edges, zero
  multiple edges, and the same source-face-5 defect, and both failed native,
  strict, and both-valid checks after STEP. Expanding a causally negative arm
  would spend cohort time without a recovery signal.
  Date/Author: 2026-09-03 / Codex.

- Decision: split active implementation into a `63055` graph-preserving
  post-sewing feasibility spike and a `47472` exact-pair pcurve reconstruction.
  Rationale: their first-bad phases are different. The former must prove that
  any post-sewing mutation preserves shared topology; the latter can intervene
  before sewing but must also eliminate the separate source-face-1 STEP-only
  regression.
  Date/Author: 2026-09-03 / Codex.

- Decision: do not hard-code CAD, face, wire, or edge identifiers in production
  repair logic.
  Rationale: signed identifiers define the diagnostic pilot, not a general
  algorithm. Runtime eligibility must come from a clean pre-sewing diagnosis,
  a newly bad post-sewing occurrence, exact lineage, and unchanged discrete
  incidence.
  Date/Author: 2026-09-03 / Codex.

## Outcomes & Retrospective

The causal observation milestone and the first one-variable sewing ablation are
complete. The ablation is negative and deliberately leaves no production API
or profile. Capacity work remains closed in favor of VQ-8192/64D. The assembly
selector remains 91/100 strict with 84/84 controls and zero regressions, so
full VQ training, sequence regeneration, boundary loss, and AR remain blocked.

## Context and Orientation

`tools/directed_trim_assembly.py::construct_brep_directed` builds fitted faces,
adds pcurves, performs one optional face repair, adds all faces to
`BRepBuilderAPI_Sewing`, and builds a solid from the resulting shell. Its
current sewing block uses tolerance `1e-3` and otherwise accepts OCC defaults.

`tools/assembly_repair.py` defines named independent switches and profiles.
`tools/run_assembly_repair_matrix.py::profile_kwargs` converts a profile to
constructor keyword arguments and keeps each native-risk profile in a one-CAD
child process. The matrix runner writes STEP, evaluates native and project
strict validity, and can apply the existing schema-v2 geometry/topology gate.
`tools/run_assembly_repair_selector.py` must not include a new fallback until a
separate signed full-cohort matrix proves it safe.

The authoritative lineage input is the immutable local directory
`downstream_bad_wire_lineage_47472_63055_20260903_v2`. Its run signature is
`38ac843ee80611615351db47f38540f7ff27a19dc1f7f1f28883e2c915069271`.
The source pickles and generated STEP files remain local and are never copied
into Git.

## Plan of Work

The SameParameter experiment is complete and negative. Retain its evidence but
do not add an assembler argument or matrix profile. For `00063055...`, first
build a diagnostic-only feasibility spike around the exact pre-sewing source
face 5 and its sewn descendant. Evaluate whether its pre-sewing trim and
pcurves can be retained or whether the sewn face can be locally reprojected
without breaking shared edge and vertex identity. Reject any operation before
validity testing if face, edge, or vertex counts change; face-edge, edge-face,
or vertex-edge incidence changes; a shared source edge splits or merges; or
sampled 3D curves move outside the existing conservative tolerance.

Independently, implement the copied-pcurve candidate for `00047472...` after
`add_pcurves_to_edges` and before optional face repair. Operate on a copied
face. Eligibility
requires a strict adjacent occurrence whose source edges are uniquely mapped,
non-periodic U/V surface status, and no topology ambiguity. Remove and rebuild
only those edges' pcurves from their retained 3D curves, with topology-changing
and loop-removal modes disabled. Accept only if 3D curve samples, discrete
incidence, conservative geometry, native validity, and strict face checks all
pass. Final STEP roundtrip must also remove the source-face-1 regression.

## Concrete Steps

Work in `D:/luolin/BrepARG2` with the `brepgen_env` interpreter. Run focused
tests after each edit:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe -m pytest -q tests/test_directed_trim_assembly.py tests/test_run_assembly_repair_matrix.py tests/test_assembly_repair.py

Run each exact-CAD feasibility candidate through one isolated child process and
a fresh output directory. Do not reuse or overwrite a signed root. A diagnostic
spike is not a selector profile. Only after an exact candidate is both-valid,
schema-v2 accepted, and graph/geometry preserving should it receive a
default-off profile and run with `--historical-invalid-only
--isolate-cad-workers --selector-geometry-gate` on the 16-case expansion,
followed by the fixed 100-CAD run without `--historical-invalid-only`.

## Validation and Acceptance

Implementation acceptance requires focused tests, `py_compile`, and
`git diff --check`. The default constructor and every prior profile must remain
behaviorally identical. The exact pilot is scientific evidence even when it
is negative; success requires a non-empty STEP, native true, strict true,
both-valid true, a complete accepted schema-v2 gate, and no worker failure.

Promotion to the selector requires a separate 100-CAD run with all 100 rows,
at least one net-new strict recovery, 84/84 historical controls, zero
regressions, zero worker/protocol failures, and accepted schema-v2 evidence for
every selected fallback. The project-wide release remains at least 95/100
strict under the final selector.

## Idempotence and Recovery

Every run uses an unused output directory and a signed run manifest. A failed
or negative run is preserved rather than overwritten. Worker crashes and
timeouts remain denominator rows. Code defaults keep current behavior, so a
rejected experimental profile can be removed from selector consideration
without changing established results.

## Artifacts and Notes

The frozen formal lineage summary is:

    cases=2 completed=2 conclusive=true
    coverage=0 observation=0 mapping=0 source_binding=0 worker_protocol=0
    00047472 first_bad=post_add_pcurves_pre_repair
    00063055 first_bad=post_sewing_pre_step

The selected representation remains:

    VQ-8192/64D strict=69/100
    same-scale bypass strict=70/100
    Delta_q=1 percentage point

## Interfaces and Dependencies

No SameParameter interface is added. Any promoted repair must be a new
default-off `RepairProfile` switch backed by diagnosis-derived eligibility,
not a CAD or face identifier. The feasibility code may use existing
pythonocc 7.7.2 edge and face builders or ShapeFix APIs together with the
current geometry/topology validator; it adds no dependency and changes no
upstream `BrepARG/` file. The exact interface will be frozen only after a spike
proves that a graph-preserving operation exists.

Revision note 2026-09-03: created after the clean, signed four-stage lineage
run separated an add-pcurves defect from a sewing-introduced defect. The plan
freezes a one-variable sewing ablation before its validity result is observed.

Revision note 2026-09-03 22:05 +08:00: Recorded the negative
SameParameterMode ablation, closed it without a production profile, and made
the two active routes explicit: a graph-preserving post-sewing feasibility
spike for `00063055...` and an exact-pair non-periodic pcurve reconstruction for
`00047472...`.
