# Minimal topology-preserving failure-family probe

This ExecPlan is a living document and follows `PLANS.md` in the repository
root. It records a small CPU-only experiment for the frozen 100-CAD assembly
cohort. The probe is deliberately separate from the upstream `BrepARG/`
directory and must never change source edge or vertex incidence.

## Purpose / Big Picture

The production selector is still below its strict-valid release gate. The
remaining historical-invalid CADs fall into curve-fit, closure/p-curve, and
adjacent-crossing families. This probe evaluates the smallest repair already
available in this repository that can affect those families: tighter surface
fitting together with interpolation of sampled edge curves. It changes only
geometric fitting; it does not remove edges, merge vertices, relax topology
checks, or change tolerances. A human can verify the result by reading the
Git-safe report under `reports/` and checking that every restored case has
unchanged topology and zero regressions.

## Progress

- [x] (2026-08-18) Confirm the current runner, calibration manifest, and
  topology/geometry gates.
- [x] (2026-08-18 08:23 +08:00) Run the 16 historical-invalid CAD probe with
  isolated workers and `--selector-geometry-gate` in a fresh output root.
- [x] (2026-08-18 08:24 +08:00) Verify the full matrix: 16 attempts, 0
  worker/protocol failures, 5 OCC both-valid, 3 gate accepted, and no new
  topology-preserving restoration.
- [x] (2026-08-18 08:26 +08:00) Produce the path-free v5 report with compact
  per-CAD gate evidence and rejection taxonomy.
- [ ] Commit and push the runner plumbing, snapshot fix, tests, plan, and
  authoritative report without touching `BrepARG/`.

## Surprises & Discoveries

- Observation: An earlier local-surface-refit run exists outside the current
  reproducible source tree and cannot be used as the sole implementation
  artifact.
  Evidence: its run manifest binds to a dirty revision and a switch absent
  from the current `REPAIR_SWITCHES` list.
- Observation: The first gated rerun (`gate_v3`) did not contain geometry-gate
  fields because `--isolate-cad-workers` was not part of the parent's dispatch
  condition for this profile.
  Evidence: its run payload lacked `selector_geometry_gate` and it had no
  `worker_logs`; the fresh single-CAD run and v4 matrix do contain the gate.
- Observation: The snapshot tool originally dropped matrix-level
  `selector_geometry_topology_gate` fields even though the worker recorded
  them.
  Evidence: v4 raw rows had five gate objects while the first snapshot had
  zero; v5 now archives all five and summarizes three accepted/two rejected.

## Decision Log

- Decision: Use the existing profile
  `directed_trim_surface_precision_curve_interpolate`.
  Rationale: It is topology-preserving, already tested, and isolates geometric
  curve/surface fitting without the unsafe global ShapeFix or near-vertex
  reconciliation paths.
  Date/Author: 2026-08-18 / Codex.
- Decision: Keep all STEP/pickle bytes outside Git and archive only hashes and
  compact JSONL diagnostics.
  Rationale: These files are large or source-sensitive and the repository
  policy permits only reproducible metadata.
  Date/Author: 2026-08-18 / Codex.
- Decision: Treat `--isolate-cad-workers` as an explicit dispatch request and
  forward `--selector-geometry-gate` to both isolated and direct paths.
  Rationale: native OCC work must remain contained even for profiles that do
  not otherwise require isolation, and gate evidence must be generated in the
  same process that writes the STEP candidate.
  Date/Author: 2026-08-18 / Codex.
- Decision: Preserve matrix-level geometry gate evidence in Git-safe snapshots
  and aggregate accepted/rejected CAD IDs plus reason counts.
  Rationale: a strict-valid candidate that changes topology is not a safe
  restoration; the rejection reason is required to order the next failure
  family experiment.
  Date/Author: 2026-08-18 / Codex.

## Outcomes & Retrospective

The authoritative v4 run completed 16 attempts with no worker or protocol
failure. It produced 14 STEP-readable files, 6 native-valid files, and 5
strict/both-valid files. The geometry gate accepted 3 of those 5; all three
are already present in the production selector's historical recovery set, so
the profile adds zero new restorations. It rejected `00032101` because OCC
merged source vertices (18 input versus 16 candidate vertices), and rejected
`00076198` because the candidate changed edge/face/vertex topology and its
bounding box exceeded the allowed relative delta. The profile is therefore a
negative diagnostic result and is not eligible for production promotion or a
100-CAD rerun.

The canonical Git-safe report is
`reports/failure_family_minimal_probe_gate_v4_20260818/`; its dedicated gate
archive validation is `valid=true`. The generic companion snapshot is
`reports/failure_family_minimal_probe_gate_v5_20260818/`; both bind the same
matrix SHA-256 and retain all five gate objects. The next experiment must
target a different failure family while keeping topology and geometry gates
enabled.

## Context and Orientation

`tools/run_assembly_repair_matrix.py` loads the frozen calibration manifest,
runs one or more `RepairProfile` values in isolated child processes, writes
local STEP files, validates native and strict BRep status, and enforces the
topology/geometry selector gates. `tools/assembly_repair.py` defines profiles;
`tools/directed_trim_assembly.py` performs the candidate construction. The
calibration manifest at
`D:/luolin/V13/local_runs/assembly_calibration_100cad_v1_20260809/calibration_manifest.jsonl`
contains the same 100 CAD identities used by the selector. The runner's
`--historical-invalid-only` option selects the 16 known strict-invalid cases.

## Plan of Work

Run the existing profile against that manifest with isolated workers and a new
output directory under `D:/luolin/V13/local_runs/`. Read the resulting JSONL,
classify each row by its recorded status and validity components, and copy only
path-free metadata into
`reports/failure_family_minimal_probe_gate_v5_20260818/`. The runner plumbing
and snapshot whitelist are covered by focused tests. Do not edit upstream code
or copy generated geometry into the Git working tree.

## Concrete Steps

From `D:/luolin/BrepARG2`, run:

    C:/Users/YU/.conda/envs/brepgen_env/python.exe tools/run_assembly_repair_matrix.py --calibration-manifest D:/luolin/V13/local_runs/assembly_calibration_100cad_v1_20260809/calibration_manifest.jsonl --breparg-root D:/luolin/V13/BrepARG --output-dir D:/luolin/V13/local_runs/failure_family_minimal_probe_gate_v4_20260818 --profile directed_trim_surface_precision_curve_interpolate --historical-invalid-only --isolate-cad-workers --selector-geometry-gate

The executable path above may be written without the visual space as
`brepgen_env`; use the existing environment's Python if the named environment
differs locally. Expect 16 attempts, no worker/protocol failures, and a report
that records every result even when construction fails.

## Validation and Acceptance

Run the focused runner tests and the new report verifier. The probe is
accepted as a repair candidate only if all of the following hold for a restored
CAD: `native_brep_valid`, `strict_brep_valid`, `both_valid`, topology counts and
incidence equal the source, and the geometry gate passes. The profile must also
preserve every historical strict-valid control when evaluated on the full
cohort; a 16-case positive is therefore evidence only, not a production
promotion. Any failure, crash, topology mismatch, or regression is a negative
result and must remain excluded from the selector.

## Idempotence and Recovery

The local run directory is disposable and may be deleted or regenerated; it is
outside Git. Re-running the command with the same manifest and source hashes
must produce the same cohort signature. If a worker crashes, keep the attempt
row and rerun with isolation; never broaden the topology gate or modify source
data to make the row pass.

## Artifacts and Notes

The Git-safe report contains `README.md`, `assembly_repair_summary.json`,
`assembly_repair_attempts.jsonl`, `assembly_repair_run.json`,
`repair_diagnostics_summary.json`, `artifact_manifest.json`, and
`archive_validation.json`. JSONL rows may include STEP size/hash and compact
gate measurements, but never STEP bytes, source paths, pickles, arrays, or
checkpoints.

## Interfaces and Dependencies

The probe depends on Python, NumPy, pythonocc-core as imported by the existing
runner, and the repository modules under `tools/`. It adds no runtime API and
does not import or modify code below `BrepARG/`.
