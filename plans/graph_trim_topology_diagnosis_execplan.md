# Diagnose Graph-Preserving Trim Candidates

This ExecPlan is a living document and follows `PLANS.md` at the repository
root.  It is intentionally limited to read-only diagnosis and fail-closed
prototypes for CAD ids `00032101` and `00076198`; it must never modify or
include the upstream `BrepARG/` tree.

## Purpose / Big Picture

The directed surface-precision and curve-interpolation candidates can be OCC
native-valid while silently merging vertices or edges.  That makes them unsafe
for the schema-v2 assembly selector even when STEP can be read and OCC reports
a valid solid.  This work will identify the first operation that changes the
source face/edge/vertex graph, record per-CAD evidence, and only retain a new
repair switch if the candidate survives the existing topology and geometry
gate before and after STEP roundtrip.  A human can verify the result by reading
the Git-safe JSON report and rerunning the isolated probe command.

## Progress

- [x] (2026-08-18 03:00 +08:00) Confirmed the worktree is `protocol-v5-graph-trim` and the
  existing selector gate rejects topology-changing OCC-valid candidates.
- [x] (2026-08-18 03:35 +08:00) Added the isolated probe and its self-contained
  execution plan without changing the production constructor interface.
- [x] (2026-08-18 03:55 +08:00) Ran stage probes and tolerance variants for both
  target CADs. Curves constructed all 28/28 and 106/106 source edges; the first
  observed graph change is in historical `ShapeFix_Face`.
- [x] (2026-08-18 04:13 +08:00) Re-ran 8 safe worker attempts (historical plus
  minimal-no-topology at three sewing tolerances): 0 worker failures and 0
  selector-eligible candidates. Mapped source vertex merges and the remaining
  self-intersection to source face 28, edges 86/91.
- [x] (2026-08-18 04:20 +08:00) Isolated 12 unsafe attempts that exited with
  Windows access violation 0xC0000005, removed those policies from the probe,
  added a Git-safe negative report and regression tests, and left production
  code unchanged. No commit or push is made from this worktree.
- [x] (2026-08-18 04:27 +08:00) Re-ran the probe and existing assembly-focused
  regression suite: 45 tests passed. JSON parsing, `git diff --check`, and the
  Git-safe artifact scan also passed.

## Surprises & Discoveries

- Existing `00032101` evidence reports candidate counts changing from 18 to 16
  vertices, while `00076198` changes 106 to 104 edges and 68 to 65 vertices.
- OCC validity is therefore not sufficient evidence of semantic topology
  preservation; the selector's v2 gate must remain authoritative.
- The safe minimal path for `00076198` preserves 106 edges and 212 face-edge
  occurrences but remains strict-invalid with one self-intersecting two-edge
  wire. The matching source face is face 28, whose source edges are
  `[86, 91, 92, 93]`; the crossing sub-loop maps to edges 86 and 91.
- Explicitly reusing source endpoint vertices made the collapse worse (14
  candidate vertices for `00032101`, 64 for `00076198`). This is a negative
  control, not a production option.

## Decision Log

- Decision: Treat any edge, face, vertex, incidence, or bounding-box change as
  a rejection, even if native and strict OCC validity are true.  Rationale:
  downstream token semantics depend on the source graph.  Date/Author:
  2026-08-18 / graph-trim probe agent.
- Decision: Keep all OCC operations in a worker subprocess and store only
  compact JSON/hash evidence.  Rationale: OCC can abort or corrupt process
  state on malformed shapes, and raw STEP/pickle artifacts are out of scope.
- Decision: Keep experimental face-fix and sewing policies out of
  `tools/directed_trim_assembly.py`; patch the imported OCC helpers only
  inside the isolated diagnostic worker and restore them in `finally`.
  Rationale: the production API must not expose known-crashing switches, while
  the negative experiment remains reproducible. Date/Author: 2026-08-18 / Codex.
- Decision: Exclude every candidate that changes discrete topology or fails any
  schema-v2 check, even when native and strict OCC validity are true. Rationale:
  source token semantics require graph identity; relaxing this gate would turn
  a repair into silent data mutation. Date/Author: 2026-08-18 / Codex.
- Decision: Do not pursue more tolerance scans or topology relaxation for these
  two CADs. Rationale: sewing 1e-3, 1e-4 and 1e-5 produced the same decisions;
  the remaining defects are curve/surface-to-graph incompatibility. Date/Author:
  2026-08-18 / Codex.

## Outcomes & Retrospective

No graph-preserving candidate exists among the tested policies. For
`00032101`, historical and minimal face repair are OCC-valid only after
merging source vertices (18→16 and 18→15), so the incidence gate rejects them.
For `00076198`, historical repair deletes two edges and three vertices
(106→104, 68→65, 212→208); the graph-preserving minimal path keeps the edge
count but remains strict-invalid with one self-intersection, 68→65 vertices,
and bbox delta 0.0496. All eight safe attempts are fail-closed and all twelve
unsafe variants are recorded separately as native access violations. The
production constructor and schema-v2 gate are unchanged. The Git-safe evidence
is in `reports/assembly_graph_preserving_trim_negative_20260818/`.

## Context and Orientation

`tools/directed_trim_assembly.py` constructs surfaces, fitted/interpolated
edges, wires, faces, and a sewn solid.  `tools/assembly_repair.py` supplies
loop ordering and the profile switch vocabulary.  The matrix runner in
`tools/run_assembly_repair_matrix.py` calls `tools/assembly_selector_geometry.py`
to compare candidate STEP signatures with the effective source topology.  The
existing gate requires equal face/edge/vertex counts and incidence vectors,
bounded edge and point deviations, no free edges or self-intersections, and a
single solid.  This plan diagnoses why a candidate fails those checks; it does
not relax them.

## Plan of Work

First inspect the frozen calibration source and all existing Git-safe reports,
then build a small subprocess probe that records topology signatures after each
constructive stage available without changing production code.  The probe will
compare source adjacency with the effective adjacency passed to OCC, inspect
the sewn shape before writing, and inspect the STEP reimport.  It will include
per-edge endpoint distances and a mapping based on sampled curve proximity so
that merges caused by interpolation, sewing, or write/read tolerance can be
distinguished.  If the first changed stage is a production operation, test a
narrow opt-in alternative in the isolated worktree only; the alternative must
be rejected unless the full v2 gate and roundtrip checks pass.

Finally write a compact report under `local_runs/` (outside Git) and, if useful
for future maintainers, a small Git-safe diagnostic fixture under `reports/`
without copying STEP, pickle, NumPy, or checkpoint data.  Do not commit or
push this branch.

## Concrete Steps

Run commands from `D:\luolin\V13\.worktrees\graph-trim-20260818` using the
`brepgen_env` Python interpreter.  All commands that import OCC must be run in
an isolated worker process.  Expected successful probes exit zero and produce
JSON; a candidate that violates topology must produce a structured rejection,
not an exception swallowed as success.

## Validation and Acceptance

The probe is accepted only when both CADs have a per-stage topology diff,
explicit rejection reasons for every unsafe candidate, and no worker/protocol
failure.  A promoted switch additionally requires equal source/candidate
face-edge and edge-vertex incidence, equal counts, bounded bbox and sampled
3-D curve error, no free edges or wire self-intersections, strict/native
validity, and the same results after STEP roundtrip.  If no candidate meets
these conditions, the correct result is a documented rejection and no code
promotion.

## Idempotence and Recovery

Probes write to a timestamped `D:\luolin\V13\local_runs` directory and can be
rerun without changing tracked files.  Never delete existing runs.  If OCC
crashes, inspect the worker log and rerun only the single CAD in a fresh
subprocess.

## Artifacts and Notes

Only compact JSON, JSONL, CSV, Markdown, command manifests, and source hashes
may be copied into a future Git report.  STEP, pickle, NumPy arrays, raw CAD,
and checkpoints remain local and untracked.

## Interfaces and Dependencies

The probe may call the existing `construct_brep_directed` and
`geometry_topology_gate` helpers, but must not import or edit `BrepARG/` source
files beyond adding its path for the existing runtime dependency.  It should
use `pythonocc-core`, NumPy, and the standard library only, matching the
repository's current assembly tools.

## Change Note

2026-08-18: Created this plan before any implementation so the diagnosis can
be resumed independently and the fail-closed topology policy is explicit.

2026-08-18 04:27 +08:00: Closed the plan with no promoted candidate, a clean
production constructor, 8/8 completed safe workers, 0 selector-eligible
candidates, and 45 passing focused tests.

2026-08-18 04:20 +08:00: Updated this living plan after the isolated probes.
The update records the first ShapeFix_Face topology changes, the source-index
mapping for the remaining self-intersection, the native-crash exclusions, and
the final decision to promote no candidate. This keeps the negative result
restartable without copying STEP, pickle, or raw geometry.
