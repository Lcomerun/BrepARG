# P0-A assembly-chain evidence

This directory closes the Git evidence gap for the frozen 16 original/GT strict-invalid CADs in the existing 100-CAD calibration cohort. The matrix contains 96 attempts: `joint_iterations` in `{200, 0}` crossed with sewing tolerance in `{1e-4, 1e-3, 1e-2}`.

The attribution gate passed at 16/16 cases (100%). The failure taxonomy is: 10 wire self-intersections, 3 curve-fit failures, 2 wire-build failures, and 1 non-unit/empty solid.

The evidence does not justify a global assembly setting change. Disabling joint optimization changed four complete outcome signatures, but only one CAD recovered to both-valid under any variant. Changing sewing tolerance altered one signature and produced no additional recoveries.

Evidence files:

- `assembly_chain_cases.jsonl`: one attribution record per invalid CAD.
- `assembly_chain_attempts_detailed.jsonl`: all 96 stage-local attempt records, with machine-local paths removed.
- `failure_taxonomy.json`: exact CAD membership for each cause family.
- `joint_optimize_ablation.json`: paired joint-on/joint-off outcomes at each tolerance.
- `tolerance_scan.json`: paired tolerance outcomes within each joint arm.
- `attempts_compact.csv`: compact analysis table.
- `step_sha256.csv`: size and SHA-256 bindings for local STEP files; no STEP bytes.
- `repair_checklist.md`: repair actions generated from the attributed causes.
- `artifact_manifest.json`: size and SHA-256 for every archived artifact.

STEP, source pickle, checkpoint, reconstructed array, and upstream `BrepARG/` bytes are not present. P0-A diagnosis is complete; this does not claim that assembly repair has been implemented.
