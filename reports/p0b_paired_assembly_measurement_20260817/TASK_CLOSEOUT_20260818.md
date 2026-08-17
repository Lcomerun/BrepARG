# P0-A/P0-B Evidence Closeout

This note records the evidence delivered for the P0-A diagnosis and the P0-B fixed-cohort measurement. It is intentionally Git-safe: checkpoint, STEP, pickle, NPZ, raw CAD, and upstream `BrepARG/` bytes remain outside the repository.

## Archived Evidence

- `reports/p0a_assembly_chain_evidence_20260817/` contains the 16-case JSONL diagnosis, 96 detailed attempts, failure taxonomy, `joint_optimize` ablation, three-tolerance scan, compact table, repair checklist, and byte manifest.
- `reports/p0b_formal_results_20260814/` contains four 100-epoch histories (bypass/VQ-4096, seeds 3 and 4), four TensorBoard event files, logs and summaries, epoch metrics, task manifests, source bindings, and `checkpoint_manifest.json`.
- The checkpoint manifest records 12 local checkpoint files totaling 4,586,357,724 bytes (about 4.27 GiB), with size and SHA-256 only. No checkpoint bytes are archived.

## Fixed 100-CAD Measurement

Both arms use the same ordered 100-CAD cohort (`identity_sha256=646693dbfde083bf16ae63f917658cc0c3b3eb71cedaeddfeea55007bd741474`), the same unchanged assembly chain, `joint_iterations=200`, and an attempts denominator of 100. The selected checkpoints are the seed-3 best checkpoints bound by SHA-256 in `measurement_runtime_manifest.json`.

| Arm | STEP readable | OCC native | Project strict | Both valid | Attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| bypass@60k | 95 | 73 | 70 | 64 | 100 |
| VQ-4096/64D@60k | 95 | 55 | 57 | 49 | 100 |

The requested strict comparison is `GT=84`, `bypass@300k=70`, `FSQ@300k=49`, `bypass@60k=70`, and `VQ@60k=57` valid CADs out of 100. Therefore:

- `Delta_q = bypass@60k - VQ@60k = 13 pp`.
- `Delta_r = GT - bypass@60k = 14 pp`.
- Registered gate: `CAPACITY_AB_FIRST` because `Delta_q > 5 pp`.

Boundary-consistency loss is not started by this closeout. The next representation experiment is the preregistered VQ-8192 versus RVQ capacity A/B; the P0-A repair track remains separately attributable.

## Verification

The focused report and measurement regression suite passes with 33 tests. The remote branch is `origin/experiment/protocol-v5-scaling-ladder`; its report manifests contain no forbidden artifacts or machine-local absolute paths.
