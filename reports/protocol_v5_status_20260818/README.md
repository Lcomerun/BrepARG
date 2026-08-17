# Protocol V5 evidence index

This page is the navigation point for the evidence requested for the Protocol V5
representation and assembly decision. It contains only Git-safe summaries,
histories, logs, TensorBoard event files, tabular measurements, and immutable
artifact hashes. Checkpoint bytes, raw CAD, STEP files, pickle files, NumPy
arrays, and the `BrepARG/` source tree remain outside Git.

## P0-A assembly-chain diagnosis

The 16 invalid calibration cases have an explicit first-cause classification
and a repair checklist. The report includes one JSONL row per case, the failure
taxonomy, the `joint_optimize` ablation, and the three-tolerance scan:

- [`p0a_assembly_chain_evidence_20260817`](../p0a_assembly_chain_evidence_20260817/README.md)

All 16/16 cases have a recorded attribution. The report is diagnostic evidence,
not permission to relax the schema-v2 topology gate.

## P0-B stability and capacity archives

The original P0-B stability matrix (continuous bypass and learned VQ,
seeds 3/4) is archived here:

- [`p0b_formal_results_20260814`](../p0b_formal_results_20260814/README.md)

The post-hardening capacity matrix contains four complete histories, four
TensorBoard event files, compact logs and statistics, and checkpoint size/SHA-256
bindings. It has 60,000 train patches, 12,000 validation patches, bf16, batch
128, and epochs 0..99 for VQ-8192 and RVQ-2x4096, seeds 3/4:

- [`capacity_ab_60k_posthardening_v2_20260818`](../capacity_ab_60k_posthardening_v2_20260818/README.md)

The archive validator reports `formal_result_eligible=true`, identical
inventories for all four tasks, and zero skipped or non-finite events. The
checkpoint manifest records 12 local checkpoint sizes and SHA-256 values; no
checkpoint bytes are present.

## Fixed 100-CAD paired measurement

The capacity measurement uses one ordered parent-isolated cohort of 100 CADs,
selection seed `20260809`, cohort identity SHA-256
`646693dbfde083bf16ae63f917658cc0c3b3eb71cedaeddfeea55007bd741474`, and the
same unchanged reconstruction/assembly/STEP/OCC audit chain for every arm. All
100 attempts are in the denominator, including construction and STEP failures.

- [`capacity_ab_assembly_measurement_20260817`](../capacity_ab_assembly_measurement_20260817/capacity_ab_assembly_measurement.md)

The requested strict-valid comparison is:

| Reference or arm | Strict valid / 100 |
| --- | ---: |
| GT historical | 84 |
| bypass@300k (historical) | 70 |
| FSQ@300k (historical) | 49 |
| bypass@60k, seed 3 best | 70 |
| learned VQ-8192@60k, seed 3 best | 69 |

The report also separates STEP-readable, native OCC-valid, strict-valid, and
both-valid counts. Its immutable checkpoint bindings identify the bypass and
VQ seed-3 best artifacts by SHA-256, while keeping their bytes local.

The registered gates are:

- `Delta_q = bypass@60k - VQ@60k = 1 pp`: the capacity/quantization gate passes;
  VQ-8192 is selected over RVQ because RVQ is not a significant positive
  paired improvement and carries the preregistered sequence cost.
- `Delta_r = GT - bypass@60k = 14 pp`: the boundary-loss numeric trigger is
  true, but execution remains held until the independent assembly-chain release
  gate is met.

## Current release status

The latest unchanged-chain assembly hardening selector reaches 91/100 strict
valid, 97/100 STEP-readable, 90/100 native-valid, and 88/100 both-valid with
zero regression of the original 84 strict-valid controls. The release gate is
still `strict >= 95/100` with all 84 controls preserved. Therefore boundary
consistency training, sequence regeneration, and AR training remain blocked;
the next work item is a safe assembly-chain repair that preserves schema-v2
topology and the zero-regression requirement.
