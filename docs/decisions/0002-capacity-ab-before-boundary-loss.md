# ADR-0002: Resolve learned-quantizer capacity before boundary loss

## Status

Accepted and executed. `vq_8192_64d_random` is the selected capacity arm.
Boundary-consistency training and autoregressive work remain blocked until the
independent assembly gate passes. A post-hardening rerun from `D:\capv5` at
commit `6f7436da50a5f455fd9af3c806676ac0f49b8f9f` reproduced the same
decision and is now the authoritative capacity record.

## Date

2026-08-17

## Context

On the same ordered 100-CAD validation cohort and through the same unchanged assembly chain, continuous bypass at 60,000 training patches achieved 70 percent project-strict validity while learned VQ-4096/64D achieved 57 percent. The paired outcomes contain 15 CADs that are strict-valid only with bypass and 2 that are strict-valid only with VQ. The two-sided exact McNemar p-value is 0.0023, so the 13 percentage-point gap is unlikely to be a cohort-count fluctuation.

Within the VQ arm, strict-invalid CADs also have about 2.7 times the median curved-surface reconstruction MSE of strict-valid CADs. This connects quantization error to the downstream assembly failure within one arm. At the same time, ground-truth input assembles at only 84 percent and bypass at 70 percent, so the assembly chain has a separate reconstruction and construction loss. Changing quantization, boundary loss, and construction together would make any improvement impossible to attribute.

## Decision

Run a capacity A/B before boundary-consistency training. Compare:

- `vq_8192_64d_random`: one learned 8192-entry, 64-dimensional codebook using the historical-feature `random` anchor; it emits one code per latent position and therefore adds no sequence-length cost relative to VQ-4096.
- `rvq_2x4096_64d_random`: two independent learned 4096-entry, 64-dimensional codebooks. Stage 1 quantizes the input and stage 2 quantizes the residual; reconstruction uses the sum. It emits two codes per latent position and is expected to increase complete CAD sequence length by about 36 percent.

Train both at seeds 3 and 4 for exactly 100 epochs with the same immutable 60,000/12,000 exact patch inventories, bf16 policy, batch size 128, learning rate `3e-4`, gradient clipping, plateau decay, non-finite fuse, checkpoint lifecycle audit, and exact resume contract used by P0-B.

Measure each candidate's seed-3 best checkpoint on the same frozen 100 CADs through the unchanged assembly chain. Selection uses project-strict paired validity and exact McNemar evidence. Curved MSE, STEP readability, native validity, both-valid, code usage, and token cost are explanatory or secondary measures.

Select VQ-8192 if the same-scale bypass-minus-VQ strict gap is at most 5 percentage points, unless RVQ exceeds VQ-8192 by at least 5 strict-validity points, has more paired wins than losses, and passes the exact paired significance gate. This quantitative exception is frozen before seeing the capacity results and prices RVQ's sequence cost explicitly. Report both RVQ stages' unique usage, entropy perplexity, coverage, and `usage_fraction = entropy perplexity / codebook size` separately every epoch; a collapsed or historically decaying second stage invalidates the claimed residual-capacity benefit.

The first schema-v1 capacity launch is diagnostic-only and must not be resumed. Its task signature claimed the plateau scheduler monitored curved-parent MSE, but the launcher omitted the environment variable and the training process used global validation MSE. Schema v2 requires the exported environment, history configuration, every epoch's `plateau_metric`, `plateau_value`, and `scheduler_metric` to agree. It also gives the two RVQ stages separate histogram namespaces and prevents the compatibility marginal from selecting checkpoints.

In parallel, repair the original assembly chain against original-control inputs with independent switches and a no-regression gate. Do not combine the repaired chain with the capacity comparison until each track has a valid unchanged-control result. After selecting capacity and accepting repairs, remeasure the winner through the repaired chain.

## Outcome

The schema-v2 formal matrix completed all four tasks at 100 epochs with
`valid=true`, identical 60,000/12,000 train/validation inventories, and
runtime-resume compatibility. RVQ seed 4 encountered one Windows
`PermissionError` while atomically replacing its rolling checkpoint at epoch
51; the same signed root resumed from epoch 50 without changing its
configuration and completed successfully on attempt 2.

The fixed ordered 100-CAD capacity measurement completed at measurement
signature `2dff0044bcebcc00edb744955393f3b8bf8a952d95f7e9c5508f98df05a7b433`.
The unchanged-chain strict results were:

- continuous bypass: `70/100`
- VQ-8192: `69/100`, a `1 pp` bypass gap
- RVQ-2x4096: `65/100`, four points below VQ-8192

The exact two-sided paired McNemar result for RVQ versus VQ-8192 was
`p=0.4239501953125`, with 5 RVQ-only and 9 VQ-only strict successes. Since
RVQ also carries the preregistered estimated `+36%` downstream sequence cost,
the registered rule selects VQ-8192 directly. The Git-safe report is
`reports/capacity_ab_assembly_measurement_20260817/`; its JSON SHA-256 is
`b3c26d56fc90b9e0c8d09bbae49650232d16e2d3c52cdaa197f18733bff50c6d`.

A later post-hardening rerun from `D:\capv5` at commit
`6f7436da50a5f455fd9af3c806676ac0f49b8f9f` completed all four tasks at exactly
100 epochs with `valid=true`, identical train/validation inventories, and
runtime-resume compatibility. The fixed ordered 100-CAD measurement in
`reports/capacity_ab_posthardening_assembly_measurement_20260818/` reproduced
bypass `70/100`, VQ-8192 `69/100`, RVQ `65/100`, exact two-sided McNemar
`p=0.4239501953125`, and the same `VQ_8192_DIRECT_WIN` decision. The report
contains only the JSON, Markdown, CSV, gate summary, and artifact manifest.
The primary artifact hashes are `capacity_ab_assembly_measurement.json` =
`cd89cf63d25f297c536f43789e32077f914b9c95a115ce5b432a28a6e7d0a9c1`,
`capacity_ab_assembly_measurement.md` =
`f4f265367737b5e5ed870992f647225ca8ff7021836d240953c84ca92dbaedda`, and
`capacity_ab_assembly_pairs.csv` =
`64a28b4c8b7c6180bb9f84e23800328a1e9fae27929e280c07c6c84ebc31e4de`.
The gate summary and manifest bind the source JSON and all four Git-safe files.

## Alternatives Considered

### Start boundary-consistency loss immediately

Rejected for now. The registered gate gives capacity precedence when `Delta_q > 5` points, and the observed value is 13 points. Boundary loss could reduce the separate bypass-to-GT gap, but starting it first would leave the measured quantization tax unresolved.

### Increase only reconstruction-loss weight or train VQ-4096 longer

Rejected as the capacity test. P0-B already gives stable 100-epoch, zero-nonfinite controls, and the downstream paired gap is large. More epochs do not isolate codebook cardinality or residual representation capacity.

### Choose by validation MSE alone

Rejected. The project needs assemblable CADs, and prior experiments repeatedly showed that aggregate reconstruction loss can improve without a corresponding BRep validity gain. MSE is retained to explain mechanisms and diagnose per-CAD failures.

### Choose RVQ for its theoretical code combinations

Rejected. A second residual codebook can collapse and its doubled surface-code stream imposes a real downstream cost. RVQ must prove independent stage-2 use and materially better paired validity.

## Consequences

The training pipeline gains a second learned-VQ cardinality and a residual quantizer with staged usage metrics. The four formal runs and fixed-cohort measurement are complete. VQ-8192 is selected because it crosses the five-point capacity gate while preserving one token per latent position; RVQ is rejected because its strict validity is lower and its paired advantage is not significant.
The 2026-08-17 measurement remains historical evidence, while the
2026-08-18 post-hardening rerun is the authoritative capacity record.

Capacity is therefore no longer the blocking decision. The independent production
assembly result remains `88/100` strict-valid against the required `95/100`
gate, so boundary loss, sequence regeneration, and autoregressive training stay
blocked until the repaired-chain gate is met.

All new checkpoints and generated STEP files remain local. Git stores
implementation, tests, complete lightweight histories, compact logs,
TensorBoard events, per-CAD statistics, experiment manifests, and checkpoint
hashes. The selected VQ-8192 arm is not released to full training, sequence
regeneration, or autoregressive training until the assembly gate is met and the
selected arm is remeasured through the accepted repaired chain.
