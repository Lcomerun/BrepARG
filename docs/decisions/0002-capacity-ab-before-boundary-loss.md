# ADR-0002: Resolve learned-quantizer capacity before boundary loss

## Status

Accepted for the next experiment. Boundary-consistency training and autoregressive work remain blocked.

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

Select VQ-8192 directly if the same-scale bypass-minus-VQ strict gap is at most 5 percentage points. Select RVQ only if it provides a material paired validity advantage sufficient to justify its sequence cost. Report stage-2 RVQ unique usage, entropy perplexity, coverage, and usage fraction separately every epoch; a collapsed second stage invalidates the claimed residual-capacity benefit.

In parallel, repair the original assembly chain against original-control inputs with independent switches and a no-regression gate. Do not combine the repaired chain with the capacity comparison until each track has a valid unchanged-control result. After selecting capacity and accepting repairs, remeasure the winner through the repaired chain.

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

The training pipeline gains a second learned-VQ cardinality and a residual quantizer with staged usage metrics. Four formal 100-epoch runs and two fixed-cohort assembly measurements are required before boundary loss can start. VQ-8192 is preferred when it crosses the five-point gate because it preserves one token per latent position. RVQ remains viable only when measured utility offsets its longer sequence.

All new checkpoints and generated STEP files remain local. Git stores implementation, tests, complete lightweight histories, compact logs, TensorBoard events, per-CAD statistics, experiment manifests, and checkpoint hashes. If neither arm crosses the capacity gate, the representation layer is not released to full training, sequence regeneration, or autoregressive training.
