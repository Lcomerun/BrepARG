# ADR-0002: Resolve learned-quantizer capacity before boundary loss

## Status

Accepted and executed. VQ-8192 is the selected learned quantizer. Boundary-consistency training and autoregressive work remain blocked by the separate assembly-chain gate.

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

The schema-v2 experiment completed both arms at seeds 3 and 4 for exactly 100 epochs on the same 60,000/12,000 patch inventories. All four formal tasks passed the finite-state and inventory validator with zero non-finite events. The fixed 100-CAD measurement then produced:

| Arm | STEP readable | Native valid | Strict valid | Both valid |
| --- | ---: | ---: | ---: | ---: |
| bypass@60k | 95 | 73 | 70 | 64 |
| VQ-8192/64D | 96 | 67 | 69 | 61 |
| RVQ-2x4096/64D | 96 | 72 | 65 | 62 |

VQ-8192 has `Delta_q = bypass - VQ = 1 percentage point`, within the pre-registered 5-point gate. RVQ is four strict-valid points worse than VQ-8192. Their paired strict outcomes contain five RVQ-only successes and nine VQ-only successes, with exact two-sided McNemar `p=0.42395`; this does not justify RVQ's estimated 36 percent sequence-length increase. The measured decision is therefore `VQ_8192_DIRECT_WIN`.

The full registered strict comparison is `GT 84 | bypass@300k 70 | FSQ@300k 49 | bypass@60k 70 | VQ-8192@60k 69`. Therefore `Delta_r = GT - bypass@60k = 14 percentage points`; the original P0-B numeric boundary-loss trigger remains true even though the capacity tax is resolved.

The capacity result resolves the learned-quantizer choice, not the assembly chain. The formal failure-triggered selector pilot restored only five of the six pre-registered invalid controls, and the best accepted no-regression assembly composition remains below the 95/100 release gate. The numeric boundary-loss trigger is recorded but its execution, sequence regeneration, and autoregressive training remain held until that separate gate and result review are resolved. Git-safe per-CAD evidence is in `reports/capacity_ab_assembly_measurement_20260817/`.

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

The training pipeline retains both capacity probes for reproducibility, but VQ-8192 is the selected learned quantizer because it crosses the five-point gate while preserving one token per latent position. RVQ is not promoted: its lower strict validity and non-significant paired result do not offset the longer sequence.

All new checkpoints and generated STEP files remain local. Git stores implementation, tests, complete lightweight histories, compact logs, TensorBoard events, per-CAD statistics, experiment manifests, and checkpoint hashes. Passing the quantizer-capacity gate does not release full training, sequence regeneration, or autoregressive training while the independent assembly-chain gate remains below target.
