# Protocol V4: 100-Epoch Three-Seed FSQ Cohort

Status on 2026-08-05: all three seeds and all three FSQ arms completed 100 epochs from clean commit `12fd520`. The cohort state is `COMPLETED`, every seed returned code 0, all stderr logs are empty, and the experiment decision is `NO_PROMOTED_ARM`.

## Result

| Arm | Checkpoint val MSE, mean +/- sample SD | Checkpoint perplexity, mean (range) | Checkpoint coverage, mean | Curved parent MSE, mean |
| --- | ---: | ---: | ---: | ---: |
| 8192/4D | `0.006035 +/- 0.001303` | `1501.43 (1369.53-1606.22)` | `44.90%` | `0.010950` |
| 4096/6D | `0.005609 +/- 0.000350` | `663.69 (619.42-746.62)` | `43.49%` | `0.010588` |
| 8192/6D | `0.006100 +/- 0.000328` | `902.64 (694.94-1089.24)` | `31.48%` | `0.011530` |

The 4096/6D arm remains the diagnostic cross-seed leader on checkpoint validation MSE and curved parent-cluster MSE mean. It is not promoted: its checkpoint perplexity is below the `800` healthy-usage reference in all three seeds, and seed 0 still selected epoch 99, so this cohort does not establish a stable healthy-usage platform. The 8192/4D arm has substantially higher perplexity but also the largest reconstruction variance, including seed-1 late degradation. No arm passes the complete representation decision.

The outcome does not authorize an all-chunk build or sequence/AR work. The full-data curved reference of `5e-5` is not used to reject this single-chunk cohort by itself, but every observed curved MSE remains around `1e-2`, so it also provides no evidence for downstream readiness.

## What this experiment changes

Protocol V4 extends the clean Protocol V3 comparison from 15 to 100 epochs and adds seed 2. All nine training runs start from scratch. It also hardens the later all-chunk data build so an isolated corrupt pickle is quarantined and excluded from every split instead of aborting an otherwise healthy build.

The all-chunk builder now checks archive and member identities before materialization. Duplicate archive basenames, duplicate normalized member identities, unsafe member paths, and duplicate materialization targets fail closed. Unpickle failures are written to both the complete protocol manifest and `quarantined_pickle_members.jsonl`; they can pass protocol construction only when both the configured count and fraction limits are satisfied. CLI defaults are at most 100 failures and at most 0.001 of scanned pickle members.

## Fixed experiment controls

- Dataset: the same verified `abc_0000` Protocol V2 cohort used by the clean Protocol V3 experiment.
- Protocol SHA-256: `43d0c5b36375cc78f3386a78a020a9baacc5a314372380f29e2eedb446345e6f`.
- Split pickle SHA-256: `df72b5757c3aabc89c707fd351c086ca8914cd96a49868decb8d15c104b17357`.
- Train/validation cap: 12,000 / 4,637 deduplicated patches.
- Train/validation parent coverage in Protocol V3 evidence: 322/323 and 100/100 after final filtering.
- Seeds: 0, 1, and 2.
- Epochs: 100 for every arm; ordinary early stopping disabled through minimum epoch and patience both equal to 100. Existing nonfinite-loss termination remains active.
- Batch size: 128.
- Learning rate: `3e-4`.
- Complex/curved oversampling: disabled in every arm.
- Complex/curved loss weights: 1.0 in every arm.
- Arms: 8192/4D `[8,8,8,16]`, 4096/6D `[4,4,4,4,4,4]`, and 8192/6D `[4,4,4,4,4,8]`.
- Execution: seeds and arms run sequentially on the single RTX 3060; the Python orchestrator stops on the first failed seed.

## Interpretation boundary

This experiment determines whether the 4096/6D arm remains ahead after longer same-data optimization and whether entropy perplexity, code coverage, and curved parent-cluster MSE reach a stable platform. The result is mixed: the reconstruction ranking remains favorable to 4096/6D in aggregate, but `ppl >= 800` is not met and the curves do not prove a stable platform.

The paper-scale `curved <= 5e-5` reference is not a hard gate for this single-chunk experiment. It becomes meaningful only after a later filtered full-ABC run reaches its own platform. No sequence regeneration or AR training is permitted from Protocol V4 alone.

If 4096/6D remains ahead with healthy utilization, the next stage is a full-chunk protocol build using the quarantine and identity safeguards added here. Only after a full-data VQ model passes its representation decision may the project regenerate sequences and enter AR. At that transition, the 4096 vocabulary must be synchronized in the relevant upstream trainer, sequence builder, and config so the vocabulary is 6198; those excluded `BrepARG/` edits are not part of this branch.

## Reproduction and local artifacts

The implementation plan is `docs/superpowers/plans/2026-08-05-protocol-v4-100epoch-fullchunk.md`. The launcher is `tools/run_protocol_v4_100epoch_cohort.py`, with Windows background wrapper `tools/start_protocol_v4_100epoch_cohort.ps1`.

The structured result is `fsq_abc_100epoch_three_seed_20260805.json`, and `artifact_manifest.json` binds every copied file by byte count and SHA-256. Complete 100-epoch histories are under `reports/protocol_v4/histories/`, and all nine curated event files are under `reports/tensorboard/protocol_v4_fsq_abc_100epoch_three_seed_20260805/`. Inspect the latter with `tensorboard --logdir reports/tensorboard/protocol_v4_fsq_abc_100epoch_three_seed_20260805`.

The nine local checkpoints total about 1.92 GiB and remain under ignored `local_runs/`; they are identified by SHA-256 in the structured result but are not committed. PID/state and stdout/stderr files also remain local.
