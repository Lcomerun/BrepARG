# Protocol V4: 100-Epoch Three-Seed FSQ Cohort

Status on 2026-08-05: implementation and pre-launch verification complete; the local RTX 3060 cohort is being launched from branch `experiment/protocol-v4-100epoch-fullchunk`. Final metrics are intentionally not reported until all requested runs finish and a separate statistics task is requested.

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

This experiment determines whether the 4096/6D arm remains ahead after longer same-data optimization and whether entropy perplexity, code coverage, and curved parent-cluster MSE reach a stable platform. `ppl >= 800` is evaluated at that platform, not at an arbitrary early epoch.

The paper-scale `curved <= 5e-5` reference is not a hard gate for this single-chunk experiment. It becomes meaningful only after a later filtered full-ABC run reaches its own platform. No sequence regeneration or AR training is permitted from Protocol V4 alone.

If 4096/6D remains ahead with healthy utilization, the next stage is a full-chunk protocol build using the quarantine and identity safeguards added here. Only after a full-data VQ model passes its representation decision may the project regenerate sequences and enter AR. At that transition, the 4096 vocabulary must be synchronized in the relevant upstream trainer, sequence builder, and config so the vocabulary is 6198; those excluded `BrepARG/` edits are not part of this branch.

## Reproduction and local artifacts

The implementation plan is `docs/superpowers/plans/2026-08-05-protocol-v4-100epoch-fullchunk.md`. The launcher is `tools/run_protocol_v4_100epoch_cohort.py`, with Windows background wrapper `tools/start_protocol_v4_100epoch_cohort.ps1`.

Raw checkpoints, changing histories, PID/state files, and stdout/stderr logs remain under ignored `local_runs/`. After all nine runs finish, a later result-statistics task may copy compact histories, aggregate JSON, and curated TensorBoard event files into `reports/`.

