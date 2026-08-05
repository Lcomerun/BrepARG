# Protocol V4 100-Epoch Cohort and Full-Chunk Safety Design

## Scope and outcome

Protocol V4 has two bounded outcomes. First, it launches the existing Protocol V3 three-arm FSQ comparison on the exact same parent-isolated `abc_0000` cohort for 100 epochs at seeds 0, 1, and 2. Second, it makes the later all-chunk protocol build safe against isolated corrupt pickle members and ambiguous archive/member identities. This phase does not regenerate sequences, train AR, change `BrepARG/`, change `papers/`, or claim full-data representation quality.

The 100-epoch experiment is a from-scratch cohort, not a continuation of the 15-epoch checkpoints. Every seed trains the same three arms: 8192/4D `[8,8,8,16]`, 4096/6D `[4,4,4,4,4,4]`, and 8192/6D `[4,4,4,4,4,8]`. It uses 12,000 train patches, all 4,637 deduplicated validation patches, batch 128, learning rate `3e-4`, no complex or curved oversampling, unit loss weights, and the same Protocol V2 split SHA. Early stopping is disabled for ordinary finite runs by setting both the minimum epoch and patience to 100. The existing nonfinite-loss safety stop remains active.

The launcher runs seeds sequentially on the single RTX 3060. It writes one output directory and one TensorBoard subtree per seed plus a lightweight cohort state file. The launcher itself can be detached after its first training process is observed alive and producing a log. Completion and result analysis are explicitly deferred until the user asks for them.

## Protocol-build safety

Every pickle member is unpickled and structurally inspected before it can be assigned to a split. A member that cannot be unpickled receives a stable `load_failed:<ExceptionType>` rejection row, appears in a dedicated quarantine JSONL output, and never appears in `split.pkl` or under the materialized split root.

An isolated bad member no longer automatically invalidates a full build. The builder accepts explicit maximum load-failure count and maximum load-failure fraction thresholds. A build is `VERIFIED` only when both thresholds are satisfied and all existing eligibility, non-empty cohort, and parent-overlap checks pass. Exceeding either threshold fails the build while preserving the manifest and quarantine evidence. The portable CLI defaults are deliberately conservative for the later approximately 680,000-member build: at most 100 bad pickle members and at most 0.1 percent of scanned pickle members. These defaults tolerate a rare corrupt record but fail on a systemic decode problem or a wrong Python environment.

Archive/member identity is checked before record decoding or materialization. Archive basenames must be unique ignoring case, every member path must be safe and relative, and every normalized `archive-name!/member-path` identity and `archive-stem/member-path` materialization target must be globally unique. Duplicate or unsafe identities fail before split construction rather than allowing a dictionary overwrite or materialized-file collision. Identical member names in different uniquely named chunk archives remain valid because the archive name is part of the identity and materialized path.

## Components and data flow

`breparg_improvements/cad_protocol.py` owns archive identity validation, corrupt-member quarantine accounting, load-failure thresholds, summary fields, and the dedicated quarantine output. `tools/build_cad_protocol.py` exposes the thresholds on the command line.

`tools/run_protocol_v4_100epoch_cohort.py` owns the experiment controls and sequential seed orchestration. It does not implement training; it launches `breparg_improvements/train.py --stage vqsweep` with an explicit environment for each seed. The training script continues to own matched model initialization, sampling, metrics, checkpoints, TensorBoard, promotion checks, and split binding.

`tools/start_protocol_v4_100epoch_cohort.ps1` is a thin Windows detached-start wrapper. It starts the Python orchestrator with `-WindowStyle Hidden`, writes the launcher PID, and returns. The task is complete after code/tests are pushed and the detached process is verified alive with the seed-0 log advancing. It is not necessary to watch all 900 arm-epochs finish in this turn.

## Error handling and recovery

The cohort orchestrator refuses a missing protocol directory, invalid/duplicate seed list, non-positive epochs/caps/batch, or a reused seed output containing an unfinished run. It writes state atomically before and after every seed and stops the cohort on the first nonzero training exit. Completed seed directories may be skipped only when their sweep JSON reports all three configured arms and the requested epoch count; partial runs are not silently treated as complete.

Protocol construction records corrupt members even if the configured tolerance is exceeded. Identity problems remain hard failures because they make provenance or materialization ambiguous. No code deletes source archives, parsed CADs, old checkpoints, or prior Protocol V3 evidence.

## Verification and decision boundary

Unit tests first prove that one corrupt member is quarantined and excluded while a sufficiently large fixture can remain verified, that excessive corruption fails, that duplicate archive/member identities and unsafe paths fail early, and that member names shared by different unique chunk archives materialize to distinct paths. Launcher tests prove the exact three arms are inherited through `vqsweep`, seeds default to 0/1/2, all scientific controls are identical except seed/output paths, and sequential state transitions stop on failure.

The 100-epoch cohort is evidence for whether 4096/6D retains its lead and whether perplexity and curved parent-cluster MSE approach a platform. `ppl >= 800` is evaluated at the observed platform, not at an arbitrary early epoch. `curved <= 5e-5` is reported but is not a promotion gate for this single-chunk cohort; that scale-dependent threshold is reserved for a later full-data plateau. Sequence regeneration and AR remain blocked until a full-data VQ checkpoint passes the representation gate and the 4096 vocabulary change is synchronized in `trainer.py`, `2sequence.py`, and `config.json` to the 6198-token vocabulary.

