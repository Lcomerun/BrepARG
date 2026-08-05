# Run the Protocol V4 100-Epoch Cohort and Harden Full-Chunk Construction

This ExecPlan is a living document and follows `AGENTS.md` and `PLANS.md` in the repository root. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be maintained as work proceeds.

## Purpose / Big Picture

After this change, the user can launch a reproducible three-arm, three-seed, 100-epoch FSQ comparison on the already verified `abc_0000` cohort with one command and leave it running on the RTX 3060. The later all-chunk protocol builder can skip individually corrupt pickle members without putting them into a split or aborting an otherwise healthy approximately 680,000-member build, while still failing when corruption is systemic or archive/member identity is ambiguous. This phase ends after the detached training process is confirmed healthy; result aggregation and any decision to expand to all ABC chunks are deferred.

## Progress

- [x] (2026-08-05 10:25 +08:00) Verified the starting point is clean branch `experiment/protocol-v3-balanced-sampling` at `8ff43d8` and created isolated branch `experiment/protocol-v4-100epoch-fullchunk`.
- [x] (2026-08-05 10:29 +08:00) Ran the four Protocol V3 focused files with the actual `brepgen_env`; all 170 tests passed.
- [x] (2026-08-05 10:35 +08:00) Inspected the existing Protocol V2 artifact and 15-epoch run manifests. The reusable split has 795/100/99 records, 323/100/99 parent CADs, protocol SHA `43d0c5b36375cc78f3386a78a020a9baacc5a314372380f29e2eedb446345e6f`, split SHA `df72b5757c3aabc89c707fd351c086ca8914cd96a49868decb8d15c104b17357`, and zero parent overlap.
- [x] (2026-08-05 10:38 +08:00) Wrote and self-reviewed the Protocol V4 design and this ExecPlan.
- [x] (2026-08-05 10:41 +08:00) Added failing tests for tolerated corrupt pickle quarantine, excessive corruption failure, and archive/member identity preflight; the red run reported eight expected behavior failures and no unrelated failure after correcting one missing test import.
- [x] (2026-08-05 10:44 +08:00) Implemented full-chunk protocol safety; all 45 protocol tests pass.
- [x] (2026-08-05 10:46 +08:00) Added failing tests for the three-seed 100-epoch cohort launcher and atomic state transitions; collection failed because the new orchestrator module did not yet exist.
- [x] (2026-08-05 10:49 +08:00) Implemented the cross-platform orchestrator and Windows detached-start wrapper; all 11 initial launcher tests pass.
- [x] (2026-08-05 10:52 +08:00) Ran the initial 188 focused tests, all passing; Python compileall and PowerShell parsing pass. The complete suite reports 430 passed and the same 16 documented baseline failures: 11 require excluded `BrepARG/`, one legacy fixture lacks `ordering`, and four use the Python 3.11 `Path.write_text(newline=...)` API under Python 3.10. No Protocol V4 test fails.
- [x] (2026-08-05 10:54 +08:00) Added and passed two further launcher regressions: inherited unlisted `NS_*` variables are removed, and non-empty seed output directories cannot be reused without state. The exact five-file focused command now reports 190 passed; compileall and PowerShell parsing still pass.
- [x] (2026-08-05 10:58 +08:00) Ran a real `abc_0000` builder smoke over 50 pickle members. It exited zero with `VERIFIED`, 24 eligible, 15 parent-complete selected records, 0 load failures, an empty-quarantine SHA, 12/2/1 split records, and zero parent overlap.
- [ ] Commit and push the new experiment branch before training so run manifests bind to an immutable clean commit.
- [ ] Start the detached seed 0/1/2 cohort and confirm the launcher PID, child training PID, GPU activity, and advancing seed-0 log.

## Surprises & Discoveries

- Observation: the default PowerShell Python is `D:\ProgramData\miniforge3\python.exe`, which has neither NumPy nor the training dependencies.
  Evidence: baseline collection failed with `ModuleNotFoundError: No module named 'numpy'`; the same command under `C:\Users\YU\.conda\envs\brepgen_env\python.exe` passed 170 tests and sees CUDA through PyTorch 2.2.2+cu118.
- Observation: Protocol V3 already preserves archive identity in selected materialization paths as `split/archive_stem/member` and raises on a duplicate resolved target.
  Evidence: `breparg_improvements/cad_protocol.py::_materialize_selected` and `test_build_protocol_preserves_archive_identity_for_duplicate_member_names` pass. Protocol V4 therefore adds an earlier global identity preflight instead of replacing this layout.
- Observation: corrupt pickle members are already structurally excluded from assignment, but any single load failure currently makes the entire protocol summary `FAILED`.
  Evidence: `build_manifest_row` sets `protocol_eligible=False` and `load_failed:<type>`, while `build_protocol` unconditionally adds `archive_member_load_failures` to `failure_reasons` whenever the count is nonzero.
- Observation: the prior clean cohort used a protocol directory in the Protocol V2 worktree, not a copied artifact in the V3 worktree.
  Evidence: the seed-0 `vqvae_hp_sweep.json` run manifest records `NS_PROTOCOL_DIR=D:\luolin\V13\.worktrees\protocol-v2-parent-isolated\local_runs\protocol_v2_smoke_20260803\protocol`; that artifact still exists with the expected hashes.

## Decision Log

- Decision: retrain every arm from scratch for seeds 0, 1, and 2 rather than continue the two 15-epoch runs.
  Rationale: matched from-scratch state makes seed variance interpretable and avoids mixing checkpoint optimizer state or earlier run commits with a new 100-epoch cohort.
  Date/Author: 2026-08-05 / Codex.
- Decision: keep data selection deterministic and identical across experiment seeds.
  Rationale: `collect_protocol_vq_data` already uses fixed sampling seeds while `NS_VQ_EXPERIMENT_SEED` controls initialization/training, so changing the experiment seed does not change the patch cohort.
  Date/Author: 2026-08-05 / Codex.
- Decision: tolerate at most 100 load failures and 0.001 of scanned pickle members by default, requiring both limits to pass.
  Rationale: one corrupt record must not abort a healthy all-chunk build, while a wrong runtime or broadly damaged archive must still fail closed. Both thresholds are explicit in the summary and overridable on the CLI.
  Date/Author: 2026-08-05 / Codex.
- Decision: keep all corrupt-member manifest rows and also write a dedicated quarantine JSONL file.
  Rationale: the main manifest remains a complete scanned-inventory audit, while operators can inspect or retry corrupt members without searching every ordinary topology rejection.
  Date/Author: 2026-08-05 / Codex.
- Decision: hard-fail archive basename collisions, duplicate normalized qualified members, unsafe member paths, and duplicate materialization targets.
  Rationale: these are provenance ambiguities, not damaged samples that can safely be skipped. Continuing could overwrite archive locations or produce a split whose path no longer identifies one source.
  Date/Author: 2026-08-05 / Codex.
- Decision: run exactly 100 finite epochs per arm with minimum epochs and patience both set to 100.
  Rationale: this produces comparable curves through the requested horizon. The existing nonfinite validation stop remains a necessary safety boundary.
  Date/Author: 2026-08-05 / Codex.
- Decision: do not apply the `curved <= 5e-5` full-data promotion threshold to the single-chunk cohort.
  Rationale: the user explicitly limited that threshold to the full-data plateau. Protocol V4 records it as a reference but does not use it to decide whether the cohort was worth running.
  Date/Author: 2026-08-05 / Codex.
- Decision: do not enter sequence or AR stages and do not edit the excluded upstream `BrepARG/` files in this phase.
  Rationale: downstream work is explicitly gated on a healthy full-data VQ result; the three 4096-vocabulary edits belong to that later transition.
  Date/Author: 2026-08-05 / Codex.

## Outcomes & Retrospective

Implementation and launch outcomes will be recorded here after verification. A successful outcome is source/tests/docs pushed on the Protocol V4 branch plus a detached cohort process proven to have passed protocol verification, loaded the expected 12,000/4,637 cohort, started seed 0 on CUDA, and advanced its log. Completed metrics are not required in this turn.

Implementation verification is complete. Corrupt pickle records are quarantined and cannot enter a split; count/fraction thresholds distinguish isolated damage from systemic failure; archive/member identities are preflighted before materialization. The three-seed orchestrator fixes every scientific control, runs seeds sequentially, writes atomic state, stops on failure, and accepts only a sweep whose three arms each report 100 epochs. Focused verification is fully green; complete-suite failures remain identical in category and count to the Protocol V3 baseline after accounting for the 18 newly passing tests.

## Context and Orientation

The repository is a Python CAD-generation research project. `breparg_improvements/cad_protocol.py` scans parsed ABC ZIP archives, rejects records outside the paper-aligned topology bounds, assigns complete parent CAD groups to train/validation/test, materializes selected pickle members, and writes `protocol_summary.json`, `protocol_manifest.jsonl`, and `split.pkl`. A parent CAD is the source design from which one or more step records or rotations originated; parent-isolated splitting prevents leakage across cohorts.

`breparg_improvements/train.py` implements a `vqsweep` stage that loads one SHA-bound protocol split, builds the exact three FSQ arms, samples and deduplicates patches, trains each arm sequentially, and writes checkpoint/history/TensorBoard output. `NS_VQ_EXPERIMENT_SEED` changes initialization and training order; the fixed collection seeds keep the data identical. The existing V3 formal cohort used seeds 0 and 1 for 15 epochs, 12,000 train patches, 4,637 validation patches, batch 128, and learning rate `3e-4`.

The real protocol artifact to reuse is outside this worktree at `D:\luolin\V13\.worktrees\protocol-v2-parent-isolated\local_runs\protocol_v2_smoke_20260803\protocol`. It is an ignored local artifact, not something to commit. The source archives live under the ignored `D:\luolin\V13\ABC` tree. Heavy checkpoints and local run directories remain ignored; only code, tests, design/plan, concise summaries, and later curated TensorBoard logs are appropriate for Git.

## Plan of Work

First extend `tests/test_cad_protocol.py`. Replace the old assertion that any bad pickle makes a protocol fail with two behaviors: a corrupt member under configured count/fraction thresholds is present in the manifest and quarantine file but absent from all split paths, while corruption above either configured threshold makes the summary fail. Add identity fixtures for same member names in distinct archives, duplicate archive basenames in distinct directories, duplicate members within one ZIP, unsafe absolute/parent-traversal member paths, and case-normalized collisions. Run the focused test before implementation and preserve the expected failures as red evidence.

Then extend `breparg_improvements/cad_protocol.py`. Introduce a pure archive-member preflight that returns sorted safe identities and raises a descriptive `ValueError` for ambiguous or unsafe sources. Make `_iter_archive_records` consume that preflight inventory. Add `max_load_failures` and `max_load_failure_fraction` arguments to `build_protocol`, validate them, compute count/fraction policy results, and fail only when a threshold is exceeded. Extend `_write_outputs` with `quarantined_pickle_members.jsonl` and bind its SHA-256/count into the summary. Keep every load-failed row in the main protocol hash and manifest. Thread the CLI options through `tools/build_cad_protocol.py` with defaults 100 and 0.001.

Next create `tests/test_run_protocol_v4_cohort.py` against a new `tools/run_protocol_v4_100epoch_cohort.py`. Tests instantiate a cohort configuration and assert seeds `(0,1,2)`, epochs/minimum/patience 100, caps 12,000/4,637, batch 128, learning rate `3e-4`, disabled sampling weights, per-seed output/TensorBoard paths, and the command `breparg_improvements/train.py --stage vqsweep`. A fake subprocess runner proves seeds are sequential, state is written before and after each seed, and the first nonzero result prevents later seeds from launching. Invalid or duplicate seeds and non-positive controls must be rejected.

Implement the Python orchestrator with dataclasses and pure environment-building helpers. It atomically writes `cohort_state.json`, redirects each seed's stdout/stderr to its own files, and validates that a returned successful sweep contains exactly the configured arms with 100 epochs each before marking the seed complete. Add `tools/start_protocol_v4_100epoch_cohort.ps1` as a thin detached wrapper that resolves paths, starts the orchestrator hidden, writes its PID, and reports where to inspect state and logs.

Run focused tests with `brepgen_env`, compile all changed Python, and run the supported complete suite. Update this plan with exact results and any known unrelated baseline failures. Commit all code and documentation and push `experiment/protocol-v4-100epoch-fullchunk` before launching, because `train.py::require_clean_vq_run` intentionally rejects a dirty or uncommitted source state.

Finally start the wrapper against the existing Protocol V2 artifact and a new ignored output root under this worktree. Wait only long enough to observe the orchestrator and child training process, an advancing seed-0 stdout log, the expected split hashes/caps in the log or state, and CUDA/GPU activity. Do not wait for 100 epochs or aggregate the final curves during this turn.

## Concrete Steps

Use this isolated worktree for all commands:

    D:\luolin\V13\.worktrees\protocol-v3-balanced-sampling

Run the red and green focused tests with:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m pytest -p no:cacheprovider --basetemp=local_runs/protocol_v4_focused tests/test_cad_protocol.py tests/test_run_protocol_v4_cohort.py tests/test_vqvae_protocol_sampling.py tests/test_vqvae_metrics.py tests/test_vqvae_protocol_training.py -q

Compile changed modules with:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m compileall -q breparg_improvements tools tests

After verification and a clean commit, push with:

    git push -u origin experiment/protocol-v4-100epoch-fullchunk

Start the cohort with the wrapper, passing the existing verified protocol directory and an ignored local output root:

    powershell -NoProfile -ExecutionPolicy Bypass -File tools\start_protocol_v4_100epoch_cohort.ps1 -ProtocolDir D:\luolin\V13\.worktrees\protocol-v2-parent-isolated\local_runs\protocol_v2_smoke_20260803\protocol -OutputRoot D:\luolin\V13\.worktrees\protocol-v3-balanced-sampling\local_runs\protocol_v4_fsq_abc_100epoch_three_seed_20260805

The wrapper must return a launcher PID. `cohort_state.json` must report seed 0 as running, the seed-0 stdout log must grow, and `nvidia-smi` plus the process table must show the training Python process. These observations are sufficient launch acceptance.

## Validation and Acceptance

The protocol changes are accepted when a below-threshold corrupt pickle creates a `load_failed:*` manifest row and quarantine row, contributes zero split paths, and leaves a healthy fixture `VERIFIED`; an above-threshold fixture is `FAILED`; all identity ambiguity fixtures fail before materialization; and the pre-existing distinct-archive member-name test still produces distinct paths.

The launcher changes are accepted when tests prove every seed uses the same scientific controls, only seed/output paths differ, all three configured arms are expected, failed seeds stop the sequence, and only a verified 100-epoch three-arm sweep can be marked complete. The real detached launch must bind to a clean pushed commit and the existing protocol/split hashes.

The experiment is not accepted as a capacity or AR-promotion conclusion merely because it starts or completes. Later analysis must examine per-seed perplexity, code coverage, curved parent-cluster MSE, curve plateaus, and seed variance. Full-data expansion is allowed only if 4096/6D remains ahead at plateau with healthy usage. Sequence/AR remains prohibited until full-data VQ passes its representation decision.

## Idempotence and Recovery

Archive scanning and quarantine outputs are deterministic for the same sorted archive inventory, protocol config, and thresholds. Atomic output writes prevent a partial JSON/JSONL file from being mistaken for a complete one. A failed identity preflight writes no materialized split records.

The cohort output root is unique to this run. Re-running the detached wrapper against an actively running root is rejected by the orchestrator state/PID guard. If a seed exits nonzero, the state names that seed and exit code and no later seed begins. Recovery uses a new output root unless a completed seed has a fully verified 100-epoch sweep; partial checkpoints are not silently continued because this experiment is defined as matched from-scratch training.

No step deletes archives, materialized protocol data, old V3 runs, or checkpoints. `BrepARG/` and `papers/` remain untouched.

## Artifacts and Notes

Track source code, tests, this plan, the design, and a concise launch/readme update. Do not commit the approximately 229 MB checkpoints, raw full histories while they are changing, local protocol pickles, raw datasets, stdout/stderr logs, or PID/state files. Curated TensorBoard event files and a compact result JSON may be copied under `reports/` in a later result-statistics task after all runs finish.

The existing 15-epoch evidence remains Protocol V3 evidence and is not overwritten. Protocol V4 uses a new `local_runs/protocol_v4_*` root and a new Git branch.

## Interfaces and Dependencies

`breparg_improvements.cad_protocol.build_protocol` retains its existing keyword-only interface and adds:

    max_load_failures: int = 100
    max_load_failure_fraction: float = 0.001

It continues to return `(rows, split, summary)`. The summary adds a `load_failure_policy` mapping and quarantine count/hash/path-name fields. `quarantined_pickle_members.jsonl` contains only rows whose `reject_reason` starts with `load_failed:`.

`tools.run_protocol_v4_100epoch_cohort.CohortConfig` holds the protocol directory, output root, Python executable, seeds, epochs, caps, batch size, and learning rate. `build_seed_environment(config, seed)` returns the exact `NS_*` environment overlay. `run_cohort(config, runner=subprocess.run)` returns the final state mapping and is dependency-injectable for tests.

The implementation uses only the Python standard library and existing NumPy/PyTorch/Diffusers/TensorBoard training environment. PowerShell is used only to detach the Windows orchestrator.

Revision note 2026-08-05: created after the user approved a same-data three-arm 100-epoch rerun with an added third seed, clarified that bad pickle members must never enter a split, limited the current task to verified launch rather than continuous monitoring, and kept sequence/AR gated behind a later full-data VQ decision.
