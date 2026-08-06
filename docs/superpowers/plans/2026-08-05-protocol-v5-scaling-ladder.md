# Run the VQ-VAE Data Scaling Ladder and Quantizer Control

This ExecPlan is a living document and follows `AGENTS.md` and `PLANS.md` in the repository root. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be maintained as work proceeds.

## Purpose / Big Picture

After this change, the user can launch one fail-closed experiment pipeline that reuses the completed 12,000-patch result, builds one parent-isolated master protocol from 10 ABC chunks, trains nested 60,000- and 300,000-patch FSQ controls with two seeds, adds an official BrepARG-style learned VQ-4096 control at 60,000 patches, and writes curved-reconstruction and normalized-code-usage scaling plots. Unknown corrupt pickle members, excessive known corruption, or ambiguous multi-chunk identities stop the pipeline before training. The pipeline never starts sequence generation or autoregressive training.

## Progress

- [x] (2026-08-05 21:27 +08:00) Established that low GPU utilization was caused by no active training process, not a slow CUDA workload.
- [x] (2026-08-05 21:35 +08:00) Located the completed Protocol V4 12k cohort: three FSQ arms, three seeds, 100 epochs, clean pushed commit, histories and TensorBoard evidence.
- [x] (2026-08-05 21:48 +08:00) Located 100 local parsed ABC archives containing 681,406 pickle members and confirmed E: has about 3.9 TB free.
- [x] (2026-08-05 21:52 +08:00) Fixed the scaling design: 12k reuses E030; 60k and 300k use nested patch caps from one chunks-0-9 master protocol capped at 15,000 eligible CAD records; A is FSQ-8192/4D; B is FSQ-4096/6D; the 60k-only C arm is learned VQ-4096/64D with history-pool random restart.
- [x] (2026-08-05 23:06 +08:00) Implemented and tested the fail-closed corrupt-pickle allowlist, empty failed manifests, and global multi-chunk identity evidence.
- [x] (2026-08-05 23:06 +08:00) Implemented and tested quantizer-selectable VQ sweeps, the 60k learned-VQ arm, and the conditional continuous bypass arm.
- [x] (2026-08-05 23:06 +08:00) Implemented and tested the unattended ladder orchestrator, plateau decisions, summaries, plots, and bypass-oracle recommendation gate.
- [x] (2026-08-05 23:25 +08:00) Ran focused, full-regression, syntax, PowerShell-parser, diff, inventory, and real CUDA forward/backward verification; committed implementation as `e771ecc` and pushed `experiment/protocol-v5-scaling-ladder` to `origin`.
- [ ] Launch the ladder and observe protocol/training health long enough to rule out an immediate failure.

## Surprises & Discoveries

- Observation: Protocol V4 already quarantines corrupt members and checks archive-qualified member/materialization uniqueness, but an unknown corrupt member below a numeric threshold still leaves the protocol `VERIFIED`.
  Evidence: `breparg_improvements/cad_protocol.py::build_protocol` only compares load failures with `max_load_failures` and `max_load_failure_fraction`.
- Observation: Protocol V4 completed more evidence than the requested 12k rung requires.
  Evidence: E030 contains nine completed 100-epoch histories for three arms and seeds 0, 1, and 2. V5 treats this as the 12k scaling point and does not spend GPU time reproducing it.
- Observation: all 100 parsed archives are present locally, although a shallow directory listing initially exposed only `ABC/processed`.
  Evidence: `ABC/processed/abc_parsed_full_archives` contains `abc_0000_parsed.zip` through `abc_0099_parsed.zip`, 681,406 pickle members and about 174 GB compressed data.
- Observation: the detached PowerShell launcher creates stdout, stderr, and PID control files before or concurrently with Python startup.
  Evidence: a strict-empty Python workspace check rejected an otherwise valid detached launch. Runtime validation now permits only those three ordinary files and still rejects every other pre-existing entry.
- Observation: the real all-archive metadata preflight completed successfully.
  Evidence: 100 archives expose 681,406 pickle members, 681,406 unique archive-qualified source keys, and 681,406 unique materialization keys; inventory SHA256 is `5c0f93b016e24a48ee638b2a97f270f020b6997aff02d326f1823baef1c33618`.
- Observation: the first C-drive retry completed protocol construction and materialized all 15,000 selected records, but failed at the 60k learned-VQ arm before producing a sweep report.
  Evidence: `C:\V13_protocol_v5_scaling_20260805_local\protocol\protocol_summary.json` is `VERIFIED` with zero load failures and zero parent overlap; `train_60k_seed0.stderr.log` reports an upstream `FeaturePool` Float destination versus AMP Half source mismatch.
- Observation: the failure is an interface dtype mismatch at the excluded upstream quantizer boundary, not a data, disk, or CUDA capacity failure.
  Evidence: FSQ arms reached epochs 64 and 75, the official VQ crash occurs in `FeaturePool.query`, and a CUDA encoder?quantizer?decoder smoke reproduces the issue only for Half latent input.

## Decision Log

- Decision: require every corrupt pickle to match an explicit allowlist entry containing its archive-qualified source key, compressed-member CRC, uncompressed size, and exception type; also require both count and fraction thresholds.
  Rationale: numeric thresholds alone tolerate previously unseen damage. Binding the allowlist to immutable ZIP metadata prevents a path-only exception from silently approving changed bytes.
  Date/Author: 2026-08-05 / Codex.
- Decision: return an empty materialized split whenever an unknown or excessive load failure is observed.
  Rationale: a failed protocol should not leave a plausible-looking training split that can be consumed accidentally. The complete manifest and quarantine evidence remain available for review and allowlist generation.
  Date/Author: 2026-08-05 / Codex.
- Decision: use one chunks-0-9 master protocol capped at 15,000 eligible CAD records for both the 60k and 300k rungs.
  Rationale: the first ten archives contain 66,879 records. The measured single-chunk cohort produced 27,604 unique train patches from 795 train records, so about 12,000 train records in a 15,000-record 8:1:1 master split leave ample margin beyond 300,000 patches. Both rungs then share exactly the same parent split and deterministic patch order, while the larger cap is a prefix extension instead of a different data distribution. Scanning and retaining patch arrays from 50 chunks would add memory risk without helping this ladder.
  Date/Author: 2026-08-05 / Codex.
- Decision: compare normalized usage `perplexity / codebook_size` on scaling plots, while preserving raw perplexity and coverage.
  Rationale: raw perplexity is not directly comparable between 4096- and 8192-entry codebooks. The normalized score has a common zero-to-one interpretation.
  Date/Author: 2026-08-05 / Codex.
- Decision: implement the learned VQ arm by importing the local BrepARG `VectorQuantiser` without modifying or tracking `BrepARG/`.
  Rationale: this preserves the official 64-dimensional free codebook, cosine distance, contrastive loss, and historical `FeaturePool` behavior while keeping all project changes in our own code. Relative to the local official trainer, only `num_embed=4096` and the requested `anchor=random` change; `embed_dim=64`, the Diffusers commitment beta, `distance=cos`, `first_batch=False`, and `contras_loss=True` remain official.
  Date/Author: 2026-08-05 / Codex.
- Decision: define “train to plateau” as a maximum of 100 epochs with minimum 40 epochs, patience 15, and minimum curved parent-cluster MSE improvement of 1e-5, while still keeping the existing non-finite stop.
  Rationale: the 3060 should not spend the full horizon after the decision metric has stabilized, but all arms receive identical stop controls.
  Date/Author: 2026-08-05 / Codex.
- Decision: conditionally run a 300k continuous-latent bypass after the scaling analysis reports `CONTINUE_CAPACITY_INVESTIGATION`.
  Rationale: task four requests the oracle only when the 300k extrapolation misses the target. The state machine can make that decision from completed evidence, then run the same encoder/decoder and data with quantization replaced by an identity pass-through for seeds 0 and 1. It skips the oracle when the projection is plausible. No AR stage is reachable from this pipeline.
  Date/Author: 2026-08-05 / Codex.
- Decision: keep the upstream `BrepARG/` source unchanged and wrap the learned quantizer in our `train.py` with a float32 internal boundary, restoring the incoming dtype for the decoder.
  Rationale: the official codebook and FeaturePool remain numerically unchanged, while both cosine-distance math and history-pool writes become valid under AMP. The failed C workspace and E recycle-bin copy remain untouched as audit evidence; the corrected run uses a new C-drive workspace.
  Date/Author: 2026-08-06 / Codex.

## Outcomes & Retrospective

Source implementation is published on `origin/experiment/protocol-v5-scaling-ladder`; implementation commit `e771ecc` contains no raw data, checkpoints, logs, or upstream `BrepARG/` source. The first C-drive retry reached the protocol and failed only at learned VQ seed0 due to the diagnosed AMP dtype boundary. The corrected adapter has 87 focused VQ/ladder tests passing, a finite CUDA end-to-end learned-VQ smoke with float32 FeaturePool, Python compilation, and diff checks passing. The full `tests/` baseline remains 444 passing plus the documented 16 unrelated failures. The corrected retry launch is the remaining runtime action; numerical scaling results do not yet exist.

## Context and Orientation

`breparg_improvements/cad_protocol.py` scans parsed ABC ZIP archives, filters CAD records by topology, groups records by parent CAD, assigns whole parents to splits, and materializes selected pickle files. `tools/build_cad_protocol.py` is its command-line wrapper. A corrupt-pickle allowlist is a JSON file that records exactly which known unreadable archive members may be skipped. “Fail closed” means absence, mismatch, or threshold excess produces a failed protocol and no consumable split.

`breparg_improvements/train.py --stage vqsweep` trains multiple VQ-VAE arms over one audited split. The encoder and decoder are identical across arms. FSQ maps a low-dimensional vector to a fixed finite grid. Learned VQ maps a 64-dimensional vector to a freely learned embedding table. `D:/luolin/V13/BrepARG/quantise.py::VectorQuantiser` is the official local reference implementation and is imported at runtime without changing the excluded upstream directory.

`tools/run_protocol_v5_scaling_ladder.py` will be the unattended state machine. It validates immutable inputs, builds the 60k and 300k protocol rungs on E:, runs each seed serially, checks complete sweep reports, aggregates results, and writes JSON/CSV/PNG evidence. A state machine is a JSON record whose `status` and current rung make interruption and failure unambiguous.

## Plan of Work

First extend protocol tests with an explicit known-corruption allowlist and an unknown-corruption failure. Add ZIP metadata to `ArchiveMember`, write candidate allowlist entries for every failure, bind the configured allowlist hash into the summary, and materialize only after all load-failure gates pass. Record global archive, member, source-key, and materialization-key counts plus an inventory hash in the protocol summary.

Then extend VQ training tests with quantizer configuration objects. Preserve the existing FSQ default, allow an explicit arm list through `NS_VQ_SWEEP_ARMS`, and add `build_learned_vqvae()` around BrepARG `VectorQuantiser`. Generalize checkpoint metadata from FSQ-only fields to a quantizer descriptor while retaining `fsq_levels` for compatibility. Make the 60k configuration expose exactly A, B, and learned VQ; make 300k expose only A and B.

Next add the V5 ladder orchestrator and analysis module. Tests will assert the nested chunk definitions, two fixed seeds, serial execution, strict protocol status/hash checks, exact arm matrices, plateau controls, complete-output validation, normalized usage calculation, log-data extrapolation, plot output, and a bypass recommendation only after 300k predicts missing the target. A Windows wrapper will start the orchestrator hidden and report its PID/state/log paths.

Finally run focused tests and the supported regression suite, compile Python, parse PowerShell, commit, and push `experiment/protocol-v5-scaling-ladder`. Launch with the archive root on D: and the heavy workspace on E:. Observe a progressing protocol state, finite resource usage, and then CUDA activity when training begins. Do not wait continuously after health is established, and do not enter sequence or AR work.

## Concrete Steps

Run from `D:\luolin\V13\.worktrees\protocol-v3-balanced-sampling` with `C:\Users\YU\.conda\envs\brepgen_env\python.exe`.

Focused verification:

    python -m pytest -p no:cacheprovider --basetemp=local_runs/protocol_v5_tests tests/test_cad_protocol.py tests/test_vqvae_metrics.py tests/test_vqvae_protocol_training.py tests/test_run_protocol_v5_scaling_ladder.py -q

Syntax verification:

    python -m compileall -q breparg_improvements tools tests
    powershell -NoProfile -Command "[System.Management.Automation.Language.Parser]::ParseFile('tools/start_protocol_v5_scaling_ladder.ps1',[ref]$null,[ref]$null)"

After a clean commit, launch:

    powershell -NoProfile -ExecutionPolicy Bypass -File tools\start_protocol_v5_scaling_ladder.ps1 -ArchiveRoot D:\luolin\V13\ABC\processed\abc_parsed_full_archives -WorkspaceRoot E:\V13_protocol_v5_scaling_20260805

## Validation and Acceptance

Protocol acceptance requires tests proving an unknown bad pickle produces `FAILED`, an empty split, and a candidate allowlist record; the exact approved record can be quarantined only while both thresholds pass; changed CRC/size/error metadata invalidates approval; no failed member enters any split; and globally ambiguous archive/member/materialization identities fail before unpickling.

Training acceptance requires a forward/backward smoke for learned VQ-4096 with valid index range, a manifest that records quantizer type/dimension/restart policy, and sweep tests proving the correct rung-specific arms. Scaling acceptance requires 12k E030 ingestion, 60k/300k result ingestion, curved MSE and normalized usage plots, and a machine-readable extrapolation with either `CONTINUE_CAPACITY_INVESTIGATION` or `TARGET_PLAUSIBLE`. The ladder must contain no sequence or AR command.

Launch acceptance requires the detached process and state file to exist, the archive inventory/protocol log to advance, no immediate exception, and later CUDA activity when a training child starts. Low GPU utilization during ZIP scanning and materialization is expected and must be labeled as a CPU/I/O phase.

## Idempotence and Recovery

The pipeline writes each rung to a new workspace and uses atomic JSON state writes. Existing non-empty output roots are rejected. Protocol outputs are deterministic for the same archive inventory, allowlist, thresholds, chunks, and seed. If unknown corrupt members are found, the run stops after writing candidate allowlist evidence; the operator can review the candidate file, place an approved copy at the configured allowlist path, and relaunch into a new workspace. No archive or prior experiment is deleted or modified.

## Artifacts and Notes

Git tracks code, tests, this plan, concise JSON/CSV/PNG summaries, histories, and curated TensorBoard logs after runs complete. It excludes raw archives, materialized split pickles, model checkpoints, stdout/stderr logs, PID files, and active state directories. The V4 E030 report remains the immutable 12k point.

## Interfaces and Dependencies

`breparg_improvements.cad_protocol.build_protocol` adds `load_failure_allowlist_path: Path | None`. Its summary adds allowlist identity/status, candidate count/path/hash, archive member inventory count/hash, and uniqueness counts.

`breparg_improvements.train.quantizer_comparison_configs(arm_names=None)` returns descriptors containing `name`, `kind`, `codebook_size`, optional `levels`, and learned-VQ settings. `build_quantized_vqvae(config)` builds the selected arm.

`tools.run_protocol_v5_scaling_ladder.LadderConfig` defines archive/workspace roots, seeds, epoch controls, and rung definitions. `run_ladder(config, runner=subprocess.run)` writes atomic state and executes serial protocol/training/analyze phases. `tools.summarize_protocol_v5_scaling` reads completed reports and emits the scaling evidence and plots.

Revision note 2026-08-05: created after verifying the completed 12k cohort, the absence of an active training process, and the availability of all 100 local parsed ABC archives.

Revision note 2026-08-05: recorded the completed implementation, launcher race regression, focused test result, and real 681,406-member archive inventory preflight.

Revision note 2026-08-05: recorded implementation commit `e771ecc` and publication to the V5 experiment branch.
