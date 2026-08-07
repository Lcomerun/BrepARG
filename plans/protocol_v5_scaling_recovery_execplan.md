# Recover Protocol V5 scaling analysis and finish the capacity oracle

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current while the work proceeds. This document follows `PLANS.md` in the repository root. `AGENTS.md` refers to `.agent/PLANS.md`, but that path is absent in this checkout; the checked-in root `PLANS.md` is therefore the available authority.

## Purpose / Big Picture

Protocol V5 compares two finite scalar quantization configurations and one learned vector quantizer at increasing training-data sizes. The numerical training runs completed, but the final reporting process crashed inside the Windows native Matplotlib renderer before the conditional continuous-latent decoder oracle could start. After this plan is complete, the existing training is preserved, the scaling report can be rendered without Matplotlib, the failed ladder can resume without repeating any successful sweep, and the two-seed continuous-latent oracle will determine whether the remaining curved-surface error is caused primarily by quantization or by the encoder/decoder itself. The workflow must not start sequence regeneration or autoregressive training.

## Progress

- [x] (2026-08-07 09:25 +08:00) Confirmed that protocol construction, both 60k seeds, and both 300k seeds completed with return code zero; only the analysis step failed.
- [x] (2026-08-07 10:10 +08:00) Reproduced Windows fatal exception `0xc06d007f` in Matplotlib transform and Agg rendering code, including when `MPLBACKEND=Agg` was forced.
- [x] (2026-08-07 10:28 +08:00) Added a failing regression test that required scaling PNG generation without importing Matplotlib, then implemented the Pillow renderer and observed the test pass.
- [x] (2026-08-07 10:42 +08:00) Added a failing recovery test that rejects replay of successful protocol/training work, then implemented `resume_ladder_after_analysis` and observed the test pass.
- [x] (2026-08-07 11:04 +08:00) Ran the focused Protocol V5 suite with a repository-specific Git configuration: 146 tests passed.
- [x] (2026-08-07 11:06 +08:00) Re-ran the real scaling analysis successfully and regenerated its PNG, CSV, and JSON artifacts in the original experiment workspace.
- [x] (2026-08-07 10:27 +08:00) Committed the renderer and recovery changes as `661e879` and pushed them, together with the preceding learned-VQ AMP fix, to `experiment/protocol-v5-scaling-ladder`.
- [ ] Resume the failed ladder at analysis, run the two continuous-bypass seeds, and validate each sweep artifact (completed: analysis retry passed and seed 0 reached its first fully finite epoch at 96-98% GPU utilization; remaining: automatic completion and validation of seed 0 and seed 1).
- [ ] Record the oracle comparison and final state while keeping `advance_to_ar=false`.

## Surprises & Discoveries

- Observation: The apparent low GPU utilization was not a slow input pipeline. All four requested quantized training jobs had already exited successfully, and the ladder was stopped in a CPU-only reporting phase.
  Evidence: `ladder_state.json` records return code zero for `train_60k_seed0`, `train_60k_seed1`, `train_300k_seed0`, and `train_300k_seed1`, followed by `analyze_scaling` return code `3228369023`, hexadecimal `0xC06D007F`.

- Observation: Selecting the non-interactive Matplotlib Agg backend does not avoid this environment failure.
  Evidence: `python -X faulthandler` first showed the fatal exception through `numpy.linalg.inv`, `matplotlib.transforms`, and `tight_layout`; after removing `tight_layout`, the same native exception occurred in `matplotlib.transforms.get_affine` during Agg `savefig`.

- Observation: The learned 64-dimensional VQ control is already materially better than the six-dimensional FSQ arm at 60k patches.
  Evidence: The scaling summary reports curved parent MSE about `0.00262545` for learned VQ versus `0.00442326` for FSQ-4096/6D, with learned-VQ perplexity about `1546.28` and normalized usage about `0.37751`.

- Observation: Increasing FSQ-4096/6D from 60k to 300k patches does not extrapolate near the full-data curved target.
  Evidence: The shared-protocol two-point projection estimates full-data curved parent MSE about `0.00249854`, far above `0.00005`.

## Decision Log

- Decision: Replace only the report PNG renderer with Pillow while retaining filenames, dimensions, axes, colors, markers, legends, and scaling semantics.
  Rationale: The crash is in Matplotlib's native rendering stack rather than the summary data. Pillow provides deterministic local raster drawing and is already available in the training environment.
  Date/Author: 2026-08-07 / Codex

- Decision: Recovery is allowed only from `status=FAILED` and `phase=ANALYSIS`, after independently revalidating the verified protocol and every completed 60k/300k sweep.
  Rationale: This prevents an accidental recovery from skipping incomplete or invalid numerical work and guarantees the existing expensive runs are reused only when their artifacts satisfy the original gates.
  Date/Author: 2026-08-07 / Codex

- Decision: Run the continuous-bypass oracle because the generated decision explicitly recommends it, but never advance to sequence or autoregressive work in this ladder.
  Rationale: The oracle separates decoder capacity from quantization capacity. Autoregressive training cannot repair a representation bottleneck and would confound the root-cause experiment.
  Date/Author: 2026-08-07 / Codex

- Decision: Use a task-specific `GIT_CONFIG_GLOBAL` file containing only this checkout's `safe.directory` entry.
  Rationale: The runtime repository is owned by a previous sandbox identity. A project-local configuration makes Git revision capture work in tests and training subprocesses without changing the user's global Git settings.
  Date/Author: 2026-08-07 / Codex

## Outcomes & Retrospective

The reporting failure is fixed and the real scaling report now renders successfully. The numerical conclusion did not change: learned VQ outperforms FSQ-4096/6D at 60k, while FSQ scaling remains orders of magnitude above the curved reconstruction target. Recovery has entered the two-seed continuous-latent oracle without replaying prior training. Seed 0 completed epoch 0 with `train=0.05907`, `val=0.02195`, `finite_train=2344/2344`, and `finite_val=94/94`, while the RTX 3060 sustained about 96-98% utilization. The background parent process will validate seed 0 and then start seed 1 automatically. When both complete, this section must record curved parent MSE relative to the 60k learned-VQ and 300k FSQ arms, the final ladder status, and whether the evidence points to quantization or shared encoder/decoder capacity.

## Context and Orientation

The source checkout is `D:\luolin\V13\local_runs\protocol_v5_runtime_20260806`. The experiment workspace is `D:\luolin\V13\local_runs\protocol_v5_scaling_run_20260806`. The workspace's `ladder_state.json` is an atomic state journal containing each command, phase, return code, and log path.

`tools/summarize_protocol_v5_scaling.py` reads the completed sweep JSON files, aggregates seed means and standard deviations, writes `scaling_points.csv` and `scaling_summary.json`, and renders `curved_mse_scaling.png` and `usage_scaling.png`. `tools/run_protocol_v5_scaling_ladder.py` builds commands and environment variables for protocol construction, quantized sweeps, analysis, and the conditional continuous-bypass sweep. `breparg_improvements/train.py --stage vqsweep` is the common training entry point; an environment-selected arm controls whether it trains FSQ, learned VQ, or a continuous latent without quantization.

The continuous-bypass oracle keeps the same parent-isolated protocol, sampling cap, architecture family, epoch gates, and two seeds, but removes the quantizer. Its result answers whether reconstruction remains poor even when no discrete bottleneck is present. A low oracle error relative to all quantized arms implicates quantization capacity; a similarly high oracle error implicates the shared encoder/decoder or data representation.

## Plan of Work

Keep the numerical aggregation unchanged. In `tools/summarize_protocol_v5_scaling.py`, implement PNG generation using Pillow drawing primitives and no Matplotlib import. Preserve logarithmic patch and curved-MSE coordinate transforms, the linear usage transform, all current arm colors, the legend, and output names.

In `tools/run_protocol_v5_scaling_ladder.py`, keep the normal fresh-run behavior and extract the common analysis/oracle tail. Add a strict recovery function that reads the existing state, permits only the known failed-analysis state, revalidates the protocol summary and all expected sweep JSON files, resets the state to running, and invokes only the shared tail. Do not rerun archive inventory, protocol construction, or the 60k/300k sweeps.

In `tests/test_run_protocol_v5_scaling_ladder.py`, exercise real PNG decoding with Pillow while blocking Matplotlib imports, and use a fake runner plus complete fixture sweeps to prove recovery calls exactly one analysis and two oracle commands. Assert that no sequence or AR command is built and that the final state always records `advance_to_ar=false`.

After tests and the real analysis pass, commit the changes and push the experiment branch. Invoke `resume_ladder_after_analysis` with the recorded runtime paths and the repository-specific Git configuration inherited by subprocesses. Confirm that analysis exits zero, the state enters `GPU_ORACLE_CONTINUOUS_BYPASS`, a CUDA Python process appears, and stderr has no immediate failure. After both seeds complete, validate their sweep JSON files and record the final decision.

## Concrete Steps

Run all commands from `D:\luolin\V13\local_runs\protocol_v5_runtime_20260806` using `C:\Users\YU\.conda\envs\brepgen_env\python.exe`.

Create a task-specific Git configuration:

    git config --file D:\luolin\V13\local_runs\protocol_v5_runtime_gitconfig --replace-all safe.directory D:/luolin/V13/local_runs/protocol_v5_runtime_20260806
    $env:GIT_CONFIG_GLOBAL = 'D:\luolin\V13\local_runs\protocol_v5_runtime_gitconfig'

Verify the implementation:

    python -m pytest tests/test_cad_protocol.py tests/test_vqvae_metrics.py tests/test_vqvae_protocol_training.py tests/test_run_protocol_v5_scaling_ladder.py -q
    python -m compileall -q breparg_improvements tools tests
    git diff --check

The expected focused test result is `146 passed`. The real analysis command must exit zero and leave nonempty PNG, CSV, and JSON files under `D:\luolin\V13\local_runs\protocol_v5_scaling_run_20260806\analysis`.

Resume by constructing `LadderConfig` with the paths stored in `ladder_state.json` and calling `resume_ladder_after_analysis(config)`. This operation is deliberately idempotent only at the known failure point: after it advances beyond failed analysis, calling it again must be rejected rather than duplicate work.

## Validation and Acceptance

The plotting fix is accepted when the regression test can block Matplotlib imports, call the renderer, and open both output PNGs with Pillow. It is additionally accepted only when the real analysis command returns zero with the original V5 sweep data.

Recovery is accepted when the recovery test shows that only analysis and two oracle commands run, while inventory, protocol creation, 60k training, 300k training, sequence generation, and AR training do not run.

The full experiment is accepted when `ladder_state.json` records `status=COMPLETED`, `phase=COMPLETED`, `continuous_bypass_oracle=COMPLETED`, and `advance_to_ar=false`; both `continuous_bypass_300k/seed0` and `seed1` contain valid `vqvae_hp_sweep.json` artifacts satisfying epoch, patch-cap, parent-coverage, and arm-name gates.

## Idempotence and Recovery

The summary renderer safely overwrites its four report artifacts. The recovery entry point refuses states other than the precise failed-analysis state, so it cannot silently duplicate a live or completed oracle. Each subprocess step writes stdout and stderr to named files and updates `ladder_state.json` atomically before and after execution. If an oracle seed fails, retain its files and state journal for diagnosis; do not restart the whole ladder or begin AR.

## Artifacts and Notes

The first successful real-data Pillow analysis printed:

    status: CONTINUE_CAPACITY_INVESTIGATION
    continuous_bypass_oracle_recommended: true
    advance_to_ar: false

The original failure is preserved in the step history even after recovery. A new successful `analyze_scaling` step is appended, which makes the audit trail show both the environmental crash and its successful retry.

## Interfaces and Dependencies

`tools.summarize_protocol_v5_scaling.render_scaling_pngs(summary: Mapping[str, Any], output_dir: Path) -> None` must create `curved_mse_scaling.png` and `usage_scaling.png` without importing Matplotlib.

`tools.run_protocol_v5_scaling_ladder.resume_ladder_after_analysis(config: LadderConfig, *, runner: Callable = subprocess.run) -> dict[str, Any]` must validate existing artifacts, append recovery steps through the common analysis/oracle tail, atomically persist state, and return the final in-memory state.

Pillow is the only added runtime dependency used by reporting, and it is already installed in the `brepgen_env` environment. Training dependencies and model behavior remain unchanged.

Revision note 2026-08-07: Created this recovery plan after the native Matplotlib crash was isolated, documenting the existing TDD evidence, safe resume design, and conditional oracle completion criteria.

Revision note 2026-08-07 10:41 +08:00: Recorded the pushed fix commit and the healthy automatic recovery state after continuous-bypass seed 0 completed its first fully finite epoch under sustained GPU load.
