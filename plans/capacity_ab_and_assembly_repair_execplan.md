# Resolve quantization capacity and repair the CAD assembly chain

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must remain current while work proceeds. This document follows `PLANS.md` in the repository root. `AGENTS.md` refers to `.agent/PLANS.md`, but that file is absent, so the checked-in root `PLANS.md` is the available authority.

## Purpose / Big Picture

The same frozen 100 validation CADs currently assemble at 70 percent strict validity through the continuous-latent bypass but only 57 percent through learned VQ-4096. The paired gap is statistically credible: 15 CADs work only through bypass and 2 only through VQ (`McNemar p=0.0023`). This plan determines whether a larger single codebook or a two-stage residual codebook removes that quantization tax, while a parallel CPU-only track repairs the original assembly chain from 84 percent strict validity toward at least 95 percent without breaking any of the 84 original successes.

At completion, the repository will contain code and Git-safe evidence for VQ-8192/64D and two-stage RVQ-4096/64D at seeds 3 and 4, each trained for exactly 100 epochs on the same frozen 60,000/12,000 patch inventories and stability recipe used by P0-B. Both candidates will be reconstructed and assembled on the same ordered 100-CAD cohort, with paired McNemar statistics and the sequence-length cost reported. The assembly repairs will be individually switchable and will map every repair to the exact CADs it restores or regresses. A failure-triggered selector will preserve the current best strict-valid construction path and may replace only failed CADs with independently valid, geometry-preserving fallback outputs. The winner will then be measured again through the repaired chain. Checkpoints, STEP files, raw arrays, pickles, source data, and upstream `BrepARG/` source remain local; histories, compact logs, TensorBoard events, statistics, hashes, manifests, tests, and documentation are committed.

Sequence regeneration, autoregressive training, and the boundary-consistency loss remain blocked until the final capacity and repaired-chain gates pass.

## Progress

- [x] (2026-08-17 10:25 +08:00) Verified the starting branch and evidence. Local HEAD is `f31bd11`, remote `experiment/protocol-v5-scaling-ladder` is at `82a236a`, and the only unpushed commit is the prior paired-gate documentation closeout.
- [x] (2026-08-17 10:25 +08:00) Verified the RTX 3060 is idle and no Python trainer is running. D: has about 43.7 GB free and E: about 3.6 TB, so new local training artifacts will use E: while the Git working tree remains on D:.
- [x] (2026-08-17 10:25 +08:00) Reconfirmed the frozen Protocol V5 identities: protocol SHA-256 `6b588ee0a9dc337a683d9cc94cde7d79a80963720d22098d99e7f6eaa8101cf3`, split SHA-256 `6ff0a0c3ee6a04ee056fa1ab982eb436a9f59d3d21f21f17babf34e6dc701d29`, train count 60,000, validation count 12,000, and zero parent overlap.
- [x] (2026-08-17 11:09 +08:00) Implemented and tested VQ-8192/64D and two-stage RVQ-4096/64D, including independent RVQ stage usage, perplexity, coverage, and collapse evidence. Preserved the legacy three-item quantizer-info contract and forced RVQ residual subtraction to fp32 under bf16 input.
- [x] (2026-08-17 11:09 +08:00) Implemented the fail-closed capacity launcher for seeds 3 and 4, 100 epochs, bf16, batch 128, LR `3e-4`, gradient clip 1.0, plateau decay, zero-nonfinite fuse, atomic resume, and identical inventory digests. A non-mutating formal dry run produced exactly four signed tasks; the later storage probe moved formal outputs from E: to D:.
- [x] (2026-08-17 11:09 +08:00) Passed 176 focused and compatibility tests covering the capacity quantizers, launcher, stability lifecycle, prior P0-B behavior, fixed-cohort measurement, and P0-A snapshot/diagnostic paths.
- [x] (2026-08-17 11:36 +08:00) Ran a first CUDA smoke. VQ-8192 was numerically finite, but fail-closed evidence validation exposed two producer omissions before RVQ started: scheduler metric and checkpoint parent-overlap counts. Corrected both at the producer, added contract regressions, and passed the 157-test capacity/stability/assembly compatibility suite.
- [x] (2026-08-17 11:51 +08:00) Completed a clean two-arm CUDA smoke on the healthy D: NTFS SSD. Both VQ-8192 and RVQ validators returned `valid=true`, with identical inventories, resume-compatible runtime evidence, and zero non-finite events.
- [x] (2026-08-17 13:00 +08:00) Stopped the first formal root at VQ-8192 seed 3 epoch 52 and classified it as diagnostic-only. Its signed scheduler declared `curved_parent_mse`, but the launcher omitted `NS_VQ_PLATEAU_METRIC`, so the actual scheduler consumed global validation MSE. No later arm or seed was started.
- [x] (2026-08-17 13:20 +08:00) Upgraded the capacity evidence contract to schema v2: the launcher explicitly injects every scheduler, sampling, loss, and stop control; history and validator bind the real plateau metric; source and clean-Git identity coverage is expanded; exact batch/lifecycle checks fail closed; and existing state rejects environment drift.
- [x] (2026-08-17 13:45 +08:00) Added and executed a face/wire-local, Git-safe diagnosis of the frozen 16 P0-A invalid controls. The report distinguishes saved-STEP cases from failures before a STEP existed; unavailable pcurve evidence is never reported as a negative finding.
- [x] (2026-08-17 13:50 +08:00) Bound the local diagnosis to the P0-A stage-aware baseline attempts rather than the earlier 100-CAD calibration manifest. The corrected rerun covers 11 saved STEP cases and 5 no-STEP cases; the stale 10/6 partial result is not used.
- [x] (2026-08-17 13:55 +08:00) Snapshotted the completed 16-case face/wire diagnosis into `reports/p0a_face_wire_diagnosis_20260817/`. Archive validation found zero forbidden model/data/CAD artifacts.
- [x] (2026-08-17 13:20 +08:00) Repaired RVQ evidence semantics. Stage indices now use independent namespaces, every stage reports `usage_fraction = entropy_perplexity / codebook_size`, best-checkpoint selection uses stage-specific historical stability instead of a conflated marginal, and the RVQ-over-VQ material advantage is frozen at 5 percentage points plus exact paired significance.
- [x] (2026-08-17 13:18 +08:00) Re-ran the two-arm real-CUDA smoke from clean schema-v2 source. Both VQ-8192 and RVQ returned `valid=true`; RVQ's two stage usage reports and scheduler bindings passed.
- [x] (2026-08-17 13:30 +08:00) Completed the forced interruption/resume behavior: epoch 0 was checkpointed, both processes were killed, and the replacement process restored full state and completed epoch 1 with `resumed=true` and `resume_from_epoch=0`. The smoke validator now distinguishes deduplicated smoke batch counts from formal exact-cap counts while still requiring all observed batches finite.
- [x] (2026-08-17 20:20 +08:00) Archived the completed P0-B 60k formal cohort as a Git-safe snapshot: four 100-epoch histories, four TensorBoard event files, compact logs, metrics, manifests, and checkpoint SHA-256/size bindings. Archive validation reports 36 files, four histories, four TensorBoard events, no forbidden artifacts, and `valid=true`; the 4.27 GiB of checkpoint bytes remain local.
- [x] (2026-08-17 20:30 +08:00) Archived the P0-A frozen-invalid16 diagnosis: all 96 paired attempts (`16 cases x 2 joint settings x 3 sewing tolerances`) completed, all 16 cases have a primary attribution, and the taxonomy is 10 wire self-intersections, 3 curve-fit failures, 2 wire-build failures, and 1 non-unit/empty-solid case. The archive contains the per-case JSONL, detailed attempts, ablation, tolerance scan, taxonomy, compact CSV, repair checklist, and byte manifest only.
- [x] (2026-08-17 21:00 +08:00) Completed the unchanged-chain paired P0-B assembly measurement on the frozen ordered 100-CAD cohort using seed-3 best checkpoints for bypass and VQ-4096. The cohort identity hash, checkpoint hashes, protocol hash, split hash, command return codes, and log hashes are bound in `measurement_runtime_manifest.json`; all 100 attempts remain in the denominator.
- [x] (2026-08-18 09:30 +08:00) Revalidated the three Git-safe report directories and their focused regression tests (`33 passed`). The archive and paired measurement commits are present in `origin/experiment/protocol-v5-scaling-ladder`; no force push or checkpoint upload is required.
- [x] (2026-08-17 13:29 +08:00) Started the new four-task formal root `D:\luolin\V13\local_runs\capacity_ab_60k_v2_20260817` from clean schema-v2 commit `76604edb4bc06f6be7f4740d44886b1099e95eaa`. The old `capacity_ab_60k_20260817` root remains quarantined and is not resumed or promoted.
- [x] (2026-08-17 14:12 +08:00) Rechecked the live schema-v2 task after epoch 30. VQ-8192 seed 3 had exact `469/469` train and `94/94` validation batches, zero non-finite counters, 97 percent GPU utilization, curved-parent MSE `0.0042640607`, validation perplexity `2411.32`, and coverage `0.822144`; the serialized scheduler metric equals curved-parent MSE.
- [x] (2026-08-17 21:23 +08:00) Finished the immutable schema-v2 formal capacity matrix. All four tasks validated at exactly 100 epochs with identical train/validation inventories and resume-compatible runtime evidence. RVQ seed 4 hit a Windows `PermissionError` while replacing its rolling checkpoint after epoch 51; attempt 2 resumed from the same signed rolling state and completed with exit code 0. The final root is `COMPLETED` and `formal_result_eligible=true`.
- [x] (2026-08-17 21:39 +08:00) Measured the two seed-3 best checkpoints through the unchanged fixed 100-CAD chain. Bypass was `70/100` strict, VQ-8192 was `69/100`, and RVQ was `65/100`; exact RVQ-vs-VQ strict McNemar `p=0.4239501953125`, with 5 RVQ-only versus 9 VQ-only successes. The preregistered decision is `VQ_8192_DIRECT_WIN`; the Git-safe report is `reports/capacity_ab_assembly_measurement_20260817/` with measurement signature `2dff0044...`.
- [x] (2026-08-18 05:47 +08:00) Re-ran the capacity A/B from post-hardening source `D:\capv5` at commit `6f7436da50a5f455fd9af3c806676ac0f49b8f9f`. All four tasks validated at exactly 100 epochs with identical inventories, and the fixed 100-CAD measurement reproduced `VQ_8192_DIRECT_WIN` in `reports/capacity_ab_posthardening_assembly_measurement_20260818/`.
- [x] (2026-08-17 14:25 +08:00) Added the independent schema-v2 crossing diagnosis and Git-safe `reports/p0a_face_wire_diagnosis_v2_20260817/` snapshot without changing the v1 report. All 16 aggregate self-intersecting wires received 1-based edge-position evidence: 11 adjacent, 5 closure, 4 non-adjacent, and 7 pcurve-gap occurrences; self-only, seam, and disconnected counts are zero, while the five pre-STEP cases remain explicitly unavailable.
- [x] (2026-08-17 15:24 +08:00) Added copied-face `local_intersection_topology` repair, one-CAD worker isolation, explicit worker sentinel/timeout handling, and fail-closed protocol tests. The first clean single-run pilot recovered two cases before geometry preservation was tightened.
- [x] (2026-08-17 15:39 +08:00) Added an exclusive signed output-root contract (`assembly-repair-run-v2`), source/cohort/runtime hashes, worker identity validation, short private worker paths, explicit mutually-exclusive OCC profiles, and copied-face geometry/topology preservation gates. The signed pilot completed 16/16 attempts without process loss and recovered one case under the conservative 0.5% geometry gate.
- [x] (2026-08-17 16:02 +08:00) Re-ran the baseline profile on all 100 frozen original-control CADs. It reproduced the historical strict-valid set exactly at 84/100, with 85 native-valid, 81 both-valid, and zero historical-vector disagreement, establishing parity for subsequent restored/regressed attribution.
- [x] (2026-08-17 16:14 +08:00) Completed the signed `local_intersection_topology` 100-CAD matrix from clean commit `2351552`: 100 unique attempts, 95 STEP-readable, 85 native-valid, 85 strict-valid, 82 both-valid, one restored CAD (`00029780...`), zero regressions, and zero worker timeout/exit/protocol failures. The switch passes the individual no-regression condition but not the 95/100 assembly gate.
- [x] (2026-08-17 16:18 +08:00) Hardened the assembly snapshot contract to require the signed run manifest, completed status, exact attempt count, and matching summary SHA-256. Generated `reports/assembly_repair_local_topology_100cad_20260817/`; archive validation reports 100 attempts and zero forbidden artifacts.
- [x] (2026-08-17 16:24 +08:00) Generated the matching Git-safe `reports/assembly_repair_baseline_100cad_20260817/` snapshot, binding the exact 84/100 baseline parity matrix to run signature `9b492577...`. This permits remote case-by-case baseline comparison without exposing STEP or pickle bytes.
- [x] (2026-08-17 16:27 +08:00) Completed and snapshotted the original `directed_trim` full 100-CAD matrix. It restored both pre-STEP wire-build cases but regressed historical-valid `00044862...`, yielding 85/100 strict. The profile is rejected for composition in this form; the Git-safe negative result is preserved under `reports/assembly_repair_directed_trim_100cad_20260817/`.
- [x] (2026-08-17 16:31 +08:00) Implemented a guarded directed-loop policy with focused topology-only tests. It uses historical-order orientation when provable, graph regrouping only when provable, and otherwise retains the historical grouping with a diagnostic record. A new clean signed 100-CAD matrix is required before this replacement can be considered no-regression.
- [x] (2026-08-17 16:40 +08:00) Completed the guarded directed-trim signed 100-CAD matrix from clean commit `6bfe0c1`: 97 STEP-readable, 88 native-valid, 86 strict-valid, 83 both-valid, the same two wire-build restorations, and zero regressions. The loop policy made 2,351 face decisions: 2,307 historical-order orientations, five directed regroupings across the two restored CADs, and 39 baseline fallbacks across four topology-unproven CADs. Generated `reports/assembly_repair_directed_trim_guarded_100cad_20260817/` with compact policy evidence and zero forbidden artifacts.
- [x] (2026-08-17 16:56 +08:00) Completed the explicit guarded-directed-trim plus local-topology signed 100-CAD matrix from clean commit `e650d74`: 97 STEP-readable, 88 native-valid, 87 strict-valid, 84 both-valid, three restored CADs (`00016845...`, `00029780...`, and `00032004...`), zero regressions, and zero worker timeout, exit, or protocol failures. The signed summary hash `bfde32b...` matches the completed run manifest; the Git-safe snapshot records compact attempts and both repair diagnostics under `reports/assembly_repair_directed_trim_local_topology_guarded_100cad_20260817/`.
- [x] (2026-08-17 17:35 +08:00) Found and corrected a worker-contract gap in `directed_trim_local_pcurve_continuity`: the CLI required `--isolate-cad-workers`, but the coordinator dispatched only `local_intersection_topology` through `run_one_isolated`. Isolation is now centralized by repair-switch predicate, with six routing regression cases; 38 focused tests passed. The earlier non-isolated 16-CAD output is diagnostic-only and is not repair evidence.
- [x] (2026-08-17 17:50 +08:00) Completed the corrected isolated pcurve-continuity pilot on all 16 historical-invalid CADs: 16/16 rows, zero worker timeout/process/protocol/spawn failures, strict/both recovery `2/16`, and no regression possible within the invalid-only cohort. The restored IDs are `00016845_33dbc9e6ea684970be61af74_step_000` and `00032004_b91639b3cc4b41c7bf6a854d_step_000`.
- [x] (2026-08-17 17:50 +08:00) Completed the conditional signed 100-CAD isolated matrix with the same profile: 97 STEP-readable, 88 native-valid, 86 strict-valid, 83 both-valid, all original 84 strict-valid controls preserved, two restorations, zero regressions, and zero worker failures. The run is `COMPLETED`; its selected/full cohort signature is `e3003eb09d8822bfb6cb0d1a97e017a57823a24e74256daf44d7daf7afee1904` and its summary SHA-256 binding verifies.
- [x] (2026-08-17 17:50 +08:00) Prepared a short-path post-hardening capacity source at `D:\capv5`, detached at merged commit `6f7436da50a5f455fd9af3c806676ac0f49b8f9f`, with a local `BrepARG` junction to `D:\luolin\V13\BrepARG`. The dry run against `D:\luolin\V13\local_runs\capacity_ab_60k_posthardening_dryrun_shortpath_20260817` planned the same four formal tasks with `strict_nonfinite=true`, `dirty=false`, configuration signature `aad3db1d...`, and no output directory, checkpoint, lock, or training process created.
- [x] (2026-08-17 17:55 +08:00) Added and tested the independent `curve_fit_rescue` switch. It preserves historical curve fitting first and only applies sanitized lower-degree fallback to the edge that failed historical fitting. The 16 historical-invalid pilot reached STEP for all 16 CADs, including the three prior `curve_fit_not_done` cases, but strict validity stayed at 3/16 with the same restored set as the prior guarded directed/local topology result. Git-safe negative evidence is archived in `reports/assembly_repair_curve_rescue_invalid16_20260817/`; this switch is not promoted to a 100-CAD acceptance matrix.
- [x] (2026-08-17 19:11 +08:00) Diagnosed the non-unit-solid case as eight mutually nearest, near-coincident endpoint-id pairs created after CPU joint placement. Added the independently switchable `single_solid` near-vertex repair, isolated it in a child process, and completed its signed frozen 100-CAD matrix: 95 STEP-readable, 87 native-valid, 85 strict-valid, 83 both-valid, one restoration (`00000444_4ed4c78d6d754aac90876fc2_step_003`), zero regressions, and zero worker failures. The `85/100` result is retained as a no-regression primitive only; it does not meet the 95/100 assembly gate.
- [x] (2026-08-17 19:11 +08:00) Generated the Git-safe `reports/assembly_repair_single_solid_100cad_20260817/` snapshot with the signed matrix binding, compact per-attempt rows, aggregated reconciliation counts, and an independently verified no-path diagnosis JSON. Snapshot validation excludes forbidden model/data/CAD artifacts.
- [x] (2026-08-17 19:55 +08:00) Composed the proven `single_solid` near-vertex reconciliation with the isolated production overlay in the matrix runner. It remaps only same-face, mutual-nearest endpoint ids within `2e-4` and snaps only accepted-pair endpoints before calling the unchanged isolated `utils.construct_brep`. The completed signed 100-CAD production run at clean commit `aeec941` wrote 100 STEP files, reached 90 native-valid, 88 strict-valid, and 86 both-valid, restored `00000444...` in addition to the prior three recoveries, preserved all original 84 strict-valid controls, and recorded zero regressions. `reports/assembly_production_single_solid_100cad_20260817/` is Git-safe and verifies that its CAD set plus parent/historical-strict map equal the prior production 100-CAD report despite sorted reconstructed input order. The 95/100 gate remains closed.
- [x] (2026-08-17 20:15 +08:00) Tested a copied-face local pcurve fallback in the isolated production overlay only. The signed 16 historical-invalid CAD pilot completed all attempts and retained exactly the four existing production restorations with no new strict-valid recovery. Instrumented checks on the four native-valid/strict-invalid pcurve targets confirmed that the fallback executed but preserved geometry while leaving the same self-intersections. Enabling `ShapeFix_Wire` closed-wire mode did not change that result. The topology candidates that did remove intersections breached the registered face-geometry gate (about 1.4 to 5.2 percent boundary-length changes, with one 3.7 percent bbox change), so no candidate advances to a 100-CAD matrix or the shared upstream source.
- [x] (2026-08-17 19:19 +08:00) Pushed the independently validated near-vertex primitive and its Git-safe 100-CAD evidence as `c2ecf7f` on `experiment/protocol-v5-scaling-ladder`. The remote report preserves the `85/100` strict, zero-regression result without tracking local STEP, pickle, or checkpoint bytes.
- [x] (2026-08-17 19:55 +08:00) Composed the proven `single_solid` near-vertex reconciliation with the isolated production overlay in the matrix runner. It remaps only same-face, mutual-nearest endpoint ids within `2e-4` and snaps only accepted-pair endpoints before calling the unchanged isolated `utils.construct_brep`. The completed signed 100-CAD production run at clean commit `aeec941` wrote 100 STEP files, reached 90 native-valid, 88 strict-valid, and 86 both-valid, restored `00000444...` in addition to the prior three recoveries, preserved all original 84 strict-valid controls, and recorded zero regressions. `reports/assembly_production_single_solid_100cad_20260817/` is Git-safe and verifies that its CAD set plus parent/historical-strict map equal the prior production 100-CAD report despite sorted reconstructed input order. The 95/100 gate remains closed.
- [x] (2026-08-17 20:05 +08:00) Implemented the pre-registered failure-triggered selector and its Git-safe evidence adapter in the selector worktree. The primary is `directed_trim_local_intersection_topology`; only a strict-invalid primary can try `near_vertex_reconciliation` and then `directed_trim_curve_interpolate_local_intersection_topology`. Every candidate runs in an isolated child. The child measures the candidate STEP and returns a path-free geometry/topology gate; the parent only evaluates the fixed route. A durable local candidate ledger records every candidate before the final denominator row, so an interrupted CAD resumes from its exact route prefix.
- [x] (2026-08-17 20:29 +08:00) Performed an independent selector review before the formal pilot and hardened the result contract. The gate now compares effective vertex count plus face/edge incidence multisets; every candidate edge must expose a projectable 3D curve and every requested curve sample must succeed. Each local candidate record has a canonical SHA-256 which the final selected row, completed manifest, and Git-safe snapshot revalidate. The signed payload now includes the actual CPU joint optimizer and strict-diagnosis sources plus a path-free SHA-256/byte binding for every executed source pickle. JSONL appends use `fsync` and a resume may discard only an unterminated final torn line. A previous failure's raw exception text is cleared on successful resume and never archived. The full focused suite passed `83` tests after these changes.
- [x] (2026-08-17 21:02 +08:00) Closed the second selector review before formal execution. A serialized positive gate is now accepted only after complete schema-v2 checks, thresholds, finite metrics, incidence, and sample-accounting validation in routing, ledger recovery, and final-row validation. The effective post-reconciliation vertex-edge incidence is measured explicitly. Parent and child bind the exact pickle bytes passed to `pickle.loads`, and code, `BrepARG/utils.py`, and every selected pickle are rehashed both after acquiring the run lock and before completion. The Git-safe gate snapshot includes all new incidence and accounting evidence. The seven-file focused suite passed `95` tests and `git diff --check` passed.
- [ ] Rebase the reviewed selector commit onto the current remote hardening branch, push it without force, then run the selector first on the frozen 16 original failures and only then on all 100 original-control CADs. Its pre-registered acceptance condition is exactly the expected six both-valid restorations, at least `90/100` strict-valid, at least `87/100` both-valid, all original 84 strict-valid controls preserved, and zero worker/protocol failures. A miss fails the selector rather than changing the gate.
- [ ] Implement the remaining assembly repairs as independent, diagnosed-entity-local switches, with tests and one commit per logically independent repair.
- [x] (2026-08-17 12:35 +08:00) Added the first CPU-only repair primitives and tests: immutable named profiles, deterministic directed loop extraction with degenerate closed-edge handling, explicit endpoint-continuity validation, bounded duplicate-point cleanup and lower-degree curve fitting fallbacks, plus explicit single-shell/single-solid checks in the combined directed assembler. The next milestone is the fixed-cohort profile runner and 100-CAD no-regression matrix.
- [x] (2026-08-17 12:51 +08:00) Added an idempotent local `run_assembly_repair_matrix.py` coordinator with independent and combined profiles, fixed 100-CAD identity checks, attempts-based strict/native/both-valid counts, and restored/unchanged/regressed CAD maps. A one-CAD real OCC smoke completed as both-valid.
- [x] (2026-08-17 13:00 +08:00) Ran development pilots on all 16 historical failures. The all-switch profile restored 0/16 and failed early on 7; independent directed trim restored exactly the two historical wire-build failures, while curve fallback, continuity-only, and single-solid restored none. This is negative pilot evidence, not the formal 100-CAD result; the next repair must target self-intersecting pcurves/wires locally rather than globally reordering every face.
- [x] (2026-08-17 13:15 +08:00) Tested an explicit OCC `ShapeFix_Wire`/`ShapeFix_Face` pcurve self-intersection mode twice on the same 16 failures. It retained the two directed-trim wire-build recoveries but restored none of the ten self-intersection cases. Generic OCC self-intersection repair is therefore rejected as the next global switch; the formal 95/100 GT gate remains unmet.
- [ ] Re-run the frozen 100 original-control CADs after each repair, preserve all 84 original strict-valid CADs, reach at least 95 strict-valid CADs, and publish the repair-to-restored/regressed-case map.
- [ ] Re-measure the selected capacity arm through the repaired chain and apply the final release gate.
- [ ] Snapshot Git-safe evidence into `reports/capacity_ab_60k_20260817/` and `reports/assembly_repair_20260817/`, validate forbidden-file exclusions, commit, and push normally to `experiment/protocol-v5-scaling-ladder`.

## Surprises & Discoveries

- Observation: GitHub was unreachable during initial setup.
  Evidence: `git push origin HEAD:experiment/protocol-v5-scaling-ladder` failed on 2026-08-17 with `Recv failure: Connection was reset`. Work continues locally, and ordinary push will be retried without force.

- Observation: The capacity candidates have different downstream sequence costs even though both use 64-dimensional latent vectors.
  Evidence: VQ-8192 still emits one token per latent position, while two-stage RVQ emits two code indices. The registered work order estimates approximately 36 percent total CAD-sequence growth after non-surface tokens are included, so RVQ must demonstrate a material validity advantage rather than a marginal MSE win.

- Observation: The formal capacity root's resume contract handled a Windows checkpoint-replace failure without changing the run identity.
  Evidence: RVQ seed 4 attempt 1 exited with `PermissionError [WinError 5]` while `os.replace` tried to swap the rolling checkpoint at epoch 51. Attempt 2 used the same signed task environment, resumed full state from epoch 50, reached epoch 99, and the validator reported `valid=true` for all four tasks with identical inventories.

- Observation: The first RVQ implementation would have subtracted the stage-one code from a bf16 latent because the AMP adapter restores the incoming dtype before returning.
  Evidence: The new bf16 regression captures the stage-two input and requires `torch.float32`; it fails with the original subtraction and passes after explicitly computing the residual from `latent.float()` and a detached fp32 stage-one code.

- Observation: Extending a `NamedTuple` with stage metadata changed `len(info)` from three to five even though positional indices zero through two still worked.
  Evidence: The legacy-contract regression requires `len(info) == 3`. `QuantizerInfo` is now a three-item tuple subclass whose `stage_indices` and `stage_perplexities` are attributes, so old positional unpacking remains exact.

- Observation: The first CUDA smoke proved the model path finite but was deliberately rejected as evidence because two signed facts were not repeated in downstream artifacts.
  Evidence: VQ-8192 completed all three train and two validation batches with no non-finite event. Validation reasons were limited to scheduler-schema and checkpoint parent-overlap mismatches. The failed root remains local diagnostic evidence; a new root is required after producer-side fixes.

- Observation: The external E: volume cannot safely host checkpoint-heavy formal training.
  Evidence: Windows reports the exFAT volume as `Warning / Full Repair Needed`; `py-spy` showed the smoke blocked inside `torch.save` at native `WriteFile`. A 256 MiB write to D: completed in about 0.18 seconds. The clean smoke and formal matrix therefore use D: and immediately restored sustained 96--98% GPU utilization after the one-time dataset scan and deduplication.

- Observation: The first assembly switch matrix disproved the assumption that the four diagnosed families can be repaired by globally enabling all four treatments.
  Evidence: On the frozen 16 historical failures, the combined profile restored 0 CADs, wrote 9 STEP files, and raised 7 assembly errors. Directed trim alone restored `00016845...` and `00032004...`, the two prior wire-build failures, but no other switch restored a case. Several data records contain degenerate or non-manifold face-loop incidence, so fail-closed global endpoint validation rejects even cases whose historical assembler reached STEP.

- Observation: OCC's generic pcurve/wire self-intersection fixer does not address the dominant ten-case failure family in this cohort.
  Evidence: Both a face-integrated wire tool configuration and a direct wire-fixer configuration produced the same outcome as directed trim alone: strict 2/16, with only the two wire-build cases recovered. All ten previously diagnosed self-intersection CADs remained strict-invalid or failed before STEP.

- Observation: The historical 100-CAD calibration manifest is not the complete P0-A baseline population.
  Evidence: It contained 10 saved invalid STEP rows, while the stage-aware P0-A baseline rerun contained 11; `00095733_8b325d2fcb27ec9e79388602_step_000` was written only by the latter. The face/wire report is therefore bound to input SHA-256 `2a996faa68bc308fea6c7e5b9061f53a37970d74d2d64fb5994ba840bf74e490`.

- Observation: The dominant post-STEP defect is localized enough for a bounded repair search.
  Evidence: The corrected report identifies 10 CADs and 17 face occurrences with self-intersecting wires, while no saved case has a wire-order failure. Five cases fail before STEP and must be handled through source-topology/curve-fit diagnostics, not pcurve surgery.

- Observation: Edge-pair probing splits the saved-STEP failures into different local modes rather than one uniform self-intersection defect.
  Evidence: The v2 report classifies all 16 aggregate self-intersecting wires. It contains 11 adjacent occurrences across 6 CADs, 5 closure occurrences across 4 CADs, and 4 non-adjacent occurrences across 2 CADs. `00032101...` and `00051587...` have non-adjacent `(2,4)` crossings; `00032101...` and `00095733...` expose 7 explicit two-dimensional pcurve gaps. There are no self-only, seam, or disconnected occurrences. The other five CADs remain pre-STEP unavailable rather than false-negative clean cases.

- Observation: A topology-changing ShapeFix treatment becomes useful only when triggered by an observed self-intersection and bounded to that face.
  Evidence: A disposable CPU probe restored `00002441...` and `00029780...` to native and strict validity with zero free edges. It did not recover `00008763...`, and a shared-process probe exited while entering the first non-adjacent/gap case `00032101...`; the implementation therefore needs copied-face rollback and one-CAD child-process isolation before it is eligible for the fixed cohort.

- Observation: A clean task signature can still misdescribe runtime behavior when the launcher does not explicitly export the signed control.
  Evidence: The diagnostic VQ-8192 seed-3 history reports `config.scheduler.metric=curved_parent_mse` while every epoch records `plateau_metric=global_val`; at epoch 52 the scheduler input was `0.00122266` while curved parent MSE was `0.0028803291`. Schema v2 validates all three values against each other.

- Observation: Concatenating two RVQ integer streams without an offset merges unrelated code identities.
  Evidence: Code 7 from stage 1 and code 7 from stage 2 occupied the same histogram bin in schema v1. Schema v2 offsets stage 2 by 4096 and treats the combined histogram as compatibility-only evidence; selection uses each stage's usage fraction independently.

- Observation: The upstream history-feature pool cannot append across its final free rows after checkpoint restore.
  Evidence: Restoring VQ-8192 with 7,748 of 8,192 rows filled and processing a 512-token batch attempted to assign 512 rows into a 444-row slice. The local learned-VQ adapter now fills the tail and randomly replaces rows with the remainder without modifying `BrepARG/`.

- Observation: Running two assembly coordinators against one output root is unsafe without an OS-level writer lock.
  Evidence: The first local topology pilot produced 32 JSONL rows for 16 profile/CAD pairs because both parents read an empty `done` set before appending. The contaminated root is diagnostic-only; the runner now binds a signed run manifest under the existing cross-process lock.

- Observation: A worker output path can fail independently of OCC geometry when the profile and long CAD identity are repeated in nested Windows paths.
  Evidence: The first signed pilot returned 16 STEP-write failures with no geometry change after a private temporary path exceeded practical `MAX_PATH`; shortening the private worker directory restored 11 STEP writes in the next signed pilot.

- Observation: Native-valid topology repair is not sufficient evidence that a copied face preserved the CAD boundary.
  Evidence: The two previously recovered faces changed perimeter by 0.24% and 5.23%, respectively; the conservative 0.5% area/perimeter and 0.1% bounding-box gate retains only the lower-change recovery and rejects the larger geometric drift.

- Observation: The conservative local topology treatment is safe on the frozen cohort but has low recall.
  Evidence: The signed full matrix preserves all 84 historical strict-valid CADs and restores only `00029780...`, producing 85/100 strict validity. It therefore remains eligible as one independently measured component, but cannot be presented as an assembly-chain solution or used to release boundary consistency, sequence regeneration, or AR.

- Observation: Global directed trim has a narrow useful effect but is not fail-safe on malformed source incidence.
  Evidence: It restores `00016845...` and `00032004...`, where historical wire construction failed, but regresses `00044862...`. Five faces in the regressed CAD contain open/disconnected endpoint components for which neither historical-order orientation nor graph regrouping can prove a closed loop; the current implementation raises instead of retaining baseline ordering.

- Observation: Evidence-gated fallback removes the directed-trim regression without losing either recovery.
  Evidence: The guarded full matrix changes the baseline strict vector at exactly two CADs, both positive restorations. `00044862...` returns exactly to its baseline status/native/strict/both tuple, and no historical-valid CAD regresses. Only five of 2,351 face decisions use directed regrouping; all five occur in the two restored wire-build CADs.

- Observation: Selective curve rescue fixes the failure stage but not strict validity.
  Evidence: `directed_trim_curve_rescue_local_intersection_topology` on the 16 historical-invalid controls wrote 16/16 STEP files and converted `00051602...`, `00061931...`, and `00087341...` from `curve_fit_not_done` to diagnosed `step_invalid`. Strict validity remained 3/16 with restored CADs `00016845...`, `00029780...`, and `00032004...`, so the 95/100 assembly gate is unchanged.

- Observation: Geometric endpoint loop fallback is not a safe promotion path for the current frozen cohort.
  Evidence: A local-only probe found tiny WCS endpoint gaps on topology-unproven faces, but when promoted to a switch it restored no additional CADs; a face-local snapped-edge variant still restored no additional CADs and introduced five builder failures on the 16-case pilot. The code path was removed and the local run roots are retained only as negative evidence.

- Observation: Relaxing the copied-face topology preservation gate would recover too little for too much semantic risk.
  Evidence: A local-only monkeypatch allowing up to 6 percent boundary-length and 4 percent bbox drift restored only `00002441...` among the remaining self-intersection cases. That would move the best no-regression result from 87/100 to at most 88/100 while accepting geometry changes beyond the documented conservative gate.

- Observation: The independently safe repairs compose additively on the frozen cohort, but their union is still far below the assembly acceptance threshold.
  Evidence: The signed explicit combination preserves all 84 original strict-valid CADs and restores the union of the guarded-trim pair and the local-topology singleton, reaching 87/100 strict-valid with zero worker failures. It remains eight CADs below the predeclared 95/100 gate, so it is evidence for repair locality rather than permission to start boundary consistency, sequence regeneration, or AR.

- Observation: Passing `--isolate-cad-workers` was not sufficient to guarantee isolation for every local OCC profile.
  Evidence: Before the routing fix, `directed_trim_local_pcurve_continuity` satisfied the CLI check but was executed by the parent through `run_one`; only `local_intersection_topology` entered `run_one_isolated`. The first 16-CAD run is retained locally as a diagnostic trace, while the corrected run records per-CAD worker return codes and logs.

- Observation: Local pcurve continuity adds no recoveries beyond guarded directed trim on the frozen cohort.
  Evidence: The corrected profile recovered exactly `00016845...` and `00032004...`, with 86/100 strict-valid and zero regressions, matching the guarded-directed-trim strict count while remaining mutually exclusive with the other OCC pcurve strategies.

- Observation: The non-unit-solid failure is caused by post-placement topology disagreement, not by the absence of an OCC solid wrapper.
  Evidence: In `00000444_4ed4c78d6d754aac90876fc2_step_003`, the historical grouping drops one edge on each of faces 28, 30, 32, and 34. After the same 200-iteration CPU joint placement used by the matrix runner, eight endpoint-id pairs are 1.0000169e-4 apart. The baseline round-trip has zero solids, seven invalid faces, and one disconnected wire; the reconciled candidate has one solid and no invalid faces or wires.

- Observation: The reconciliation rule is conservative but not an assembly-chain release.
  Evidence: It applies to six cohort CADs under same-face, mutually unique nearest-neighbor, cluster-size-two constraints. The signed matrix preserves all 84 original strict-valid controls and restores exactly one case, reaching 85/100 rather than 95/100 strict validity.

- Observation: Tightening surface fitting can improve the project strict predicate without producing a native-valid BRep.
  Evidence: The 16-failure `directed_trim_surface_fit_tight_local_intersection_topology` pilot restores `00032101...` as strict-valid but records `native_brep_valid=false` and `both_valid=false`. It is diagnostic evidence for pcurve sensitivity, not an eligible production fallback.

- Observation: Independent no-regression primitives cannot be inferred safe when composed inside one construction call.
  Evidence: Near-vertex reconciliation remaps endpoint identifiers before loop grouping and directed trim. That changes the input to the primary loop policy, so the `87/100` directed/local-topology result and the `85/100` near-vertex result do not prove that their direct code-level combination has zero regressions.

- Observation: A parent-process STEP geometry gate would bypass the existing native OCC failure containment.
  Evidence: `run_one_isolated` deliberately treats a child native exit, timeout, spawn failure, or protocol failure as one denominator row. Reading fallback STEP files and projecting curves in the selector parent would reintroduce those native operations outside that boundary. The selector now forwards `--selector-geometry-gate` to the existing child worker; the child returns only JSON metrics and a fail-closed gate decision.

- Observation: `BRepBndLib.Add` is too loose for a valid interpolated B-spline candidate bbox gate.
  Evidence: On fixed CAD `00002441_179a8df075c94d0596b1f20d_step_010`, the input-to-candidate edge residual was below `1.7e-13` normalized and edge-length delta was `0.000171`, but `Add` reported a `0.05823` bbox delta because of a loose control-polygon bound. OCC `AddOptimal(shape, box, True, False)` measures the trimmed shape and reports `0.00634`; the fixed 2 percent bbox threshold is retained.

- Observation: The first selector gate could prove close global geometry while silently omitting candidate boundary elements.
  Evidence: The original v1 signature recorded input vertex and face-edge incidence information, but its candidate signature compared only face/edge totals. It also skipped null 3D curves and failed `curve.Value()` calls while reporting zero candidate-to-input projection failures. The v2 gate rejects any topology-degree mismatch, non-projectable edge, accounting mismatch, or unsampled requested curve point.

- Observation: A signed start-of-run source hash does not prove which pickle bytes the isolated worker actually deserialized.
  Evidence: The selector payload was originally built before worker launch, while the child reopened the source path later. The hardened worker hashes one byte string, deserializes that same string with `pickle.loads`, rehashes the path after load, and returns both bindings; the parent checks those plus a post-worker hash against the signed payload. Any mismatch becomes a protocol failure rather than an assembly result.

## Decision Log

- Decision: Compare one 8192-entry learned codebook with two sequential 4096-entry residual codebooks, both using 64-dimensional code vectors and `anchor='random'`.
  Rationale: VQ-8192 tests codebook cardinality without increasing downstream token count. RVQ tests whether a second residual stage recovers detail beyond a single nearest-neighbor assignment, at a known sequence-length cost.
  Date/Author: 2026-08-17 / Codex.

- Decision: Reuse the exact P0-B 60k/12k patch inventories and stability recipe, not merely the same source split or sample counts.
  Rationale: Ordered and sorted exact-patch SHA-256 digests are required to isolate quantizer capacity from data selection, augmentation order, numerical precision, optimizer policy, and training duration.
  Date/Author: 2026-08-17 / Codex.

- Decision: Treat RVQ stage 2 as a separate measured codebook and fail the scientific interpretation if it collapses.
  Rationale: Aggregate or tuple-code perplexity can hide an unused residual stage. Every epoch must report each stage's unique codes, entropy perplexity, coverage, and usage fraction; non-finite values or missing stage-2 metrics fail closed.
  Date/Author: 2026-08-17 / Codex.

- Decision: Select capacity using paired 100-CAD strict validity and McNemar evidence, with reconstruction MSE as explanatory evidence only.
  Rationale: The objective is usable BRep assembly. Existing within-arm evidence shows strict-invalid VQ CADs have about 2.7 times the median curved MSE of strict-valid VQ CADs, but only the assembly outcome proves that lower error converts to utility.
  Date/Author: 2026-08-17 / Codex.

- Decision: Give VQ-8192 the default win if its repaired quantization gap is at most 5 percentage points. Choose RVQ only when it is materially and pairwise better enough to justify the estimated 36 percent sequence growth.
  Rationale: VQ-8192 has no token-count penalty. RVQ's extra code must buy a clear validity improvement, not just a small mean-MSE reduction within n=100 sampling noise.
  Date/Author: 2026-08-17 / Codex.

- Decision: Select `vq_8192_64d_random` as the capacity winner and reject RVQ for this release path.
  Rationale: The completed fixed 100-CAD measurement gives bypass `70/100` strict-valid, VQ-8192 `69/100`, and RVQ `65/100`. VQ-8192's bypass gap is only 1 percentage point, inside the registered 5-point gate. RVQ is 4 points worse than VQ-8192 on strict validity, has 5 RVQ-only versus 9 VQ-only strict successes, and McNemar `p=0.4239501953125`, so its extra sequence cost is not justified.
  Date/Author: 2026-08-17 / Codex.

- Decision: Keep assembly repair and representation capacity as parallel implementations but merge their effects only after each has an unchanged-chain control result.
  Rationale: Changing quantization and assembly simultaneously would make recovered CADs impossible to attribute. The capacity A/B first uses the current chain; the repair track first uses original controls; only the final winner is measured through the repaired chain.
  Date/Author: 2026-08-17 / Codex.

- Decision: Treat the paired 60k strict-validity gap as a capacity gate before starting boundary-consistency loss.
  Rationale: On the identical 100-CAD cohort, bypass@60k is 70/100 strict and VQ-4096@60k is 57/100 strict, so `Delta_q=13 pp` exceeds the registered 5 pp noise band. The reconstruction-to-assembly loss is also 14 pp (`84-70`), but the preregistered order resolves the representation confound first.
  Date/Author: 2026-08-18 / Codex.

- Decision: Keep schema-v2 formal training immutable at commit `76604edb4bc06f6be7f4740d44886b1099e95eaa` and add capacity-aware Git-safe archival support without touching that running worktree's signed training sources.
  Rationale: Editing the detached training worktree would invalidate its source hashes. The archival code is exercised independently and will only read the completed state after all four tasks validate.
  Date/Author: 2026-08-17 / Codex.

- Decision: Keep all post-launch development in the assembly worktree and leave `D:\luolin\V13\.worktrees\capacity-ab-v2-20260817` detached at `76604ed` through all four tasks.
  Rationale: Even when signed training-file hashes are unchanged, a later task launched from a different Git HEAD would weaken the four-task provenance claim. The launcher starts each serial task from this immutable path, while the assembly worktree can advance independently.
  Date/Author: 2026-08-17 / Codex.

- Decision: Promote topology-changing ShapeFix only as a copied-face, diagnosis-triggered candidate with rollback, and execute each CAD in a child process.
  Rationale: The local treatment recovered adjacent-crossing examples, but one closure example remained invalid and the first non-adjacent/gap example terminated a shared probe. Copying the face permits rejection without mutating the baseline shape; process isolation retains a failed attempt instead of losing the whole matrix to a native OCC exit.
  Date/Author: 2026-08-17 / Codex.

- Decision: Reject the partially trained schema-v1 formal root rather than resume it.
  Rationale: Scheduler state and checkpoint selection were already driven by a different metric than the registered protocol. Reusing that state would preserve the confound even after fixing the launcher.
  Date/Author: 2026-08-17 / Codex.

- Decision: Preserve the original face/wire report as schema v1 and publish crossing positions as a separate schema-v2 report restricted to v1 aggregate self-intersecting wires.
  Rationale: Scanning two-dimensional gaps across every otherwise clean wire produced unrelated topology evidence. Restricting the detailed checks to the 16 wires already identified by OCC `CheckSelfIntersection` answers the repair-localization question without rewriting the historical v1 evidence or conflating other defects. Occurrence kinds remain independent, so a wire can carry both a crossing and a pcurve gap.
  Date/Author: 2026-08-17 / Codex.

- Decision: Require RVQ to exceed VQ-8192 by at least 5 strict-validity points, have more paired wins than losses, and pass the exact paired significance gate even when VQ-8192 itself is within 5 points of bypass.
  Rationale: This resolves the earlier documentation/code ambiguity before results are observed and prices RVQ's estimated 36 percent sequence-length increase explicitly.
  Date/Author: 2026-08-17 / Codex.

- Decision: Treat `pcurve_self_intersection` and `local_intersection_topology` as mutually exclusive OCC strategies, and keep them out of the generic `combined` profile until an individual no-regression gate passes.
  Rationale: The assembler dispatches these strategies through an `if/elif`; claiming that both are enabled would silently ignore one. Explicit profiles preserve the mapping between a reported switch and the code path that actually ran.
  Date/Author: 2026-08-17 / Codex.

- Decision: Require every assembly output root to carry a SHA-256 run signature covering the calibration manifest, ordered cohort, profiles, iteration count, worker policy, repair source files, Git status, and BrepARG utility hash.
  Rationale: A new directory alone prevents the last accident but does not make resume or parameter drift auditable. The signed manifest fails closed when an existing root belongs to another run or contains unsigned artifacts.
  Date/Author: 2026-08-17 / Codex.

- Decision: Accept a copied-face topology candidate only when wire count, edge count, positive area, perimeter, and bounding-box invariants pass in addition to OCC native validity and removal of the diagnosed self-intersection.
  Rationale: ShapeFix can create a native-valid but semantically altered face. The fixed tolerances (0.5% area/perimeter, 0.1% coordinate scale) keep this local repair conservative and make rejected candidates explicit rather than hiding geometry drift in the final strict-valid count.
  Date/Author: 2026-08-17 / Codex.

- Decision: Retain `local_intersection_topology` as a no-regression repair primitive, not as an accepted final profile.
  Rationale: Its full-cohort result is 85/100 with one restoration and zero regressions. It may be combined only with other independently no-regressing switches and every explicit combination must still run its own 100-CAD matrix; the final assembly gate remains at least 95/100.
  Date/Author: 2026-08-17 / Codex.

- Decision: Reject the original global `directed_trim` profile and replace it with a fail-closed directed-loop policy.
  Rationale: A directed strategy may orient or regroup a face only when it can prove a closed walk. If both directed strategies fail, the face must retain the historical baseline grouping and record the fallback. This preserves the two known recoveries while targeting the sole observed regression without changing tolerances or hard-coding CAD identities.
  Date/Author: 2026-08-17 / Codex.

- Decision: Retain the explicit guarded-directed-trim plus local-topology profile as the current best no-regression assembly control, but do not promote it to an accepted chain.
  Rationale: Its independent signed result is 87/100 strict-valid, preserving all original 84 successes and restoring exactly three previously invalid CADs. The profile misses the fixed 95/100 gate by eight CADs, so subsequent work must target the remaining diagnosed failure families rather than treating composition as a release condition.
  Date/Author: 2026-08-17 / Codex.

- Decision: Route every profile containing `local_intersection_topology` or `local_pcurve_continuity` through the isolated worker path, and retain pcurve-continuity only as a diagnosed no-regression negative result.
  Rationale: Both switches invoke native OCC face-level operations and must preserve the signed one-CAD failure denominator. The corrected pcurve profile has zero worker failures and no regressions, but its 86/100 result and exact overlap with guarded directed trim provide no evidence for promotion or for starting boundary consistency.
  Date/Author: 2026-08-17 / Codex.

- Decision: Bind near-vertex reconciliation to the existing `single_solid` repair profile, but keep its implementation as an explicit `solid_topology_repair` flag and force that profile through isolated workers.
  Rationale: The external profile name is already part of prior matrix evidence, while the new low-level behavior must remain independently visible and contained. A repair may merge only a same-face, positive-distance, mutually unique nearest pair within 2e-4; any ambiguity, non-finite geometry, or cluster larger than two returns the original adjacency unchanged.
  Date/Author: 2026-08-17 / Codex.

- Decision: Archive the non-unit-solid diagnosis only through the assembly snapshot's path-rejecting copier.
  Rationale: The report must preserve the geometry-stage, BRepCheck, loop-coverage, size, and SHA-256 evidence without serializing source-pickle paths, STEP paths, CAD bytes, or machine-specific directories.
  Date/Author: 2026-08-17 / Codex.

- Decision: Use a failure-triggered selector rather than composing the guarded directed/local-topology, near-vertex, and interpolation switches in one assembler invocation.
  Rationale: The primary `directed_trim_local_intersection_topology` output is retained whenever it is strict-valid, preserving the existing `87/100` strict control vector. Only a strict-invalid primary output may attempt the independently measured near-vertex primitive and then the interpolation primitive. A fallback replaces the primary only after native validity, strict validity, both-valid status, and input/output geometry-topology checks pass. This routes by observed output validity and fixed invariants, never by CAD identifier or historical label.
  Date/Author: 2026-08-17 / Codex.

- Decision: Run all selector candidate geometry measurements inside the existing isolated child worker and use a bidirectional boundary gate.
  Rationale: The child already owns source-pickle loading, CPU joint placement, OCC construction, STEP write/read, and strict validation. It now also records unique topology counts, optimized input and candidate shape bounds, edge-length agreement, input-to-candidate curve projection residual, and candidate-to-input polyline residual. The pure parent route never reads a STEP or calls OCC. The gate requires equal face/edge counts, one solid, no free edges/order/self-intersection failures, bbox delta at most 2 percent, edge length delta at most 5 percent, and both normalized RMS/max residual checks at 1 percent/5 percent.
  Date/Author: 2026-08-17 / Codex.

- Decision: Treat selector evidence as formal only after full topological, input, candidate, and snapshot binding checks; do not promote the earlier two one-CAD smoke runs.
  Rationale: A numeric geometry residual alone does not prove that every output boundary element was measured, and a profile-name-only ledger link cannot protect a resumed output from content drift. The v2 contract requires vertex and incidence invariants, full curve/sample accounting, canonical candidate fingerprints, source-pickle and code hashes, durable JSONL prefixes, and snapshot revalidation of both local ledgers before a result can be Git-safe.
  Date/Author: 2026-08-17 / Codex.

## Outcomes & Retrospective

The face/wire diagnosis milestone and P0-B stability archive are complete and
Git-safe. Schema v1/v2 localizes all 16 invalid controls and the archive records
all 96 diagnosis attempts. The P0-B archive contains four 100-epoch histories,
four TensorBoard event files, compact logs, metrics, manifests, and checkpoint
SHA-256/size bindings; the 4.27 GiB checkpoint bytes remain local. The original
unchanged-chain capacity measurement (bypass@60k 70/100, VQ-4096@60k 57/100)
correctly triggered the preregistered capacity-first decision.

The authoritative post-hardening rerun from `D:\capv5` at commit
`6f7436da50a5f455fd9af3c806676ac0f49b8f9f` completed all four VQ-8192/RVQ
tasks at exactly 100 epochs with `valid=true`, identical inventories, and
runtime-resume compatibility. On the fixed cohort, bypass@60k is 70/100 strict,
VQ-8192 is 69/100, and RVQ is 65/100; exact RVQ-versus-VQ McNemar is
`p=0.4239501953125`. The registered decision is `VQ_8192_DIRECT_WIN` with
`Delta_q=1 pp`; the post-hardening report is Git-safe and retains only JSON,
Markdown, and CSV summaries. The 2026-08-17 measurement remains historical.

Baseline parity is exact at 84/100. The best isolated production composition
reaches 100 STEP-readable, 90 native-valid, 88 strict-valid, and 86 both-valid
with all 84 original controls preserved and zero regressions. The assembly gate
therefore remains closed at seven cases below 95/100. Boundary consistency,
sequence regeneration, and AR remain blocked; the plan is not complete until a
repair meeting the immutable gate and its repaired-chain winner measurement are
verified.

The fail-closed selector implementation is now part of this branch. It preserves
strict-valid primary outputs, records every isolated candidate in a durable
ledger, and accepts a fallback only after complete source-topology and
bidirectional geometry checks. Its formal 16-CAD and 100-CAD evidence is tracked
as a separate milestone because implementation alone does not satisfy the 95/100
release gate.

## Context and Orientation

`breparg_improvements/train.py` defines the VQ-VAE architecture, quantizer configurations, weighted training loop, stability fuses, validation buckets, code usage metrics, atomic checkpoints, and auto-resume. Its existing `vq_4096_64d_random` arm wraps the upstream `BrepARG/quantise.py::VectorQuantiser` so quantizer math remains float32 under mixed precision. The new quantizers belong in this local modified project, not in the ignored upstream `BrepARG/` tree.

`tools/run_p0b_stability_retest.py` is the authoritative P0-B launcher and validator. It freezes the Protocol V5 hashes, exact patch counts, seeds, training recipe, finite-state lifecycle, writer lock, checkpoint deserialization, and cross-task inventory equality. The capacity launcher should reuse its validated helpers or mirror its checks without weakening them.

`tools/run_p0b_vq_assembly_measurement.py` reconstructs a frozen 100-CAD cohort from selected checkpoints and audits STEP readability, OpenCascade native validity, project strict validity, and their intersection. The capacity coordinator must generalize this contract to VQ-8192 and RVQ while retaining all failures in the denominator and preserving the cohort identity SHA-256 `646693dbfde083bf16ae63f917658cc0c3b3eb71cedaeddfeea55007bd741474`.

`tools/diagnose_assembly_chain.py` and `reports/p0a_assembly_chain_evidence_20260817/` provide the 16 original failures and their taxonomy: ten wire self-intersections, three curve-fit failures, two wire-build failures, and one non-unit or empty solid. `tools/directed_trim_assembly.py` contains an earlier topology-directed prototype but is not yet the accepted production chain. Repairs must remain outside upstream source and be individually selectable.

`tools/diagnose_assembly_face_wires.py` is the local P0-A narrowing tool. Schema v1 records the precise face and wire index for every self-intersection or ordering failure observed in a saved original STEP file. Schema v2 keeps that report immutable and adds independent adjacent, closure, non-adjacent, single-edge, pcurve-gap, seam, and disconnected occurrences with one-based OCC edge positions. For the five stage-aware baseline failures without a STEP, it records only source-topology clues and an explicit unavailable occurrence; this prevents missing pcurve evidence from being mistaken for a clean result.

A learned vector quantizer replaces each continuous latent vector with one nearest code vector. VQ-8192 raises the number of available vectors from 4096 to 8192. Residual vector quantization, abbreviated RVQ, first quantizes the latent, then quantizes the residual error with a second independent codebook; reconstruction uses the sum of both selected code vectors. Each stage therefore needs its own usage health report.

## Plan of Work

First add the two quantizer configurations and an AMP-safe two-stage residual module to `breparg_improvements/train.py`. Preserve the current model encoder, decoder, 64-dimensional bottleneck, optimizer, initialization seeding, and reconstruction loss. Return an information object that the training and validation aggregators can interpret without confusing two stage indices with one joint code. Add focused CPU and CUDA tests for shapes, gradients, state dictionaries, dtype behavior, exact residual composition, stage isolation, metadata, finite-state scanning, and metrics.

Next add a capacity launcher and snapshot tool under `tools/`. It must freeze exactly two arms and seeds 3 and 4; validate the same Protocol V5 hash, split hash, ordered/sorted patch inventory hashes, precision and optimizer recipe; reject foreign or incomplete outputs; use per-task writer-safe output directories; support exact automatic resume; and produce compact status. Use the D: output root because the E: filesystem requires repair. Run a small real-CUDA probe before formal work, including forced interruption followed by exact resume.

Then generalize the assembly measurement path for the two capacity checkpoints. Select the best checkpoint by the same curved-parent reconstruction rule used by P0-B, bind its SHA-256 and epoch, reconstruct the frozen cohort in identical order, run the unchanged chain, and calculate exact paired discordances and two-sided McNemar p-values for strict and native outcomes. Preserve the VQ-4096 and bypass P0-B results as historical references, not newly mixed arms.

In parallel, implement assembly repair switches in a local module. Start with trim-loop ordering, edge orientation, and self-intersection prevention; then bounded curve fitting and degenerate-edge handling; then endpoint continuity and topology validation before wire build; finally enforce exactly one closed shell and one solid after sewing. Each treatment runs alone against the original chain, records changed CAD identities and failure stages, and must not regress any of the 84 original strict-valid controls. Combine only switches that individually pass the no-regression rule.

The failure-triggered selector is a separate local runner, not a new global construction profile. `tools/run_assembly_repair_selector.py` always executes the guarded primary in a child process. A primary strict-valid result is copied to the final selector row immediately and no fallback is started. For a strict-invalid primary, it attempts near-vertex reconciliation, then interpolation only if near-vertex is not eligible. Each fallback must have a saved STEP, status `both_valid`, native validity, strict validity, conjunction validity, and an accepted gate. `tools/assembly_selector_geometry.py` measures the gate in the child: it compares the post-joint-placement input with the actual STEP using unique face/edge/solid counts, strict validity-component counts, optimal trimmed-shape bounds, total edge length, input-to-candidate curve projection, and candidate-to-input polyline distance. The parent process does not import OCC for this stage.

Finally run the chosen capacity checkpoint through the accepted repaired chain on the same cohort. Archive source changes, complete histories, compact logs, TensorBoard events, metrics, paired rows, manifests, and checkpoint hashes. Generate forbidden-artifact validation that rejects model/data/CAD bytes from both report directories. Update this plan and the ADR with actual outcomes and commit in logically reviewable units.

## Concrete Steps

Use `D:\luolin\V13\.worktrees\assembly-repair-20260817` for assembly development and `D:\luolin\V13\v6git` for branch integration. Use `C:\Users\YU\.conda\envs\brepgen_env\python.exe` for tests and training. The frozen protocol is `D:\luolin\V13\local_runs\protocol_v5_scaling_run_20260806\protocol`; the upstream runtime dependency is `D:\luolin\V13\BrepARG`. Formal capacity source is immutable at `D:\luolin\V13\.worktrees\capacity-ab-v2-20260817` commit `76604ed`, with outputs under `D:\luolin\V13\local_runs\capacity_ab_60k_v2_20260817`. The E: volume is excluded because its filesystem needs repair.

Run focused tests after each implementation increment:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m pytest -q tests\test_vqvae_protocol_training.py tests\test_training_stability.py tests\test_run_p0b_stability_retest.py
    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m pytest -q tests\test_run_p0b_vq_assembly_measurement.py tests\test_diagnose_assembly_chain.py tests\test_directed_trim_assembly.py

The exact capacity launcher command will be recorded here after its CLI is implemented. It must expose `probe`, `run`, `status`, and `validate` operations and must never silently change the registered formal constants. The completed P0 archive and paired-measurement verification used:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m pytest -q tests\test_snapshot_p0a_assembly_chain.py tests\test_snapshot_p0b_formal_results.py tests\test_run_p0b_vq_assembly_measurement.py tests\test_audit_assembly_step_validity.py

The expected result is `33 passed`; the report-level validators must also show `forbidden_artifacts=[]`, four histories, four TensorBoard events, cohort attempts `100` per arm, and `decision=CAPACITY_AB_FIRST`.

Run the selector from `D:\luolin\V13\.worktrees\assembly-selector-20260817` using the frozen original-control manifest that produces cohort signature `e3003eb09d8822bfb6cb0d1a97e017a57823a24e74256daf44d7daf7afee1904`:

    C:\Users\YU\.conda\envs\brepgen_env\python.exe tools\run_assembly_repair_selector.py --calibration-manifest D:\luolin\V13\local_runs\assembly_calibration_100cad_v1_20260809\calibration_manifest.jsonl --breparg-root D:\luolin\V13\BrepARG --output-dir D:\luolin\V13\local_runs\assembly_selector_invalid16_20260817 --historical-invalid-only

The pilot must print exactly six final both-valid restores and a `selector_protocol_passed: true` record. If it passes, repeat the same command without `--historical-invalid-only` into a new `assembly_selector_100cad_20260817` root. The full run must preserve all 84 historical strict-valid CADs, have at least 90 strict-valid and 87 both-valid final rows, exactly the registered six both-valid restorations and three fallback-accepted IDs, and no worker/protocol failures. A selector proof does not waive the separate 95/100 assembly release gate.

## Validation and Acceptance

Both quantizer arms must complete seeds 3 and 4 through epoch 99. Every history must report exactly 100 epochs, zero skipped batches, zero non-finite loss/gradient/state/validation events, effective gradient clipping, finite lifecycle audits, and successful best/final/rolling checkpoint deserialization. All four tasks must share exact train and validation inventory digests. RVQ must report valid per-stage usage for all epochs; stage 2 collapse is reported explicitly and cannot be hidden by stage 1 or combined indices.

The unchanged-chain capacity report must contain exactly 100 attempts for each arm with identical ordered CAD IDs and cohort hash. It must separately report STEP-readable, native, strict, and both-valid counts; discordant pairs; exact McNemar p-values; curved reconstruction distributions; and estimated token cost. If VQ-8192 reduces the bypass-minus-candidate strict gap to at most 5 points, select it unless RVQ is materially pairwise better under the registered rule. If neither arm meets the gate, do not start boundary consistency or AR; record that capacity remains unresolved.

The assembly repair report must cover all 100 original-control CADs. Acceptance is strict validity of at least 95/100, all 84 original strict-valid CADs still strict-valid, and a machine-readable map from each switch to restored, unchanged, and regressed CAD IDs. Broad tolerance changes or global joint-optimization changes are not accepted substitutes.

The selector must first pass the 16 historical-invalid pilot and then the full 100-CAD routing proof. Its local `assembly_selector_candidates.jsonl` may contain private paths and is never archived. Its final `assembly_repair_matrix.jsonl` has one selected result per CAD, while its `selection` object contains only a whitelist of profile names, booleans, SHA-256/size, numerical gate metrics, and rejection codes. The candidate ledger hash and final matrix hash are bound into `assembly_repair_run.json`. Snapshot validation must reject a selector selection object containing a path field, absolute path text, raw exception text, or a forbidden data/CAD/model artifact.

The final repaired-chain report must use the selected capacity arm and same cohort. Representation may proceed to full-scale training only if the candidate's strict gap to the same-scale bypass is at most 5 points on the repaired chain and the assembly repair meets its GT gate. Otherwise this plan ends with a fail-closed representation review, not sequence or AR work.

Git acceptance requires `git diff --check`, focused tests, report manifest revalidation, no tracked checkpoint/raw-data/STEP/pickle/array files, a clean worktree after commits, and a normal push to `origin/experiment/protocol-v5-scaling-ladder`.

## Idempotence and Recovery

All formal tasks use immutable signed manifests and task-specific output directories. Repeating an identical command resumes from the atomic rolling checkpoint. A changed arm, seed, protocol hash, inventory digest, precision, optimizer recipe, epoch target, or code revision must fail rather than overwrite evidence. Snapshot generation writes to a temporary file or directory and atomically replaces only a validated Git-safe target. Network push failure does not alter local commits and is retried without force.

## Artifacts and Notes

Starting paired evidence:

    GT strict:              84/100
    bypass@60k strict:      70/100
    VQ-4096@60k strict:     57/100
    bypass-only strict:     15
    VQ-only strict:          2
    strict McNemar p:        0.0023
    Delta_q:                13 percentage points
    Delta_r:                14 percentage points

No new local checkpoint is copied into Git. For every checkpoint used in a conclusion, the report records its absolute local role, byte size, SHA-256, arm, seed, epoch, experiment signature, protocol identity, and exact inventory identity.

## Interfaces and Dependencies

`breparg_improvements/train.py::quantizer_comparison_configs` must expose `vq_8192_64d_random` and `rvq_2x4096_64d_random`. `build_quantized_vqvae` must construct both. The RVQ module must behave like the existing quantizer from the VQ-VAE's perspective by returning `(quantized, loss, info)`, but `info` must retain separate stage index tensors. Metrics code must consume those tensors through an explicit staged schema rather than guessing from tensor shape.

The capacity launcher must be a standalone Python CLI in `tools/` and use the frozen constants above. The assembly repair module and coordinator must also live in this repository and accept explicit repair switches. No implementation may edit or track `D:\luolin\V13\BrepARG`.

`tools/assembly_selector_geometry.py::input_geometry_signature` returns a path-free signature of the post-joint-placement source grids and edges. `candidate_step_signature` is called only inside the isolated matrix worker and returns unique topology counts, optimal bounds, edge length, and both boundary residual directions. `geometry_topology_gate` is pure and fail-closed. `tools/run_assembly_repair_selector.py::route_selector_candidates` accepts injected candidate/gate functions for non-OCC tests and enforces the primary, near-vertex, interpolation ordering. Its `fallback_eligible` predicate requires saved STEP, `both_valid` status, all three validity booleans, and the worker-generated gate. `snapshot_assembly_repair.py::compact_selection` removes candidate paths, log paths, and raw error text before any Git-safe copy.

Revision note 2026-08-17 10:25 +08:00: Created this plan after the paired VQ/bypass gate selected capacity A/B. It freezes the capacity candidates, P0-B recipe, paired assembly decision, independent repair design, storage policy, final merge gate, and Git-safe evidence contract.

Revision note 2026-08-17 11:09 +08:00: Recorded completion of the capacity implementation, the three-item quantizer-info compatibility fix, fp32 residual handling under bf16, the 176-test regression, and the non-mutating four-task formal dry run. The next capacity milestone is a clean-commit CUDA smoke followed by the resumable formal matrix.

Revision note 2026-08-17 11:36 +08:00: Recorded the bounded CUDA smoke result and the evidence-schema repair. The next step remains a clean-commit two-arm CUDA smoke in a new output root, followed only on success by formal training.

Revision note 2026-08-17 12:16 +08:00: Recorded the successful D:-hosted two-arm smoke, E: filesystem diagnosis, formal launch and first healthy metrics. Added the immutable-source decision and clarified that formal training is now an automatic background task rather than an unstarted milestone.

Revision note 2026-08-17 12:35 +08:00: Recorded the first independently switchable assembly-repair primitives and the provenance correction that separates the running formal worktree from subsequent development before the second formal task begins.

Revision note 2026-08-17 13:00 +08:00: Recorded the real OCC runner smoke and the 16-case independent/combined pilot. The pilot recovered the two wire-build cases only under directed trim and established that global switch composition is unsafe; future work narrows trim/pcurve repair to diagnosed failing entities before the formal 100-CAD no-regression matrix.

Revision note 2026-08-17 13:15 +08:00: Recorded two negative generic pcurve self-intersection repair pilots. They added no restoration beyond the two directed-trim wire-build cases, so the plan rejects broad OCC ShapeFix as a route to the 95/100 gate and leaves boundary loss and downstream training blocked.

Revision note 2026-08-17 13:20 +08:00: Recorded the schema-v1 scheduler mismatch, stopped and quarantined the diagnostic-only run, froze schema-v2 runtime/evidence controls, repaired RVQ stage accounting and selection, and made a new CUDA smoke plus clean-root restart mandatory.

Revision note 2026-08-17 14:40 +08:00: Recorded the healthy schema-v2 live state, completed the v2 edge-position diagnosis, corrected stale formal-source and storage text, and constrained the next topology-changing repair to copied-face rollback plus one-CAD process isolation after bounded probes produced two recoveries and one native-process exit.

Revision note 2026-08-17 13:25 +08:00: Recorded the successful schema-v2 two-arm CUDA smoke and the FeaturePool tail-overflow found by the forced-resume smoke. The fix remains in the local adapter and must pass a second forced-resume smoke before formal launch.

Revision note 2026-08-17 13:30 +08:00: Recorded successful forced full-state resume across epoch 0 to epoch 1 and corrected the validator so formal runs retain exact 469/94 batch gates while bounded smoke runs use their realized deduplicated counts.

Revision note 2026-08-17 13:45 +08:00: Added a face/wire-local P0-A diagnostic milestone. Generic ShapeFix and global tolerance experiments were negative, so subsequent repair candidates must be tied to named failing faces or pre-STEP source-topology defects.

Revision note 2026-08-17 13:50 +08:00: The first face/wire execution used the older calibration manifest and therefore omitted one saved P0-A baseline STEP. The tool now accepts and prefers the stage-aware P0-A baseline-attempt manifest, which fixes the evidence population before any repair decision is made.

Revision note 2026-08-17 14:25 +08:00: Added and validated the separate face/wire crossing schema v2. It freezes the 16/16 stage-aware population, 11 STEP / 5 pre-STEP split, one-based edge-position semantics, independent occurrence taxonomy, and Git-safe snapshot before any new geometry mutation.

Revision note 2026-08-17 15:39 +08:00: Recorded the contaminated duplicate pilot, the signed assembly-run contract and worker protocol hardening, the Windows private-path correction, explicit OCC strategy exclusivity, and the conservative copied-face geometry-preservation gate. The signed topology pilot is diagnostic-only at 1/16 recovery; baseline parity and the remaining failure-mode repairs are still required.

Revision note 2026-08-17 16:18 +08:00: Recorded exact baseline parity and the completed local-topology 100-CAD no-regression matrix. The switch restores one CAD without regression but reaches only 85/100, so it is retained as an independent primitive while the 95/100 gate stays closed. The Git-safe snapshot now carries the signed run manifest and verifies its summary hash.

Revision note 2026-08-17 16:24 +08:00: Added the matching baseline parity snapshot so the remote evidence chain now contains both the unchanged 84/100 control vector and the 85/100 local-topology vector under their signed run manifests.

Revision note 2026-08-17 16:27 +08:00: Recorded the full original directed-trim matrix as a rejected profile: two wire-build recoveries, one historical-valid regression, and 85/100 strict overall. The next implementation increment is a topology-evidence fallback to baseline grouping, followed by a new signed 100-CAD no-regression matrix.

Revision note 2026-08-17 16:40 +08:00: Recorded the guarded directed-trim full matrix at 86/100 with both target restorations and zero regressions. Added Git-safe aggregation of the face-level loop-policy decisions, preserving the exact evidence that directed regrouping was limited to the two restored CADs.

Revision note 2026-08-17 16:56 +08:00: Recorded the completed signed explicit composition of guarded directed trim and copied-face local topology. The full 100-CAD result is 87 strict-valid, 84 both-valid, three restorations, zero regressions, zero worker failures, and a verified summary-to-manifest SHA-256 binding. The Git-safe snapshot includes the local-topology acceptance/rejection evidence and guarded loop-policy aggregation; the 95/100 gate remains closed.

Revision note 2026-08-17 17:50 +08:00: Fixed the local OCC worker-routing gap, validated it with 38 focused tests, and completed the corrected isolated pcurve-continuity pilot plus conditional 100-CAD matrix. The formal result is 97 STEP-readable, 88 native-valid, 86 strict-valid, 83 both-valid, two restorations, zero regressions, zero worker/protocol failures, and a verified summary SHA-256. The profile adds no recovery beyond guarded directed trim, so the 95/100 gate and all downstream representation work remain closed.

Revision note 2026-08-17 19:11 +08:00: Diagnosed and repaired the isolated non-unit-solid failure by reconciling eight same-face, mutually nearest endpoint-id pairs after the real CPU joint-placement stage. Recorded the signed 100-CAD no-regression result at 95 STEP-readable, 87 native-valid, 85 strict-valid, and 83 both-valid with one restoration and zero regressions. Added path-rejecting Git-safe diagnosis archival and explicitly kept the 95/100 assembly gate closed.

Revision note 2026-08-17 19:17 +08:00: Ported the bounded face-ID, closed-edge scale, guarded trim-loop, curve-fit, copied-face topology, and OCC completion checks into an isolated upstream production overlay. The signed production-backend matrix wrote 100/100 STEP files, preserved all 84 original strict-valid controls, restored the existing three CADs, and reached 87/100 strict-valid with zero regressions. It is eight below the predeclared 95/100 gate, so ADR-0003 holds the overlay out of the shared `D:\luolin\V13\BrepARG` source. The report snapshot and exported patch are Git-safe evidence for a future targeted residual repair.

Revision note 2026-08-17 19:55 +08:00: Composed the independently proven one-to-one near-vertex reconciliation with the isolated production backend at clean commit `aeec941`. The full 100-CAD signed matrix writes 100/100 STEP files, reaches 90 native-valid, 88 strict-valid, and 86 both-valid, restores `00000444...` plus the prior three cases, and has zero regressions while preserving the original 84 strict controls. The Git-safe snapshot binds SHA-256 artifacts and proves CAD-set plus parent/historical-strict equivalence to the prior production cohort, although its input order is reconstructed and sorted. The result remains seven below the immutable 95/100 gate; ADR-0003 continues to hold all production source changes outside `D:\luolin\V13\BrepARG`.

Revision note 2026-08-17 20:15 +08:00: Recorded a signed invalid-only production-overlay pcurve fallback pilot and a direct four-target instrumented follow-up. The 16-CAD pilot was `4/16` strict-valid, exactly the four recoveries already present in the 88/100 production composition, and left all other failures unchanged. The direct check proved that the fallback ran; geometry-preserving pcurve candidates did not reduce wire self-intersections, and the only topology candidates that removed them exceeded the face-geometry gate. Toggling `ShapeFix_Wire` closed-wire mode made no difference. The experiment is rejected without a full matrix, commit, or shared-source change; residual repair must move to a different diagnosed failure family.

Revision note 2026-08-17 21:39 +08:00: Completed the schema-v2 capacity root and unchanged-chain capacity measurement. RVQ seed 4 recovered from a Windows rolling-checkpoint replace failure by exact auto-resume in the same signed root. The final validator reports four valid 100-epoch tasks with consistent inventories. The fixed 100-CAD capacity report selects VQ-8192: bypass `70/100` strict, VQ-8192 `69/100`, RVQ `65/100`, RVQ-vs-VQ strict McNemar `p=0.4239501953125`, and estimated RVQ sequence cost `+36%`. Capacity is resolved, but sequence regeneration, boundary consistency, and AR stay blocked because the best production assembly chain remains `88/100`, below the `95/100` gate.

Revision note 2026-08-18 05:47 +08:00: Re-ran the capacity A/B from post-hardening source `D:\capv5` at commit `6f7436da50a5f455fd9af3c806676ac0f49b8f9f`. All four tasks validated at exactly 100 epochs with identical inventories and runtime-resume compatibility, and the fixed 100-CAD measurement reproduced `VQ_8192_DIRECT_WIN`. The Git-safe snapshot at `reports/capacity_ab_posthardening_assembly_measurement_20260818/` contains only the JSON, Markdown, and CSV artifacts and binds the post-hardening capacity root as authoritative.

Revision note 2026-08-17 20:05 +08:00: Added the failure-triggered selector, its child-worker-only bidirectional geometry gate, durable dynamic candidate ledger, Git-safe selection serializer, and 70 focused tests. One near-vertex and one interpolation fallback smoke both returned native/strict/both-valid and passed the fixed gate. Replaced loose `BRepBndLib.Add` with `AddOptimal` after proving the former reports a control-polygon bbox rather than the trimmed shape; the threshold itself was not relaxed. The next assembly milestone is the registered 16-invalid selector pilot followed only on success by the full 100-CAD matrix.

Revision note 2026-08-17 20:29 +08:00: Recorded the pre-pilot selector review and closed its evidence blockers. Geometry-gate schema v2 binds vertices and face/edge incidence and rejects every unprojectable or unsampled boundary element; candidate results, source pickles, execution sources, matrix/ledger hashes, and Git-safe snapshots are now cross-bound. JSONL resume is durable and may repair only an unterminated tail write. The earlier one-CAD smoke outputs remain development-only, and the next formal action is a clean-commit 16-invalid pilot.

Revision note 2026-08-17 21:02 +08:00: Closed the second pre-pilot review findings. Positive geometry gates now require full semantic validation, effective vertex-edge incidence is bound after topology remapping, workers prove the exact pickle bytes they deserialize, completion rechecks every code/runtime/input hash, and the Git-safe snapshot retains the new proof fields. The focused suite now passes 95 tests; the next action remains a clean commit followed by a new-root 16-invalid pilot.
