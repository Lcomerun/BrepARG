# Build and validate a parent-CAD-isolated Protocol V2

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document is maintained in accordance with `PLANS.md` at the repository root. `AGENTS.md` mentions `.agent/PLANS.md`, but that path is absent; the checked-in authority is the root `PLANS.md`.

## Purpose / Big Picture

Historical V13 training used record-level train/validation/test splits and then split VQ patches a second time inside the training set. As a result, validation metrics can contain geometry derived from the same parent CAD as training, and the current metrics cannot establish generalization. After this plan is complete, an operator can build one explicit CAD protocol manifest, prove that all STEP parts from a parent CAD stay in one split, train and validate the FSQ VQ-VAE from disjoint CAD sets, and inspect a lightweight report from a small reproducible experiment.

The observable result is a `protocol_summary.json` whose accepted rows satisfy 10 to 50 faces, at most 150 global edges and at most 30 edges on every face; an integrity report whose three pairwise parent overlaps are zero; focused tests that fail on the historical behavior and pass on the new behavior; and a small FSQ smoke run that emits finite reconstruction and code-usage metrics. No file under `BrepARG/` or `papers/` is modified or committed.

## Progress

- [x] (2026-08-03 20:44 +08:00) Created isolated worktree `D:\luolin\V13\.worktrees\protocol-v2-parent-isolated` on branch `experiment/protocol-v2-parent-isolated`.
- [x] (2026-08-03 20:48 +08:00) Ran the focused baseline in `brepgen_env`; observed 122 passing tests and two unrelated existing failures.
- [x] (2026-08-03 20:53 +08:00) Mapped the historical record-level split, patch-level validation leakage, and shard/cache provenance gap.
- [x] (2026-08-03 21:00 +08:00) Wrote the approved design in `docs/superpowers/specs/2026-08-03-protocol-v2-parent-isolated-design.md`.
- [ ] Add failing protocol eligibility and parent-group split tests.
- [ ] Implement the standard-library Protocol V2 core and manifest CLI.
- [ ] Add failing VQ CAD-level isolation and exact-deduplication tests.
- [ ] Integrate isolated train/validation patch collection and FSQ validation metrics.
- [ ] Run a real-data protocol smoke and a tiny FSQ training smoke.
- [ ] Write aggregate experiment reports, run final reviews, and push the feature branch.

## Surprises & Discoveries

- Observation: The repository instruction points to `.agent/PLANS.md`, but the only plan authority present is `PLANS.md` at the repository root.
  Evidence: `rg --files -g PLANS.md` returned only `PLANS.md`.

- Observation: Running tests with the default `C:\Python314\python.exe` produced dependency failures because it has neither NumPy nor PyTorch; the existing `brepgen_env` has NumPy 1.26.4, PyTorch 2.2.2+cu118, Diffusers 0.27.0, and visible CUDA.
  Evidence: `C:\Users\YU\.conda\envs\brepgen_env\python.exe` imported all three libraries and reported `torch.cuda.is_available() == True`.

- Observation: The correct-environment focused baseline has two existing failures unrelated to Protocol V2.
  Evidence: `python -m pytest tests\test_audit_split_integrity.py tests\test_local_pipeline_helpers.py -q` reported `122 passed, 2 failed`; one test directly expects the intentionally excluded worktree path `BrepARG/2sequence.py`, and one old sequence-shard fixture omits the already-required `ordering` metadata.

- Observation: Old patch shard and sample cache inputs cannot prove split isolation.
  Evidence: `breparg_improvements/train.py::collect_se` ignores CAD paths when a shard or cache is configured, and their current records do not carry a verified protocol hash and split assignment.

- Observation: The only complete healthy parsed source is an archive root, not the unpacked paths referenced by historical splits.
  Evidence: `D:\luolin\V13\ABC\processed\abc_parsed_full_archives` contains 100 ZIP files with 681,406 parsed pickle members and 174,374,900,417 compressed bytes; the unpacked `abc_parsed_full` directory is empty. The first archive alone contains 5,943 records.

## Decision Log

- Decision: Implement data protocol repair before AR training changes.
  Rationale: Context length, learning rate, accumulation, ordering and sampling experiments cannot be interpreted while train/validation identity is contaminated. Keeping this branch focused also prevents a multi-variable experiment.
  Date/Author: 2026-08-03 / Codex, based on the user's A-E priorities.

- Decision: Create `breparg_improvements/cad_protocol.py` rather than strengthening a historical BrepARG same-data helper.
  Rationale: V13 needs one reusable protocol authority without changing the legacy comparison path or importing NumPy/PyTorch for metadata audit.
  Date/Author: 2026-08-03 / Codex.

- Decision: Reject unresolved parent identity from Protocol V2.
  Rationale: Falling back to a basename can make an unverifiable split appear leak-free. Fail-closed behavior is required for an independent validation claim.
  Date/Author: 2026-08-03 / Codex.

- Decision: Balance split targets by record count while never splitting a parent group.
  Rationale: Parent groups have unequal numbers of STEP parts. Balancing only parent count can create a much more skewed training or validation record count.
  Date/Author: 2026-08-03 / Codex.

- Decision: Use exact patch hashes for training deduplication and rounded hashes only for audit in this milestone.
  Rationale: Rounding at four decimals may erase small but meaningful curved geometry differences. Exact duplicate removal is safe; approximate duplicate policy needs measured evidence.
  Date/Author: 2026-08-03 / Codex.

- Decision: Fail closed on legacy single-root patch shards and sample caches in Protocol V2.
  Rationale: A directory name does not prove cohort identity. Split-specific versioned assets with protocol hashes are a separate future change.
  Date/Author: 2026-08-03 / Codex.

## Outcomes & Retrospective

Implementation is in progress. This section will record the exact accepted/rejected real-data counts, split balance, parent overlap, duplicate rates, FSQ smoke metrics, test totals, commit range and remote branch after validation.

## Context and Orientation

The repository root for this worktree is `D:\luolin\V13\.worktrees\protocol-v2-parent-isolated`. V13-owned training code is in `breparg_improvements/`; repository tools are in `tools/`; tests are in `tests/`; durable summaries are in `reports/` and `reproducibility/`. The ignored sibling checkout `D:\luolin\V13\BrepARG` is upstream source used at runtime by legacy training, but it is outside this change.

A parsed CAD record is a pickle dictionary. `surf_ncs` holds one normalized 32 by 32 by 3 point grid per face. `edge_ncs` holds one normalized 32 by 3 point sequence per global edge. `faceEdge_adj` is a list whose item for each face lists global edge indices on that face. A parent CAD is the UUID portion of names like `00000140_2fc54fcd110d4f49969163c4_step_003.pkl`; multiple `_step_NNN` records may originate from the same parent.

`breparg_improvements/train.py::stage_split` currently shuffles individual pickle paths and slices them 90/5/5. `stage_vqvae` and `stage_vqsweep` currently collect patches from train paths and slice the resulting patch array, so patches from one CAD cross the VQ train/validation boundary. `tools/audit_split_integrity.py` can detect parent overlap after the fact but is not used to construct a safe split.

Protocol V2 means a manifest schema and filtering rules, not a model architecture. The rules are 10 through 50 faces inclusive, no more than 150 global edges, no more than 30 edges on any face, complete `faceEdge_adj`, legal edge indices, resolvable parent identity, and parent-grouped 8:1:1 splitting. A protocol hash is the SHA-256 of the canonical configuration and ordered manifest identity; it ties later artifacts to the exact cohort.

## Plan of Work

First, add protocol tests in `tests/test_cad_protocol.py`. Fixtures will exercise both inclusive boundaries and each reject reason. Other tests will construct unequal parent groups in different input orders and prove deterministic assignment, zero pairwise parent overlap, no parent fragmentation, and explicit reporting when too few parents can populate every split. Run only this new test file and observe missing-module or missing-symbol failures before implementation.

Second, create `breparg_improvements/cad_protocol.py`. Define an immutable `ProtocolConfig`, parent/source normalization helpers, a pure record inspection function, manifest-row creation, deterministic parent-group selection and splitting, canonical hashing, summary construction, and JSONL/JSON/pickle writers. Loading failures become rows rather than aborting the scan. Create `tools/build_cad_protocol.py` as a thin CLI that recursively discovers ordinary `.pkl` files or streams `.pkl` members from `abc_XXXX_parsed.zip` archives, optionally applies an explicitly labeled smoke scan limit, materializes only accepted selected rows when requested, builds outputs atomically where practical, calls the existing split integrity audit, and returns nonzero when acceptance gates fail.

Third, add patch inventory tests in `tests/test_vqvae_protocol_sampling.py`. They will prove that exact duplicate patches within a split are merged without losing source provenance; surface and edge kinds hash separately; rounded hashes are measured independently; and train/validation sources and parents are disjoint. A pure preparation helper will be tested rather than launching a GPU model.

Fourth, extend `breparg_improvements/vqvae_sampling.py` with canonical patch hashing, split-local deduplication, provenance merge, and duplicate summaries. Extend `collect_se` or replace it with a records-preserving helper so `train.py` retains source identity until train and validation inventories have been built and audited. Parsed-file training collects train patches from `split['train']` and validation patches from `split['val']`; it never slices one combined patch array. The sweep path receives the same repair. Legacy unsplit shard/cache configuration raises a clear error under Protocol V2.

Fifth, create `breparg_improvements/vqvae_metrics.py` and focused tests. It will assign `surface_planar_like`, `surface_curved_proxy`, or `edge` labels and compute code usage from aggregate validation counts. Integrate these helpers into `_train_vqvae`: collect per-sample reconstruction loss by bucket, accumulate all validation encoding indices, then write unique bins, coverage, entropy perplexity and bucket MSE into every history record. Batch perplexity returned by the quantizer is not averaged.

Sixth, run the protocol CLI against a deterministic small subset of actual parsed data found on this machine. Store raw manifest and split outputs under ignored `local_runs/`; copy only aggregate JSON and a short Markdown interpretation into `reports/protocol_v2/`. The report must include source root, scan cap, git commit, config, counts, rejection reasons, distributions, protocol hash and integrity gate. Absolute paths may remain in the local raw artifact; the committed report redacts machine-specific roots to a descriptive label.

Seventh, run a tiny FSQ smoke in the existing `brepgen_env` using the generated split. Use a fixed seed, a small train and validation patch cap, one to a few epochs and a batch that fits the available GPU. The smoke must prove finite forward/backward loss, checkpoint writing, validation buckets and aggregate code usage. It must be labeled engineering smoke and must not be compared with the paper's Valid score. If a second capacity arm is affordable, change only FSQ levels while keeping all other inputs and controls fixed.

Finally, update this living plan, the design document if implementation decisions change, `reproducibility/reports/current_conclusions.md`, and a lightweight experiment report. Run focused and broader tests, Python compilation, JSON parsing, Git diff checks and an independent code review. Commit focused changes, push `experiment/protocol-v2-parent-isolated` to `origin`, and do not merge `main`.

## Concrete Steps

All implementation commands run from `D:\luolin\V13\.worktrees\protocol-v2-parent-isolated` in PowerShell. Use the established environment explicitly:

    $python = 'C:\Users\YU\.conda\envs\brepgen_env\python.exe'

The protocol red phase is:

    & $python -m pytest tests\test_cad_protocol.py -q

Before implementation, expect collection or assertion failures because `breparg_improvements.cad_protocol` and its interfaces do not exist. After implementation, expect every test in that file to pass.

The patch isolation and metrics red phases are:

    & $python -m pytest tests\test_vqvae_protocol_sampling.py tests\test_vqvae_metrics.py -q

Before implementation, expect missing-symbol failures. After implementation, expect zero failures and explicit assertions for source/parent disjointness, exact hash behavior, rounded audit behavior, bucket labels and aggregate perplexity.

Run the protocol CLI using the actual data root discovered during execution. A representative invocation is:

    & $python tools\build_cad_protocol.py --archive-root D:\luolin\V13\ABC\processed\abc_parsed_full_archives --chunks 0-0 --output-dir E:\V13_protocol_v2_smoke_20260803\protocol --materialize-root E:\V13_protocol_v2_smoke_20260803\parsed_pool --max-scan-records 2000 --max-eligible-records 1000 --seed 20260803

Expect `protocol_manifest.jsonl`, `protocol_summary.json`, `split.pkl` and `split_integrity.json`. The summary status must be `VERIFIED`; a scan-limited report must contain `experiment_scale: smoke`. Archive members that are not selected are never extracted, and no command rewrites the source ZIP.

Run the tiny training through `breparg_improvements/train.py` with explicit environment variables. Exact values will be updated after inspecting eligible counts and GPU memory. The command must set `NS_POOL`, `NS_OUTBASE`, `NS_OUT`, `NS_N`, `NS_VQ_SAMPLES`, `NS_VQ_VAL_SAMPLES`, `NS_VQ_EPOCHS`, `NS_VQ_BS`, `NS_LEVELS`, protocol thresholds and a fixed seed. Run `--stage split` first, inspect the protocol gate, then run `--stage vqvae`. Do not run sequence or AR in this smoke.

Focused verification is:

    & $python -m pytest tests\test_cad_protocol.py tests\test_audit_split_integrity.py tests\test_vqvae_protocol_sampling.py tests\test_vqvae_metrics.py -q
    & $python -m compileall -q breparg_improvements tools tests
    git diff --check

The broader regression command is:

    & $python -m pytest tests -q

Known baseline failures must be compared by exact test name rather than hidden. The worktree-specific `BrepARG/2sequence.py` test and the pre-existing sequence-shard `ordering` fixture are baseline exceptions; no new failures are accepted.

## Validation and Acceptance

Protocol eligibility is accepted when boundary fixtures at faces 10/50, global edges 150 and per-face edges 30 pass, while values outside each boundary fail with stable reasons. Missing or malformed adjacency and unresolved parent IDs must fail closed.

Splitting is accepted when changing input order does not change any source assignment, every parent appears in exactly one split, and `tools/audit_split_integrity.py` reports zero parent overlap for train/validation, train/test and validation/test. Summary actual counts may differ from exact 8:1:1 on small cohorts, but the deviation and parent counts must be visible.

VQ isolation is accepted when train records come only from train CAD paths, validation records come only from validation CAD paths, and source-key and parent-ID intersections are empty before tensors are created. Exact duplicates may be removed only inside a split. Cross-split exact hash overlap is reported and blocks an independent validation claim until investigated.

FSQ monitoring is accepted when a synthetic known histogram produces the mathematically expected unique count, coverage and entropy perplexity; the training smoke writes these metrics from aggregate validation counts and writes finite reconstruction loss for every non-empty bucket.

The experiment is accepted as a smoke, not as a quality conclusion, when it uses actual parsed CAD inputs, writes a checkpoint and history, contains only finite reported metrics, records git/protocol/config identity, and publishes an aggregate report small enough for Git. It does not need to meet the later heuristic target of perplexity 800 to 1500 or curved reconstruction `5e-5`; failure to meet those targets in a tiny smoke is not evidence against the architecture.

The branch is ready to push when `git status --short` is clean after commits, no tracked path is under `BrepARG/`, `papers/`, datasets or checkpoint patterns, focused tests pass, broader tests introduce no failures relative to baseline, and `git ls-remote --heads origin experiment/protocol-v2-parent-isolated` shows the pushed branch.

## Idempotence and Recovery

Manifest construction is repeatable with the same source set, configuration and seed. Writers use temporary files followed by replacement so an interrupted run does not leave a valid-looking partial summary. An existing output directory is not silently reused unless an explicit overwrite or resume option verifies the protocol hash.

No source data is moved, deleted or rewritten. Raw protocol and training artifacts live under ignored `local_runs/`; rerunning a smoke uses a new run name or an explicit exact output directory. Checkpoints remain ignored and are never added to Git.

If Protocol V2 rejects too many records, inspect reject counts and sample paths in the local manifest. Do not loosen limits automatically. If parent parsing fails because a real naming variant was not covered, add a failing fixture for that exact form, extend the parser narrowly, and regenerate the entire manifest because its protocol hash changes.

If the training smoke fails after the protocol gate passes, retain its log under `local_runs/`, record the first failing stage in the report, and diagnose it without falling back to the old patch-level split. Legacy shard/cache use can be restored only by explicitly running a historical branch; it is not a recovery path for a Protocol V2 result.

The Git worktree can be removed after the remote branch is verified, but no destructive cleanup is part of this plan. The main worktree and ignored upstream/data trees remain untouched.

## Artifacts and Notes

Baseline commit before feature work is `16f976eadc6bd095ec1e43a7e6201f9fcd411eb4`.

The focused baseline transcript is:

    122 passed, 2 failed in 6.77s
    FAILED test_breparg_sequence_groups_preserve_source_path_metadata
    FAILED test_sequence_shards_merge_preserves_order_and_metadata

The first failure addresses a directly excluded upstream file absent from the isolated worktree. The second fixture omits existing required metadata and is not caused by this work. Both remain visible in final regression comparisons.

Historical evidence motivating the change is in `docs/audits/v13_sequence_split_integrity_20260731.json`, `docs/audits/breparg_same_data_split_integrity_20260731.json`, `docs/full_experiment_postmortem_20260731.md`, and `reproducibility/project_history/07_data_and_protocol/data_and_evaluation_protocol.md`.

## Interfaces and Dependencies

In `breparg_improvements/cad_protocol.py`, define a frozen `ProtocolConfig` with `min_faces`, `max_faces`, `max_global_edges`, `max_edges_per_face`, `train_ratio`, `val_ratio`, `test_ratio`, `seed` and `version`. Define pure interfaces equivalent to:

    def parent_cad_id(source_path: str) -> str | None
    def inspect_cad_record(data: Mapping[str, Any], config: ProtocolConfig) -> dict[str, Any]
    def build_manifest_row(source_path: str, data: Mapping[str, Any] | None, config: ProtocolConfig, load_error: str | None = None) -> dict[str, Any]
    def assign_parent_splits(rows: Sequence[Mapping[str, Any]], config: ProtocolConfig) -> dict[str, str]
    def build_protocol(paths: Sequence[Path], config: ProtocolConfig, max_eligible_records: int = 0) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]
    def write_protocol_outputs(output_dir: Path, rows: Sequence[Mapping[str, Any]], split: Mapping[str, Sequence[str]], summary: Mapping[str, Any]) -> dict[str, Path]

The module may use `dataclasses`, `hashlib`, `json`, `numbers`, `pickle`, `random`, `re`, `statistics`, `tempfile`, `zipfile` and `pathlib`. It must not require NumPy, PyTorch or BrepARG.

In `breparg_improvements/vqvae_sampling.py`, define canonical hashing and deduplication interfaces that retain `record_id`, `source_path`, `parent_id`, `kind`, array shape, duplicate count and source provenance. Training integration may use NumPy because this module already does.

In `breparg_improvements/vqvae_metrics.py`, define bucket labeling and aggregate code-usage functions. Entropy perplexity is `exp(-sum(p_i * log(p_i)))` for nonzero probabilities computed from the complete validation count vector. Coverage is `unique_nonzero / codebook_size`.

`tools/build_cad_protocol.py` is the supported CLI. `breparg_improvements/train.py` consumes its split semantics and rejects legacy unverified patch assets in protocol mode. Tests use pytest and the existing `brepgen_env`; the real smoke additionally uses the installed CUDA-enabled PyTorch and Diffusers.

Revision note 2026-08-03: Created after repository inspection, baseline execution, and design selection. It intentionally limits the first implementation and experiment to data protocol, VQ isolation, deduplication and FSQ observability; AR and generation changes remain a later dependent milestone.

Revision note 2026-08-03 21:05 +08:00: Added direct ZIP archive scanning and selected-record materialization after discovering that all historical unpacked split paths are stale and the archive root is the only healthy full-data authority.
