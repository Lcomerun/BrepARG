# V13 Lightweight Reproducibility Source Package Design

Date: 2026-08-02

Status: Approved design, pending implementation

Target platform: Linux GPU servers, with AutoDL as the primary deployment environment

## 1. Purpose

This design defines a lightweight, source-only reproducibility package for the V13/BrepARG research project. A future researcher must be able to unpack the archive, understand the full project history, verify that required external data and model artifacts are the intended ones, inspect every documented experiment, and launch supported experiments without relying on the original Windows drive layout or the historical `/root/autodl-tmp` layout.

The package is not merely a source-code snapshot. It is a layered research handoff containing runnable source, a clean reference revision, environment locks, normalized experiment definitions, external-artifact contracts, the full chain postmortem, experiment records, failure evidence, and verification tooling.

The package must preserve the distinction between three facts:

1. The current working tree contains the most complete runnable implementation, including tracked modifications and relevant untracked source files.
2. Outer Git commit `16cf19b` is the clean historical reference revision, not the default runnable implementation.
3. The nested BrepARG source has its own provenance at commit `07970a4`; its nested `.git` directory must not be shipped.

## 2. Goals

The completed package must support the following outcomes.

- A new operator can read one entry document and understand what V13 changes relative to BrepARG, what was trained, what failed, what evidence exists, and what work remains.
- A new operator can configure all machine-specific paths in one file and run a global preflight without editing project source.
- A new operator can list, explain, smoke-test, run, monitor, and verify any supported experiment through one launcher interface.
- Missing, substituted, corrupted, or incompatible external artifacts are detected before expensive training or evaluation begins.
- Historical failed experiments remain available for audit but cannot be launched accidentally.
- Every new run freezes the effective source identity, environment, command, parameters, input hashes, random seeds, and expected outputs.
- The archive can be rebuilt deterministically from the same source state and produces a verifiable ZIP checksum.
- The project timeline, experiment ledger, postmortem, decisions, failure incidents, and evidence paths remain readable without the original conversation.

## 3. Non-Goals

The package will not contain large or machine-specific artifacts. In particular, it will not contain checkpoints, sequence packages, parsed ABC archives, parsed shards, VQ patch shards, generated STEP/STL files, generated PNG images, TensorBoard event files, caches, or full large training logs.

The package will not provide Docker images, silently download large data, clean the current workspace, move files between drives, or delete any original material. It will not claim reproduction of official BrepARG paper metrics unless the official protocol and compatible official weights have actually been verified. It will not convert historical failed runs into recommended experiments merely because their commands can be reconstructed.

## 4. Package Layout

The archive root is `v13_repro_source_20260802/` and contains the following stable structure.

    v13_repro_source_20260802/
    |-- START_HERE.md
    |-- reproduce.sh
    |-- PACKAGE_MANIFEST.json
    |-- SHA256SUMS
    |-- source/
    |   |-- current/
    |   `-- clean_head_16cf19b/
    |-- provenance/
    |-- experiments/
    |   |-- recommended/
    |   |-- baselines/
    |   |-- diagnostics/
    |   `-- historical_failed/
    |-- configs/
    |-- environments/
    |-- launchers/
    |-- artifact_specs/
    |-- reports/
    |-- project_history/
    |-- tests/
    `-- BUILD_REPORT.md

All package-level file names are ASCII. Chinese documentation content is retained as UTF-8 because the project history and operator guidance are primarily Chinese.

## 5. Source Layers and Provenance

### 5.1 Current runnable source

`source/current/` is the default source used by launchers. It is a filtered snapshot of the current working tree at package-build time. It includes tracked files plus relevant untracked code, tests, configuration templates, plans, and documentation. It excludes Git internals, generated outputs, large data, caches, local environments, secrets, and artifacts prohibited by this design.

The snapshot must include a complete file manifest with relative path, byte count, SHA-256, source classification, and inclusion reason. The package builder must sort paths before hashing and archiving.

### 5.2 Clean outer reference

`source/clean_head_16cf19b/` is exported from outer Git commit `16cf19bb79b6bfa8beb4660e88f8d9dc813216e2`. It is reference-only and is never selected by a normal `run` command. A researcher can use it to inspect the last clean committed baseline and compare it with the current runnable snapshot.

### 5.3 Provenance records

`provenance/` contains at least:

- outer repository HEAD and commit metadata;
- outer tracked diff against `16cf19b`, including binary-diff metadata where applicable;
- outer `git status --short` captured at build time;
- an untracked-file manifest containing hashes and inclusion/exclusion decisions;
- the complete source manifest for `source/current/`;
- the complete source manifest for `source/clean_head_16cf19b/`;
- nested BrepARG commit `07970a4`, nested status, nested patch, and nested source manifest;
- a machine-readable build provenance JSON containing the fixed package release epoch, builder version, source identities, and filtering rules. Wall-clock build time and host-specific execution details are written beside the ZIP rather than embedded in the deterministic archive.

Nested `.git` directories and repository credentials are never copied. The package must fail its acceptance check if any `.git` directory or credential-shaped file is present.

## 6. Experiment Taxonomy and Registry

Every known experiment is represented by a structured JSON descriptor under one of four categories.

### 6.1 Recommended

`experiments/recommended/` contains the current scientifically defensible next-step workflows. These prioritize component isolation and protocol repair before further long training. Examples include parent-isolated split auditing, the ground-truth/continuous/FSQ/teacher-argmax oracle ladder, the promoted complex-curved VQ evaluation, and only those training workflows whose inputs and success rules are fully specified.

### 6.2 Baselines

`experiments/baselines/` contains V13 reference runs, same-data BrepARG retraining, and any official-weight probe or compatible official baseline that can be described honestly. A descriptor must distinguish `official_weight`, `same_data_self_trained`, and `protocol_probe`; these labels cannot be used interchangeably.

### 6.3 Diagnostics

`experiments/diagnostics/` contains FSQ-only reconstruction, true-token reconstruction, teacher-forcing CE, teacher-forced argmax reconstruction, DFS versus RCM ordering, sequence-length buckets, data leakage audits, checkpoint health checks, generation quality gates, and comparable focused tests.

### 6.4 Historical failed

`experiments/historical_failed/` contains the commands and evidence for known failed or invalid runs, including AR divergence/nonfinite incidents, OOM-prone configurations, stale path assumptions, incompatible checkpoint loading, incomplete sequence pipelines, and generation-only tuning that did not solve the underlying distribution problem.

Historical failed descriptors are disabled by default. Running one requires `--allow-historical-failed`, and the resulting run manifest must retain a warning and the documented failure reason.

### 6.5 Descriptor schema

Every experiment descriptor contains:

- stable experiment ID and human-readable title;
- category and scientific role;
- runnable state: `runnable`, `documentary`, or `blocked_missing_evidence`;
- default source layer, entry point, working directory, and command arguments;
- environment requirements and optional capabilities such as OCC;
- required and optional artifact IDs;
- path variables consumed from `configs/paths.env`;
- default parameters, smoke parameters, and random-seed policy;
- expected outputs and machine-checkable success rules;
- resume policy and checkpoint compatibility rules;
- known risks, historical result summary, and evidence references;
- confidence label: `confirmed`, `strong_inference`, or `insufficient_evidence`;
- whether explicit historical-failure opt-in is required.

Documentary experiments appear in `list` and `explain` but refuse `run` with an actionable reason. An experiment must not be labeled runnable when its required code, artifact identity, or command cannot be verified.

### 6.6 Coverage accounting

The package builder creates an experiment-source inventory from plans, reports, run summaries, manifests, and known launcher scripts. Every candidate record must be mapped to one or more experiment descriptors or explicitly classified as non-experiment context. The build fails when an inventory item remains unclassified. This makes the requirement to include all historical experiments auditable instead of relying on an informal claim of completeness.

## 7. Unified Launcher

The public interface is:

    bash reproduce.sh preflight
    bash reproduce.sh bootstrap
    bash reproduce.sh list
    bash reproduce.sh explain <experiment-id>
    bash reproduce.sh smoke <experiment-id>
    bash reproduce.sh run <experiment-id>
    bash reproduce.sh status <experiment-id>
    bash reproduce.sh verify <experiment-id>

Historical failed execution additionally requires:

    bash reproduce.sh run <experiment-id> --allow-historical-failed

`reproduce.sh` is a small LF-terminated shell wrapper. It delegates structured work to a Python standard-library launcher under `launchers/`, allowing `list`, `explain`, package-integrity checks, and missing-artifact diagnostics to work before the project environment is installed.

### 7.1 Command behavior

`preflight` validates package integrity, path configuration, platform, available disk space, NVIDIA driver visibility, CUDA/PyTorch compatibility when installed, required command-line tools, external artifacts, and optional OCC capability. It separates package-level success from experiment-level readiness instead of reporting one ambiguous pass/fail state.

`bootstrap` creates or updates the named Conda environment idempotently, installs the pinned pip layer, and runs import probes. It never edits global Python and never hides dependency failures.

`list` prints every experiment with category, runnable state, required artifact readiness, and a one-line purpose.

`explain` prints the experiment's scientific question, exact source layer, inputs, parameter defaults, expected outputs, success rules, evidence, and known risks.

`smoke` uses the descriptor's reduced workload. It checks imports, data loading, checkpoint compatibility, one or a few batches, output creation, and finite metrics. It is not allowed to silently substitute synthetic data for a real scientific input, although package-internal fixtures may test launcher mechanics.

`run` creates an isolated run directory, freezes metadata, verifies inputs, and executes the descriptor. Existing run directories are not overwritten. Resume behavior must be explicit and descriptor-controlled.

`status` reports process state where available, latest log activity, completed stages, last finite metric, produced outputs, and any recorded failure.

`verify` checks required outputs, hashes, JSON schemas, finite metrics, checkpoint health, and experiment-specific acceptance rules. A process exit code of zero alone is not sufficient for verified status.

## 8. Runtime Path Configuration

No package control file, experiment descriptor, or public launcher may embed `C:\`, `D:\`, `E:\`, `/root/autodl-tmp`, or another host-specific absolute path as a runtime default. The package provides `configs/paths.env.example`; the operator creates `configs/paths.env` locally. Legacy source snapshots and historical evidence may retain old absolute paths for provenance, but those files are not exposed as the supported launch interface and are labeled accordingly.

The stable variables include:

    V13_DATA_ROOT
    V13_ARTIFACT_ROOT
    V13_RUN_ROOT
    V13_PARSED_ARCHIVE_ROOT
    V13_PARSED_SHARD_ROOT
    V13_PATCH_SHARD_ROOT
    V13_VQ_CHECKPOINT
    V13_AR_CHECKPOINT
    V13_SEQUENCE_PACKAGE

Experiment descriptors may introduce additional named paths only when the path has a documented purpose and an artifact specification. Launchers resolve and normalize every path, reject empty required values, and write the resolved values to the run manifest without modifying the source snapshot.

## 9. Environment Locking

The package uses a Conda plus pip dual lock for Linux GPU servers.

`environments/environment.linux-gpu.yml` pins the environment name, Python version, Conda channels, and packages best managed by Conda. `environments/requirements.linux-gpu.lock.txt` contains exact pip versions and the appropriate PyTorch CUDA wheel source. Broad unbounded requirements are not accepted in the final lock.

The lock is selected for Linux x86_64, RTX 5090-class GPUs, and a CUDA runtime compatible with the pinned PyTorch build. The package also contains:

- an environment provenance report derived from available historical environment evidence;
- a minimal import probe for PyTorch, Transformers, NumPy, SciPy, zstandard, project modules, and other required libraries;
- a CUDA probe that reports device name, compute capability, PyTorch CUDA build, driver visibility, and a small tensor operation;
- an optional OCC installation guide and an OCC probe that constructs, writes, reads, and validates a trivial STEP solid;
- explicit handling for optional `chamferdist`, including whether each experiment requires it or uses a verified fallback.

If current Windows cannot resolve or execute the Linux environment, local acceptance verifies syntax and lock completeness; the final report lists real Linux GPU creation and CUDA execution as target-machine checks rather than pretending they were completed.

## 10. External Artifact Contracts

Large inputs remain outside the ZIP. Each external artifact has a JSON contract under `artifact_specs/` with:

- stable artifact ID and role;
- expected file or directory layout;
- exact byte size and SHA-256 for known files;
- a manifest hash for directory artifacts;
- producer experiment or historical source;
- compatible source/checkpoint/tokenizer/vocabulary identity;
- experiments that require it;
- expected placement relative to configured roots;
- verification mode and command;
- recovery or regeneration instructions when available.

The core contracts cover parsed archives, parsed shards, VQ patch shards, sequence packages, selected VQ checkpoints, selected finite AR checkpoints, and BrepARG baseline artifacts.

An artifact absent from the build machine may be documented as unresolved evidence, but it cannot carry a fabricated hash and cannot satisfy a runnable experiment. Recommended runnable experiments must have complete contracts. A checksum mismatch, size mismatch, vocabulary mismatch, incompatible `max_seq_len`, or nonfinite model tensor stops before training or generation.

## 11. Project History and Experiment Archive

The package includes a self-contained project-history layer:

    project_history/
    |-- 00_READ_ME_FIRST.md
    |-- 01_full_postmortem/
    |-- 02_timeline/
    |-- 03_experiment_ledger/
    |-- 04_plans_and_decisions/
    |-- 05_original_records/
    |-- 06_failure_incidents/
    |-- 07_data_and_protocol/
    `-- 08_evidence_index/

### 11.1 Full postmortem

The package includes `docs/full_experiment_postmortem_20260731.md` and its supporting audit records. The copied document must pass UTF-8 decoding and a mojibake check so Chinese content remains readable on Linux.

### 11.2 Timeline

The timeline records major data, VQ-VAE, sequence, AR, generation, baseline, diagnostics, storage, recovery, and governance events in chronological order. Each event includes date or date range, action, observed outcome, confidence, and evidence IDs. Where the only surviving source is user-provided terminal text, the event is explicitly labeled `conversation_derived_audit_record` rather than presented as an original log.

### 11.3 Experiment ledger

The normalized ledger is provided in both human-readable Markdown and machine-readable JSON. Every entry records experiment identity, source version where known, data cohort, split, sample count, model configuration, training epochs, checkpoint, metrics with numerator and denominator, outcome, failure mode, and evidence. Unknown values remain explicitly unknown; they are not guessed.

The ledger preserves the distinction between all-attempt success rate, generated-survivor validity, STEP readability, raw BRep validity, repaired validity, closed-kernel status, complexity, and geometric uniqueness.

### 11.4 Original records and large logs

Complete plans, core documentation, textual local reports, small logs, manifests, histories, summaries, and audits are copied when allowed by the lightweight policy. Large logs are represented by a record containing original path, byte count, SHA-256 when available, selected event lines, and head/tail excerpts. Heavy binary outputs are represented only by artifact or evidence records.

### 11.5 Evidence index

Every conclusion in the postmortem and ledger can point to an evidence ID. The evidence index maps that ID to packaged path, original path, content hash, evidence type, availability, and any caveat. Missing original evidence remains visible as missing rather than disappearing from the history.

## 12. Run Isolation and Failure Protection

Every run is written below `V13_RUN_ROOT` using a stable experiment ID plus UTC timestamp and a short configuration hash. Before execution, the launcher writes an immutable pre-run manifest containing:

- experiment descriptor hash;
- current source manifest hash and provenance identity;
- environment and CUDA summary;
- resolved paths;
- external artifact hashes;
- full command and environment overrides;
- model and data parameters;
- seed and determinism settings;
- expected outputs and verification rules.

The launcher uses strict shell error propagation and writes stage status atomically. It refuses to overwrite an existing run. It refuses implicit checkpoint fallback, silent scratch restart, architecture mismatch, vocabulary mismatch, incompatible positional embeddings, and use of known nonfinite checkpoints.

Training and evaluation verification explicitly checks finite losses and finite floating-point checkpoint tensors. OOM, nonfinite metrics, incomplete sequence assembly, missing output files, or a downstream step attempted after an upstream failure result in failed status. A failed run remains inspectable and is never relabeled successful by a later partial output.

## 13. Security and Privacy Filtering

The builder scans candidate files for common secret-bearing names and token patterns. It excludes `.env`, private keys, credentials, shell histories, nested environment directories, and machine-specific authentication files. Any suspicious included file blocks the build and is listed in the build report for manual review.

The package may preserve original local paths in historical evidence when necessary for provenance, but runnable configuration and commands must be path-portable. Historical path strings are labeled as evidence and are not interpreted by launchers.

## 14. Deterministic Build

The repository gains a package builder that uses sorted traversal, explicit inclusion rules, normalized ZIP timestamps, normalized package-level text line endings, and deterministic JSON serialization. The builder uses a fixed release epoch supplied through `SOURCE_DATE_EPOCH`, with the 2026-08-02 package release epoch as the documented default. Wall-clock time, temporary paths, and host-specific facts are excluded from the ZIP payload. Rebuilding with the same source inputs, artifact specifications, builder version, and release epoch must produce identical file manifests and an identical ZIP hash.

The build output is:

    dist/v13_repro_source_20260802.zip
    dist/v13_repro_source_20260802.zip.sha256
    dist/v13_repro_source_20260802.build-execution.json

The host-specific `build-execution.json` records actual wall-clock time, build host, temporary validation path, and command outcome, but is not part of the deterministic ZIP. The builder never deletes, moves, or modifies original project files. It constructs a staging directory, validates it, creates the ZIP, verifies extraction in a fresh temporary directory, writes the deterministic in-package `BUILD_REPORT.md`, and only then publishes the final archive, checksum, and host execution record.

## 15. Testing and Acceptance

### 15.1 Automated package tests

The implementation includes tests for:

- experiment descriptor schema and unique IDs;
- category and historical opt-in enforcement;
- path configuration parsing and missing-variable messages;
- artifact file and directory-manifest verification;
- checkpoint metadata and finite-tensor validation where fixtures permit;
- source inclusion/exclusion rules;
- absolute runtime-path scanning;
- forbidden extension and nested `.git` scanning;
- secret-pattern scanning;
- deterministic manifest and ZIP creation;
- launcher `list`, `explain`, `preflight`, `smoke`, `status`, and `verify` dispatch;
- run-directory collision and explicit resume behavior;
- project-history ledger and evidence-index referential integrity.

Tests use small package fixtures and do not ship real model weights or datasets.

### 15.2 Archive acceptance

The archive is accepted only when all of the following are true.

1. `SHA256SUMS` verifies every payload file except `SHA256SUMS` itself, while the external `.zip.sha256` verifies the complete ZIP including the checksum file.
2. The ZIP extracts into a fresh path, including a path containing spaces, without escaping the destination.
3. No prohibited model/data/output/cache extension, nested `.git`, secret-bearing file, or unintended file larger than the configured lightweight threshold is present.
4. No package control file, experiment descriptor, or public launcher contains a host-specific Windows drive or `/root/autodl-tmp` runtime default. Any such path retained in source provenance or historical evidence is outside the supported control plane and is clearly classified as historical.
5. `bash reproduce.sh list` and `bash reproduce.sh explain <id>` work without the research environment.
6. Global `preflight` distinguishes package integrity from missing external artifacts and exits with actionable diagnostics.
7. A historical failed experiment is blocked without explicit opt-in.
8. A deliberately corrupted fixture is rejected by artifact verification.
9. The current and clean source manifests, outer diff, untracked manifest, BrepARG provenance, and experiment coverage inventory are present and internally consistent, with no unclassified historical experiment candidate.
10. The full postmortem, timeline, experiment ledger, failure incidents, and evidence index are present and UTF-8 readable.
11. The archive remains lightweight: no heavy artifacts are included and the final size is reported. A size above 100 MiB requires an explicit build failure and review rather than silent publication.
12. Shell syntax checks and package-level smoke tests pass locally. Linux GPU environment creation, CUDA smoke, OCC STEP smoke, and at least one real experiment smoke are listed as required target-server acceptance steps when they cannot be executed on the Windows build host.

## 16. Entry Documentation

`START_HERE.md` is written for a researcher with no conversation history. It starts with the project's current scientific conclusion and the difference between runnable source, clean reference source, and external artifacts. It provides three short paths:

1. Read the project history and current root-cause conclusions.
2. Configure and verify external artifacts on a Linux GPU server.
3. Launch the recommended diagnostic, baseline, or training workflow.

It also states the limitations that must not be hidden: current generated CAD quality remains inadequate, the official BrepARG metric reproduction is not yet established, parent-CAD split leakage affected historical validation interpretation, and longer training alone did not resolve the dominant reconstruction/assembly failures.

## 17. Implementation Boundaries

The implementation should reuse existing project tools where they are sound, especially the server package, transfer manifest, transfer verification, recovery packet, and training-readiness utilities. It should not copy the stale `dist/v13_server_ready_20260710.zip` as the new package because that archive omits the required history/provenance layers and contains obsolete path assumptions.

New package-building and launcher code should use Python standard-library structured parsing rather than ad hoc text substitution. Existing experiment commands may be wrapped, but their scientific parameters must remain explicit in descriptors. Unrelated training refactors, model redesign, data migration, and workspace cleanup are outside this implementation.

## 18. Approved Decisions

The user approved the following decisions during design review:

- lightweight source package rather than a data-inclusive archive;
- Linux GPU/AutoDL as the only supported runtime target;
- inclusion of all historical experiments, categorized as recommended, baselines, diagnostics, or historical failed;
- current dirty working-tree snapshot as the default runnable source;
- clean outer Git HEAD `16cf19b` as a separate reference snapshot;
- nested BrepARG provenance at commit `07970a4` without nested Git metadata;
- Conda plus pip dual environment locking;
- one unified `reproduce.sh` command surface;
- complete full-chain postmortem, timeline, experiment ledger, plans, decisions, failure incidents, original textual records, data/protocol records, and evidence index;
- external large artifacts represented by specifications, hashes, sizes, expected paths, and verification rules rather than bundled content;
- explicit historical-failure opt-in and strict failure propagation;
- no destructive cleanup as part of packaging.

## 19. Completion Definition

Implementation is complete when the deterministic ZIP and checksum exist, all package-level automated acceptance checks pass, the build report states exactly what was included and excluded, the archive can be unpacked and navigated without the original conversation, and a Linux GPU operator has unambiguous commands for environment setup, artifact verification, experiment inspection, smoke testing, execution, monitoring, and result verification.
