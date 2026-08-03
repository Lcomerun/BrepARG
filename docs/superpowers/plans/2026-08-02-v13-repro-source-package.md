# Build the V13 Lightweight Reproducibility Source Package

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a lightweight deterministic ZIP that lets a Linux GPU researcher understand the full V13/BrepARG project history, verify external artifacts, and launch supported experiments without relying on the original machine paths.

**Architecture:** The repository gains a static `reproducibility/` control layer and a Python standard-library package builder. The builder assembles a filtered current source snapshot, exports clean Git commit `16cf19b`, records outer and nested BrepARG provenance, normalizes experiment and artifact catalogs, curates historical textual evidence, creates deterministic manifests and ZIP metadata, and verifies a fresh extraction. The archive's `reproduce.sh` delegates to a package-contained Python launcher that performs package checks, artifact checks, guarded execution, status reporting, and output verification.

**Tech Stack:** Python 3 standard library, Bash, PowerShell only for the host invocation, JSON, SHA-256, ZIP, pytest for repository tests, Conda and pip metadata for the target Linux environment.

This ExecPlan is a living document and must be maintained according to the repository-root `PLANS.md`. The approved design is `docs/superpowers/specs/2026-08-02-v13-repro-source-package-design.md`. The current working tree is intentionally dirty because its complete state is the default runnable source to package. Implementation therefore stays in the current tree and only adds isolated packaging files; a Git worktree would omit the uncommitted source state that the package is required to preserve.

## Purpose / Big Picture

After this work, the user receives `dist/v13_repro_source_20260802.zip`, `dist/v13_repro_source_20260802.zip.sha256`, and a host build-execution report. Unpacking the ZIP on Linux exposes `START_HERE.md` and a single `reproduce.sh` interface. Without installing the ML environment, a researcher can verify package integrity, list experiments, inspect their purpose and evidence, and learn which external assets are missing. After configuring paths and creating the pinned environment, the same interface can smoke-test, run, monitor, and verify supported diagnostics or training workflows. Historical failures remain inspectable but cannot run without explicit opt-in.

The package is a scientific handoff, not a claim that current generation quality is satisfactory. It explicitly preserves the established findings: reconstruction and BRep assembly are the strongest upstream bottlenecks, AR adds independent failures, DFS has a modest teacher-forcing advantage over RCM, parent-CAD leakage weakened historical validation claims, and official BrepARG metric reproduction remains incomplete.

## Progress

- [x] (2026-08-02 09:25Z) Approved and committed the full package design as commit `26cc2e1`.
- [x] (2026-08-02 09:45Z) Audited repository shape, current and nested Git identities, historical text sources, key local artifacts, and dependency-file gaps.
- [x] (2026-08-02 10:05Z) Wrote runtime-launcher tests first and observed the expected `ModuleNotFoundError: No module named 'reproducibility'` collection failure.
- [x] (2026-08-02 10:15Z) Implemented the package-contained runtime launcher; `python -m pytest tests/test_repro_runtime.py -q` passes 9 tests.
- [x] (2026-08-02 10:25Z) Wrote package-builder tests first and observed the expected `ModuleNotFoundError: No module named 'tools.repro_package_builder'` collection failure.
- [x] (2026-08-02 10:40Z) Implemented deterministic JSON/ZIP, source filtering, history evidence summaries, clean commit export, checksum creation, and stage validation; `python -m pytest tests/test_repro_package_builder.py -q` passes 19 tests.
- [x] (2026-08-02 12:40Z) Added 29 experiment descriptors, 22 external-artifact contracts, the pinned Linux GPU environment, bootstrap and OCC probes, entry documentation, conclusions, timeline, machine-readable ledger, failure incidents, data protocol, and evidence index.
- [x] (2026-08-02 13:25Z) Repaired real-build portability failures with test-first changes: Windows history-path compaction, control-plane host-path scanning without embedding a legacy path, and UTF-8 `SHA256SUMS` support.
- [x] (2026-08-02 13:37Z) Ran the complete packaging-focused suite; 57 tests passed. Catalog validation reported 29 experiments, 22 artifacts, 20 referenced artifacts, and 11 history-coverage rules.
- [x] (2026-08-02 13:43Z) Built the real source package from the dirty working-tree snapshot plus `D:\V13_rootcause_recovery_20260717`; the payload contains 1,260 files, 851 history records, and no heavy model, data, CAD, image, PDF, cache, or nested Git artifacts.
- [x] (2026-08-02 13:46Z) Verified a fresh extraction in a path containing spaces: all 1,259 internal checksums, five package-contained tests, Python compilation, Bash syntax, `list`, `explain`, missing-asset preflight, and the historical-failure opt-in guard behaved as designed.
- [x] (2026-08-02 13:47Z) Rebuilt into a second, longer output path with the same `SOURCE_DATE_EPOCH`; independent byte comparison confirmed the two ZIP files were identical. The published archive identity is recorded externally in `dist/v13_repro_source_20260802.zip.sha256` and `dist/v13_repro_source_20260802.build-execution.json`.

## Surprises & Discoveries

- Observation: The outer repository's current HEAD is now `26cc2e1`, while the design requires clean reference commit `16cf19b` and the current dirty tree as separate source layers.
  Evidence: `git rev-parse HEAD` returned `26cc2e1a182088989b79cb787dde8437aa690c30`; `git log` shows `16cf19b` immediately before the design commit.

- Observation: The nested BrepARG repository identifies commit `07970a4`, has five modified Python files, one untracked PDF, and corrupted AppleDouble pack-index noise.
  Evidence: `git -C BrepARG status --short` reported modifications to `2sequence.py`, `generate_brep.py`, `train_vqvae.py`, `trainer.py`, and `utils.py`; Git also reported `non-monotonic index .git/objects/pack/._pack-...idx`. The builder must avoid depending on a full nested Git object traversal and must exclude `._*`, the PDF, caches, and nested `.git`.

- Observation: Three primary V13 artifacts are present locally, while parsed archives occupy about 174.4 GB and cannot be hashed naively as part of every package build.
  Evidence: the selected VQ checkpoint is 228,776,414 bytes, AR checkpoint is 113,859,999 bytes, sequence package is 1,507,004,496 bytes, and 101 parsed archive files total 174,374,900,417 bytes. File artifacts can receive exact SHA-256; the archive directory should use an existing manifest when available or a deterministic name/size inventory with an explicit checksum-strength label.

- Observation: Historical textual evidence is broader than the tracked repository.
  Evidence: `local_runs` contains 524 JSON/JSONL/Markdown/log/text files and `D:\V13_rootcause_recovery_20260717` contains 144 such files. The builder must include small evidence directly and summarize large logs without pulling binary outputs into the ZIP.

- Observation: The only top-level environment file is intentionally incomplete and stale.
  Evidence: `environment.server.yml` exists, while no pip lock or project metadata file is present. The new package needs its own explicit Linux GPU reference lock and must label target-server resolution checks honestly.

- Observation: Windows' traditional 260-character path limit affected historical evidence staging even though the final ZIP format supports long paths.
  Evidence: the first real build failed at an absolute path of exactly 260 characters, and a deterministic rebuild under a longer output root exposed a second 262-character path. History targets are now compacted below a package-relative threshold and stored under `project_history/_compacted/`, while `history_inventory.json` preserves each full original path.

- Observation: A checksum manifest whose digest fields are ASCII can still require UTF-8 because relative file names are part of each line.
  Evidence: the real project contains Chinese file names; writing `SHA256SUMS` as ASCII raised `UnicodeEncodeError`. Builder and runtime regression tests now verify UTF-8 file-name round trips without changing the conventional `hash  path` format.

- Observation: Package integrity and target-machine readiness are independent states.
  Evidence: fresh-extraction preflight reported `package_integrity: ok` over 1,259 files while returning exit code 2 and `target_ready: false` because `paths.env`, PyTorch, CUDA runtime imports, and OCC were not configured in the Windows build environment.

## Decision Log

- Decision: Keep implementation in the current dirty working tree and isolate only the build staging area.
  Rationale: The current dirty tree is itself the approved default source snapshot. A clean worktree would package the wrong source state. New implementation files are kept separate and existing user changes are not reverted.
  Date/Author: 2026-08-02 / Codex

- Decision: Use Python standard-library modules for both the builder and runtime control plane.
  Rationale: Package inspection and missing-artifact diagnostics must work before Conda/PyTorch installation. Structured JSON parsing is safer than shell text substitution and remains portable across Linux and the Windows build host.
  Date/Author: 2026-08-02 / Codex

- Decision: Represent experiment coverage through a catalog plus a generated source inventory, not by copying every historical command as executable shell.
  Rationale: Historical records have different evidence quality and several are known unsafe. A descriptor can be runnable, documentary, or blocked while still preserving the record and its evidence mapping.
  Date/Author: 2026-08-02 / Codex

- Decision: Hash the three present primary file artifacts exactly and allow directory artifacts to declare `manifest_sha256` with a verification-strength field.
  Rationale: Re-reading 174 GB during each build is disproportionate. Scientific honesty requires distinguishing full content hashing from deterministic inventory hashing rather than fabricating equivalence.
  Date/Author: 2026-08-02 / Codex

- Decision: Use a 256 KiB direct-copy threshold for historical text and produce hash/excerpt evidence records for larger text files. Individual explicitly important summaries may be allowlisted regardless of location.
  Rationale: This retains detailed project history while enforcing the 100 MiB source-package ceiling.
  Date/Author: 2026-08-02 / Codex

- Decision: Pin the Linux reference environment to Python 3.11.13, PyTorch 2.8.0+cu128, Transformers 4.57.3, Diffusers 0.35.1, and NumPy 2.2.6, with `pythonocc-core` and `occwl` installed and probed separately.
  Rationale: The package must define a reproducible target while keeping optional CAD dependencies diagnosable. The stale pip package named `OCC` is explicitly excluded because it is not the required OpenCascade binding.
  Date/Author: 2026-08-02 / Codex

- Decision: Store internal `SHA256SUMS` as UTF-8 and keep the external ZIP checksum sidecar ASCII.
  Rationale: UTF-8 is required for real source file names, while the release ZIP name itself is ASCII. Both use the standard two-space delimiter and are verified independently.
  Date/Author: 2026-08-02 / Codex

- Decision: Keep the final ZIP SHA-256 outside the ZIP in `.zip.sha256` and the host build-execution report.
  Rationale: Embedding an archive's own digest in its payload changes the archive and makes a fixed-point identity impossible. The package instead contains complete per-file checksums, while the sidecar authenticates the ZIP as a whole.
  Date/Author: 2026-08-02 / Codex

## Outcomes & Retrospective

The lightweight reproducibility handoff is implemented and independently exercised. The package exposes one public `reproduce.sh` interface, 29 normalized experiment records, 22 external-artifact contracts, current and clean-reference source layers, pinned environment setup, and 851 curated historical records. It preserves the complete scientific postmortem and does not imply that current V13 or BrepARG generation quality is satisfactory.

The packaging-focused repository suite passed 57 tests. A fresh extraction passed all five package-contained tests, verified 1,259 payload checksums, compiled the Python control and source layers, parsed the Bash entry point, listed all experiments, explained the primary complex-curved diagnostic, rejected a historical failed experiment without explicit opt-in, and reported missing external paths without crashing. An independent ZIP scan found no forbidden heavy extensions, nested `.git`, caches, secret-shaped files, unsafe paths, duplicate entries, or files above 16 MiB. Two builds under different output-root lengths and the same release epoch were byte-identical.

The remaining acceptance work is intentionally target-specific rather than a packaging defect. On a Linux x86_64 NVIDIA host, a researcher must create the Conda environment, confirm CUDA and the real OCC STEP round trip, configure paths to the excluded artifacts, and run at least one real smoke experiment. Those checks were not claimed on the Windows build host. The final whole-archive size and SHA-256 are authoritative in `dist/v13_repro_source_20260802.zip.sha256` and the adjacent build-execution report; internal payload identity is authoritative in `SHA256SUMS` inside the archive.

## Context and Orientation

The project root is `D:\luolin\V13`. V13 training and generation code primarily lives in `breparg_improvements/`; upstream and locally modified BrepARG code lives in the nested `BrepARG/` checkout. Utility scripts live in `tools/`, tests in `tests/`, historical execution plans in `plans/`, audit reports in `docs/` and `local_reports/`, and generated run evidence in ignored `local_runs/`. A separate recovery evidence tree is present at `D:\V13_rootcause_recovery_20260717`.

An “external artifact” means a large input or trained output intentionally excluded from the ZIP, such as a `.pt` checkpoint, `.pkl` sequence package, parsed `.7z` archive, or shard directory. An “artifact contract” is a small JSON file that describes the expected identity, size, layout, compatibility, and configured runtime path for one such artifact. A “descriptor” is a JSON experiment definition consumed by the launcher. A “documentary” experiment is visible for research audit but cannot run because a trustworthy command or required evidence is incomplete.

The three locally available selected artifacts are under `ABC/processed/train_outputs/ubuntu/`. The package builder may read these files to calculate hashes but never copies them. The parsed archives are under `ABC/processed/abc_parsed_full_archives/`. No operation in this plan deletes or moves those files.

The current full postmortem is `docs/full_experiment_postmortem_20260731.md`. Existing plans and reports already contain the core timeline and experimental findings; the package adds a normalized ledger and evidence index so a new reader does not have to infer chronology from file names.

## File Structure

The implementation creates the following focused files.

`reproducibility/reproduce.sh` is the public Bash wrapper copied to the package root.

`reproducibility/launchers/repro_cli.py` parses public commands and renders concise output. `reproducibility/launchers/repro_runtime.py` owns descriptor loading, path parsing, package checksums, artifact validation, run-directory creation, guarded subprocess execution, status, and verification.

`reproducibility/launchers/bootstrap.sh` creates or updates the Linux Conda environment and runs import probes.

`reproducibility/catalog/experiments.json` is the normalized source catalog for all public descriptors. `reproducibility/catalog/artifacts.json` declares external artifacts and their build-host candidate paths. The builder expands these catalogs into category directories and individual artifact contracts inside the archive.

`reproducibility/configs/paths.env.example`, `reproducibility/environments/environment.linux-gpu.yml`, `reproducibility/environments/requirements.linux-gpu.lock.txt`, and environment verification scripts define the target server setup.

`reproducibility/docs/START_HERE.md` and files under `reproducibility/project_history/` provide the curated narrative, timeline, ledger, failure incidents, data/evaluation protocol, and evidence-reading guide.

`tools/repro_package_builder.py` contains deterministic hashing, source filtering, clean-tree export, provenance capture, history curation, catalog expansion, ZIP writing, and validation functions. `tools/build_repro_source_package.py` is the host CLI.

`tests/test_repro_runtime.py` and `tests/test_repro_package_builder.py` verify runtime and builder behavior with small fixtures. `tests/test_repro_package_integration.py` validates the real staged archive without using model/data contents.

## Plan of Work

### Milestone 1: Runtime launcher with explicit safety behavior

Write `tests/test_repro_runtime.py` before runtime code. Tests create a tiny package fixture with one recommended runnable descriptor, one documentary descriptor, one historical failed descriptor, one artifact contract, a paths file, and `SHA256SUMS`. They assert that catalog loading rejects duplicate IDs, path parsing does not execute shell syntax, checksum corruption fails, artifact mismatch is actionable, historical runs require opt-in, run directories do not overwrite, and verification rejects nonfinite JSON metrics.

Run the focused tests and confirm collection fails because `reproducibility.launchers.repro_runtime` does not exist. Then implement `repro_runtime.py`, `repro_cli.py`, and `reproduce.sh` minimally until those tests pass. The subprocess interface accepts only command arrays from validated descriptors; it expands `${NAME}` placeholders from a controlled mapping and never invokes `shell=True`.

Acceptance for this milestone is a passing focused test file plus direct CLI output for `list`, `explain`, and historical-failure blocking in the fixture package.

### Milestone 2: Deterministic package builder

Write `tests/test_repro_package_builder.py` before builder code. Tests cover stable JSON, stable ZIP timestamps/order, forbidden extensions, nested `.git`, secret-shaped names, package-control absolute paths, source filtering, clean-export behavior through a fixture repository, history direct-copy versus excerpt records, and checksum generation that excludes `SHA256SUMS` itself.

Run the focused tests and confirm they fail because `tools.repro_package_builder` does not exist. Implement the builder library and CLI. Build staging always occurs below an explicit output directory, and recursive removal is permitted only after resolving and proving the staging path is inside that output directory and has the exact expected package-stage name. The final implementation uses temporary sibling files and atomic rename for ZIP publication.

Acceptance for this milestone is two byte-identical ZIP files built from the same fixture and all focused builder tests passing.

### Milestone 3: Catalogs, environment, and project history

Add the experiment and artifact catalogs plus entry documentation and environment files. Catalog validation is run after each addition. Experiment descriptors must identify commands and required artifacts exactly; experiments lacking enough evidence are marked documentary rather than guessed into runnable form.

The history layer includes the full postmortem, a chronological summary, a normalized experiment ledger, plans and decisions, original report records, failure incidents, data/evaluation protocol, and an evidence index. The builder scans plans, local reports, local run roots, and the recovery tree to generate coverage inventory. Every candidate maps to a descriptor or the explicit classification `non_experiment_context`; unclassified candidates fail the build.

Acceptance for this milestone is a catalog report with unique IDs, all referenced artifact IDs present, all evidence references resolvable or explicitly marked externally unavailable, no unknown coverage item, and UTF-8 decoding of every package-level Markdown/JSON file.

### Milestone 4: Real build and archive acceptance

Compute exact hashes for the selected VQ, AR, and sequence files while keeping them outside the stage. Build `dist/v13_repro_source_20260802.zip`, verify it against its external checksum, extract it to a fresh path containing spaces, verify internal `SHA256SUMS`, and execute package-level `list`, `explain`, `preflight`, and historical guard checks. Run shell syntax checking when Bash is available and always compile package Python sources.

Scan the extraction for forbidden heavy extensions, nested `.git`, caches, secrets, path traversal, files above the allowed threshold, and host paths in package control files. Build the same package a second time with the same release epoch and compare SHA-256 values. Remove only the validated temporary extraction and second-build staging after evidence is captured; never remove existing project outputs.

Acceptance for this milestone is a ZIP below 100 MiB, an exact checksum, a build report covering every design acceptance rule, passing package tests, and an honest target-server checklist for Conda creation, CUDA, OCC, and one real experiment smoke that Windows cannot execute.

## Concrete Steps

All commands run from `D:\luolin\V13` unless noted.

1. Create runtime tests with `apply_patch`, then run:

       python -m pytest tests/test_repro_runtime.py -q

   Expected initial result: failure during import because the runtime module has not yet been created.

2. Implement runtime files with `apply_patch`, then rerun the same command. Expected final result: all runtime tests pass.

3. Create builder tests with `apply_patch`, then run:

       python -m pytest tests/test_repro_package_builder.py -q

   Expected initial result: failure during import because the builder module has not yet been created.

4. Implement builder files with `apply_patch`, then rerun focused runtime and builder tests. Expected result: all focused tests pass.

5. Add catalogs and templates, then run:

       python tools/build_repro_source_package.py --validate-only

   Expected result: a JSON or text summary reporting unique experiment IDs, complete artifact references, zero unclassified coverage items, and no package-control path violations.

6. Run the relevant suite:

       python -m pytest tests/test_repro_runtime.py tests/test_repro_package_builder.py tests/test_repro_package_integration.py -q

   Expected result: all tests pass with no warnings that indicate skipped package behavior.

7. Build the package:

       python tools/build_repro_source_package.py --output-dir dist --package-name v13_repro_source_20260802.zip

   Expected result: the ZIP, `.sha256`, and `.build-execution.json` paths are printed, with final archive size below 100 MiB.

8. Run independent verification from a fresh extraction. On Windows, use the builder's `--verify-archive` command and direct Python launcher. If `bash` is available, additionally run:

       bash -n <extracted>/v13_repro_source_20260802/reproduce.sh
       bash <extracted>/v13_repro_source_20260802/reproduce.sh list
       bash <extracted>/v13_repro_source_20260802/reproduce.sh preflight

   `preflight` may return a nonzero readiness code because the build host is Windows and external paths are intentionally not configured, but output must report package integrity separately and explain every missing target requirement.

9. Rebuild to a temporary output directory using the same `SOURCE_DATE_EPOCH`, compare ZIP hashes, and record the result in the ExecPlan and build report.

## Validation and Acceptance

Validation is layered. Unit tests prove parser, guard, hash, and deterministic-archive behavior. Integration tests prove a fixture archive can be built, extracted, and inspected. Real-build validation proves that the actual dirty source tree and historical evidence conform to inclusion rules. Fresh-extraction commands prove the final object works independently of the repository working directory.

The final acceptance report must state counts for current-source files, clean-reference files, experiments by category and state, artifact contracts by verification strength, copied historical records, excerpted historical records, missing evidence records, and total payload bytes. It must list the three primary external artifact hashes without copying their contents. It must show that the known nonfinite AR latest checkpoint is not selected as the recommended AR artifact.

No completion claim is valid without fresh outputs for the full focused test command, real package build, archive checksum, fresh-extraction package verification, forbidden-content scan, and deterministic rebuild comparison.

## Idempotence and Recovery

The builder is idempotent for identical inputs and `SOURCE_DATE_EPOCH`. It writes into an exact temporary stage below the requested output directory. Before removing a prior temporary stage, it resolves the path, verifies that its parent is the intended output directory, and checks the expected stage basename. It writes final ZIP and metadata through temporary sibling files and atomically replaces only files with the requested final package names.

If hashing a large external file is interrupted, rerunning the build recomputes it without changing the artifact. If catalog validation fails, no final ZIP is published. If fresh extraction fails, the failed temporary extraction is retained until its path and error are written to the host build-execution report. Existing `dist/v13_server_ready_20260710.zip` is not deleted or overwritten because the new archive has a different name.

Run execution is similarly additive. Each experiment gets a new timestamped run directory. Resume requires an explicit checkpoint and compatibility checks. Failed runs remain available for inspection; verification writes a result record rather than modifying original outputs.

## Artifacts and Notes

Approved package root:

    v13_repro_source_20260802/
    |-- START_HERE.md
    |-- reproduce.sh
    |-- PACKAGE_MANIFEST.json
    |-- SHA256SUMS
    |-- source/current/
    |-- source/clean_head_16cf19b/
    |-- provenance/
    |-- experiments/{recommended,baselines,diagnostics,historical_failed}/
    |-- configs/
    |-- environments/
    |-- launchers/
    |-- artifact_specs/
    |-- reports/
    |-- project_history/
    |-- tests/
    `-- BUILD_REPORT.md

Known primary external artifact sizes at plan creation:

    fsq_vqvae_best.pt       228,776,414 bytes
    ar_best.pt              113,859,999 bytes
    sequences_fsq_rcm.pkl 1,507,004,496 bytes
    parsed archive root   174,374,900,417 bytes across 101 files

The nested BrepARG `.git` corruption warning is provenance evidence, not a reason to copy Git metadata. The builder should capture working-tree files directly and use `git diff --no-ext-diff --binary` when it succeeds; if nested diff fails, it must create a manifest and explicit failure record rather than aborting the whole source package.

## Interfaces and Dependencies

In `reproducibility/launchers/repro_runtime.py`, define stable public functions used by tests and the CLI:

    load_paths(path: pathlib.Path) -> dict[str, str]
    load_experiments(package_root: pathlib.Path) -> dict[str, dict]
    load_artifact_specs(package_root: pathlib.Path) -> dict[str, dict]
    verify_package_checksums(package_root: pathlib.Path) -> list[dict]
    verify_artifact(spec: dict, paths: dict[str, str]) -> dict
    create_run_context(package_root: pathlib.Path, experiment: dict, paths: dict[str, str], now: datetime | None = None) -> dict
    run_experiment(package_root: pathlib.Path, experiment_id: str, allow_historical_failed: bool = False, smoke: bool = False) -> int
    verify_run(run_dir: pathlib.Path, experiment: dict) -> dict

The launcher returns zero only for completed commands. Readiness and user-input failures use distinct nonzero codes and structured messages. Descriptors contain command argument arrays, never executable shell strings.

In `tools/repro_package_builder.py`, define:

    sha256_file(path: pathlib.Path) -> str
    stable_json_bytes(value: object) -> bytes
    write_checksum_manifest(root: pathlib.Path) -> pathlib.Path
    write_deterministic_zip(source_root: pathlib.Path, zip_path: pathlib.Path, epoch: int) -> str
    validate_stage(stage_root: pathlib.Path) -> dict
    build_package(repo_root: pathlib.Path, output_dir: pathlib.Path, package_name: str, epoch: int) -> dict

The builder depends only on Python's standard library and the local `git` command. PyTorch is not imported by the builder. Checkpoint finite-tensor inspection is deferred to the target environment or to existing project verification tools; build-time contracts still record exact checkpoint hashes and known metadata evidence.

Revision note (2026-08-02): Initial self-contained implementation plan written after approval of the package design and direct user authorization to build the archive.

Revision note (2026-08-02 13:48Z): Updated the living plan after implementation and independent acceptance. This revision records the completed catalogs, environment, project-history layer, real-build portability fixes, test counts, safety scans, deterministic rebuild, scientific scope, and the Linux target checks that remain external to the Windows build host.
