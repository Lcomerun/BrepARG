# Reinitialize a source-and-summary-only Git repository

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This document is maintained in accordance with `PLANS.md` at the repository root.

## Purpose / Big Picture

The workspace currently mixes approximately 195 GiB of datasets, checkpoints, generated CAD, logs, upstream BrepARG source, and papers with the V13 project code. After this work, a fresh local Git repository will contain only the V13-owned source code, tests, reproducibility controls, portable configuration templates, project documentation, and lightweight experiment summaries. The old Git metadata will remain recoverable in a timestamped ignored backup until the user confirms the new repository and GitHub upload.

The observable result is that `git status --short` lists only explicitly approved source-and-summary paths, `git check-ignore` proves that `ABC/`, `BrepARG/`, `papers/`, `local_runs/`, model weights, datasets, CAD outputs, and the old Git backup are excluded, and the staged file set contains no prohibited extension or file larger than 5 MiB.

## Progress

- [x] (2026-08-03 Asia/Shanghai) Audited the existing repository, nested BrepARG repository, workspace sizes, tracked files, remotes, secret-shaped values, and Git history blob sizes.
- [x] (2026-08-03 Asia/Shanghai) Confirmed with the user that `BrepARG/` and `papers/` are wholly out of scope.
- [x] (2026-08-03 Asia/Shanghai) Replaced `.gitignore` with the approved source-and-summary boundary.
- [x] (2026-08-03 Asia/Shanghai) Created `local_training_config.example.json` and ignored the unchanged machine-local configuration.
- [x] (2026-08-03 Asia/Shanghai) Created lightweight experiment-summary navigation in `reports/` without copying raw outputs.
- [x] (2026-08-03 Asia/Shanghai) Validated JSON, report-index targets, ignored roots and formats, and eligible source paths before changing Git metadata.
- [x] (2026-08-03 Asia/Shanghai) Moved the old `.git` contents to a recoverable ignored backup and initialized a fresh `main` repository.
- [x] (2026-08-03 Asia/Shanghai) Staged only explicit allowlisted paths; the first complete audit found 211 files totaling 2.58 MiB with no prohibited root, extension, or file above 5 MiB.
- [x] (2026-08-03 Asia/Shanghai) Ran focused source-tree tests, compilation, JSON parsing, and staged-text credential-pattern scanning.
- [ ] Present the final staged inventory for user approval before the first commit or push.

## Surprises & Discoveries

- Observation: The workspace was already a local Git repository at commit `648d113`, but it had no configured remote, no upstream branch, and no remote-tracking references.
  Evidence: `git remote -v`, `git config --local --get-regexp '^remote\.'`, and `git for-each-ref refs/remotes` produced no entries.

- Observation: The old Git history contains no blob larger than 5 MiB, so the old metadata does not require Git LFS migration or history filtering before being archived.
  Evidence: A `git rev-list --objects --all` and `git cat-file --batch-check` size scan returned no blob at or above 5 MiB.

- Observation: `ABC/` is approximately 171 GiB and `local_runs/` is approximately 24 GiB, while all currently tracked files are smaller than 1 MiB.
  Evidence: Recursive PowerShell size inventory and `git ls-files` file-size scan.

- Observation: `BrepARG/` is a nested Git repository with five locally modified upstream files, but the user confirmed those changes only supported the local comparison and are not part of this repository.
  Evidence: `git -C BrepARG status --short` and the user's scope decision.

- Observation: Windows `Move-Item` transferred all 494 old Git metadata files to the backup but could not remove the now-empty hidden `.git` source directory.
  Evidence: `.git.backup-20260803` contained 494 files and 1,346,031 bytes including `HEAD`, `config`, `index`, objects, and refs; the residual `.git` contained zero entries. The empty shell was renamed to `.git.backup-20260803-empty-shell`, avoiding deletion.

- Observation: The package-control tests are designed for a built reproducibility package, not the source-tree `reproducibility/` directory.
  Evidence: A source-tree run produced 16 passes and 4 expected failures because generated `PACKAGE_MANIFEST.json`, `SHA256SUMS`, and `experiments/` were absent. The source-tree runtime, builder, integration, template, and build tests passed 55/55.

## Decision Log

- Decision: Reinitialize rather than preserve the existing outer Git history.
  Rationale: The local repository has no evidence of an active GitHub relationship, and the user explicitly requested a complete reinitialization when upload status cannot be confirmed.
  Date/Author: 2026-08-03 / Codex and user.

- Decision: Preserve the old `.git` directory as `.git.backup-20260803` rather than delete it.
  Rationale: Moving is recoverable and protects the existing commits while allowing a clean root commit. The backup is ignored by the new repository.
  Date/Author: 2026-08-03 / Codex.

- Decision: Exclude `BrepARG/` and `papers/` as complete directory trees without analyzing or migrating their contents.
  Rationale: The user explicitly stated that BrepARG changes were local comparison adaptations and that paper writing has not begun.
  Date/Author: 2026-08-03 / User.

- Decision: Commit experiment summaries, aggregate metrics, artifact manifests, and experiment history, but exclude raw run directories and logs.
  Rationale: This gives other devices enough evidence to understand results without cloning datasets, weights, generated CAD, or machine-specific operational state.
  Date/Author: 2026-08-03 / Codex and user.

- Decision: Stop before the first commit and remote push.
  Rationale: The previously approved workflow requires the user to inspect the final staged file list and size before creating the new root commit.
  Date/Author: 2026-08-03 / Codex.

## Outcomes & Retrospective

The workspace now has a fresh, uncommitted `main` repository and a readable old-history backup at `.git.backup-20260803`, whose old tip remains `648d113`. The approved source-and-summary allowlist staged 211 files totaling 2.58 MiB before this plan's final progress update. No staged path came from an excluded root; no staged file had a prohibited data, model, archive, CAD, log, or PDF extension; and no file exceeded 5 MiB.

The source-tree validation passed 55 tests, compiled all Python under `breparg_improvements/`, `tools/`, and `reproducibility/`, parsed the key JSON controls, and found no common credential format in 209 staged text files. Four package-control assertions remain intentionally runnable only against a generated package, because their required manifest, checksum file, and experiment descriptors are build outputs excluded from this source repository. No initial commit, remote configuration, or push has occurred.

## Context and Orientation

The repository root is `D:\luolin\V13`. V13-owned implementation lives primarily in `breparg_improvements/`; operational, evaluation, audit, and packaging programs live in `tools/`; automated tests live in `tests/`; and portable environment, catalog, experiment history, and launcher controls live in `reproducibility/`. Technical documentation is in `docs/`, and prior execution plans are in `plans/`.

The directories `ABC/` and `local_runs/` contain nearly all of the workspace's large data and model artifacts. `BrepARG/` is an upstream checkout with its own `.git`, and `papers/` is future paper work. These four directory trees must not enter the new repository. `local_reports/` and `breparg_improvements/repro_outputs/` contain machine-local reports and raw run outputs; durable conclusions already exist in `docs/full_experiment_postmortem_20260731.md` and `reproducibility/reports/current_conclusions.md`, so raw report directories remain excluded.

A Git metadata backup is the old `.git` directory moved to `.git.backup-20260803`. It is not a copy of the 195 GiB workspace; it is approximately 1.3 MiB of commits and indexes. Restoring it means removing the new `.git` directory and moving `.git.backup-20260803` back to `.git` after first verifying both exact paths.

## Plan of Work

First, replace the root `.gitignore` with rules that exclude complete out-of-scope roots, machine-local configuration, datasets, model formats, archives, raw run outputs, logs, CAD outputs, caches, build products, and old Git backups. The rules keep ordinary source, Markdown, JSON, JSONL, CSV, YAML, shell scripts, and PowerShell scripts eligible for explicit staging.

Second, create `local_training_config.example.json` from the shape of `local_training_config.json`, replacing machine drive paths with descriptive placeholders. Leave `local_training_config.json` untouched on disk and ignore it. Create `reports/README.md` and `reports/experiment_index.json` as lightweight navigation into the already-curated conclusion, audit, and experiment-ledger documents; do not copy large artifacts.

Third, validate the intended repository surface before changing Git metadata. Check that every required path exists, that the JSON files parse, and that prohibited roots are matched by `.gitignore`. Then resolve the exact `.git` and backup paths, verify the backup target does not exist, move `.git` to `.git.backup-20260803`, and run `git init` followed by `git branch -M main`.

Fourth, stage only `.gitignore`, root project documentation, `breparg_improvements/`, `tools/`, `tests/`, `reproducibility/`, `docs/`, `plans/`, `reports/`, the example configuration, and the server environment definition. Do not use `git add .` or `git add -A`. Inspect every staged path, reject prohibited extensions, reject any file over 5 MiB, confirm that neither `BrepARG/` nor `papers/` is staged, and calculate the total staged size.

Finally, run the repository's lightweight tests and Python compilation checks that do not require datasets, checkpoints, OCC, or a GPU. Update this plan with exact results and present the staged inventory to the user. Do not create the initial commit and do not configure or push a GitHub remote until the user approves the inventory.

## Concrete Steps

All commands run from `D:\luolin\V13` in PowerShell.

After the configuration and summary files are written, validate JSON syntax with:

    Get-Content -Raw local_training_config.example.json | ConvertFrom-Json | Out-Null
    Get-Content -Raw reports\experiment_index.json | ConvertFrom-Json | Out-Null

Verify ignored roots and formats with representative paths:

    git check-ignore -v ABC\processed\sample.pkl
    git check-ignore -v BRepARG\generate_brep.py
    git check-ignore -v papers\draft.tex
    git check-ignore -v local_runs\run\model.pt
    git check-ignore -v local_reports\summary.json
    git check-ignore -v .git.backup-20260803\config

Before moving Git metadata, resolve and compare the source and target:

    $repoRoot = (Resolve-Path -LiteralPath .).Path
    $oldGit = (Resolve-Path -LiteralPath .git).Path
    $backupGit = Join-Path $repoRoot '.git.backup-20260803'
    if ($oldGit -ne (Join-Path $repoRoot '.git')) { throw 'Unexpected Git path' }
    if (Test-Path -LiteralPath $backupGit) { throw 'Backup target already exists' }
    Move-Item -LiteralPath $oldGit -Destination $backupGit
    git init
    git branch -M main

Stage the explicit allowlist:

    git add -- .gitignore AGENTS.md PLANS.md README.md PROJECT_INDEX.md environment.server.yml local_training_config.example.json
    git add -- breparg_improvements tools tests reproducibility docs plans reports

Audit the staged set using `git diff --cached --name-only`, reject files with data/model/CAD/log/PDF extensions, reject files larger than 5 MiB, and ensure staged paths do not begin with `ABC/`, `BrepARG/`, `papers/`, `local_runs/`, `local_reports/`, `processed_local/`, or `.git.backup-`.

Run focused validation:

    python -m pytest reproducibility\tests tests\test_repro_runtime.py tests\test_repro_package_control.py -q
    python -m compileall -q breparg_improvements tools reproducibility

If a test requires a missing optional dependency or artifact, record the exact failure rather than changing unrelated code.

## Validation and Acceptance

The new repository is ready for user review when `git rev-parse --show-toplevel` returns `D:/luolin/V13`, `git branch --show-current` returns `main`, and `git log` reports that the branch has no commits. The staged inventory must contain no path from excluded roots, no prohibited model/data/archive/CAD/log/PDF extension, no file larger than 5 MiB, and no file matching a common credential or private-key pattern.

`reports/experiment_index.json` and `local_training_config.example.json` must parse as JSON. `reports/README.md` must tell a user on another machine where to read current conclusions and experiment history without requiring local artifact paths. The final handoff must state the staged file count and total size, test output, old Git backup location, and that no commit or push occurred.

## Idempotence and Recovery

The file-generation edits are repeatable through `apply_patch`. The Git move is intentionally guarded: it stops if `.git.backup-20260803` exists or if `.git` does not resolve exactly under the workspace root. If initialization fails after the move, retry `git init` without moving the backup again.

Before any commit, rollback is straightforward: remove only the newly created `.git` directory after resolving it as `D:\luolin\V13\.git`, then move `.git.backup-20260803` back to `.git`. Do not delete the backup until the GitHub push has been independently confirmed.

Staging is reversible with `git rm --cached` in the new repository because the new repository has no commits. This removes files only from the index and does not remove the working copies.

## Artifacts and Notes

The audit established these key boundaries:

    ABC/             approximately 171 GiB, excluded
    local_runs/      approximately 24 GiB, excluded
    BRepARG/         nested upstream Git repository, excluded by user decision
    papers/          future paper work, excluded by user decision
    old .git/        approximately 1.3 MiB, preserved as an ignored backup

No common API token, private key, credential file, or private IPv4 address was found in the intended source-and-document surface. Machine-specific absolute paths remain in some historical operational scripts and documents; these are not secrets, and broader portability refactoring is deferred from this repository-boundary task.

## Interfaces and Dependencies

This work uses Git, PowerShell, Python, and pytest already present in the environment. It does not add a runtime library. The public configuration interface is `local_training_config.example.json` plus `reproducibility/configs/paths.env.example`; a developer creates ignored local copies and supplies machine paths there.

The durable experiment-navigation interface is `reports/experiment_index.json`. Each entry has `id`, `status`, `category`, `summary`, `metrics`, and `artifacts_external` fields. `summary` and `metrics` are repository-relative paths or null; `artifacts_external=true` states that checkpoints, datasets, and generated CAD are deliberately absent from Git.

Revision note 2026-08-03: Created this ExecPlan after the user approved clean reinitialization, full exclusion of `BrepARG/` and `papers/`, and a source-plus-lightweight-summary repository boundary.
