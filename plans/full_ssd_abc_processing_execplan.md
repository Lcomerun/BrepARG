# Full SSD ABC Processing and Training

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md` in the repository root. The user has copied all ABC STEP chunk archives to `E:\ABC\step` on a 1 TB SSD and wants the machine to process as many files as possible without filling the drive or hanging. The desired observable result is a resumable long-running job that extracts one chunk archive, deletes the archive after extraction, parses STEP files with multiple workers and per-file timeouts, deletes the extracted chunk after parsing, and starts training after all chunks have been processed.

## Purpose / Big Picture

The local SSD has enough room for compressed archives and processed data, but not enough to keep every chunk decompressed at once. This work creates and starts a guarded pipeline so the machine can process the full corpus chunk by chunk. A user can monitor progress through a state file and logs, and if the job stops, rerun the same command to resume from existing parsed pkl files.

## Progress

- [x] (2026-06-25 01:31 +08:00) Removed previous smoke artifacts from `D:\luolin\V13\processed_local` and `.pytest_cache`.
- [x] (2026-06-25 01:32 +08:00) Confirmed `E:\ABC\step` contains 100 `abc_XXXX_step_v00.7z` archives totaling 106,651,993,150 bytes.
- [x] (2026-06-25 01:32 +08:00) Confirmed E drive free space is about 886 GB.
- [x] (2026-06-25 01:33 +08:00) Confirmed Windows `tar.exe` can list the `.7z` archives.
- [x] (2026-06-25 01:36 +08:00) Added tests for archive discovery and output path calculation.
- [x] (2026-06-25 01:52 +08:00) Implemented `tools/run_ssd_archive_pipeline.py`.
- [x] (2026-06-25 01:58 +08:00) Ran helper tests and dry-run verification.
- [x] (2026-06-25 02:02 +08:00) Ran a tiny real chunk0 smoke: extracted chunk0, parsed 10 files, 7 ok and 3 multi.
- [x] (2026-06-25 02:03 +08:00) Started the long-running pipeline in the background with PID 24452.
- [x] (2026-06-25 02:07 +08:00) Confirmed chunk0 is parsing with 4 workers and parsed pkl count is increasing.
- [ ] Monitor chunk0 completion, then confirm archive/extracted-dir deletion and chunk1 start.
- [ ] Record training start after all chunks are processed.

## Surprises & Discoveries

- Observation: There is no `7z`, `7za`, or `7zr` command on PATH.
  Evidence: `Get-Command 7z`, `7za`, and `7zr` returned nothing.
- Observation: Windows `tar.exe` can read the chunk `.7z` files.
  Evidence: `tar -tf E:\ABC\step\abc_0000_step_v00.7z` listed model directories.
- Observation: `E:\ABC\step` also contains `success.zip` and `batch_unzip_7z.sh`.
  Evidence: both files were listed beside the 100 chunk archives. They are not chunk archives and must not be deleted by the ABC chunk pipeline.
- Observation: The parser log only updates every 1000 STEP files by default, so pkl counts are a better short-interval progress signal.
  Evidence: after start, the pipeline log stayed at `steps=10000`, while `E:\ABC\processed\abc_parsed_full\abc_0000` increased from 32 to 130 pkls.

## Decision Log

- Decision: Process one chunk archive at a time.
  Rationale: This allows deleting the compressed archive and decompressed chunk directory at clear checkpoints, keeping peak SSD usage lower than decompressing all chunks.
  Date/Author: 2026-06-25 / Codex.
- Decision: Delete a chunk archive only after extraction succeeds and the extracted directory contains STEP files.
  Rationale: This honors the user's storage-saving requirement while avoiding deletion after a failed extraction.
  Date/Author: 2026-06-25 / Codex.
- Decision: Delete the extracted chunk directory only after the parser command exits successfully.
  Rationale: If parsing crashes, keeping the extracted chunk allows a retry without needing the deleted archive.
  Date/Author: 2026-06-25 / Codex.
- Decision: Ignore `success.zip` and `batch_unzip_7z.sh`.
  Rationale: They are not named ABC chunk archives and should not be treated as raw data chunks.
  Date/Author: 2026-06-25 / Codex.
- Decision: Use 4 parser workers and 60 second per-file timeout for the first long run.
  Rationale: Multiple workers increase throughput, while 4 is a conservative starting point for Windows, OCC memory use, and SSD contention.
  Date/Author: 2026-06-25 / Codex.

## Outcomes & Retrospective

The pipeline is being prepared. The previous local verification artifacts have been removed, and the SSD archive inventory is confirmed. The next result should be a resumable job with logs.
The pipeline is now running in the background. It has deleted `abc_0000_step_v00.7z` after verifying the existing extraction and is parsing chunk0 with 4 workers.

## Context and Orientation

Each archive `abc_XXXX_step_v00.7z` contains one ABC raw STEP chunk. A STEP file is a CAD exchange file. The parser turns each STEP file into a parsed pkl dictionary with surfaces, edges, bounding boxes, and adjacency fields. The parsed output for chunk `XXXX` will be written under `E:\ABC\processed\abc_parsed_full\abc_XXXX`. The existing improved training script can read this one-level chunk subdirectory layout.

The safe Windows parser is `tools/process_abc_windows.py`. It parses STEP files in child Python processes so a per-file timeout can kill a stuck parser without freezing the whole job. The full archive orchestrator will call this parser chunk by chunk.

## Plan of Work

Add `tools/run_ssd_archive_pipeline.py`. It discovers chunk archives, extracts each archive into its same-named directory using Windows `tar.exe`, deletes the archive after verifying extraction, runs `tools/process_abc_windows.py` with workers and timeout, deletes the extracted chunk directory after parsing succeeds, records state in JSON after every stage, and optionally launches `breparg_improvements/train.py --stage all` after all chunks finish.

## Concrete Steps

Run from `D:\luolin\V13`.

Verify helper tests:

    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q

Dry-run the pipeline:

    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' tools\run_ssd_archive_pipeline.py --archive-root E:\ABC\step --parsed-root E:\ABC\processed\abc_parsed_full --train-out-root E:\ABC\processed\train_outputs --chunks 0-2 --dry-run

Start the full background job:

    Start-Process -WindowStyle Hidden -FilePath 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -ArgumentList @('tools\run_ssd_archive_pipeline.py','--archive-root','E:\ABC\step','--parsed-root','E:\ABC\processed\abc_parsed_full','--train-out-root','E:\ABC\processed\train_outputs','--chunks','all','--workers','4','--timeout','60','--delete-archive-after-extract','--delete-extracted-after-process','--train-after')

## Validation and Acceptance

The preparation is accepted when tests pass, dry-run reports the 100 archives without modifying files, and the background job writes a PID/log/state file. Full completion is accepted when every archive is either processed or explicitly recorded as failed, no extracted chunk directories remain for successful chunks, parsed pkl directories exist under `E:\ABC\processed\abc_parsed_full`, and training has started after processing.

## Idempotence and Recovery

The job is safe to rerun. Parsed pkl files already present are skipped by `tools/process_abc_windows.py`. If a run stops after archive extraction but before parsing, the extracted directory remains and the pipeline can continue parsing it. If a run stops after parsing but before deleting the extracted directory, rerun will skip existing pkls and then delete the extracted directory after a successful parser pass.

## Artifacts and Notes

Expected artifacts are:

    D:\luolin\V13\tools\run_ssd_archive_pipeline.py
    E:\ABC\processed\logs\archive_pipeline_<timestamp>.log
    E:\ABC\processed\logs\archive_pipeline_state.json
    E:\ABC\processed\abc_parsed_full\abc_XXXX\*.pkl
    E:\ABC\processed\train_outputs\newscheme_full_local\...

## Interfaces and Dependencies

Use `C:\Users\YU\.conda\envs\brepgen_env\python.exe` directly with `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`. Use Windows `tar.exe` for extraction because no `7z` executable is available on PATH.

Revision note: Initial ExecPlan created for the destructive SSD archive processing pipeline.

Revision note: Updated after implementing the orchestrator, running dry-run and smoke verification, and starting the background job.

Revision note: Hardened the Windows parser and SSD orchestrator for long local runs. The parser now writes per-chunk incremental reports under `E:\ABC\processed\abc_parsed_full\abc_XXXX\_reports`, accepts a small configurable `error + timeout` rate instead of failing an entire chunk for a few bad STEP files, and reuses the manifest on rerun so known non-output files are skipped. The orchestrator now discovers already-extracted chunk directories when the original archive has already been deleted, records per-chunk parse summaries in the state file, and skips training if any chunk is incomplete or failed. The active final background job is PID `26036`, with log `E:\ABC\processed\logs\archive_pipeline_20260625_023733.log`.
