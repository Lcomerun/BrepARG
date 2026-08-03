# Sharded Sequence Generation and Merge

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md` in the repository root. The user wants to replace the risky all-in-memory sequence generation path with a per-chunk sharded path that preserves intermediate shard files, records progress, can resume after interruption, and can merge shards into a single AR-compatible `sequences_fsq_rcm.pkl` style file.

## Purpose / Big Picture

The current `sequence` stage converts parsed ABC `.pkl` CAD files into autoregressive token sequences by loading the trained VQ-VAE checkpoint and encoding each model's surface and edge geometry. The old implementation keeps every generated sequence in memory and writes one final `sequences_fsq_rcm.pkl` only after all train, validation, and test files are processed. On full ABC scale this is fragile: if the process crashes after many hours, no intermediate work can be reused.

After this change, the user can generate sequence shards such as `sequence_shards/abc_0000.pkl`, `sequence_shards/abc_0001.pkl`, and so on, with `_manifest.jsonl` and `_summary.json` records. A later merge step creates a single downstream-compatible merged file while retaining all shard files. This preserves the current AR training interface and creates a safer recovery point for future runs.

## Progress

- [x] (2026-06-28 01:10 +08:00) Confirmed the previous all-in-memory `sequence` path does not write progress files before the final pickle dump.
- [x] (2026-06-28 01:47 +08:00) Confirmed the previous all-in-memory sequence process completed successfully and wrote `E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm.pkl`.
- [x] (2026-06-28 01:55 +08:00) Confirmed the final report says `sequence.status=VERIFIED`, `sequences=425120`, `out_of_vocab=0`, `se_tokens_per_element=4`, and `ordering=RCM`.
- [x] (2026-06-28 02:00 +08:00) Add tests for grouping split paths by ABC chunk, merging shard files, preserving metadata, and writing summary records.
- [x] (2026-06-28 02:01 +08:00) Implement a small testable helper module for shard grouping, manifest rows, and merge validation.
- [x] (2026-06-28 02:02 +08:00) Implement a sharded sequence runner that can process chunk shards, skip completed shards, and merge them.
- [x] (2026-06-28 02:02 +08:00) Add a Windows launcher for the full epoch100 sharded sequence run.
- [x] (2026-06-28 02:03 +08:00) Validate tests, compilation, and PowerShell parser checks before starting the full run.
- [x] (2026-06-28 02:03 +08:00) Start the sharded run and monitor shard creation.
- [x] (2026-06-28 02:08 +08:00) Restart the sharded run with `NS_SEQ_WORKERS=5 --resume` after confirming 3 completed shards; existing shards were skipped instead of recomputed.
- [x] (2026-06-28 03:33 +08:00) Complete all 100 per-chunk shard files under `sequence_shards`.
- [x] (2026-06-28 03:34 +08:00) Merge the 100 shards into `sequences_fsq_rcm_sharded_merged.pkl` while preserving the canonical `sequences_fsq_rcm.pkl`.
- [x] (2026-06-28 03:37 +08:00) Verify manifest, summary, report fields, pickle readability, tests, and Python compilation.

## Surprises & Discoveries

- Observation: The all-in-memory sequence run finished after roughly 3 hours 46 minutes, faster than the conservative estimate from the older Linux `/data` run.
  Evidence: monitor log recorded `sequence_process_missing` at `2026-06-28 01:47:02`, while `sequences_fsq_rcm.pkl` has `LastWriteTime=2026-06-28 01:45:04`.
- Observation: The all-in-memory result is valid and can be used for AR training now.
  Evidence: `train_report.json` contains `sequence.status=VERIFIED`, `sequences=425120`, `out_of_vocab=0`, `max_token=10292`, `vocab_size=10294`, and `se_tokens_per_element=4`.
- Observation: A merge from shards can be schema-compatible with current AR training, but may not be byte-identical to the already generated monolithic file if chunks are processed in parallel.
  Evidence: `BrepARG\2sequence.py` uses `random.randint` for a face-index cyclic offset inside `_encode_single_rotation`. Parallel workers do not consume random numbers in the exact same global order as the old single process. This is not expected to harm training because the offset is an augmentation-like re-indexing, but it means byte-for-byte equality is not the acceptance criterion.
- Observation: Five sequence workers kept the RTX 3060 busy without exhausting memory.
  Evidence: the resumed run used `--workers 5`; GPU samples were commonly 80-100% with about 5.4 GB of 12 GB VRAM used, and the five worker processes each held roughly 1.9-2.0 GB system RAM.
- Observation: The final shard-derived merge has the same split counts and token validity as the canonical monolithic sequence file, but a slightly different file size.
  Evidence: canonical `sequences_fsq_rcm.pkl` length is `1414689276`; shard-derived `sequences_fsq_rcm_sharded_merged.pkl` length is `1419629996`. Both report `sequences=425120`, `train=382720`, `val=21124`, `test=21276`, `vocab_size=10294`, `max_token=10292`, `out_of_vocab=0`, and `se_tokens_per_element=4`.

## Decision Log

- Decision: Keep the current verified monolithic `sequences_fsq_rcm.pkl` as the immediate AR-compatible file.
  Rationale: It is already complete and verified. There is no need to throw away a good artifact while adding sharded generation for recoverability.
  Date/Author: 2026-06-28 / Codex.
- Decision: The sharded merge will write a separate merged file first, rather than overwriting the current canonical file immediately.
  Rationale: This preserves both the current monolithic result and the new shard-derived result. After validation, either file can be chosen for AR.
  Date/Author: 2026-06-28 / Codex.
- Decision: For full-scale speed, allow multiple worker processes but keep worker count configurable.
  Rationale: Each worker loads the VQ-VAE on the single RTX 3060. Too many workers can exhaust VRAM or reduce throughput, while two or three may improve CPU/I/O overlap. The script should default conservatively and expose `NS_SEQ_WORKERS`.
  Date/Author: 2026-06-28 / Codex.
- Decision: Do not modify the AR loader yet.
  Rationale: The user asked whether not merging affects training. Current AR code expects one `sequences_fsq_rcm.pkl`; keeping the merge step avoids delaying AR training with a larger loader refactor.
  Date/Author: 2026-06-28 / Codex.

## Outcomes & Retrospective

The sharded sequence workflow is implemented and verified. It generated 100 per-chunk shard files under `E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequence_shards`, wrote an append-only `_manifest.jsonl`, wrote `_summary.json`, and produced `E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm_sharded_merged.pkl`.

The current verified single-file sequence output remains available for AR training at `E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm.pkl`. The shard-derived merged file is schema-compatible with the current AR loader but is intentionally stored separately so both artifacts are preserved.

Final validation evidence:

    manifest lines=103
    manifest done=100
    manifest skipped_existing=3
    manifest error/stale=0
    shard_count=100
    merged_output length=1419629996
    merged sequences=425120
    merged train=382720
    merged val=21124
    merged test=21276
    merged vocab_size=10294
    merged max_token=10292
    merged out_of_vocab=0
    merged se_tokens_per_element=4
    helper tests: 19 passed in 1.11s
    py_compile: passed for sequence_sharding.py, run_sharded_sequence.py, and train.py

## Context and Orientation

The repository root is `D:\luolin\V13`. The parsed ABC data is under `E:\ABC\processed\abc_parsed_full`, organized as chunk directories such as `abc_0000`, `abc_0001`, through `abc_0099`. The current training output directory is `E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100`.

The current sequence file is `E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm.pkl`. It is a pickle containing keys `train`, `val`, `test`, vocabulary sizes, token offsets, and special token IDs. The AR stage in `breparg_improvements\train.py` reads this single file through `_load_ar_seqs`.

The VQ-VAE checkpoint used for sequence encoding is `E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\fsq_vqvae_best.pt`.

## Plan of Work

First, create `breparg_improvements\sequence_sharding.py` with pure helper functions. These helpers must be testable without GPU. They will extract chunk IDs from parsed paths, group `split.pkl` paths by chunk, validate common sequence metadata across shards, merge shard pickles into one sequence pickle, and write a summary JSON.

Second, add tests in `tests\test_local_pipeline_helpers.py`. The tests will create small fake shard pickle files and prove that merge order is deterministic, train/val/test lists are concatenated, metadata is preserved, and inconsistent vocab metadata raises a clear error.

Third, create `tools\run_sharded_sequence.py`. This script will load the epoch100 VQ-VAE checkpoint, create one shard per ABC chunk, write each shard to `sequence_shards/abc_XXXX.pkl`, append parent-side manifest rows as shards finish, and merge all completed shards into `sequences_fsq_rcm_sharded_merged.pkl`. It will support `--workers`, `--resume`, and `--merge-only` so the run is restartable.

Fourth, create `tools\run_sharded_sequence_epoch100.ps1` as the Windows launcher. It will set `NS_POOL`, `NS_OUTBASE`, `NS_OUT`, `CUDA_VISIBLE_DEVICES`, and `NS_SEQ_WORKERS`. It will write logs under `E:\ABC\processed\logs`.

Fifth, validate with tests and syntax checks. Do not start the full sharded run until the helper tests pass, Python compilation passes, and the PowerShell script parses.

## Concrete Steps

Run commands from `D:\luolin\V13`.

After tests are added but before implementation, run:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py::LocalPipelineHelperTests::test_sequence_shards_merge_preserves_order_and_metadata tests/test_local_pipeline_helpers.py::LocalPipelineHelperTests::test_sequence_shards_reject_inconsistent_metadata -q

The expected RED result is import or missing function failure.

After implementation, run:

    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m pytest tests/test_local_pipeline_helpers.py -q
    & 'C:\Users\YU\.conda\envs\brepgen_env\python.exe' -m py_compile breparg_improvements\sequence_sharding.py tools\run_sharded_sequence.py
    [System.Management.Automation.Language.Parser]::ParseFile('D:\luolin\V13\tools\run_sharded_sequence_epoch100.ps1', [ref]$null, [ref]$null)

To run the sharded sequence job:

    Start-Process powershell.exe -ArgumentList '-ExecutionPolicy','Bypass','-File','D:\luolin\V13\tools\run_sharded_sequence_epoch100.ps1' -WindowStyle Hidden

Expected output paths:

    E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequence_shards\abc_0000.pkl
    E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequence_shards\_manifest.jsonl
    E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequence_shards\_summary.json
    E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm_sharded_merged.pkl

## Validation and Acceptance

The sharded implementation is accepted when the helper tests pass, compilation passes, the launcher parses, shard files are written per chunk, the manifest records each chunk's status and counts, and the merged file has the same schema as the existing `sequences_fsq_rcm.pkl`. A merge does not need to be byte-identical to the old monolithic output because the original algorithm contains randomized face-index offsets, but it must have valid token ranges, `out_of_vocab=0`, `se_tokens_per_element=4`, `ordering=RCM`, and consistent vocabulary metadata.

The AR training is not delayed if the canonical `sequences_fsq_rcm.pkl` remains present. If the user wants AR to train from the shard-derived merge, copy or rename `sequences_fsq_rcm_sharded_merged.pkl` to `sequences_fsq_rcm.pkl` after validation while preserving the original monolithic file under a backup name.

## Idempotence and Recovery

The sharded runner should be safe to rerun. If `--resume` is used, existing verified shard files are skipped. The manifest is append-only for traceability, and `_summary.json` records the latest aggregate state. The current monolithic `sequences_fsq_rcm.pkl` remains untouched by default.

## Artifacts and Notes

Current verified monolithic evidence:

    sequences_fsq_rcm.pkl length=1414689276
    sequence.status=VERIFIED
    sequences=425120
    train=382720
    val=21124
    test=21276
    vocab_size=10294
    max_token=10292
    out_of_vocab=0
    se_tokens_per_element=4
    ordering=RCM

Monitor evidence for the old run:

    2026-06-28 01:42 process pid=30816 working_set_gb=14.68 private_gb=15.81
    2026-06-28 01:47 sequence_process_missing; monitor_stop
    sequences_fsq_rcm.pkl LastWriteTime=2026-06-28 01:45:04

Sharded full-run evidence:

    launcher log: E:\ABC\processed\logs\sharded_sequence_epoch100_20260628_020816.log
    run started with workers=5 after resume
    first resumed skip rows: abc_0000, abc_0001, abc_0002
    final shard: abc_0099.pkl LastWriteTime=2026-06-28 03:33:45
    merge log line: [03:34:52] merged shards=100 sequences=425120 out_of_vocab=0 -> E:\ABC\processed\train_outputs\newscheme_full_vqvae_epoch100\sequences_fsq_rcm_sharded_merged.pkl
    report stage: sequence_sharded.status=VERIFIED

## Interfaces and Dependencies

In `breparg_improvements\sequence_sharding.py`, define:

    chunk_id_from_path(path)
    group_split_paths_by_chunk(split)
    sequence_metadata(package)
    merge_sequence_shards(shard_paths, output_path)
    summarize_sequence_package(package)

In `tools\run_sharded_sequence.py`, use the existing `BrepARG\2sequence.py` `ARDataPreprocessor._process_single_cad` method to preserve tokenization behavior as closely as possible. Use `breparg_improvements\train.py` helpers to build and load the VQ-VAE model.

Revision note: Created this plan after the all-in-memory full sequence completed successfully, making sharded generation an engineering improvement rather than an emergency recovery.
