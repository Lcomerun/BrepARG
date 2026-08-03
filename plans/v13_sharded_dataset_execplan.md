# V13 sharded dataset compression and server training preparation

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows the repository-level instructions in `AGENTS.md` and the ExecPlan requirements in `PLANS.md`.

## Purpose / Big Picture

The user needs to move the V13 project back to a server for VQ-VAE recovery training, but AutoDL has a practical file-count limit of about 200,000 files. The parsed ABC pool has hundreds of thousands of small `.pkl` files, so directly uploading the extracted parsed tree is not viable. After this work, the user will have a two-layer sharded data pipeline: parsed shards that preserve the original parsed pickle payloads and source metadata, and VQ patch shards that the VQ-VAE trainer can read directly on the server. The workflow must also reclaim local disk by compressing each extracted chunk, verifying the new shard, and deleting only the extracted chunk directory that was proven to be covered by the shard.

## Progress

- [x] (2026-07-08 00:40 +08:00) Inspected current local state. `ABC/processed/abc_parsed_full` contains extracted directories `abc_0000` through `abc_0004`. `ABC/processed/abc_parsed_full_archives` contains 100 zip archives from `abc_0000_parsed.zip` through `abc_0099_parsed.zip`. D: has about 32.5 GB free before deleting extracted chunks.
- [x] (2026-07-08 00:48 +08:00) Added `breparg_improvements/sharded_data.py`, a shared streaming reader/writer for `.pkl.zst`, `.pkl.gz`, and uncompressed `.pkl` shard files.
- [x] (2026-07-08 00:52 +08:00) Added `tools/build_parsed_shards.py` and `tools/verify_parsed_shards.py` so extracted chunk directories can be converted to parsed shards, verified by payload hash, and optionally deleted after verification.
- [x] (2026-07-08 00:58 +08:00) Added `tools/build_vqvae_patch_shards.py` to generate VQ patch shards from parsed shards inside the numpy-capable training environment.
- [x] (2026-07-08 01:02 +08:00) Added patch-shard sampling support to `breparg_improvements/vqvae_sampling.py` and `breparg_improvements/train.py` through `NS_VQ_PATCH_SHARD_ROOT` and `NS_VQ_PATCH_SHARDS`.
- [x] (2026-07-08 01:04 +08:00) Added focused tests for parsed shard construction and patch-shard sampling to `tests/test_local_pipeline_helpers.py`.
- [x] (2026-07-08 01:08 +08:00) Ran syntax checks and targeted tests for the new sharding code. `py_compile` passed and `pytest tests/test_local_pipeline_helpers.py -k "parsed_shard or patch_shard" -q` reported `2 passed, 116 deselected`.
- [x] (2026-07-08 01:12 +08:00) Wrote `docs/v13_sharded_dataset_operator_guide.md`, an operator guide for local shard building, AutoDL upload, server patch-shard generation, VQ-VAE launch, recovery, and cleanup policy.
- [x] (2026-07-08 01:32 +08:00) Built parsed shards for currently extracted chunks `abc_0000` through `abc_0004`, verified all five with `tools/verify_parsed_shards.py`, and deleted the extracted directories after each verified shard.
- [x] (2026-07-08 01:36 +08:00) Added `tools/run_parsed_shard_cycle.py` so remaining zip archives can be processed one chunk or range at a time without keeping the full extracted pool on disk.
- [x] (2026-07-08 01:42 +08:00) Exercised the cycle driver on `abc_0005`. It extracted `abc_0005_parsed.zip`, built and verified `parsed_abc_0005.pkl.zst`, and deleted the extracted `abc_0005` directory.
- [x] (2026-07-08 01:48 +08:00) Wrote `local_reports/v13_project_cleanup_manifest_20260708.md` and deleted `.pytest_cache`, the only immediately safe cleanup target identified in this pass.
- [x] (2026-07-08 03:40 +08:00) Decided the safe storage strategy for all parsed shards: use `C:\V13_abc_parsed_shards` as the authoritative parsed-shard root, because D: could not safely retain all original zip archives and all parsed shards at once.
- [x] (2026-07-08 01:49 +08:00) Prepared a conservative documentation cleanup manifest. Broad report/local-run deletion remains intentionally deferred until canonical evidence is retained.
- [x] (2026-07-08 04:35 +08:00) Refreshed and verified `abc_0000` through `abc_0064` at `C:\V13_abc_parsed_shards`. The refreshed summary reported 65 shards, 447,764 sources, 424,181,164,521 payload bytes, and 64,636,147,236 shard bytes.
- [x] (2026-07-08 08:03 +08:00) Resumed after the first long foreground run hit the outer tool timeout. Evidence showed no running Python process, no `.tmp` shard, no extracted chunk directory, and completed shards through `abc_0074`; continued safely with smaller monitored batches.
- [x] (2026-07-08 08:43 +08:00) Built and verified `abc_0075` through `abc_0079`; refreshed `abc_0000` through `abc_0079`. The refreshed summary reported `status=VERIFIED`, 80 shards, 549,655 sources, 520,124,922,907 payload bytes, and 79,786,855,602 shard bytes.
- [x] (2026-07-08 10:44 +08:00) Built and verified `abc_0080` through `abc_0089`; each chunk was deleted after verification and no extracted directories or `.tmp` files remained.
- [x] (2026-07-08 11:12 +08:00) Built and verified `abc_0090` through `abc_0099`; each chunk was deleted after verification and no extracted directories or `.tmp` files remained.
- [x] (2026-07-08 11:32 +08:00) Refreshed the final manifest at `C:\V13_abc_parsed_shards\_manifest.jsonl` and wrote `local_reports/v13_parsed_shards_manifest_croot_0000_0099_20260708.json`. Final status is `VERIFIED` for 100 shards, 681,406 sources, 648,659,550,518 payload bytes, and 99,966,076,379 shard bytes.
- [x] (2026-07-08 11:36 +08:00) Performed final conservative cleanup: deleted `.pytest_cache` after the final pytest run recreated it, and deleted the empty legacy directory `ABC/processed/abc_parsed_shards`. Updated `local_reports/v13_project_cleanup_manifest_20260708.md` with the final cleanup actions and C-root retention policy.

## Surprises & Discoveries

- Observation: The default Python on this machine has `zstandard` installed but lacks `numpy` and `torch`; the `brepgen_env` Python has `numpy` and `torch` but lacks `zstandard`.
  Evidence: `python -c "import importlib.util; ..."` reported `zstandard True`, `torch False`, `numpy False`, while `C:\Users\YU\.conda\envs\brepgen_env\python.exe` reported `zstandard False`, `torch True`, `numpy True`.
- Observation: The extracted five chunks are large enough that all compression must be sequential and verified before deletion.
  Evidence: `abc_0000` through `abc_0004` contain 5,945, 6,514, 6,780, 6,873, and 6,779 files respectively, totaling about 34.8 GB of extracted data.
- Observation: The first five parsed shards compress much smaller than the extracted source but still require a storage decision for all 100 chunks.
  Evidence: `tools/verify_parsed_shards.py` reported five verified shards with `total_sources=32881`, `total_payload_bytes=34787456054`, and `total_shard_bytes=5318248544`. D: had about 62.1 GB free after deleting extracted directories, while extrapolating from the first five shards suggests all 100 parsed shards may need about 106 GB.
- Observation: The final parsed-shard set fit on C: but not comfortably on D: while retaining the original zip archives.
  Evidence: The final `C:\V13_abc_parsed_shards` set contains 100 shard files totaling 99,966,076,379 bytes. After final verification, C: had about 33.7 GB free and D: had about 67.4 GB free, while the original 100 zip archives were retained under `ABC/processed/abc_parsed_full_archives`.
- Observation: The long `abc_0065` through `abc_0079` foreground run exceeded the outer command timeout, but the sharding workflow recovered cleanly.
  Evidence: After timeout, completed shards existed through `parsed_abc_0074.pkl.zst`, `_cycle_manifest.jsonl` showed `built_verified` rows through `abc_0074`, and there were no running Python processes, `.tmp` shard files, or extracted chunk directories. The remaining chunks were resumed with `--resume`.

## Decision Log

- Decision: Parsed shards store raw source pickle bytes instead of unpickled numpy arrays.
  Rationale: This lets the local compression step run without importing numpy and avoids accidental shape or dtype mutations. Source identity is preserved by metadata fields such as `source_relpath`, `chunk_id`, and `source_sha256`. The server can later unpickle the payloads when building VQ patch shards.
  Date/Author: 2026-07-08 / Codex.
- Decision: Use streaming pickle objects inside compressed `.pkl.zst` or `.pkl.gz` files rather than one giant in-memory list.
  Rationale: A whole ABC chunk can be several GB uncompressed. Streaming avoids loading the entire chunk into RAM and makes verification possible record by record.
  Date/Author: 2026-07-08 / Codex.
- Decision: The VQ-VAE trainer reads VQ patch shards, not parsed shards, during training.
  Rationale: Parsed shards are the durable source layer. Patch shards are the training layer. This keeps training fast and lets the trainer reuse existing patch-level sampling, complex/curved weighting, and source caps.
  Date/Author: 2026-07-08 / Codex.
- Decision: The first deletion target is only extracted chunk directories that have a verified parsed shard.
  Rationale: The user explicitly asked to delete extracted chunks after compression, and the existing zip archives remain as a recovery source. Broad report/document cleanup is less deterministic and must be handled by a manifest first.
  Date/Author: 2026-07-08 / Codex.
- Decision: Use `C:\V13_abc_parsed_shards` as the authoritative local parsed-shard root for the completed dataset.
  Rationale: D: could not safely hold all original zip archives plus all parsed shards. C: had enough capacity, and using a single root keeps upload and server verification simple.
  Date/Author: 2026-07-08 / Codex.

## Outcomes & Retrospective

Final local outcome 2026-07-08: The parsed-shard compression task is complete. `C:\V13_abc_parsed_shards` contains 100 verified parsed shard files from `parsed_abc_0000.pkl.zst` through `parsed_abc_0099.pkl.zst`, totaling 99,966,076,379 bytes. The final summary `local_reports/v13_parsed_shards_manifest_croot_0000_0099_20260708.json` reports `status=VERIFIED`, 681,406 source records, and 648,659,550,518 payload bytes. `ABC/processed/abc_parsed_full` has no extracted `abc_XXXX` directories, there are no `.tmp` shard files, and the 100 original zip archives remain under `ABC/processed/abc_parsed_full_archives`.

The remaining server-side work is to upload `C:\V13_abc_parsed_shards` to `/workspace/ABC/processed/abc_parsed_shards`, build VQ patch shards on the server, and launch VQ-VAE training with `NS_VQ_PATCH_SHARD_ROOT` or `NS_VQ_PATCH_SHARDS`. Broad cleanup of reports and local runs remains intentionally conservative; only `.pytest_cache` was deleted automatically.

## Context and Orientation

The repository root is `D:\luolin\V13`. The current parsed source archives live in `ABC/processed/abc_parsed_full_archives` as one zip per ABC chunk, for example `abc_0000_parsed.zip`. The currently extracted parsed tree is `ABC/processed/abc_parsed_full`, and at the start of this plan it contains only `abc_0000` through `abc_0004`.

A parsed `.pkl` file is the output of STEP parsing for one CAD model. It normally contains arrays such as `surf_ncs` and `edge_ncs`. A parsed shard is a compressed file that stores many of those original parsed `.pkl` payloads as raw bytes plus metadata. A VQ patch shard is a compressed file that stores individual surface and edge patches already extracted from parsed records. VQ-VAE training should read VQ patch shards through `NS_VQ_PATCH_SHARD_ROOT` or `NS_VQ_PATCH_SHARDS`.

The existing VQ-VAE sampler lives in `breparg_improvements/vqvae_sampling.py`. The existing training entry lives in `breparg_improvements/train.py`. The new shard utilities are `breparg_improvements/sharded_data.py`, `tools/build_parsed_shards.py`, `tools/verify_parsed_shards.py`, and `tools/build_vqvae_patch_shards.py`.

## Plan of Work

First, verify the new sharding utilities with targeted tests. The tests should prove that a chunk directory can be converted into a parsed shard without unpickling the original records, that the shard verifier catches and counts the payloads, and that the VQ-VAE sampler can read a patch shard directly.

Second, write an operator guide in `docs/v13_sharded_dataset_operator_guide.md`. The guide must explain local compression, verification, deletion, AutoDL upload, server-side patch shard generation, VQ-VAE training environment variables, and recovery after interruption.

Third, process the currently extracted chunk directories one at a time. For each chunk, run `tools/build_parsed_shards.py` with `--delete-after-verify`. The tool must write a manifest row, verify the shard by re-reading payload hashes, and only then delete the extracted `abc_XXXX` directory. After each chunk, check that the parsed shard exists and that the extracted directory no longer exists.

Fourth, continue with remaining chunks by using `tools/run_parsed_shard_cycle.py` to extract one zip archive at a time from `ABC/processed/abc_parsed_full_archives`, build the parsed shard, verify it, and delete the extracted directory. If local disk cannot hold both the original zip archives and all parsed shards, stop before risking data loss and switch the shard output root to a larger drive or the server.

Fifth, prepare documentation cleanup conservatively. The first cleanup artifact should be a manifest that says which local reports are retained as canonical server-start evidence and which stale reports are safe to delete. Deleting broad untracked evidence files before that manifest exists is not acceptable.

## Concrete Steps

Run syntax and tests from the repository root:

    cd D:\luolin\V13
    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m py_compile breparg_improvements\sharded_data.py breparg_improvements\vqvae_sampling.py breparg_improvements\train.py tools\build_parsed_shards.py tools\verify_parsed_shards.py tools\build_vqvae_patch_shards.py
    C:\Users\YU\.conda\envs\brepgen_env\python.exe -m pytest tests\test_local_pipeline_helpers.py -k "parsed_shard or patch_shard" -q

Build and delete the five extracted chunks:

    cd D:\luolin\V13
    python tools\build_parsed_shards.py --parsed-root ABC\processed\abc_parsed_full --shard-root C:\V13_abc_parsed_shards --manifest C:\V13_abc_parsed_shards\_manifest.jsonl --chunks 0-4 --compression zstd --compression-level 10 --resume --delete-after-verify
    python tools\verify_parsed_shards.py C:\V13_abc_parsed_shards\parsed_abc_0000.pkl.zst C:\V13_abc_parsed_shards\parsed_abc_0001.pkl.zst C:\V13_abc_parsed_shards\parsed_abc_0002.pkl.zst C:\V13_abc_parsed_shards\parsed_abc_0003.pkl.zst C:\V13_abc_parsed_shards\parsed_abc_0004.pkl.zst --output local_reports\v13_parsed_shards_verify_0000_0004_20260708.json

Continue remaining chunks with the disk-safe cycle driver:

    cd D:\luolin\V13
    python tools\run_parsed_shard_cycle.py --archive-root ABC\processed\abc_parsed_full_archives --parsed-root ABC\processed\abc_parsed_full --shard-root C:\V13_abc_parsed_shards --manifest C:\V13_abc_parsed_shards\_manifest.jsonl --chunks 5-9 --compression zstd --compression-level 10 --resume --delete-after-verify

Build patch shards on the server after uploading parsed shards:

    cd /workspace/V13
    python -m pip install zstandard
    python tools/build_vqvae_patch_shards.py --parsed-shard-root /workspace/ABC/processed/abc_parsed_shards --patch-shard-root /workspace/ABC/processed/vqvae_patch_shards --manifest /workspace/ABC/processed/vqvae_patch_shards/_manifest.jsonl --compression zstd --patches-per-shard 100000 --complex-min-faces 12 --complex-min-edges 20 --max-source-faces 50 --max-source-edges 150

Start VQ-VAE training from patch shards:

    cd /workspace/V13
    export NS_VQ_PATCH_SHARD_ROOT=/workspace/ABC/processed/vqvae_patch_shards
    export NS_OUTBASE=/workspace/ABC/processed/train_outputs
    export NS_OUT=newscheme_vqvae_sharded_recovery
    export NS_VQ_SAMPLES=300000
    export NS_VQ_EPOCHS=120
    export NS_VQ_BS=128
    export NS_VQ_COMPLEX_FRACTION=0.4
    export NS_VQ_CURVED_FRACTION=0.35
    export NS_VQ_MAX_SOURCE_FACES=50
    export NS_VQ_MAX_SOURCE_EDGES=150
    python breparg_improvements/train.py --stage vqvae

## Validation and Acceptance

The parsed shard layer is accepted when every intended chunk has a parsed shard file, the verifier reports `status: VERIFIED`, the manifest records the same source count as the original chunk directory, and the extracted chunk directory has been deleted only after verification.

The patch shard layer is accepted when `tools/build_vqvae_patch_shards.py` writes `_summary.json` with nonzero `patch_shards`, `patches`, `surfaces`, and `edges`, and `breparg_improvements/train.py --stage vqvae` logs `VQ patch-shard sampling selected=...` instead of scanning parsed `.pkl` files.

The server preparation documentation is accepted when a user can read `docs/v13_sharded_dataset_operator_guide.md` and know how to build parsed shards locally, upload them, build patch shards on AutoDL, set training environment variables, and resume safely after interruption.

## Idempotence and Recovery

`tools/build_parsed_shards.py --resume` is safe to rerun. If a verified shard already exists, it skips rebuilding and can still delete the extracted chunk directory if `--delete-after-verify` is set. Temporary shard files use `.tmp` suffix and are replaced atomically after writing. If a run fails before replacement, delete the `.tmp` file and rerun the same command.

Do not delete original zip archives unless a separate decision is made and verified. The current safe deletion scope is only `ABC/processed/abc_parsed_full/abc_XXXX` directories after a matching parsed shard has been verified.

## Artifacts and Notes

Initial extracted chunk sizes:

    abc_0000: 5,945 files, 6,895,935,497 bytes
    abc_0001: 6,514 files, 6,979,303,822 bytes
    abc_0002: 6,780 files, 7,151,947,792 bytes
    abc_0003: 6,873 files, 6,948,059,155 bytes
    abc_0004: 6,779 files, 6,845,874,694 bytes

The local default Python can build `.pkl.zst` parsed shards. The conda training environment currently needs `pip install zstandard` before reading `.zst` shards, or the tools can be run with `--compression gzip` for stdlib-only compressed shards.

## Interfaces and Dependencies

`breparg_improvements/sharded_data.py` provides `open_shard_writer(path, compression, level)`, `open_shard_reader(path)`, `dump_shard_record(handle, record)`, `iter_shard_records(path)`, `PARSED_SHARD_FORMAT`, and `PATCH_SHARD_FORMAT`.

`tools/build_parsed_shards.py` provides `build_chunk_shard(chunk_dir, parsed_root, shard_root, compression, compression_level, resume, delete_after_verify)` and a CLI for chunk conversion.

`tools/verify_parsed_shards.py` verifies parsed shard payload hashes and can optionally deep-unpickle records in an environment with numpy.

`tools/build_vqvae_patch_shards.py` reads parsed shards and writes patch shards with records of type `vq_patch`.

`breparg_improvements/vqvae_sampling.py` provides `collect_vqvae_patch_shard_records(paths, cap, ...)`.

`breparg_improvements/train.py` reads patch shards when `NS_VQ_PATCH_SHARD_ROOT` or `NS_VQ_PATCH_SHARDS` is set. Server training should invoke `python breparg_improvements/train.py --stage vqvae` rather than `--stage all` when using patch shards.

Revision note 2026-07-08: Created this ExecPlan after discovering the local file-count and disk-space constraints. The plan records why parsed shards store raw pickle bytes and why VQ patch shards are the direct training input.

Final note 2026-07-08: Completed and verified all 100 parsed shards at `C:\V13_abc_parsed_shards`; this root is the local upload source for the server-side patch-shard build.
