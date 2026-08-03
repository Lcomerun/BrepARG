# Local ABC Chunk4 Storage-Safe Processing

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md` in the repository root. The work matters because the local machine cannot comfortably hold all 100 ABC chunks decompressed plus all processed artifacts. After this plan, a user can inspect the local skills inventory, understand which data format is safe for VQVAE training, and see a bounded chunk4 processing proof without filling the disk.

## Purpose / Big Picture

The repository contains BrepARG and related improvements for ABC B-rep data. The local `ABC` folder currently contains compressed chunks and one decompressed chunk4 directory. The goal is to avoid creating a second huge copy of the dataset while still proving that local processing can work. The observable result is a small set of reports under `local_reports` and `processed_local/abc_0004_smoke`, plus a clear recommendation about storage.

## Progress

- [x] (2026-06-24 23:45 +08:00) Read `AGENTS.md` and `PLANS.md`; confirmed significant data processing should use an ExecPlan.
- [x] (2026-06-24 23:50 +08:00) Mapped top-level repository contents: `BrepARG`, `breparg_improvements`, `ABC`, `AGENTS.md`, and `PLANS.md`.
- [x] (2026-06-24 23:55 +08:00) Read the project documentation and data pipeline notes in `BrepARG/README.md`, `breparg_improvements/README.md`, and `breparg_improvements/docs/DATA_AND_PROCESSING.md`.
- [x] (2026-06-24 23:58 +08:00) Confirmed chunk4 has 8,175 `.step` files totaling about 8.98 GB, not a full 10,000-file chunk.
- [x] (2026-06-24 23:59 +08:00) Confirmed local disk free space is about 149.8 GB on `D:\`.
- [x] (2026-06-25 00:04 +08:00) Confirmed `brepgen_env` has `numpy`, `torch`, `occwl`, `OCC`, `diffusers`, `transformers`, and `tqdm`; only `shutup` is missing.
- [x] (2026-06-25 00:08 +08:00) Added `tools/inventory_skills.py` and `tools/local_abc_chunk4_audit.py`.
- [x] (2026-06-25 00:10 +08:00) Generated skills inventory into `local_reports/local_skills_inventory.md` and `.json`.
- [x] (2026-06-25 00:12 +08:00) Ran chunk4 audit-only scan and chunk4 smoke parsing into `processed_local/abc_0004_smoke`.
- [x] (2026-06-25 00:17 +08:00) Ran `.npz` compression probe on the 10 successful parsed pkls.
- [x] (2026-06-25 00:20 +08:00) Wrote `local_reports/storage_plan_and_chunk4_result.md` with final evidence and recommendations.

## Surprises & Discoveries

- Observation: The workspace root is not a git repository, so changes cannot be committed here.
  Evidence: `git status --short` returned `fatal: not a git repository`.
- Observation: `ABC` contains `.7z` files for chunks 0, 1, and 2 in addition to the decompressed chunk4 directory.
  Evidence: `ABC` contains `abc_0000_step_v00.7z`, `abc_0001_step_v00.7z`, `abc_0002_step_v00.7z`, and `abc_0004_step_v00`.
- Observation: Chunk4 has outlier files, including a zero-byte STEP and several STEP files larger than 100 MB.
  Evidence: size scan found one zero-byte file and files up to 277,855,398 bytes.
- Observation: The base Python is unsuitable for this project, but `brepgen_env` is close to usable.
  Evidence: base Python lacks `numpy`, `torch`, `occwl`, and `OCC`; `brepgen_env` has all of them but lacks `shutup`.
- Observation: Smoke parsing succeeded without field mismatches.
  Evidence: `processed_local/abc_0004_smoke/summary.json` reports `ok=10`, `badfields=0`, and `error=0` for the 20 selected files.
- Observation: Parsed pkl can be much larger than the source STEP, but compresses well.
  Evidence: The 10 successful smoke samples had 863,925 input STEP bytes and 9,666,622 parsed pkl bytes, an 11.19x expansion. Gzip-6 probe size was 2,744,261 bytes, 0.284x of pkl.
- Observation: `.npz` compressed size was effectively the same as gzip pkl for this sample, but it does not eliminate object serialization for ragged list fields.
  Evidence: `npz_probe.jsonl` totals 2,766,474 bytes for 9,666,622 bytes of pkl, or 0.286x. The probe needed object arrays for list fields such as `faceEdge_adj`.

## Decision Log

- Decision: Keep parsed outputs as one pkl per CAD for the initial local proof.
  Rationale: Existing VQVAE, deduplication, and sequence code use `pickle.load()` on per-CAD dictionaries. This preserves compatibility and resumability.
  Date/Author: 2026-06-25 / Codex.
- Decision: Do not merge processed CADs into one large pkl.
  Rationale: One large pkl increases corruption blast radius, makes skip/resume awkward, and requires loader changes for partial reads.
  Date/Author: 2026-06-25 / Codex.
- Decision: Use manifest files and smoke parsing before proposing a new storage backend such as npz, hdf5, or zarr.
  Rationale: The first risk is local disk usage and parser viability. Format migration should only happen after measuring real parsed size and confirming training loader changes.
  Date/Author: 2026-06-25 / Codex.
- Decision: Inject a no-op `shutup` module inside the smoke script rather than installing packages.
  Rationale: `shutup` only suppresses warnings in `process_brep.py`; avoiding installation keeps the local environment unchanged.
  Date/Author: 2026-06-25 / Codex.
- Decision: Recommend per-CAD pkl as the active training format and compressed archives or a later `.pkl.gz` adapter for storage relief.
  Rationale: This avoids breaking `pickle.load()`-based readers while acknowledging measured compression savings of about 72%.
  Date/Author: 2026-06-25 / Codex.

## Outcomes & Retrospective

The work produced a local skills inventory, a chunk4 audit, 10 successful parsed pkl samples, compression probes, and a human-readable storage report. Evidence supports a conservative storage plan: process one chunk at a time, preserve per-CAD pkl compatibility first, emit manifests and summaries, and only introduce compressed alternate formats after implementing and testing a loader adapter.

## Context and Orientation

`BrepARG/process_data/process_brep.py` parses a STEP file into a Python dictionary. A STEP file is a CAD exchange file. A parsed pkl is a Python pickle file containing numpy arrays and adjacency lists for one CAD model. The VQVAE training code reads surface and edge arrays from these dictionaries. `BrepARG/dataset.py` defines `CombinedData`, which loads deduplicated surface and edge pickle files for VQVAE training. `BrepARG/2sequence.py` loads the parsed CAD dictionaries to create autoregressive token sequences. `breparg_improvements/train.py` also expects parsed pkl files and recursively globs either flat or one-level subdirectory layouts.

The fields that matter for compatibility are `surf_ncs`, `edge_ncs`, `surf_bbox_wcs`, `edge_bbox_wcs`, `edgeFace_adj`, `faceEdge_adj`, `surf_wcs`, `edge_wcs`, `corner_wcs`, `edgeCorner_adj`, and `corner_unique`. The geometry arrays are expected as float32 arrays. Changing these field names or converting values to a quantized dtype would require loader and possibly training changes.

## Plan of Work

First, create a local skills inventory so the user has an easy reference for later prompts. Second, audit chunk4 file counts and sizes. Third, use `brepgen_env` to parse a small number of non-empty STEP files smaller than 2 MB into per-CAD pkls. Fourth, write a manifest, compression probe, and summary report. Finally, compare observed pkl size to source STEP size and explain whether a different storage container is worth implementing.

## Concrete Steps

Run from `D:\luolin\V13`.

Generate the local skills inventory:

    conda run -n brepgen_env python tools/inventory_skills.py

Expected output includes:

    skills=<number>
    markdown=D:\luolin\V13\local_reports\local_skills_inventory.md
    json=D:\luolin\V13\local_reports\local_skills_inventory.json

Run the chunk4 smoke audit:

    conda run -n brepgen_env python tools/local_abc_chunk4_audit.py --limit 20 --max-step-bytes 2000000

Expected output is a JSON summary naming:

    D:\luolin\V13\processed_local\abc_0004_smoke\summary.json
    D:\luolin\V13\processed_local\abc_0004_smoke\manifest.jsonl
    D:\luolin\V13\processed_local\abc_0004_smoke\parsed_pkl

## Validation and Acceptance

The skills inventory is accepted when `local_reports/local_skills_inventory.md` exists and includes every discovered `SKILL.md` file under the local Codex and plugin skill roots.

The chunk4 smoke proof is accepted when `processed_local/abc_0004_smoke/summary.json` exists, `parse_counts.ok` is greater than zero, and the `manifest.jsonl` rows for successful parses include all expected parsed fields. It is acceptable for some files to be filtered or fail because ABC contains multi-solid, empty, very large, or parser-hostile STEP files.

The storage recommendation is accepted when it explicitly states whether merged pkl, per-CAD pkl, compressed pkl, or another format should be used, and explains the training compatibility impact for VQVAE.

## Idempotence and Recovery

The skills inventory can be regenerated safely. The chunk4 audit can also be rerun safely. It writes only under `processed_local/abc_0004_smoke` and does not delete or modify source STEP files. If the smoke run fails because a STEP file cannot be parsed, rerun with a lower `--limit`, a lower `--max-step-bytes`, or `--audit-only` to produce the file-size report without parsing.

## Artifacts and Notes

Primary artifacts are:

    D:\luolin\V13\local_reports\local_skills_inventory.md
    D:\luolin\V13\local_reports\local_skills_inventory.json
    D:\luolin\V13\processed_local\abc_0004_smoke\step_file_sizes.csv
    D:\luolin\V13\processed_local\abc_0004_smoke\summary.json
    D:\luolin\V13\processed_local\abc_0004_smoke\manifest.jsonl
    D:\luolin\V13\processed_local\abc_0004_smoke\compression_probe.jsonl
    D:\luolin\V13\processed_local\abc_0004_smoke\parsed_pkl\*.pkl

## Interfaces and Dependencies

Use `conda run -n brepgen_env python` for parsing because that environment contains `numpy`, `torch`, `occwl`, `OCC`, `diffusers`, `transformers`, and `tqdm`. The script imports `BrepARG/process_data/process_brep.py` and `occwl.io.load_step`. It preserves `pickle.dump(data, protocol=pickle.HIGHEST_PROTOCOL)` output as the compatibility format.

Revision note: Initial ExecPlan created to guide the local storage-safe chunk4 proof and capture decisions before running data processing.

Revision note: Updated after running the skills inventory, chunk4 smoke parse, gzip probe, and npz probe. Added final evidence and pointed to the result report.
