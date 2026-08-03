# V13 parsed shard + VQ patch shard operator guide

This guide explains how to replace hundreds of thousands of small parsed `.pkl` files with a server-friendly two-layer dataset:

- parsed shards: compressed chunk files that preserve the original parsed `.pkl` bytes and source metadata.
- VQ patch shards: compressed patch files generated from parsed shards and read directly by VQ-VAE training.

Use this guide when preparing V13 for AutoDL or any server with a strict file-count limit.

## Why this format

AutoDL can limit practical storage to about 200,000 files. The full ABC parsed pool has far more small files than that. Uploading the extracted tree is fragile and slow.

Do not merge the whole dataset into one giant pickle. A single giant file is difficult to verify, easy to corrupt, and likely to require too much RAM. Instead, keep shard files small enough to verify and retry. The recommended layout is one parsed shard per ABC chunk, followed by many VQ patch shards for training.

## Directory layout

Local source state:

```text
D:\luolin\V13\ABC\processed\abc_parsed_full_archives\
  abc_0000_parsed.zip
  ...
  abc_0099_parsed.zip

D:\luolin\V13\ABC\processed\abc_parsed_full\
  abc_0000\
  abc_0001\
  ...
```

Local parsed-shard output:

```text
C:\V13_abc_parsed_shards\
  parsed_abc_0000.pkl.zst
  parsed_abc_0001.pkl.zst
  ...
  _manifest.jsonl
```

The final local parsed-shard root for the 2026-07-08 run is `C:\V13_abc_parsed_shards`. It was moved off the repository's D: drive so the machine could retain both the original zip archives and the verified parsed shards.

Server patch-shard output:

```text
/workspace/ABC/processed/vqvae_patch_shards/
  vq_patch_shard_0000.pkl.zst
  vq_patch_shard_0001.pkl.zst
  ...
  _manifest.jsonl
  _summary.json
```

## Data model

A parsed shard stores a header record followed by many `parsed_source` records. Each `parsed_source` record contains:

```text
chunk_id
source_relpath
source_name
source_bytes
source_sha256
payload
```

`payload` is the original parsed `.pkl` file as raw bytes. The local parsed-shard build step does not unpickle it, so it can run without numpy.

A VQ patch shard stores a header record followed by many `vq_patch` records. Each patch record contains:

```text
source_path
chunk_id
kind
array
curvature_score
n_faces
n_edges
is_complex_source
```

VQ-VAE training reads the VQ patch shard records directly.

## Local prerequisites

From `D:\luolin\V13`, check whether the default Python can write zstd shards:

```powershell
python -c "import zstandard; print('zstandard ok')"
```

If this fails, either install it:

```powershell
python -m pip install zstandard
```

or use `--compression gzip` in the commands below. Gzip is slower and usually larger, but it uses only Python's standard library.

## Build parsed shards for already extracted chunks

At the time this guide was written, five chunk directories were already extracted:

```text
ABC\processed\abc_parsed_full\abc_0000
ABC\processed\abc_parsed_full\abc_0001
ABC\processed\abc_parsed_full\abc_0002
ABC\processed\abc_parsed_full\abc_0003
ABC\processed\abc_parsed_full\abc_0004
```

Run this command from the repository root:

```powershell
cd D:\luolin\V13
python tools\build_parsed_shards.py `
  --parsed-root ABC\processed\abc_parsed_full `
  --shard-root C:\V13_abc_parsed_shards `
  --manifest C:\V13_abc_parsed_shards\_manifest.jsonl `
  --chunks 0-4 `
  --compression zstd `
  --compression-level 10 `
  --resume `
  --delete-after-verify
```

The command writes one shard per chunk, verifies it, appends a row to `_manifest.jsonl`, and deletes the extracted `abc_XXXX` directory only after verification passes.

After the command, verify the new shards:

```powershell
python tools\verify_parsed_shards.py `
  C:\V13_abc_parsed_shards\parsed_abc_0000.pkl.zst `
  C:\V13_abc_parsed_shards\parsed_abc_0001.pkl.zst `
  C:\V13_abc_parsed_shards\parsed_abc_0002.pkl.zst `
  C:\V13_abc_parsed_shards\parsed_abc_0003.pkl.zst `
  C:\V13_abc_parsed_shards\parsed_abc_0004.pkl.zst `
  --output local_reports\v13_parsed_shards_verify_0000_0004_20260708.json
```

Expected result:

```text
"status": "VERIFIED"
```

Also confirm that the extracted directories are gone:

```powershell
Get-ChildItem ABC\processed\abc_parsed_full -Directory
```

## Process the remaining chunks one at a time

For each remaining chunk, extract the original zip archive, build a parsed shard, verify it, and delete the extracted directory. The safest path is to use `tools/run_parsed_shard_cycle.py`, which performs those actions one chunk at a time and leaves the original zip archives untouched.

Example for `abc_0005` through `abc_0009`:

```powershell
cd D:\luolin\V13
python tools\run_parsed_shard_cycle.py `
  --archive-root ABC\processed\abc_parsed_full_archives `
  --parsed-root ABC\processed\abc_parsed_full `
  --shard-root C:\V13_abc_parsed_shards `
  --manifest C:\V13_abc_parsed_shards\_manifest.jsonl `
  --chunks 5-9 `
  --compression zstd `
  --compression-level 10 `
  --resume `
  --delete-after-verify
```

Repeat with later ranges until `99`.

Important: keep only one newly extracted chunk on disk at a time unless you have verified there is enough free space.

## Disk-space rule

Before extracting a new chunk, check free space:

```powershell
Get-PSDrive D
```

A chunk can temporarily require roughly:

```text
zip archive already present: about 1.6-2.0 GB
extracted chunk: about 6-8 GB
new parsed shard: likely 1.5-2.5 GB, measure after the first shard
```

If keeping both all original zip archives and all parsed shards does not fit locally, do not delete original zip archives automatically. Choose one of these safer options:

- write parsed shards to another local drive with more space,
- upload verified parsed shards to the server and remove local parsed shards after upload verification,
- ask before deleting original zip archives.

## Upload parsed shards to AutoDL

After parsed shards are built, upload this directory:

```text
C:\V13_abc_parsed_shards
```

to:

```text
/workspace/ABC/processed/abc_parsed_shards
```

On Linux, verify the file count is low:

```bash
find /workspace/ABC/processed/abc_parsed_shards -type f | wc -l
```

For 100 chunks, expect about 101 files: 100 parsed shards plus `_manifest.jsonl`.

Install zstandard in the training environment if using `.pkl.zst`:

```bash
python -m pip install zstandard
```

Then verify the parsed shards:

```bash
cd /workspace/V13
python tools/verify_parsed_shards.py /workspace/ABC/processed/abc_parsed_shards/parsed_abc_*.pkl.zst \
  --output /workspace/V13/local_reports/v13_parsed_shards_verify_server.json
```

Expected result:

```text
"status": "VERIFIED"
```

## Build VQ patch shards on the server

Run this on AutoDL after parsed shards are uploaded:

```bash
cd /workspace/V13
python tools/build_vqvae_patch_shards.py \
  --parsed-shard-root /workspace/ABC/processed/abc_parsed_shards \
  --patch-shard-root /workspace/ABC/processed/vqvae_patch_shards \
  --manifest /workspace/ABC/processed/vqvae_patch_shards/_manifest.jsonl \
  --compression zstd \
  --compression-level 6 \
  --patches-per-shard 100000 \
  --complex-min-faces 12 \
  --complex-min-edges 20 \
  --max-source-faces 50 \
  --max-source-edges 150
```

Check the summary:

```bash
cat /workspace/ABC/processed/vqvae_patch_shards/_summary.json
```

Expected fields should be nonzero:

```text
patch_shards
patches
surfaces
edges
```

## Train VQ-VAE directly from patch shards

Use `--stage vqvae`, not `--stage all`, because patch shards bypass the parsed-pool split step.

Example:

```bash
cd /workspace/V13
export NS_VQ_PATCH_SHARD_ROOT=/workspace/ABC/processed/vqvae_patch_shards
export NS_OUTBASE=/workspace/ABC/processed/train_outputs
export NS_OUT=newscheme_vqvae_sharded_recovery
export NS_VQ_SAMPLES=300000
export NS_VQ_EPOCHS=120
export NS_VQ_BS=128
export NS_VQ_LR=1e-5
export NS_DISABLE_AMP_VQVAE=1
export NS_VQ_COMPLEX_FRACTION=0.4
export NS_VQ_CURVED_FRACTION=0.35
export NS_VQ_MAX_SOURCE_FACES=50
export NS_VQ_MAX_SOURCE_EDGES=150
export NS_VQ_COMPLEX_LOSS_WEIGHT=1.15
export NS_VQ_CURVED_LOSS_WEIGHT=1.5

python breparg_improvements/train.py --stage vqvae
```

The log should include:

```text
VQ patch-shard sampling selected=...
```

That proves the trainer is reading patch shards rather than scanning extracted `.pkl` files.

## Resume and recovery

If parsed shard building stops:

```powershell
python tools\build_parsed_shards.py ... --resume --delete-after-verify
```

The tool skips verified existing shards. If an extracted chunk still exists and the shard verifies, it can delete the extracted chunk during the resumed run.

If a `.tmp` shard remains after a crash, delete only the `.tmp` file and rerun the same command:

```powershell
Remove-Item C:\V13_abc_parsed_shards\*.tmp
```

Do not delete extracted chunk directories manually unless the matching parsed shard has been verified.

## Cleanup policy

Safe to delete automatically:

- `ABC\processed\abc_parsed_full\abc_XXXX` after `parsed_abc_XXXX.pkl.zst` verifies.
- `.tmp` shard files after a failed run, before rerunning.

Do not delete automatically:

- `ABC\processed\abc_parsed_full_archives\abc_XXXX_parsed.zip`.
- model checkpoints.
- `local_reports` evidence files.
- paper artifacts.

For project document cleanup, first create a manifest that lists retained canonical reports and stale reports. Delete stale reports only after the manifest is reviewed or after a command verifies the retained server-start guide, transfer manifest, and sharding guide exist.

## Completion checklist

The compression task is complete when:

- 100 parsed shard files exist.
- `tools/verify_parsed_shards.py` reports `VERIFIED` for all 100 parsed shards.
- `ABC\processed\abc_parsed_full` has no extracted `abc_XXXX` directories left.
- AutoDL has the parsed shards below the file-count limit.
- VQ patch shards are built on the server.
- VQ-VAE training logs `VQ patch-shard sampling selected=...`.

## Current local completion snapshot

As of 2026-07-08, the local parsed-shard layer is complete and verified:

```text
shard_root: C:\V13_abc_parsed_shards
summary: local_reports\v13_parsed_shards_manifest_croot_0000_0099_20260708.json
status: VERIFIED
shards: 100
chunks: abc_0000 through abc_0099
total_sources: 681406
total_payload_bytes: 648659550518
total_shard_bytes: 99966076379
extracted chunk directories: 0
temporary shard files: 0
original zip archives retained: 100
```
