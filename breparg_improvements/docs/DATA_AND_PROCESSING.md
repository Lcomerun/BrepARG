# Data format & processing (ABC, new scheme)

This merges the old `DATA_FORMAT_COMPARISON.md` + `UNIFIED_PROCESSING_PLAN.md`.

## Does the new scheme need a different data format than the existing `/data/.../breparg_data`?

The pipeline has three artifacts:

    raw .step ──process_brep.py──▶ parsed pkl ──2sequence.py + SE-VQVAE──▶ AR sequences ──▶ AR model
                 (geometry)          (per-CAD)         (tokens + ordering)      (token ids)

- **raw → parsed: NO difference.** None of the three method modules (`fsq_quantise.py`, `gnn_ordering.py`,
  `constrained_decoding.py`) touches the parser; parsed pkls are scheme-agnostic geometry
  (`surf_ncs, edge_ncs, surf_bbox_wcs, edge_bbox_wcs, edgeFace_adj, faceEdge_adj, …`). **So the 131,857
  parsed CADs in `/data/public/luol/breparg_data/abc_parsed_50c` are reusable — no re-parse needed.**
- **parsed → sequences + VQ-VAE: DIFFERENCE.** The stored `abc_sequences_50c.pkl` is the OLD scheme:
  geometry tokens from the learnable VQ codebook (`VectorQuantiser`) and face order from DFS. The new
  scheme uses FSQ tokens (different ids) + RCM order (different order; on 400 real CADs RCM differs from
  DFS on 394/400 and lowers MLA 0.88×). So the stored sequences and the SE-VQVAE checkpoint are old-scheme
  and **must be regenerated/retrained** for the new scheme.

Net: reuse parsed geometry, regenerate sequences + VQ-VAE.

## The full corpus

- Raw ABC: `/data/public/luol/ABC/step/abc_0000.._0099_step_v00/<model>/*.step` = **100 chunks × 10,000 ≈
  1,000,000 models, 1012 GB**.
- Already parsed (flat): `abc_parsed_50c` = **131,857** pkls (≤50-face subset, 54 GB).
  - NOTE (observed 2026-06-20): this pool dropped to ~91,878 at some point (no active deleter found; now
    stable). Training tolerates missing files (skips with a WARN), but prefer `abc_parsed_100c` going forward.
- **Being parsed now (full corpus)**: `abc_parsed_100c` — the user's `process_brep.py --output abc_parsed_100c
  --chunk_start 0 --chunk_end 99` run, **per-chunk subdir layout** `abc_parsed_100c/abc_XXXX/*.pkl`.
- `/data` free: ~520 GB. `/home` is 99% full — **never write checkpoints/parsed to /home**.

### VERIFIED: the in-progress `abc_parsed_100c` only needs post-processing (no re-parse, no scheme change)

Field-level comparison of an `abc_parsed_100c` pkl vs an `abc_parsed_50c` pkl (2026-06-20):

- **Identical 11 keys, identical dtypes**: `surf_ncs (f32, N×32×32×3)`, `edge_ncs (f32, M×32×3)`,
  `surf_bbox_wcs`, `edge_bbox_wcs`, `edgeFace_adj (i64)`, `faceEdge_adj (list)`, `corner_*`, `edge_wcs`,
  `surf_wcs`, `edgeCorner_adj`. Only per-model face/edge counts differ (different CADs, not format).
- The **only** difference vs 50c is the directory layout (subdirs vs flat).

⇒ The new scheme touches nothing in `process_brep.py`. Once `abc_parsed_100c` is parsed, we **only run the
post-processing** (regenerate FSQ+RCM sequences + retrain FSQ-VQVAE + AR). `train.py` reads the subdir layout
via `NS_POOL` + recursive glob (added 2026-06-20):

    NS_POOL=/data/public/luol/breparg_data/abc_parsed_100c \
    NS_OUT=newscheme_full NS_N=<count> NS_VQ_SAMPLES=300000 NS_VQ_EPOCHS=200 NS_AR_EPOCHS=120 \
      CUDA_VISIBLE_DEVICES=<free gpu> python train.py --stage all

## Two tracks to build the new-scheme dataset

### Track A (recommended) — reuse existing parsed, no re-parse

    conda activate brepgen; cd breparg_improvements
    # reuse the flat 50c pool ...
    NS_OUT=newscheme_full NS_N=131857 NS_VQ_SAMPLES=300000 NS_VQ_EPOCHS=200 NS_AR_EPOCHS=120 \
      CUDA_VISIBLE_DEVICES=1 python train.py --stage all
    # ... or the full subdir pool once abc_parsed_100c is ready:
    NS_POOL=/data/public/luol/breparg_data/abc_parsed_100c NS_OUT=newscheme_full NS_N=999999 \
      NS_VQ_SAMPLES=300000 NS_VQ_EPOCHS=200 NS_AR_EPOCHS=120 \
      CUDA_VISIBLE_DEVICES=1 python train.py --stage all
    # outputs -> /data/public/luol/breparg_data/newscheme_full/

Budget on one free GPU: FSQ-VQVAE a few hours; sequence regen ~1–2 h; AR ~several hours. ~1 GPU-day, resumable per stage.

**GPU note (2026-06-20):** GPUs here are shared and fluctuate. AR OOM'd on a GPU that another user's job grew
to fill (only ~1 GiB free); retry on a GPU with ≥10 GiB free and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Each stage is independently resumable, so on OOM just re-run that one stage (`--stage ar`) on a freer GPU.

### Track B (optional) — grow the parsed pool toward 1M (multi-day OCC parse)

    python process_abc.py --chunks all --out /data/public/luol/breparg_data/abc_parsed_full --workers 48
    # resumable (skips existing pkls); do it chunk-range by chunk-range, e.g. --chunks 0-9 then 10-19 ...
    # then point train.py's PARSED_POOL at abc_parsed_full and run Track A over it.

## Parser robustness (validated on 400 models) — "correct + won't get stuck"

- **No hangs**: per-file SIGALRM timeout (default 60s). 3 pathological files auto-skipped in the 400 sample
  (ok=244 multi=131 filtered=22 timeout=3 error=0).
- **Correct**: new parsed fields identical to `abc_parsed_50c`; mismatches flagged `badfields`, not written.
- **Resumable** (re-run all-skip), **atomic** (temp+rename, no half pkl), **memory-stable** (`maxtasksperchild`).
- ~3 models/s on 16 workers → full 1M ≈ ~4 days (split by chunk range / machines).
