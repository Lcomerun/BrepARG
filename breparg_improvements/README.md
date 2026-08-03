# BrepARG improvements — start here

Three drop-in improvements to BrepARG (`../BrepARG/`, CVPR 2026) and the scripts to train/validate them on ABC:
- **① FSQ** quantiser (replaces the VQ codebook) — `fsq_quantise.py`
- **② RCM** face ordering (replaces DFS) — `gnn_ordering.py`
- **③ Constrained decoding** (inference-time grammar) — `constrained_decoding.py`

Environment: `conda activate brepgen` (torch 2.8 / transformers / diffusers / occwl + OCC all present).
Heavy data/checkpoints live on **`/data/public/luol/breparg_data/`** — never write them to `/home` (99% full).

## File map (this is the whole project)

| File | What it is | You run it? |
|---|---|---|
| `fsq_quantise.py` | Method ① FSQ quantiser | no — imported |
| `gnn_ordering.py` | Method ② RCM (and optional GNN) ordering | no — imported |
| `constrained_decoding.py` | Method ③ constrained-decoding LogitsProcessor | no — imported |
| `test_all.py` | **Unit tests** for all 3 methods (synthetic, fast) | `python test_all.py` |
| `process_abc.py` | **Data step**: raw `.step` → parsed pkl (resumable, parallel, no-hang) | `python process_abc.py ...` |
| `train.py` | **Training**: parsed → FSQ+RCM sequences → FSQ-VQVAE → AR (+ HP sweeps) | `python train.py --stage ...` |
| `方案.md` | The original design/proposal (Chinese) | read |
| `HANDOFF.md` | Running status log across sessions — **read this first to catch up** | read |
| `docs/DATA_AND_PROCESSING.md` | Data-format answer + full-corpus processing plan (Track A/B) | read |
| `docs/REVIEW.md` | Scheme/code review + real-data validation numbers | read |
| `repro_outputs/<run>/` | Auto-generated run reports (train_report.json + SUMMARY.md) | output |

That's it — **3 runnable files** (`test_all.py`, `process_abc.py`, `train.py`), 3 imported method modules, and reference docs.

## Which file / how to prompt — common tasks

- "**Check the methods still work**" → I run `python test_all.py` (expect `通过: 58`). Just say *"run the unit tests"*.
- "**Train the new scheme on existing data**" → `train.py`. Say *"train the new scheme on N parsed CADs"* (I set `NS_N`, `NS_OUT`, epochs). Reuses `abc_parsed_50c`; no re-parse.
- "**Hyperparameter test**" → `train.py --stage vqsweep` (VQ-VAE) / `--stage ar_sweep` (AR). Say *"sweep AR/VQVAE hyperparameters"*.
- "**Process more raw ABC data**" → `process_abc.py`. Say *"parse chunks 0-9 of raw ABC"* (it's the multi-day 1M job; do it chunk-range by range).
- "**Generate B-reps / measure Valid**" → needs OCC + a trained model; say *"generate with constrained decoding from <run>"* (still TODO — see HANDOFF).
- "**What's the status / what was decided**" → read `HANDOFF.md` (or say *"summarize HANDOFF"*).

When in doubt, point me at **`train.py`** (everything model-side) or **`process_abc.py`** (everything data-side), and I'll pick the stage.

## Quick commands

    conda activate brepgen
    cd breparg_improvements

    python test_all.py                                   # 1) unit tests (58 checks)

    # 2) train new scheme on existing parsed data (Track A, no re-parse). Scale via NS_* env vars:
    NS_OUT=myrun NS_N=20000 NS_VQ_EPOCHS=80 \
      CUDA_VISIBLE_DEVICES=1 python train.py --stage all
    # stages: split | vqvae | sequence | ar | vqsweep | ar_sweep   (run all or one)

    # 3) (optional) grow the parsed pool from raw ABC (multi-day):
    python process_abc.py --chunks 0-9 --out /data/public/luol/breparg_data/abc_parsed_full --workers 48

Key fact: raw→parsed is identical to the old scheme, so existing `abc_parsed_50c` (131,857 CADs) is reusable;
the new scheme only regenerates **sequences + VQ-VAE**. Details in `docs/DATA_AND_PROCESSING.md`.
