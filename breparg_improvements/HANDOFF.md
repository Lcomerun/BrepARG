# Session Handoff — BrepARG improvements (V13)

Read this first in a new session to continue without re-deriving context.
Companion docs: `README.md` (integration), `方案.md` (design + pitfalls), `SKILLS.md` (../SKILLS.md),
`repro_outputs/` (real-data evidence).

## Goal

Improve BrepARG (CVPR 2026, autoregressive B-rep generation) on three axes, raising generated-B-rep
Valid rate, via **drop-in** modules that don't rewrite the upstream pipeline:
- 方案① FSQ quantiser (replaces VQ codebook)
- 方案② RCM / geometry-GNN face ordering (replaces DFS ordering)
- 方案③ topology-constrained decoding (inference-time LogitsProcessor)

## Layout (simplified 2026-06-19 — see README.md for the full file map + "which file/how to prompt")

    V13/
    ├── BrepARG/                     upstream source + paper PDF (READ-ONLY, do not edit)
    ├── ABC/                         raw ABC .step (small local batch; full 1M is on /data, see below)
    ├── SKILLS.md                    skills catalogue + per-session selections
    └── breparg_improvements/        OUR work (git-tracked)
        ├── README.md                ⭐ THE guide (file map + how to prompt)
        ├── 方案.md  HANDOFF.md(this)
        ├── fsq_quantise.py gnn_ordering.py constrained_decoding.py   methods ①②③ (imported)
        ├── test_all.py              58 synthetic unit tests  → `python test_all.py`
        ├── process_abc.py           raw .step → parsed (resumable/parallel/no-hang)
        ├── train.py                 parsed → FSQ+RCM seq → FSQ-VQVAE → AR (+sweeps)
        ├── docs/                    DATA_AND_PROCESSING.md, REVIEW.md (merged refs)
        └── repro_outputs/<run>/     auto run reports (train_report.json + SUMMARY.md)

    Only 3 runnable files: test_all.py / process_abc.py / train.py.

## Environment & real assets (CRITICAL — heavy data lives on /data, not /home)

- Conda env: **`brepgen`** = `/home/luolin/miniforge3/envs/brepgen/bin/python`
  (torch 2.8.0+cu128, transformers 4.57, diffusers 0.35.1, einops). README pins torch 2.1.2 but 2.8 works.
- `occ` env has OCC but **no torch** → OCC STEP reconstruction not runnable in brepgen yet.
- Real data root: `/data/public/luol/breparg_data/`
  - `abc_parsed_50c/` — 131,857 parsed CAD pkls (surf_ncs/edge_ncs/edgeFace_adj/surf_bbox_wcs/…)
  - `abc_sequences_50c.pkl` — 830M real AR sequences (train 49010 / val 3553 / test 3485). Vocab: face 50, **SE codebook 8192**, bbox 2048, vocab_size 10294, 4 SE tok/elem, START 10290/SEP 10291/END 10292/PAD 10293.
  - `checkpoint/se_abc_50c/abc_se_vqvae_epoch_*.pt` — trained SE VQ-VAE (use epoch_85; ABC codebook 8192)
  - `checkpoints/ar_abc_50c/abc_ar_vqvae_best_model.pt` (+ HF dir) — trained AR
  - A `train_ar.py --env ar_abc_50c` was running on GPU during this session → **validation ran on CPU** to avoid disturbing it.
- Sample cache: `repro_outputs/_abc_seq_sample.pkl` (300 real seqs + meta) so you don't reload 830M.

## Key facts learned (don't re-discover)

- ABC uses **se_codebook_size = 8192** (config.json `abc`, 2sequence.py:211, utils.load_se_vqvae_model).
  deepcad uses 4096. FSQ for ABC needs `prod(levels)=8192`, e.g. `fsq_levels=(8,8,8,16)` (still 4 tok/elem).
- Pipeline: encoder → quant_conv(128→64) → quantize(2×2 → 4 tokens) → post_quant_conv → decoder.
- Face ordering happens **offline** in 2sequence.py before AR training → ordering GNN can only be
  pre-trained then dropped in; it CANNOT be end-to-end trained with the AR loss.
- Grammar: `START face_block* SEP edge_block* END`; face_block = [bbox×6][geo×4][idx×1] (11);
  edge_block = [idx×2][bbox×6][geo×4] (12).

## Status

DONE (session 2026-06-18 #1):
- Evaluated 方案.md against real upstream code; fixed the one failing test (stale `/mnt/project` path).
  `python test_all.py` → **58/58 pass**.
- Organized code into `breparg_improvements/` package. Committed (commit `5de43dc`, branch master).
- Real ABC validation on existing assets (`python real_e2e_abc.py`):
  - 方案② VERIFIED: 400 real CADs, RCM MLA 307.1 vs DFS 348.5 (**0.88×**, wins 329/400). Margin smaller than synthetic (0.65×) — real graphs less grid-like.
  - 方案③ VERIFIED: 120 real GT seqs, 103,506 steps, **0 wrong-masking** (real 8192 vocab).
  - 方案① PARTIAL: real SE-VQVAE encodes real geometry; FSQ drop-in gives 4 tok/elem in [0,8192). Full Valid needs retrain.

DONE (session 2026-06-18 #2) — **fresh-ABC end-to-end with the NEW scheme** (`python e2e_pipeline_abc.py --stage all`):
  Processed raw `ABC/*.step` from scratch and ran FSQ + RCM + constrained decoding through training. All 5 stages VERIFIED.
  Evidence: `repro_outputs/e2e/SUMMARY.md` + `e2e_report.json`. ExecPlan: `EXECPLAN_e2e_abc.md`.
  - Stage0 process: 128 parsed, split 102/12/14. Stage1 FSQ-VQVAE loss 0.173→0.121.
  - Stage2 RCM+FSQ sequences: 79 seqs, **0 out-of-vocab** (max 10292 < vocab 10294), 4 tok/elem — **fresh data fits the scheme**.
  - Stage3 AR CE 8.59→2.55. Stage4 constrained gen **8/8 grammar-valid**.
  - Gap: OCC reconstruction 0/8 — upstream `joint_optimize` hardcodes `.cuda()` (run was CPU) + tiny under-trained AR; needs GPU + real training, not a scheme issue.
  - Key: FSQ levels **(8,8,8,16)=8192** to match ABC se_codebook_size → zero offset surgery.

DONE (session 2026-06-18 #3) — **data-format check + 5K new-scheme training**:
  - **Task 1 (format diff?)**: raw→parsed is IDENTICAL (reuse `abc_parsed_50c`, no re-parse needed); but
    `abc_sequences_50c.pkl` + SE-VQVAE are old-scheme (VQ+DFS) and CANNOT be reused → regenerate.
    Evidence: `repro_outputs/DATA_FORMAT_COMPARISON.md`.
  - **Task 2 (review)**: `repro_outputs/CODE_SCHEME_REVIEW.md`; unit suite 58/58 (no regression).
  - **Bug fixed (real GPU+AMP)**: FSQ index packing overflowed in fp16 (codebook 8192 > fp16 exact-int 2048)
    → `one_hot` device assert. Fixed `fsq_quantise.py` to do FSQ math in fp32 + clamp. **CPU tests never
    caught this** — only real GPU training did. Also: `lr1e-3+(8,8,8,16)` NaNs under AMP → use lr3e-4 +
    grad-clip + NaN-skip (`train_newscheme_abc.py`).
  - **Task 3 (plan)**: `EXECPLAN_train_5k.md` (experiment-design structure + HP sweep).
  - **5K GPU1 training was a FEASIBILITY validation only** ("跑通验证"): proved the new scheme trains
    end-to-end on fresh data (FSQ-VQVAE val recon 0.157→0.00074; FSQ+RCM seqs out_of_vocab=0; AR overfits
    at 5K) AND caught the FSQ AMP overflow bug. The 5K model is NOT a production model →
    its heavy artifacts (`/data/.../newscheme_5k/`) + `EXECPLAN_train_5k.md` + `repro_outputs/newscheme_5k/`
    were **deleted** (user: keep only what's useful going forward). Durable outcomes kept: the AMP fix in
    `fsq_quantise.py`, `CODE_SCHEME_REVIEW.md`, `DATA_FORMAT_COMPARISON.md`. `train_newscheme_abc.py` kept
    as the reusable engine for the eventual full run (scale via NS_* env vars).
  - **Task 8**: background GPU0 job = old-scheme VQ+DFS AR **baseline** (user's own run, KEPT).

DONE (session 2026-06-18 #4) — **prep full ABC reprocessing (correct + no-hang)**:
  - Located full raw ABC: `/data/public/luol/ABC/step` = 100 chunks × 10000 ≈ **1,000,000 models (1012 GB)**.
  - `process_abc_unified.py` (resumable, parallel, **per-file SIGALRM timeout**, atomic writes,
    field-correctness check, `maxtasksperchild`). Validated on 400 models: 3 hang-prone files auto-skipped,
    parsed fields identical to `abc_parsed_50c`, re-run all-skip, 0 `.tmp` left. ~3/s → full 1M ≈ 4 days.
  - Plan: `repro_outputs/UNIFIED_PROCESSING_PLAN.md` — **Track A** (reuse 131,857 parsed → full FSQ-VQVAE +
    FSQ/RCM sequences + AR, ~1 GPU-day, no re-parse) vs **Track B** (grow parsed pool toward 1M, multi-day).
    Key: raw→parsed is identical to old scheme, so existing parsed is reusable; new scheme only reprocesses
    sequences + VQ-VAE. Awaiting user's go on which track/scale.

### How to resume the 5K training
    conda activate brepgen; cd V13/breparg_improvements
    CUDA_VISIBLE_DEVICES=1 python train_newscheme_abc.py --stage all   # or split|vqsweep|vqvae|sequence|ar
    # outputs: /data/public/luol/breparg_data/newscheme_5k/{fsq_vqvae_best.pt,sequences_fsq_rcm.pkl,ar_best.pt}

DONE (session 2026-06-19 #5) — **preliminary Track-A validation + AR HP test + file cleanup**:
  - Ran `train.py` on **20,000 reused parsed CADs** (Track A, no re-parse), GPU1, 89.7 min, all VERIFIED:
    FSQ-VQVAE val recon **0.122→0.00086**; FSQ+RCM sequences **10,629, out_of_vocab=0**; AR HP sweep.
  - **AR HP winner: d256/L8/lr5e-4 (val CE 0.868)** > d256/L6/lr1e-3 (0.949) > d384/L8/lr5e-4 (1.004).
    Note: d384 has lowest train CE but worst val → overfits; scale model only with more data/regularization.
    Evidence: `repro_outputs/newscheme_prelim/{SUMMARY.md,train_report.json}` (heavy /data artifacts deleted).
  - **Simplified the repo** (user: too many files): now only 3 runnable files (`test_all.py`, `process_abc.py`,
    `train.py`) + 3 method modules + `README.md` (the guide) + `方案.md`/`HANDOFF.md` + `docs/`. Deleted/merged
    `e2e_pipeline_abc.py`, `real_e2e_abc.py`, `EXECPLAN_*`, and 6 redundant repro_outputs docs (conclusions
    folded into `docs/REVIEW.md` + `docs/DATA_AND_PROCESSING.md`). Renamed `process_abc_unified.py→process_abc.py`,
    `train_newscheme_abc.py→train.py`, `SESSION_HANDOFF.md→HANDOFF.md`. Added `train.py --stage ar_sweep`.

## Next steps (to reach paper-headline Valid)

1. Wire `rcm_face_ordering` into `BrepARG/2sequence.py` (replace `dfs_face_ordering_from_core`),
   regenerate sequences, retrain AR, compare Valid vs DFS baseline.
2. Wire `FSQQuantiser` into `BrepARG/trainer.py` VQVAE (ABC `fsq_levels=(8,8,8,16)`), retrain VQVAE+AR.
3. Make OCC + torch coexist (add torch to `occ`, or OCC to `brepgen`), then run `generate_brep.py`
   with `constrained_decoding` to measure Valid + Novelty/Unique/COV (check no diversity loss).

## Reproduce evidence

    conda activate brepgen
    cd V13/breparg_improvements
    python test_all.py        # 58/58 synthetic
    python real_e2e_abc.py    # real ABC; writes repro_outputs/real_e2e_abc_report.json
