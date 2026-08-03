# Scheme & code review + validation evidence

Merges the old `CODE_SCHEME_REVIEW.md` and the real-data validation that used to live in `real_e2e_abc.py`.
Verdict: **scheme is internally consistent and correct for training.** Unit suite: **58/58**.

## 方案① FSQ (`fsq_quantise.py`)

- Exact drop-in for `quantise.VectorQuantiser.forward`: returns `(z_q, loss, (perplexity, min_encodings,
  encoding_indices))`, same `z_q` shape, scalar-zero loss. Token packing flattens `(b,h,w)` row-major like
  VQ, so `2sequence.reshape(N,4)` aligns. Straight-through grad; `proj_in/proj_out` learnable.
- **Hard requirement**: `prod(fsq_levels)` must equal the dataset `se_codebook_size`. ABC = 8192 ⇒
  `(8,8,8,16)` (asserted in `train.py`).
- **AMP bug (found on real GPU, fixed)** — corrects the earlier review's wrong "AMP-safe" claim:
  FSQ packs indices into an 8192 codebook, but fp16 represents integers exactly only up to 2048, so under
  autocast `codes_to_indices` overflowed → `one_hot` CUDA device-side assert. **CPU tests never caught it.**
  Fix: compute the FSQ round/index math in fp32 + clamp (`fsq_quantise.py`). Also `lr1e-3+(8,8,8,16)` NaNs
  under AMP → `train.py` uses lr3e-4 + grad-clip + NaN-skip.
- Decode-to-STEP needs a 64-dim decode embedding materialized via `proj_out`, plus a GPU
  (`utils.joint_optimize` hardcodes `.cuda()`). Inference concern, orthogonal to training.

## 方案② RCM ordering (`gnn_ordering.py`)

- `rcm_face_ordering` is a deterministic valid permutation; handles N=0/1, disconnected, no-edge graphs.
  Monkeypatches `2sequence.dfs_face_ordering_from_core` (no upstream edit); downstream edge ordering stays correct.
- **Real-data validation (400 real ABC CADs)**: all 400 valid permutations; mean MLA RCM **307.1 vs DFS
  348.5 = 0.88×**; RCM better on **329/400** (DFS better 65, tie 6). On synthetic grids the margin is larger
  (0.65×) — real CAD graphs are less grid-regular. Learnable GNN optional, not used by default.

## 方案③ Constrained decoding (`constrained_decoding.py`)

- Stateless prefix re-parse + `LogitsProcessor`, 6 constraint classes, never deadlocks (END fallback).
- **Real-data validation**: 120 real ABC GT sequences, **103,506 token steps, 0 wrong-masking** (real 8192
  vocab). Plus 50 synthetic GT-safety. Inference-only; `prompt_len=1` for the START-only prompt.
- Boundary: blocks local illegality (undeclared-face refs, etc.), not global non-manifoldness.

## Integration / training-readiness

- We don't use `trainer.py::VQVAE` (VQ + hardcoded 4096); `train.py` builds a diffusers `VQModel` with
  `num_vq_embeddings=8192` and swaps in `FSQQuantiser((8,8,8,16))`. FSQ needs no codebook-restart/contrastive
  logic (no collapse), so the simplified loop is appropriate.
- Checkpoint hygiene: `ARModel.state_dict()` has aliased keys that break its own `load_state_dict`;
  `train.py` saves/loads the inner GPT-2 (`model.model.state_dict()`).
- `2sequence.ARDataPreprocessor` hardcodes `se_codebook_size=8192` → matches FSQ (8,8,8,16); AR vocab 10294
  lines up (regenerated sequences had 0 out-of-vocab).
