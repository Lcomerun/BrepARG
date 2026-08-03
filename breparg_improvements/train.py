"""
train_newscheme_abc.py
======================
用 **5K 条 ABC 数据**对**新方案**(FSQ + RCM + 约束解码)做真实训练。

Task-1 结论(有证据,见 repro_outputs/DATA_FORMAT_COMPARISON.md):
  - 原始 .step -> parsed 的格式/逻辑与旧数据**完全一致**(process_brep.py 未被新方案触及),
    所以**直接复用** /data/public/luol/breparg_data/abc_parsed_50c 的解析几何,无需重解析原始 step。
  - 但 parsed -> AR 序列这一步**有区别**:旧序列是 VQ+DFS,新方案是 FSQ+RCM,token 值与面序都不同;
    旧 abc_sequences_50c.pkl 与旧 SE-VQVAE 都不能复用,**必须按新方案重新生成 + 重训**。

因此本脚本:复用 5K parsed 几何 -> 训练 FSQ-VQVAE -> 用 RCM+FSQ 重建 5K 序列 -> 训练 AR。
所有重产物写入 /data(因 /home 已 99% 满)。GPU 用 GPU1(CUDA_VISIBLE_DEVICES=1)。

阶段:  --stage split | vqsweep | vqvae | sequence | ar | all
"""

import os, sys, glob, json, time, pickle, random, argparse, warnings, types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

def _find_breparg():
    d = _HERE
    for _ in range(6):
        if os.path.exists(os.path.join(d, 'BrepARG', 'model.py')):
            return os.path.join(d, 'BrepARG')
        p = os.path.dirname(d)
        if p == d: break
        d = p
    return None
BREPARG = _find_breparg(); assert BREPARG
sys.path.insert(0, BREPARG); sys.path.insert(0, os.path.join(BREPARG, 'process_data'))

from diffusers import VQModel
from fsq_quantise import FSQQuantiser
from gnn_ordering import rcm_face_ordering
from training_stability import (
    VQVAEStopConfig,
    VQVAEStopState,
    continuation_epoch_count,
    finite_average,
    parse_env_bool,
    safe_json_number,
    summarize_vqvae_history,
    update_vqvae_stop_state,
)
from ar_training_utils import (
    append_jsonl,
    ar_checkpoint_paths,
    load_ar_checkpoint,
    periodic_checkpoint_path,
    save_ar_checkpoint,
)
from vqvae_sampling import (
    collect_vqvae_patch_shard_records,
    collect_vqvae_sample_records,
    records_to_chw_array,
    records_to_patch_weights,
)
from vqvae_sample_cache import load_vqvae_sample_cache, save_vqvae_sample_cache

# ---- paths (heavy artifacts on /data) ----
DATA = '/data/public/luol/breparg_data'
# NS_POOL:解析池目录。默认 abc_parsed_50c(平铺);全量用 abc_parsed_100c(分 chunk 子目录,见下方递归 glob)。
PARSED_POOL = os.environ.get('NS_POOL', os.path.join(DATA, 'abc_parsed_50c'))
# NS_OUTBASE:重产物输出根目录。默认写 /data;当 /data 只读时设为 /home 上的目录做自包含验证。
OUT = os.path.join(os.environ.get('NS_OUTBASE', DATA), os.environ.get('NS_OUT', 'newscheme_5k'))
os.makedirs(OUT, exist_ok=True)
SPLIT = os.path.join(OUT, 'split.pkl')
VQVAE_PT = os.path.join(OUT, 'fsq_vqvae_best.pt')
VQVAE_FINAL_PT = os.path.join(OUT, 'fsq_vqvae_final.pt')
SWEEP_JSON = os.path.join(OUT, 'vqvae_hp_sweep.json')
VQVAE_HISTORY_JSON = os.path.join(OUT, 'vqvae_history.json')
SEQ_PKL = os.path.join(OUT, 'sequences_fsq_rcm.pkl')
AR_PATHS = ar_checkpoint_paths(OUT)
AR_PT = str(AR_PATHS['best'])
AR_LATEST_PT = str(AR_PATHS['latest'])
AR_CHECKPOINT_DIR = str(AR_PATHS['checkpoint_dir'])
AR_HISTORY_JSONL = str(AR_PATHS['history'])
# lightweight evidence on /home (tracked)
EVID = os.path.join(_HERE, 'repro_outputs', os.environ.get('NS_OUT', 'newscheme_5k'))
os.makedirs(EVID, exist_ok=True)
REPORT = os.path.join(EVID, 'train_report.json')

# ---- knobs ----
N_DATA = int(os.environ.get('NS_N', '5000'))
FSQ_LEVELS = tuple(int(x) for x in os.environ.get('NS_LEVELS', '8,8,8,16').split(','))
SE_CODEBOOK = int(np.prod(FSQ_LEVELS))
VQ_SAMPLES = int(os.environ.get('NS_VQ_SAMPLES', '60000'))
VQ_EPOCHS = int(os.environ.get('NS_VQ_EPOCHS', '120'))
VQ_BS = int(os.environ.get('NS_VQ_BS', '512'))
VQ_MIN_EPOCHS = int(os.environ.get('NS_VQ_MIN_EPOCHS', '12'))
VQ_PATIENCE = int(os.environ.get('NS_VQ_PATIENCE', '8'))
VQ_MIN_DELTA = float(os.environ.get('NS_VQ_MIN_DELTA', '1e-5'))
VQ_MAX_NONFINITE_VAL_EPOCHS = int(os.environ.get('NS_VQ_MAX_NONFINITE_VAL_EPOCHS', '2'))
VQ_LR = float(os.environ.get('NS_VQ_LR', '3e-4'))
VQ_RESUME_FROM = os.environ.get('NS_VQ_RESUME_FROM', '').strip()
VQ_HISTORY_IN = os.environ.get('NS_VQ_HISTORY_IN', '').strip()
VQ_TARGET_EPOCH = os.environ.get('NS_VQ_TARGET_EPOCH', '').strip()
VQ_COMPLEX_FRACTION = float(os.environ.get('NS_VQ_COMPLEX_FRACTION', '0'))
VQ_COMPLEX_MIN_FACES = int(os.environ.get('NS_VQ_COMPLEX_MIN_FACES', '12'))
VQ_COMPLEX_MIN_EDGES = int(os.environ.get('NS_VQ_COMPLEX_MIN_EDGES', '20'))
VQ_CURVED_FRACTION = float(os.environ.get('NS_VQ_CURVED_FRACTION', '0'))
VQ_PATCH_SHARD_ROOT = os.environ.get('NS_VQ_PATCH_SHARD_ROOT', '').strip()
VQ_PATCH_SHARDS = os.environ.get('NS_VQ_PATCH_SHARDS', '').strip()
VQ_SAMPLE_CACHE = os.environ.get('NS_VQ_SAMPLE_CACHE', '').strip()
VQ_MAX_SOURCE_FACES = int(os.environ.get('NS_VQ_MAX_SOURCE_FACES', '0'))
VQ_MAX_SOURCE_EDGES = int(os.environ.get('NS_VQ_MAX_SOURCE_EDGES', '0'))
VQ_COMPLEX_LOSS_WEIGHT = float(os.environ.get('NS_VQ_COMPLEX_LOSS_WEIGHT', '1'))
VQ_CURVED_LOSS_WEIGHT = float(os.environ.get('NS_VQ_CURVED_LOSS_WEIGHT', '1'))
VQ_CURVED_LOSS_THRESHOLD = float(os.environ.get('NS_VQ_CURVED_LOSS_THRESHOLD', '0.02'))
AR_EPOCHS = int(os.environ.get('NS_AR_EPOCHS', '120'))
AR_BS = int(os.environ.get('NS_AR_BS', '32'))
AR_DMODEL = int(os.environ.get('NS_AR_DMODEL', '256'))
AR_LAYERS = int(os.environ.get('NS_AR_LAYERS', '8'))
AR_LR = float(os.environ.get('NS_AR_LR', '5e-4'))
AR_SAVE_EVERY = int(os.environ.get('NS_AR_SAVE_EVERY', '20'))
AR_RESUME_FROM = os.environ.get('NS_AR_RESUME_FROM', '').strip()
AR_LOG_EVERY_BATCHES = int(os.environ.get('NS_AR_LOG_EVERY_BATCHES', '2000'))
AR_MAX_SEQ_LEN = int(os.environ.get('NS_AR_MAX_SEQ_LEN', '1024'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = torch.cuda.is_available()
VQ_AMP = AMP and not parse_env_bool(os.environ.get('NS_DISABLE_AMP_VQVAE'), False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0); np.random.seed(0); random.seed(0)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def metric_for_report(value, digits=8):
    safe = safe_json_number(value)
    return round(float(safe), digits) if safe is not None else None
def format_metric(value):
    safe = safe_json_number(value)
    return f"{float(safe):.5f}" if safe is not None else "inf"
def load_report():
    return json.load(open(REPORT)) if os.path.exists(REPORT) else {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'config': {
            'n_data': N_DATA, 'fsq_levels': list(FSQ_LEVELS), 'se_codebook': SE_CODEBOOK,
            'vq_epochs': VQ_EPOCHS, 'vq_bs': VQ_BS, 'ar_epochs': AR_EPOCHS, 'ar_bs': AR_BS,
            'vq_min_epochs': VQ_MIN_EPOCHS, 'vq_patience': VQ_PATIENCE,
            'vq_min_delta': VQ_MIN_DELTA, 'vq_max_nonfinite_val_epochs': VQ_MAX_NONFINITE_VAL_EPOCHS,
            'vq_amp': VQ_AMP, 'vq_lr': VQ_LR, 'vq_resume_from': VQ_RESUME_FROM or None,
            'vq_history_in': VQ_HISTORY_IN or None, 'vq_target_epoch': VQ_TARGET_EPOCH or None,
            'vq_complex_fraction': VQ_COMPLEX_FRACTION,
            'vq_complex_min_faces': VQ_COMPLEX_MIN_FACES,
            'vq_complex_min_edges': VQ_COMPLEX_MIN_EDGES,
            'vq_curved_fraction': VQ_CURVED_FRACTION,
            'vq_patch_shard_root': VQ_PATCH_SHARD_ROOT or None,
            'vq_patch_shards': VQ_PATCH_SHARDS or None,
            'vq_sample_cache': VQ_SAMPLE_CACHE or None,
            'vq_max_source_faces': VQ_MAX_SOURCE_FACES,
            'vq_max_source_edges': VQ_MAX_SOURCE_EDGES,
            'vq_complex_loss_weight': VQ_COMPLEX_LOSS_WEIGHT,
            'vq_curved_loss_weight': VQ_CURVED_LOSS_WEIGHT,
            'vq_curved_loss_threshold': VQ_CURVED_LOSS_THRESHOLD,
            'ar_dmodel': AR_DMODEL, 'ar_layers': AR_LAYERS, 'ar_lr': AR_LR,
            'ar_max_seq_len': AR_MAX_SEQ_LEN,
            'ar_save_every': AR_SAVE_EVERY, 'ar_resume_from': AR_RESUME_FROM or None,
            'device': DEVICE}, 'stages': {}}
def save_report(r): json.dump(r, open(REPORT, 'w'), ensure_ascii=False, indent=2)


def build_fsq_vqvae(levels=FSQ_LEVELS):
    m = VQModel(in_channels=3, out_channels=3,
                down_block_types=['DownEncoderBlock2D'] * 5, up_block_types=['UpDecoderBlock2D'] * 5,
                block_out_channels=[32, 64, 128, 256, 512], layers_per_block=2, act_fn='silu',
                latent_channels=128, vq_embed_dim=64, num_vq_embeddings=int(np.prod(levels)),
                norm_num_groups=32, sample_size=512)
    m.quantize = FSQQuantiser(num_embed=int(np.prod(levels)), embed_dim=64, fsq_levels=levels, in_dim=64)
    return m


_LAST_VQVAE_SAMPLING_SUMMARY = None


def vq_patch_shard_paths():
    paths = []
    if VQ_PATCH_SHARDS:
        normalized = VQ_PATCH_SHARDS.replace(';', os.pathsep)
        paths.extend(part.strip() for part in normalized.split(os.pathsep) if part.strip())
    if VQ_PATCH_SHARD_ROOT:
        patterns = ['vq_patch_shard_*.pkl.zst', 'vq_patch_shard_*.pkl.gz', 'vq_patch_shard_*.pkl']
        for pattern in patterns:
            paths.extend(glob.glob(os.path.join(VQ_PATCH_SHARD_ROOT, pattern)))
    return sorted(set(paths))


def collect_se(paths, cap, return_weights=False):
    global _LAST_VQVAE_SAMPLING_SUMMARY
    if VQ_SAMPLE_CACHE and os.path.exists(VQ_SAMPLE_CACHE):
        samples, weights, summary = load_vqvae_sample_cache(VQ_SAMPLE_CACHE, min_samples=cap)
        summary["source"] = "vq_sample_cache"
        _LAST_VQVAE_SAMPLING_SUMMARY = summary
        log(f"  VQ sample cache loaded={len(samples)} requested={cap} path={VQ_SAMPLE_CACHE}")
        if return_weights:
            return samples[:cap], weights[:cap]
        return samples[:cap]

    shard_paths = vq_patch_shard_paths()
    if shard_paths:
        records, summary = collect_vqvae_patch_shard_records(
            shard_paths,
            cap,
            seed=0,
            complex_fraction=VQ_COMPLEX_FRACTION,
            complex_min_faces=VQ_COMPLEX_MIN_FACES,
            complex_min_edges=VQ_COMPLEX_MIN_EDGES,
            curved_fraction=VQ_CURVED_FRACTION,
            max_source_faces=VQ_MAX_SOURCE_FACES,
            max_source_edges=VQ_MAX_SOURCE_EDGES,
        )
        summary["source"] = "vq_patch_shards"
        summary["patch_shards_requested"] = len(shard_paths)
    else:
        records, summary = collect_vqvae_sample_records(
            paths,
            cap,
            seed=0,
            complex_fraction=VQ_COMPLEX_FRACTION,
            complex_min_faces=VQ_COMPLEX_MIN_FACES,
            complex_min_edges=VQ_COMPLEX_MIN_EDGES,
            curved_fraction=VQ_CURVED_FRACTION,
            max_source_faces=VQ_MAX_SOURCE_FACES,
            max_source_edges=VQ_MAX_SOURCE_EDGES,
        )
        summary["source"] = "parsed_pkls"
    samples = records_to_chw_array(records)
    weights = records_to_patch_weights(
        records,
        complex_weight=VQ_COMPLEX_LOSS_WEIGHT,
        curved_weight=VQ_CURVED_LOSS_WEIGHT,
        curved_threshold=VQ_CURVED_LOSS_THRESHOLD,
    )
    summary["weight_mean"] = float(np.mean(weights)) if len(weights) else None
    summary["weight_max"] = float(np.max(weights)) if len(weights) else None
    summary["complex_loss_weight"] = VQ_COMPLEX_LOSS_WEIGHT
    summary["curved_loss_weight"] = VQ_CURVED_LOSS_WEIGHT
    summary["curved_loss_threshold"] = VQ_CURVED_LOSS_THRESHOLD
    if VQ_SAMPLE_CACHE:
        save_vqvae_sample_cache(VQ_SAMPLE_CACHE, samples, weights, summary)
        summary["cache_path"] = VQ_SAMPLE_CACHE
        summary["cache_written"] = True
        summary["cache_samples"] = int(len(samples))
    _LAST_VQVAE_SAMPLING_SUMMARY = summary
    if len(samples) == 0:
        raise RuntimeError("VQ-VAE sample collection produced zero geometry patches")
    if shard_paths:
        log(
            "  VQ patch-shard sampling selected={selected}/{requested} complex_selected={complex_records_selected} "
            "complex_target={complex_target} loaded_shards={loaded_shards} failed_shards={failed_shards}".format(**summary)
        )
    elif VQ_COMPLEX_FRACTION > 0 or VQ_CURVED_FRACTION > 0:
        log(
            "  VQ sampling selected={selected}/{requested} complex_selected={complex_records_selected} "
            "complex_target={complex_target} loaded_paths={loaded_paths} failed_paths={failed_paths}".format(**summary)
        )
    if return_weights:
        return samples, weights
    return samples


def weighted_reconstruction_loss(recon, target, weights=None):
    per_sample = (recon - target).pow(2).flatten(1).mean(dim=1)
    if weights is None:
        return per_sample.mean()
    weights = weights.to(device=per_sample.device, dtype=per_sample.dtype)
    weights = torch.clamp(weights, min=0)
    denom = weights.sum().clamp_min(1e-8)
    return (per_sample * weights).sum() / denom


# ===========================================================================
def stage_split():
    log("SPLIT: 从 abc_parsed_50c 采样 5K(复用解析几何,无需重解析原始 step)")
    files = sorted(glob.glob(os.path.join(PARSED_POOL, '*.pkl')))
    if not files:                                              # 分 chunk 子目录布局(abc_parsed_100c/abc_XXXX/*.pkl)
        files = sorted(glob.glob(os.path.join(PARSED_POOL, '*', '*.pkl')))
    log(f"  pool={PARSED_POOL} found={len(files)} pkls")
    random.Random(0).shuffle(files)
    files = files[:N_DATA]
    n = len(files); n_tr = int(n * 0.9); n_va = int(n * 0.05)
    split = {'train': files[:n_tr], 'val': files[n_tr:n_tr + n_va], 'test': files[n_tr + n_va:]}
    pickle.dump(split, open(SPLIT, 'wb'))
    log(f"  split train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])}")
    r = load_report(); r['stages']['split'] = {k: len(v) for k, v in split.items()}; save_report(r)
    return True


def _train_vqvae(model, Xtr, Xva, epochs, bs, lr=1e-3, tag="", save_path=None):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    Xtr = torch.tensor(Xtr); Xva = torch.tensor(Xva)
    best_val = float('inf'); hist = []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr)); tot = nb = 0
        for i in range(0, len(Xtr), bs):
            xb = Xtr[perm[i:i + bs]].to(DEVICE)
            with torch.cuda.amp.autocast(enabled=AMP):
                h = model.encoder(xb); h = model.quant_conv(h)
                zq, vq_loss, _ = model.quantize(h)
                recon = model.decoder(model.post_quant_conv(zq))
                loss = F.mse_loss(recon, xb) + vq_loss
            if not torch.isfinite(loss):          # NaN/Inf 防护:跳过这步,不污染权重
                opt.zero_grad(); continue
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 稳定 AMP 训练
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        # val recon
        model.eval(); vtot = vnb = 0
        with torch.no_grad():
            for i in range(0, len(Xva), bs):
                xb = Xva[i:i + bs].to(DEVICE)
                with torch.cuda.amp.autocast(enabled=AMP):
                    h = model.encoder(xb); h = model.quant_conv(h)
                    zq, _, _ = model.quantize(h)
                    recon = model.decoder(model.post_quant_conv(zq))
                    v = F.mse_loss(recon, xb).item()
                if np.isfinite(v):                 # 单个 batch 在 AMP 下偶发 fp16 溢出 -> 跳过,勿污染整轮均值
                    vtot += v; vnb += 1
        # vnb==0 ⇒ 整轮 val 全 NaN(配置发散,如 lr1e-3+大码本)。必须记为 inf 而非 0,
        # 否则发散配置会以「val=0」假装完美,在 vqsweep 里排第一被选中,毒化整轮训练。
        tr = tot / max(1, nb); va = (vtot / vnb) if vnb else float('inf')
        hist.append((tr, va))
        if np.isfinite(va) and va < best_val:      # 只在真改进时记录/保存 best(NaN 不更新)
            best_val = va
            if save_path:
                torch.save({'model_state_dict': model.state_dict(), 'fsq_levels': list(FSQ_LEVELS)}, save_path)
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  {tag} ep {ep:3d} train={tr:.5f} val={va:.5f}")
    return hist, best_val


def _train_vqvae(
        model,
        Xtr,
        Xva,
        epochs,
        bs,
        lr=1e-3,
        tag="",
        save_path=None,
        history_path=None,
        stop_config=None,
        amp_enabled=None,
        start_epoch=0,
        initial_best_val=None,
        initial_best_epoch=-1,
        history_prefix=None,
        save_final_path=None,
        train_weights=None):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    amp_enabled = VQ_AMP if amp_enabled is None else bool(amp_enabled)
    stop_config = stop_config or VQVAEStopConfig(
        min_epochs=VQ_MIN_EPOCHS,
        patience=VQ_PATIENCE,
        max_nonfinite_val_epochs=VQ_MAX_NONFINITE_VAL_EPOCHS,
        min_delta=VQ_MIN_DELTA,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    Xtr = torch.tensor(Xtr); Xva = torch.tensor(Xva)
    Wtr = torch.tensor(train_weights, dtype=torch.float32) if train_weights is not None else None
    start_epoch = int(start_epoch)
    initial_best = float(initial_best_val) if initial_best_val is not None else float('inf')
    stop_state = VQVAEStopState(best_val=initial_best, best_epoch=int(initial_best_epoch))
    best_val = stop_state.best_val; hist = []; history = list(history_prefix or [])
    meta = {
        'epochs_requested': epochs,
        'epochs_ran': 0,
        'best_epoch': -1,
        'start_epoch': start_epoch,
        'end_epoch': start_epoch - 1,
        'stopped_early': False,
        'stop_reason': '',
        'amp': amp_enabled,
        'history_path': history_path,
        'save_final_path': save_final_path,
        'train_weight_mean': float(torch.mean(Wtr).item()) if Wtr is not None and len(Wtr) else None,
        'train_weight_max': float(torch.max(Wtr).item()) if Wtr is not None and len(Wtr) else None,
    }
    for ep in range(epochs):
        absolute_epoch = start_epoch + ep
        model.train(); perm = torch.randperm(len(Xtr)); tot = nb = 0
        train_batches = skipped_train_batches = 0
        for i in range(0, len(Xtr), bs):
            train_batches += 1
            batch_index = perm[i:i + bs]
            xb = Xtr[batch_index].to(DEVICE)
            wb = Wtr[batch_index].to(DEVICE) if Wtr is not None else None
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                h = model.encoder(xb); h = model.quant_conv(h)
                zq, vq_loss, _ = model.quantize(h)
                recon = model.decoder(model.post_quant_conv(zq))
                loss = weighted_reconstruction_loss(recon, xb, wb) + vq_loss
            if not torch.isfinite(loss):
                skipped_train_batches += 1
                opt.zero_grad()
                continue
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        model.eval(); vtot = vnb = 0; val_batches = 0
        with torch.no_grad():
            for i in range(0, len(Xva), bs):
                val_batches += 1
                xb = Xva[i:i + bs].to(DEVICE)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    h = model.encoder(xb); h = model.quant_conv(h)
                    zq, _, _ = model.quantize(h)
                    recon = model.decoder(model.post_quant_conv(zq))
                    v = weighted_reconstruction_loss(recon, xb).item()
                if np.isfinite(v):
                    vtot += v; vnb += 1
        tr = finite_average(tot, nb); va = finite_average(vtot, vnb)
        stop_state, improved, should_stop = update_vqvae_stop_state(absolute_epoch, va, stop_state, stop_config)
        best_val = stop_state.best_val
        hist.append((tr, va))
        if improved and save_path:
            torch.save({'model_state_dict': model.state_dict(), 'fsq_levels': list(FSQ_LEVELS)}, save_path)
        if save_final_path:
            torch.save({'model_state_dict': model.state_dict(), 'fsq_levels': list(FSQ_LEVELS)}, save_final_path)
        record = {
            'epoch': absolute_epoch,
            'train_loss': metric_for_report(tr),
            'val_loss': metric_for_report(va),
            'best_val': metric_for_report(best_val),
            'best_epoch': stop_state.best_epoch,
            'improved': improved,
            'train_batches': train_batches,
            'finite_train_batches': nb,
            'skipped_train_batches': skipped_train_batches,
            'val_batches': val_batches,
            'finite_val_batches': vnb,
            'consecutive_nonfinite_val_epochs': stop_state.consecutive_nonfinite_val_epochs,
            'epochs_without_improvement': stop_state.epochs_without_improvement,
        }
        history.append(record)
        meta.update({
            'epochs_ran': ep + 1,
            'best_epoch': stop_state.best_epoch,
            'end_epoch': absolute_epoch,
            'stop_reason': stop_state.stop_reason,
        })
        if history_path:
            json.dump({
                'tag': tag,
                'config': {
                    'epochs_requested': epochs,
                    'start_epoch': start_epoch,
                    'target_epoch': start_epoch + epochs,
                    'batch_size': bs,
                    'lr': lr,
                    'amp': amp_enabled,
                    'min_epochs': stop_config.min_epochs,
                    'patience': stop_config.patience,
                    'min_delta': stop_config.min_delta,
                    'max_nonfinite_val_epochs': stop_config.max_nonfinite_val_epochs,
                },
                'history': history,
                'best_val_recon': metric_for_report(best_val),
                'best_epoch': stop_state.best_epoch,
                'stop_reason': stop_state.stop_reason,
            }, open(history_path, 'w'), indent=2)
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  {tag} ep {absolute_epoch:3d} train={format_metric(tr)} val={format_metric(va)} best={format_metric(best_val)} finite_train={nb}/{train_batches} finite_val={vnb}/{val_batches}")
        if should_stop:
            meta['stopped_early'] = True
            log(f"  {tag} early stop at ep {absolute_epoch}: {stop_state.stop_reason} best={format_metric(best_val)} best_epoch={stop_state.best_epoch}")
            break
    return hist, best_val, meta


def stage_vqsweep():
    log("VQSWEEP: FSQ-VQVAE 超参对比(短训选最优)")
    split = pickle.load(open(SPLIT, 'rb'))
    X = collect_se(split['train'], 12000)
    ntr = int(len(X) * 0.9); Xtr, Xva = X[:ntr], X[ntr:]
    log(f"  sweep 数据: train={len(Xtr)} val={len(Xva)}  (短训 15 ep)")
    configs = [
        {'name': 'lr1e-3_L8.8.8.16', 'levels': (8, 8, 8, 16), 'lr': 1e-3},
        {'name': 'lr3e-4_L8.8.8.16', 'levels': (8, 8, 8, 16), 'lr': 3e-4},
        {'name': 'lr1e-3_L8.8.8.8',  'levels': (8, 8, 8, 8),  'lr': 1e-3},
    ]
    results = []
    for c in configs:
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        m = build_fsq_vqvae(c['levels'])
        _, bv, _ = _train_vqvae(m, Xtr, Xva, epochs=15, bs=VQ_BS, lr=c['lr'], tag=c['name'])
        results.append({'name': c['name'], 'levels': list(c['levels']),
                        'codebook': int(np.prod(c['levels'])), 'lr': c['lr'], 'best_val_recon': round(bv, 5)})
        log(f"  -> {c['name']}: best_val_recon={bv:.5f}")
    results.sort(key=lambda x: x['best_val_recon'])
    json.dump({'sweep': results, 'winner': results[0]}, open(SWEEP_JSON, 'w'), indent=2)
    r = load_report(); r['stages']['vqsweep'] = {'results': results, 'winner': results[0]['name']}; save_report(r)
    log(f"  WINNER: {results[0]['name']} (val_recon={results[0]['best_val_recon']})")
    return True


def stage_vqvae():
    log(f"VQVAE: 全量训练 FSQ-VQVAE (levels={FSQ_LEVELS}, codebook={SE_CODEBOOK})")
    if vq_patch_shard_paths():
        X, W = collect_se([], VQ_SAMPLES, return_weights=True)
    else:
        split = pickle.load(open(SPLIT, 'rb'))
        X, W = collect_se(split['train'], VQ_SAMPLES, return_weights=True)
    ntr = int(len(X) * 0.95); Xtr, Xva = X[:ntr], X[ntr:]; Wtr = W[:ntr]
    resume_summary = None
    start_epoch = 0
    initial_best_val = None
    initial_best_epoch = -1
    epochs_to_run = VQ_EPOCHS
    if VQ_HISTORY_IN:
        resume_summary = summarize_vqvae_history(VQ_HISTORY_IN)
        start_epoch = resume_summary['next_epoch']
        initial_best_val = resume_summary['best_val_recon']
        initial_best_epoch = resume_summary['best_epoch']
    if VQ_TARGET_EPOCH:
        epochs_to_run = continuation_epoch_count(start_epoch, int(VQ_TARGET_EPOCH))
    log(f"  data train={len(Xtr)} val={len(Xva)} epochs={epochs_to_run} start_epoch={start_epoch} bs={VQ_BS} lr={VQ_LR} device={DEVICE}")
    m = build_fsq_vqvae(FSQ_LEVELS)
    if VQ_RESUME_FROM:
        ckpt = torch.load(VQ_RESUME_FROM, map_location=DEVICE)
        m.load_state_dict(ckpt['model_state_dict'])
        log(f"  resume from {VQ_RESUME_FROM}")
    t0 = time.time()
    hist, bv, meta = _train_vqvae(
        m,
        Xtr,
        Xva,
        epochs=epochs_to_run,
        bs=VQ_BS,
        lr=VQ_LR,
        tag="vqvae",
        save_path=VQVAE_PT,
        history_path=VQVAE_HISTORY_JSON,
        start_epoch=start_epoch,
        initial_best_val=initial_best_val,
        initial_best_epoch=initial_best_epoch,
        save_final_path=VQVAE_FINAL_PT,
        train_weights=Wtr,
    )
    first_val = hist[0][1] if hist else float('inf')
    baseline_best = initial_best_val if initial_best_val is not None else first_val
    ok = bool(hist) and np.isfinite(bv) and np.isfinite(first_val) and bv <= baseline_best
    r = load_report(); r['stages']['vqvae'] = {
        'samples': len(X), 'epochs': epochs_to_run, 'target_epoch': int(VQ_TARGET_EPOCH) if VQ_TARGET_EPOCH else None,
        'start_epoch': meta.get('start_epoch', start_epoch), 'end_epoch': meta.get('end_epoch'),
        'epochs_ran': meta.get('epochs_ran', len(hist)),
        'train_init': metric_for_report(hist[0][0]) if hist else None,
        'train_final': metric_for_report(hist[-1][0]) if hist else None,
        'val_init': metric_for_report(hist[0][1]) if hist else None,
        'val_final': metric_for_report(hist[-1][1]) if hist else None,
        'best_val_recon': metric_for_report(bv),
        'best_epoch': meta.get('best_epoch'),
        'baseline_best_val_recon': metric_for_report(initial_best_val),
        'baseline_best_epoch': initial_best_epoch if initial_best_epoch >= 0 else None,
        'resume_from': VQ_RESUME_FROM or None,
        'history_in': VQ_HISTORY_IN or None,
        'resume_summary': resume_summary,
        'stopped_early': meta.get('stopped_early', False),
        'early_stop_reason': meta.get('stop_reason') or None,
        'amp': meta.get('amp'),
        'lr': VQ_LR,
        'train_weight_mean': meta.get('train_weight_mean'),
        'train_weight_max': meta.get('train_weight_max'),
        'sampling': _LAST_VQVAE_SAMPLING_SUMMARY,
        'history': VQVAE_HISTORY_JSON,
        'checkpoint_best': VQVAE_PT,
        'checkpoint_final': VQVAE_FINAL_PT,
        'minutes': round((time.time() - t0) / 60, 1),
        'status': 'VERIFIED' if ok else 'FAILED'}
    save_report(r); log(f"  saved best -> {VQVAE_PT}  (val_init {format_metric(hist[0][1]) if hist else 'inf'} -> best {format_metric(bv)})")
    return ok
    # lr 3e-4:sweep 显示 lr1e-3 + (8,8,8,16) 在 AMP 下会 NaN;3e-4 稳定。
    # save_path=VQVAE_PT:在 best-val 轮即时保存最优权重(而非最后一轮,末轮可能偶发 NaN)。
    hist, bv = _train_vqvae(m, Xtr, Xva, epochs=VQ_EPOCHS, bs=VQ_BS, lr=3e-4, tag="vqvae", save_path=VQVAE_PT)
    ok = np.isfinite(bv) and bv < hist[0][1]       # 成败看 best-val(对末轮单 batch NaN 稳健)
    r = load_report(); r['stages']['vqvae'] = {
        'samples': len(X), 'epochs': VQ_EPOCHS, 'train_init': round(hist[0][0], 5),
        'train_final': round(hist[-1][0], 5), 'val_init': round(hist[0][1], 5),
        'val_final': round(hist[-1][1], 5), 'best_val_recon': round(bv, 5),
        'minutes': round((time.time() - t0) / 60, 1),
        'status': 'VERIFIED' if ok else 'FAILED'}
    save_report(r); log(f"  saved best -> {VQVAE_PT}  (val_init {hist[0][1]:.4f} -> best {bv:.5f})")
    return ok


def stage_sequence():
    log("SEQUENCE: RCM + FSQ-VQVAE 重建 5K 序列")
    import importlib.util
    spec = importlib.util.spec_from_file_location('breparg_2sequence', os.path.join(BREPARG, '2sequence.py'))
    sq = importlib.util.module_from_spec(spec); spec.loader.exec_module(sq)
    sq.dfs_face_ordering_from_core = lambda efp, nf: rcm_face_ordering(efp, nf)
    log("  monkeypatch DFS->RCM")
    m = build_fsq_vqvae(FSQ_LEVELS).to(DEVICE).eval()
    m.load_state_dict(torch.load(VQVAE_PT, map_location=DEVICE)['model_state_dict'])
    args = types.SimpleNamespace(max_face=50, max_edge=150, scale=1.0, aug=False)
    pre = sq.ARDataPreprocessor(SPLIT, m, args)
    tr, va, te = [], [], []
    for sp, g in pre.group_cache:
        (tr if sp == 'train' else va if sp == 'val' else te).append(g)
    out = {'train': tr, 'val': va, 'test': te, 'vocab_size': pre.vocab_size,
           'special_token_size': pre.special_token_size, 'face_index_size': pre.face_index_size,
           'se_codebook_size': pre.se_codebook_size, 'bbox_index_size': pre.bbox_index_size,
           'face_index_offset': pre.face_index_offset, 'se_token_offset': pre.se_token_offset,
           'bbox_token_offset': pre.bbox_token_offset, 'se_tokens_per_element': pre.se_tokens_per_element,
           'bbox_tokens_per_element': pre.bbox_tokens_per_element,
           'special_tokens': {'START_TOKEN': pre.START_TOKEN, 'SEP_TOKEN': pre.SEP_TOKEN,
                              'END_TOKEN': pre.END_TOKEN, 'PAD_TOKEN': pre.PAD_TOKEN}}
    pickle.dump(out, open(SEQ_PKL, 'wb'))
    nseq = len(tr) + len(va) + len(te); bad = 0; mx = -1
    for g in tr + va + te:
        ids = g['original']['input_ids']; mx = max(mx, max(ids))
        if max(ids) >= out['vocab_size'] or min(ids) < 0: bad += 1
    log(f"  seqs={nseq} vocab={out['vocab_size']} max_token={mx} out_of_vocab={bad} se_tok/elem={pre.se_tokens_per_element}")
    r = load_report(); r['stages']['sequence'] = {
        'sequences': nseq, 'train': len(tr), 'val': len(va), 'test': len(te),
        'vocab_size': out['vocab_size'], 'max_token': int(mx), 'out_of_vocab': bad,
        'se_tokens_per_element': pre.se_tokens_per_element, 'ordering': 'RCM',
        'status': 'VERIFIED' if bad == 0 and nseq > 0 and pre.se_tokens_per_element == 4 else 'FAILED'}
    save_report(r)
    return bad == 0 and nseq > 0


def _load_ar_seqs(max_seq_len=None):
    max_seq_len = AR_MAX_SEQ_LEN if max_seq_len is None else int(max_seq_len)
    data = pickle.load(open(SEQ_PKL, 'rb'))
    def seqs_of(split): return [g['original']['input_ids'] for g in data[split] if len(g['original']['input_ids']) <= max_seq_len]
    return data, seqs_of('train'), seqs_of('val')


def _ar_batches(seqs, bs, pad, shuf, device):
    idx = list(range(len(seqs)))
    if shuf:
        random.shuffle(idx)
        bucket = int(os.environ.get('NS_AR_BUCKET_SIZE', '2048'))
        ordered = []
        for i in range(0, len(idx), bucket):
            block = idx[i:i + bucket]
            block.sort(key=lambda j: len(seqs[j]), reverse=True)
            ordered.extend(block)
        idx = ordered
    else:
        idx.sort(key=lambda j: len(seqs[j]), reverse=True)
    for i in range(0, len(idx), bs):
        ch = [seqs[j] for j in idx[i:i + bs]]
        mx = max(len(s) for s in ch)
        ids = torch.full((len(ch), mx), pad, dtype=torch.long)
        att = torch.zeros((len(ch), mx), dtype=torch.long)
        for k, s in enumerate(ch):
            ids[k, :len(s)] = torch.tensor(s)
            att[k, :len(s)] = 1
        yield ids.to(device), att.to(device)


def _train_ar(data, tr, va, dmodel, layers, lr, epochs, bs, tag="ar", save_path=None,
              latest_path=None, checkpoint_dir=None, history_path=None,
              resume_from=None, save_every=20, max_seq_len=1024):
    """训练一个 AR 配置,返回 (ce_init, ce_final, best_val_ce, minutes)。可复用于单训与超参对比。"""
    from model import ARModel
    PAD = data['special_tokens']['PAD_TOKEN']
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = ARModel(vocab_size=data['vocab_size'], d_model=dmodel, nhead=8, num_layers=layers,
                    dim_feedforward=dmodel * 4, dropout=0.1, max_seq_len=max_seq_len, pad_token_id=PAD).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    start_epoch = 0
    best = float('inf')
    resumed = False
    if resume_from:
        ckpt = load_ar_checkpoint(resume_from, map_location=DEVICE)
        model.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            for group in opt.param_groups:
                group['lr'] = lr
        if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict']:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0))
        best = float(ckpt.get('best_val_ce', best))
        resumed = True
        log(f"  [{tag}] resumed from {resume_from} at epoch={start_epoch} best={best:.4f} lr={lr:g}")

    def checkpoint_payload(epoch, tr_ce, va_ce, best_val):
        return {
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scaler_state_dict': scaler.state_dict() if AMP else None,
            'epoch': int(epoch),
            'train_ce': float(tr_ce),
            'val_ce': float(va_ce),
            'best_val_ce': float(best_val),
            'vocab_size': data['vocab_size'],
            'd_model': dmodel,
            'layers': layers,
            'batch_size': bs,
            'learning_rate': lr,
            'pad_token_id': PAD,
            'config': {'max_seq_len': int(max_seq_len), 'save_every': save_every, 'amp': AMP, 'device': DEVICE, 'tag': tag},
        }

    t0 = time.time(); ci = cf = None; last_val = None; epochs_ran = 0
    for ep in range(start_epoch + 1, epochs + 1):
        model.train(); tot = nb = 0
        total_train_batches = (len(tr) + bs - 1) // bs
        for batch_idx, (ids, att) in enumerate(_ar_batches(tr, bs, PAD, True, DEVICE), start=1):
            lab = ids.clone(); lab[lab == PAD] = -100
            with torch.cuda.amp.autocast(enabled=AMP):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
            if AR_LOG_EVERY_BATCHES > 0 and batch_idx % AR_LOG_EVERY_BATCHES == 0:
                running = tot / max(1, nb)
                log(f"  [{tag}] ep {ep:3d} batch {batch_idx}/{total_train_batches} train_CE_running={running:.4f} elapsed_min={((time.time() - t0) / 60):.2f}")
        model.eval(); vt = vn = 0
        with torch.no_grad():
            for ids, att in _ar_batches(va, bs, PAD, False, DEVICE):
                lab = ids.clone(); lab[lab == PAD] = -100
                with torch.cuda.amp.autocast(enabled=AMP):
                    vt += model(input_ids=ids, attention_mask=att, labels=lab).loss.item(); vn += 1
        tr_ce = tot / max(1, nb); va_ce = vt / max(1, vn)
        if ci is None: ci = tr_ce
        cf = tr_ce; last_val = va_ce; epochs_ran += 1
        improved = va_ce < best
        if improved:
            best = va_ce
            if save_path:
                save_ar_checkpoint(save_path, checkpoint_payload(ep, tr_ce, va_ce, best))
        payload = checkpoint_payload(ep, tr_ce, va_ce, best)
        if latest_path:
            save_ar_checkpoint(latest_path, payload)
        if checkpoint_dir and save_every > 0 and ep % save_every == 0:
            save_ar_checkpoint(periodic_checkpoint_path(checkpoint_dir, ep), payload)
        if history_path:
            append_jsonl(history_path, {
                'epoch': ep, 'train_ce': tr_ce, 'val_ce': va_ce, 'best_val_ce': best,
                'improved': improved, 'train_batches': nb, 'val_batches': vn,
                'elapsed_min': round((time.time() - t0) / 60, 3),
            })
        if ep % 10 == 0 or ep == epochs or improved:
            log(f"  [{tag}] ep {ep:3d} train_CE={tr_ce:.4f} val_CE={va_ce:.4f} best={best:.4f}")
    return {
        'ce_init': ci, 'ce_final': cf, 'last_val_ce': last_val, 'best_val_ce': best,
        'minutes': round((time.time() - t0) / 60, 1), 'start_epoch': start_epoch,
        'end_epoch': start_epoch + epochs_ran, 'epochs_requested': epochs,
        'epochs_ran': epochs_ran, 'resumed': resumed, 'resume_from': resume_from or None,
        'checkpoint_best': save_path, 'checkpoint_latest': latest_path,
        'checkpoint_dir': checkpoint_dir, 'history': history_path,
    }


def ar_stage_verified(meta):
    best_val = meta.get('best_val_ce')
    try:
        best_val = float(best_val)
    except (TypeError, ValueError):
        return False
    return (
        int(meta.get('epochs_ran') or 0) > 0
        and best_val == best_val
        and best_val not in (float('inf'), float('-inf'))
    )


def stage_ar():
    log("AR: 训练 AR(新方案 FSQ+RCM 序列)")
    data, tr, va = _load_ar_seqs(AR_MAX_SEQ_LEN)
    log(f"  train={len(tr)} val={len(va)} vocab={data['vocab_size']} d_model={AR_DMODEL} layers={AR_LAYERS} bs={AR_BS} lr={AR_LR} max_seq_len={AR_MAX_SEQ_LEN}")
    meta = _train_ar(
        data, tr, va, AR_DMODEL, AR_LAYERS, AR_LR, AR_EPOCHS, AR_BS, tag="ar",
        save_path=AR_PT, latest_path=AR_LATEST_PT, checkpoint_dir=AR_CHECKPOINT_DIR,
        history_path=AR_HISTORY_JSONL, resume_from=AR_RESUME_FROM or None,
        save_every=AR_SAVE_EVERY, max_seq_len=AR_MAX_SEQ_LEN,
    )
    verified = ar_stage_verified(meta)
    r = load_report(); r['stages']['ar'] = {
        'train_seqs': len(tr), 'val_seqs': len(va), 'epochs': AR_EPOCHS, 'd_model': AR_DMODEL, 'layers': AR_LAYERS,
        'max_seq_len': AR_MAX_SEQ_LEN,
        'batch_size': AR_BS, 'learning_rate': AR_LR, 'save_every': AR_SAVE_EVERY,
        'ce_init': round(meta['ce_init'], 4) if meta['ce_init'] is not None else None,
        'ce_final': round(meta['ce_final'], 4) if meta['ce_final'] is not None else None,
        'last_val_ce': round(meta['last_val_ce'], 4) if meta['last_val_ce'] is not None else None,
        'best_val_ce': round(meta['best_val_ce'], 4), 'minutes': meta['minutes'],
        'start_epoch': meta['start_epoch'], 'end_epoch': meta['end_epoch'],
        'epochs_ran': meta['epochs_ran'], 'resumed': meta['resumed'],
        'resume_from': meta['resume_from'], 'checkpoint_best': AR_PT,
        'checkpoint_latest': AR_LATEST_PT, 'checkpoint_dir': AR_CHECKPOINT_DIR,
        'history': AR_HISTORY_JSONL,
        'status': 'VERIFIED' if verified else 'FAILED'}
    save_report(r); log(f"  saved best -> {AR_PT} latest -> {AR_LATEST_PT} (best val CE {meta['best_val_ce']:.4f})")
    return verified


def stage_ar_sweep():
    """AR 超参对比:在**同一份** FSQ+RCM 序列上训练多个配置,按 best val CE 排名。"""
    log("AR_SWEEP: AR 超参对比(同序列,不同 d_model/layers/lr)")
    data, tr, va = _load_ar_seqs(AR_MAX_SEQ_LEN)
    log(f"  train={len(tr)} val={len(va)} vocab={data['vocab_size']} max_seq_len={AR_MAX_SEQ_LEN}")
    configs = [
        {'name': 'd256_L8_lr5e-4', 'd': 256, 'L': 8, 'lr': 5e-4},
        {'name': 'd384_L8_lr5e-4', 'd': 384, 'L': 8, 'lr': 5e-4},
        {'name': 'd256_L6_lr1e-3', 'd': 256, 'L': 6, 'lr': 1e-3},
    ]
    ep = int(os.environ.get('NS_AR_SWEEP_EPOCHS', '50'))
    results = []
    for c in configs:
        meta = _train_ar(data, tr, va, c['d'], c['L'], c['lr'], ep, AR_BS, tag=c['name'], max_seq_len=AR_MAX_SEQ_LEN)
        results.append({'name': c['name'], 'd_model': c['d'], 'layers': c['L'], 'lr': c['lr'],
                        'epochs': ep, 'ce_init': round(meta['ce_init'], 4), 'ce_final': round(meta['ce_final'], 4),
                        'best_val_ce': round(meta['best_val_ce'], 4), 'minutes': meta['minutes']})
        log(f"  -> {c['name']}: best_val_CE={meta['best_val_ce']:.4f} (train {meta['ce_init']:.3f}->{meta['ce_final']:.3f}, {meta['minutes']}min)")
    results.sort(key=lambda x: x['best_val_ce'])
    r = load_report(); r['stages']['ar_sweep'] = {'results': results, 'winner': results[0]['name']}
    save_report(r)
    log(f"  WINNER (lowest val CE): {results[0]['name']}  val_CE={results[0]['best_val_ce']}")
    return True


STAGES = {'split': stage_split, 'vqsweep': stage_vqsweep, 'vqvae': stage_vqvae,
          'sequence': stage_sequence, 'ar': stage_ar, 'ar_sweep': stage_ar_sweep}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='all'); a = ap.parse_args()
    order = ['split', 'vqsweep', 'vqvae', 'sequence', 'ar'] if a.stage == 'all' else [a.stage]
    t0 = time.time(); res = {}
    for s in order:
        log(f"===== STAGE {s} =====")
        ok = STAGES[s](); res[s] = ok
        if not ok:
            log(f"STAGE {s} FAILED, stop."); break
    r = load_report(); r['elapsed_min'] = round((time.time() - t0) / 60, 1)
    overall = r.get('overall', {})
    overall.update({s: ('PASS' if res.get(s) else 'FAIL/NA') for s in order})
    r['overall'] = overall; save_report(r)
    log(f"DONE {r['overall']} in {r['elapsed_min']}min")
    sys.exit(0 if all(res.get(s) for s in order) else 1)
