"""Optional on-disk cache for expensive VQ-VAE patch sampling."""

from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_npz_path(path: Path | str) -> Path:
    """np.savez_compressed 会自动追加 .npz;save/load 必须落在同一个真实文件名上。"""
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


def _fingerprint_text(fingerprint: Any) -> str:
    return json.dumps(fingerprint, ensure_ascii=True, sort_keys=True, default=_json_default)


def save_vqvae_sample_cache(
    path: Path | str,
    samples: np.ndarray,
    weights: np.ndarray,
    summary: dict[str, Any],
    fingerprint: Any = None,
) -> None:
    path = _normalize_npz_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(samples) != len(weights):
        raise ValueError(f"samples and weights length mismatch: {len(samples)} != {len(weights)}")
    payload_summary = dict(summary or {})
    payload_summary["cache_samples"] = int(len(samples))
    if fingerprint is not None:
        payload_summary["cache_fingerprint"] = _fingerprint_text(fingerprint)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    try:
        np.savez_compressed(
            tmp,
            samples=samples,
            weights=weights,
            summary_json=np.array(json.dumps(payload_summary, ensure_ascii=True, default=_json_default)),
        )
        last = None
        for _ in range(50):
            try:
                os.replace(tmp, path)
                last = None
                break
            except PermissionError as exc:  # Windows: 读者持有目标句柄
                last = exc
                time.sleep(0.2)
        if last is not None:
            raise last
    finally:
        if tmp.exists():
            tmp.unlink()


def load_vqvae_sample_cache(
    path: Path | str,
    min_samples: int = 0,
    expected_fingerprint: Any = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw_path = Path(path)
    path = _normalize_npz_path(raw_path)
    if not path.exists():
        raise FileNotFoundError(path if path == raw_path else f"{raw_path} (resolved to {path})")
    try:
        with np.load(path, allow_pickle=False) as data:
            samples = np.asarray(data["samples"], dtype=np.float32)
            weights = np.asarray(data["weights"], dtype=np.float32)
            summary_json = str(np.asarray(data["summary_json"]).item())
    except (zipfile.BadZipFile, ValueError, KeyError, EOFError) as exc:
        raise ValueError(
            f"corrupt VQ sample cache {path}: {type(exc).__name__}: {exc}; "
            "delete the file and rebuild the cache"
        ) from exc
    if len(samples) != len(weights):
        raise ValueError(f"cache samples and weights length mismatch: {len(samples)} != {len(weights)}")
    min_samples = max(0, int(min_samples))
    if len(samples) < min_samples:
        raise ValueError(f"cache has fewer samples than requested: {len(samples)} < {min_samples}")
    summary = json.loads(summary_json)
    if expected_fingerprint is not None:
        expected_text = _fingerprint_text(expected_fingerprint)
        stored_text = summary.get("cache_fingerprint")
        if stored_text is None:
            # 旧版工具构建的缓存无指纹:放行但显式告警,配置漂移风险由使用者确认
            print(f"[vq_sample_cache] WARNING legacy cache without fingerprint: {path}",
                  flush=True)
        elif stored_text != expected_text:
            raise ValueError(
                f"stale VQ sample cache {path}: fingerprint mismatch "
                f"(stored={stored_text!r}, expected={expected_text!r}); "
                "sampling configuration changed — delete the cache and rebuild"
            )
    summary["cache_path"] = str(path)
    summary["cache_samples"] = int(len(samples))
    summary["cache_loaded"] = True
    return samples, weights, summary
