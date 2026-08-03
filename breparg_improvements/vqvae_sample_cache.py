"""Optional on-disk cache for expensive VQ-VAE patch sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_vqvae_sample_cache(path: Path | str, samples: np.ndarray, weights: np.ndarray, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(samples) != len(weights):
        raise ValueError(f"samples and weights length mismatch: {len(samples)} != {len(weights)}")
    payload_summary = dict(summary or {})
    payload_summary["cache_samples"] = int(len(samples))
    np.savez_compressed(
        path,
        samples=samples,
        weights=weights,
        summary_json=np.array(json.dumps(payload_summary, ensure_ascii=True, default=_json_default)),
    )


def load_vqvae_sample_cache(path: Path | str, min_samples: int = 0) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        samples = np.asarray(data["samples"], dtype=np.float32)
        weights = np.asarray(data["weights"], dtype=np.float32)
        summary_json = str(np.asarray(data["summary_json"]).item())
    if len(samples) != len(weights):
        raise ValueError(f"cache samples and weights length mismatch: {len(samples)} != {len(weights)}")
    min_samples = max(0, int(min_samples))
    if len(samples) < min_samples:
        raise ValueError(f"cache has fewer samples than requested: {len(samples)} < {min_samples}")
    summary = json.loads(summary_json)
    summary["cache_path"] = str(path)
    summary["cache_samples"] = int(len(samples))
    summary["cache_loaded"] = True
    return samples, weights, summary
