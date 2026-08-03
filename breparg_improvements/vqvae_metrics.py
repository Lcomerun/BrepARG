"""Pure helpers for interpretable VQ-VAE validation metrics."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


VQ_BUCKETS = ("surface_planar_like", "surface_curved_proxy", "edge")


class VQValidationAccumulator:
    """Accumulate complete-validation code counts and per-sample bucket losses."""

    def __init__(self, codebook_size: int, buckets: Sequence[str]):
        self.codebook_size = int(codebook_size)
        if self.codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        self.buckets = list(buckets)
        self.offset = 0
        self.counts = np.zeros(self.codebook_size, dtype=np.int64)
        self.bucket_names: list[str] = []
        self.per_sample_mse: list[float] = []

    def update(self, per_sample_mse: object, encoding_indices: object) -> None:
        losses = np.asarray(_to_numpy(per_sample_mse), dtype=np.float64).reshape(-1)
        indices = np.asarray(_to_numpy(encoding_indices), dtype=np.int64).reshape(-1)
        end = self.offset + len(losses)
        if end > len(self.buckets):
            raise ValueError("validation losses exceed configured bucket labels")
        if np.any(indices < 0) or np.any(indices >= self.codebook_size):
            raise ValueError("encoding index outside configured codebook")
        self.counts += np.bincount(indices, minlength=self.codebook_size)
        self.bucket_names.extend(self.buckets[self.offset:end])
        self.per_sample_mse.extend(float(value) for value in losses)
        self.offset = end

    def summary(self) -> dict[str, object]:
        if self.offset != len(self.buckets):
            raise ValueError(
                f"validation samples do not match bucket labels: {self.offset} != {len(self.buckets)}"
            )
        return {
            "code_usage": aggregate_code_usage(self.counts, self.codebook_size),
            "reconstruction_mse": summarize_bucket_mse(
                self.bucket_names, self.per_sample_mse
            ),
        }


def _to_numpy(value: object) -> object:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    return numpy() if callable(numpy) else value


def surface_span_proxy(surface: np.ndarray) -> float:
    """Return the smallest-to-largest axis span ratio used as a curvature proxy."""
    points = np.asarray(surface, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return 0.0
    spans = np.ptp(points, axis=0)
    largest = float(np.max(spans))
    if not math.isfinite(largest) or largest <= 1e-8:
        return 0.0
    return float(np.min(spans) / largest)


def patch_bucket(record: Mapping[str, object], curved_threshold: float = 0.02) -> str:
    """Assign a validation patch to one of the three reporting buckets."""
    if str(record.get("kind", "")) == "edge":
        return "edge"
    score = surface_span_proxy(np.asarray(record.get("array"), dtype=np.float32))
    if score >= float(curved_threshold):
        return "surface_curved_proxy"
    return "surface_planar_like"


def aggregate_code_usage(counts: Sequence[int], codebook_size: int) -> dict[str, float | int]:
    """Summarize one histogram accumulated over the complete validation set."""
    codebook_size = int(codebook_size)
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive")
    histogram = np.asarray(counts, dtype=np.int64).reshape(-1)
    if len(histogram) != codebook_size:
        raise ValueError(
            f"histogram length does not match codebook: {len(histogram)} != {codebook_size}"
        )
    if np.any(histogram < 0):
        raise ValueError("code counts must be non-negative")
    total = int(histogram.sum())
    unique = int(np.count_nonzero(histogram))
    perplexity = 0.0
    if total:
        probabilities = histogram[histogram > 0].astype(np.float64) / float(total)
        perplexity = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "tokens": total,
        "unique_bins": unique,
        "coverage": float(unique / codebook_size),
        "entropy_perplexity": perplexity,
    }


def summarize_bucket_mse(
    buckets: Iterable[str], per_sample_mse: Iterable[float]
) -> dict[str, dict[str, float | int | None]]:
    """Compute sample-weighted MSE while retaining empty expected buckets."""
    buckets = list(buckets)
    per_sample_mse = list(per_sample_mse)
    if len(buckets) != len(per_sample_mse):
        raise ValueError("buckets and per_sample_mse must have the same length")
    totals = {name: 0.0 for name in VQ_BUCKETS}
    counts = {name: 0 for name in VQ_BUCKETS}
    for bucket, value in zip(buckets, per_sample_mse):
        if bucket not in totals:
            raise ValueError(f"unknown VQ validation bucket: {bucket}")
        value = float(value)
        if not math.isfinite(value):
            continue
        totals[bucket] += value
        counts[bucket] += 1
    return {
        name: {
            "samples": counts[name],
            "mse": totals[name] / counts[name] if counts[name] else None,
        }
        for name in VQ_BUCKETS
    }
