"""Pure helpers for interpretable VQ-VAE validation metrics."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


VQ_BUCKETS = ("surface_planar_like", "surface_curved_proxy", "edge")


def _symmetric_3x3_eigenvalues(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return sorted eigenvalues for a 3x3 symmetric matrix without LAPACK."""
    values = [[float(matrix[row, column]) for column in range(3)] for row in range(3)]
    for _ in range(24):
        pivot = max(
            ((abs(values[row][column]), row, column)
             for row in range(3) for column in range(row + 1, 3)),
            key=lambda item: item[0],
        )
        magnitude, row, column = pivot
        if magnitude <= 1e-14:
            break
        angle = 0.5 * math.atan2(2.0 * values[row][column], values[column][column] - values[row][row])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        for index in range(3):
            if index in (row, column):
                continue
            first = values[index][row]
            second = values[index][column]
            values[index][row] = values[row][index] = cosine * first - sine * second
            values[index][column] = values[column][index] = sine * first + cosine * second
        diagonal_row = values[row][row]
        diagonal_column = values[column][column]
        off_diagonal = values[row][column]
        values[row][row] = cosine * cosine * diagonal_row - 2.0 * sine * cosine * off_diagonal + sine * sine * diagonal_column
        values[column][column] = sine * sine * diagonal_row + 2.0 * sine * cosine * off_diagonal + cosine * cosine * diagonal_column
        values[row][column] = values[column][row] = 0.0
    return tuple(sorted(values[index][index] for index in range(3)))


class VQValidationAccumulator:
    """Accumulate complete-validation code counts and per-sample bucket losses."""

    def __init__(
        self,
        codebook_size: int,
        buckets: Sequence[str],
        parent_ids: Sequence[str | Sequence[str]] | None = None,
    ):
        self.codebook_size = int(codebook_size)
        if self.codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        self.buckets = list(buckets)
        self.parent_ids = list(parent_ids or [])
        if self.parent_ids and len(self.parent_ids) != len(self.buckets):
            raise ValueError("parent_ids and bucket labels must have the same length")
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
        result = {
            "code_usage": aggregate_code_usage(self.counts, self.codebook_size),
            "reconstruction_mse": summarize_bucket_mse(
                self.bucket_names, self.per_sample_mse
            ),
        }
        if self.parent_ids:
            result["parent_cluster_mse"] = summarize_parent_cluster_mse(
                self.parent_ids, self.per_sample_mse
            )
            result["parent_cluster_reconstruction_mse"] = {
                name: summarize_parent_cluster_mse(
                    [parent for parent, bucket in zip(self.parent_ids, self.bucket_names) if bucket == name],
                    [value for bucket, value in zip(self.bucket_names, self.per_sample_mse) if bucket == name],
                )
                for name in VQ_BUCKETS
            }
        return result


def _to_numpy(value: object) -> object:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    return numpy() if callable(numpy) else value


def surface_plane_residual(surface: np.ndarray) -> float:
    """Return a rotation-invariant smallest-to-largest PCA spread ratio."""
    points = np.asarray(surface, dtype=np.float32).reshape(-1, 3)
    if points.size == 0:
        return 0.0
    centered = points.astype(np.float64) - points.astype(np.float64).mean(axis=0)
    x_values, y_values, z_values = centered.T
    divisor = max(1, len(centered))
    covariance = np.asarray(
        [
            [np.sum(x_values * x_values) / divisor, np.sum(x_values * y_values) / divisor, np.sum(x_values * z_values) / divisor],
            [np.sum(x_values * y_values) / divisor, np.sum(y_values * y_values) / divisor, np.sum(y_values * z_values) / divisor],
            [np.sum(x_values * z_values) / divisor, np.sum(y_values * z_values) / divisor, np.sum(z_values * z_values) / divisor],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.maximum(np.asarray(_symmetric_3x3_eigenvalues(covariance)), 0.0)
    largest = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
    if not math.isfinite(largest) or largest <= 1e-8:
        return 0.0
    return float(math.sqrt(float(eigenvalues[0]) / largest))


def surface_span_proxy(surface: np.ndarray) -> float:
    """Backward-compatible alias for the rotation-invariant plane residual."""
    return surface_plane_residual(surface)


def patch_bucket(record: Mapping[str, object], curved_threshold: float = 0.02) -> str:
    """Assign a validation patch to one of the three reporting buckets."""
    if str(record.get("kind", "")) == "edge":
        return "edge"
    score = surface_span_proxy(np.asarray(record.get("array"), dtype=np.float32))
    if score >= float(curved_threshold):
        return "surface_curved_proxy"
    return "surface_planar_like"


def summarize_parent_cluster_mse(
    parent_ids: Iterable[str | Sequence[str]], per_sample_mse: Iterable[float]
) -> dict[str, float | int | None]:
    """Expand provenance groups, then average within and across original parents."""
    parent_groups = list(parent_ids)
    per_sample_mse = list(per_sample_mse)
    if len(parent_groups) != len(per_sample_mse):
        raise ValueError("parent_ids and per_sample_mse must have the same length")
    finite_by_parent: dict[str, list[float]] = {}
    nonfinite_parents = set()
    nonfinite_samples = 0
    nonfinite_parent_contributions = 0
    finite_samples = 0
    parent_patch_contributions = 0
    for parent_group, value in zip(parent_groups, per_sample_mse):
        raw_parents = [parent_group] if isinstance(parent_group, str) else list(parent_group)
        parents = tuple(dict.fromkeys(str(parent) for parent in raw_parents))
        if not parents:
            raise ValueError("each validation patch must identify at least one parent")
        value = float(value)
        if not math.isfinite(value):
            nonfinite_samples += 1
            nonfinite_parent_contributions += len(parents)
            nonfinite_parents.update(parents)
            continue
        for parent in parents:
            finite_by_parent.setdefault(parent, []).append(value)
        parent_patch_contributions += len(parents)
        finite_samples += 1
    parent_means = [sum(values) / len(values) for values in finite_by_parent.values()]
    return {
        "unique_patch_samples": finite_samples,
        "parent_patch_contributions": parent_patch_contributions,
        "parent_clusters": len(parent_means),
        "nonfinite_samples": nonfinite_samples,
        "nonfinite_parent_contributions": nonfinite_parent_contributions,
        "nonfinite_parents": len(nonfinite_parents),
        "mse": sum(parent_means) / len(parent_means) if parent_means else None,
    }


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
    nonfinite = {name: 0 for name in VQ_BUCKETS}
    for bucket, value in zip(buckets, per_sample_mse):
        if bucket not in totals:
            raise ValueError(f"unknown VQ validation bucket: {bucket}")
        value = float(value)
        if not math.isfinite(value):
            nonfinite[bucket] += 1
            continue
        totals[bucket] += value
        counts[bucket] += 1
    return {
        name: {
            "samples": counts[name],
            "nonfinite_samples": nonfinite[name],
            "mse": totals[name] / counts[name] if counts[name] else None,
        }
        for name in VQ_BUCKETS
    }
