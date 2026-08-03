"""Bucketed VQ-VAE reconstruction diagnostics.

This tool evaluates one or more FSQ VQ-VAE checkpoints on validation geometry
patches and reports reconstruction MSE by source complexity and curvature
buckets. It is intended to answer whether a held checkpoint failed uniformly or
mainly on complex/curved CAD geometry.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"


BUCKET_ORDER = [
    "all",
    "surface",
    "edge",
    "simple_source",
    "complex_source",
    "flat_patch",
    "curved_patch",
    "simple_flat_patch",
    "simple_curved_patch",
    "complex_flat_patch",
    "complex_curved_patch",
]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def buckets_for_record(record: dict[str, Any], curved_threshold: float = 0.02) -> list[str]:
    """Return diagnostic bucket names for a sampled patch record."""

    is_complex = bool(record.get("is_complex_source", False))
    curvature = _finite_float(record.get("curvature_score")) or 0.0
    is_curved = curvature >= float(curved_threshold)
    kind = str(record.get("kind", "")).strip().lower()

    buckets = ["all"]
    if kind in {"surface", "edge"}:
        buckets.append(kind)
    buckets.append("complex_source" if is_complex else "simple_source")
    buckets.append("curved_patch" if is_curved else "flat_patch")
    buckets.append(
        "{}_{}_patch".format(
            "complex" if is_complex else "simple",
            "curved" if is_curved else "flat",
        )
    )
    return buckets


def _loss_stats(values: list[float], target_loss: float) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
            "target_ratio": None,
            "above_target_rate": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    target = float(target_loss)
    mean = float(np.mean(arr))
    return {
        "count": int(len(values)),
        "mean": mean,
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "target_ratio": (mean / target) if target > 0 else None,
        "above_target_rate": float(np.mean(arr > target)) if target > 0 else None,
    }


def summarize_bucket_losses(
    records: list[dict[str, Any]],
    losses: list[float],
    target_loss: float = 1e-6,
    curved_threshold: float = 0.02,
) -> dict[str, dict[str, Any]]:
    """Summarize per-patch losses into complexity and curvature buckets."""

    if len(records) != len(losses):
        raise ValueError(f"records/losses length mismatch: {len(records)} != {len(losses)}")

    bucket_values: dict[str, list[float]] = {bucket: [] for bucket in BUCKET_ORDER}
    for record, loss in zip(records, losses):
        number = _finite_float(loss)
        if number is None:
            continue
        for bucket in buckets_for_record(record, curved_threshold=curved_threshold):
            bucket_values.setdefault(bucket, []).append(number)

    return {
        bucket: _loss_stats(bucket_values.get(bucket, []), target_loss=target_loss)
        for bucket in BUCKET_ORDER
        if bucket_values.get(bucket)
    }


def compare_bucket_summaries(
    summaries: dict[str, dict[str, dict[str, Any]]],
    reference_label: str,
) -> dict[str, Any]:
    """Compare checkpoint bucket summaries against a reference label."""

    if reference_label not in summaries:
        raise ValueError(f"missing reference summary: {reference_label}")

    reference = summaries[reference_label]
    comparisons: dict[str, dict[str, Any]] = {}
    regressions: list[tuple[float, str, str]] = []

    for label, summary in summaries.items():
        if label == reference_label:
            continue
        checkpoint_comparison: dict[str, Any] = {}
        for bucket in BUCKET_ORDER:
            if bucket not in reference or bucket not in summary:
                continue
            ref_mean = _finite_float(reference[bucket].get("mean"))
            cur_mean = _finite_float(summary[bucket].get("mean"))
            if ref_mean is None or cur_mean is None:
                continue
            delta = cur_mean - ref_mean
            relative = cur_mean / ref_mean if ref_mean > 0 else None
            checkpoint_comparison[bucket] = {
                "count": int(summary[bucket].get("count", 0)),
                "reference_count": int(reference[bucket].get("count", 0)),
                "mean": cur_mean,
                "reference_mean": ref_mean,
                "delta_mean": delta,
                "relative_mean": relative,
            }
            if delta > 0 and relative is not None:
                regressions.append((relative, label, bucket))
        comparisons[label] = checkpoint_comparison

    regressions.sort(reverse=True)
    worst_buckets = []
    seen = set()
    for _, _, bucket in regressions:
        if bucket not in seen:
            worst_buckets.append(bucket)
            seen.add(bucket)
        if len(worst_buckets) >= 5:
            break

    return {
        "reference_label": reference_label,
        "checkpoints": comparisons,
        "worst_regression_buckets": worst_buckets,
    }


def top_loss_records(
    records: list[dict[str, Any]],
    losses_by_label: dict[str, list[float]],
    limit: int = 10,
    curved_threshold: float = 0.02,
) -> dict[str, list[dict[str, Any]]]:
    """Return the highest-loss records for each evaluated checkpoint."""

    limit = max(0, int(limit))
    output: dict[str, list[dict[str, Any]]] = {}
    if limit == 0:
        return {label: [] for label in losses_by_label}

    for label, losses in losses_by_label.items():
        if len(records) != len(losses):
            raise ValueError(f"records/losses length mismatch for {label}: {len(records)} != {len(losses)}")
        ranked: list[tuple[float, int]] = []
        for index, loss in enumerate(losses):
            number = _finite_float(loss)
            if number is not None:
                ranked.append((number, index))
        ranked.sort(reverse=True)
        rows: list[dict[str, Any]] = []
        for loss, index in ranked[:limit]:
            record = records[index]
            rows.append(
                {
                    "record_id": str(record.get("record_id", "")),
                    "source_path": str(record.get("source_path", "")),
                    "kind": str(record.get("kind", "")),
                    "n_faces": int(record.get("n_faces", 0)),
                    "n_edges": int(record.get("n_edges", 0)),
                    "curvature_score": _finite_float(record.get("curvature_score")) or 0.0,
                    "is_complex_source": bool(record.get("is_complex_source", False)),
                    "buckets": buckets_for_record(record, curved_threshold=curved_threshold),
                    "loss": loss,
                }
            )
        output[label] = rows
    return output


def filter_records_by_source_caps(
    records: list[dict[str, Any]],
    max_faces: int | None = None,
    max_edges: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop patch records from sources outside downstream face/edge caps."""

    face_cap = None if max_faces is None or int(max_faces) <= 0 else int(max_faces)
    edge_cap = None if max_edges is None or int(max_edges) <= 0 else int(max_edges)
    kept: list[dict[str, Any]] = []
    dropped_faces = 0
    dropped_edges = 0

    for record in records:
        n_faces = int(record.get("n_faces", 0))
        n_edges = int(record.get("n_edges", 0))
        too_many_faces = face_cap is not None and n_faces > face_cap
        too_many_edges = edge_cap is not None and n_edges > edge_cap
        if too_many_faces or too_many_edges:
            if too_many_faces:
                dropped_faces += 1
            if too_many_edges:
                dropped_edges += 1
            continue
        kept.append(record)

    return kept, {
        "input_records": int(len(records)),
        "kept_records": int(len(kept)),
        "max_faces": face_cap,
        "max_edges": edge_cap,
        "dropped_records": int(len(records) - len(kept)),
        "dropped_too_many_faces": int(dropped_faces),
        "dropped_too_many_edges": int(dropped_edges),
    }


def load_split_paths(split_path: Path, split_name: str) -> list[Path]:
    with open(split_path, "rb") as handle:
        split = pickle.load(handle)
    if split_name not in split:
        raise ValueError(f"split {split_name!r} not found in {split_path}")
    return [Path(path) for path in split[split_name]]


def collect_records(
    paths: list[Path],
    sample_cap: int,
    seed: int,
    complex_fraction: float,
    curved_fraction: float,
    complex_min_faces: int,
    complex_min_edges: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sys.path.insert(0, str(IMPROVEMENTS_DIR))
    from vqvae_sampling import collect_vqvae_sample_records

    return collect_vqvae_sample_records(
        paths,
        cap=sample_cap,
        seed=seed,
        complex_fraction=complex_fraction,
        curved_fraction=curved_fraction,
        complex_min_faces=complex_min_faces,
        complex_min_edges=complex_min_edges,
    )


def load_vqvae_model(checkpoint: Path, device: str):
    sys.path.insert(0, str(IMPROVEMENTS_DIR))
    import torch
    from train import build_fsq_vqvae

    ckpt = torch.load(checkpoint, map_location=device)
    levels = tuple(int(x) for x in ckpt.get("fsq_levels", [8, 8, 8, 16]))
    model = build_fsq_vqvae(levels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate_checkpoint_losses(
    checkpoint: Path,
    records: list[dict[str, Any]],
    batch_size: int,
    device: str,
) -> list[float]:
    import torch

    sys.path.insert(0, str(IMPROVEMENTS_DIR))
    from vqvae_sampling import records_to_chw_array

    model = load_vqvae_model(checkpoint, device=device)
    patches = records_to_chw_array(records)
    losses: list[float] = []

    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            xb = torch.from_numpy(patches[start : start + batch_size]).to(device)
            h = model.encoder(xb)
            h = model.quant_conv(h)
            zq, _, _ = model.quantize(h)
            recon = model.decoder(model.post_quant_conv(zq))
            per_sample = (recon - xb).pow(2).flatten(1).mean(dim=1)
            losses.extend(float(x) for x in per_sample.detach().cpu().tolist())
    return losses


def parse_checkpoint_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty checkpoint label in {value!r}")
    return label, Path(path)


def write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# VQ-VAE Bucket Diagnostic",
        "",
        f"- split: `{report['split_path']}` / `{report['split_name']}`",
        f"- sample count: `{report['sample_count']}`",
        f"- target loss: `{report['target_loss']}`",
        f"- decision: `{report['decision']['status']}`",
        "",
        "## Sampling",
        "",
    ]
    for key, value in report["sampling_summary"].items():
        lines.append(f"- {key}: `{value}`")
    if report.get("source_cap_filter"):
        lines.extend(["", "## Source Cap Filter", ""])
        for key, value in report["source_cap_filter"].items():
            lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Checkpoint Bucket Means", ""])
    buckets = ["all", "simple_source", "complex_source", "flat_patch", "curved_patch", "complex_curved_patch"]
    header = "| Checkpoint | " + " | ".join(buckets) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in buckets) + " |"
    lines.extend([header, sep])
    for label, summary in report["checkpoints"].items():
        row = [label]
        for bucket in buckets:
            stat = summary.get(bucket, {})
            mean = stat.get("mean")
            row.append("n/a" if mean is None else f"{float(mean):.6g}")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Comparison", ""])
    comparison = report["comparison"]
    for label, bucket_map in comparison["checkpoints"].items():
        lines.append(f"### {label} vs {comparison['reference_label']}")
        lines.append("")
        lines.append("| Bucket | Mean | Reference | Delta | Relative |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for bucket, item in bucket_map.items():
            lines.append(
                "| {bucket} | {mean:.6g} | {ref:.6g} | {delta:.6g} | {rel:.3f} |".format(
                    bucket=bucket,
                    mean=float(item["mean"]),
                    ref=float(item["reference_mean"]),
                    delta=float(item["delta_mean"]),
                    rel=float(item["relative_mean"]),
                )
            )
        lines.append("")

    if report.get("top_losses"):
        lines.extend(["## Top Loss Records", ""])
        for label, rows in report["top_losses"].items():
            lines.append(f"### {label}")
            lines.append("")
            lines.append("| Loss | Kind | Faces | Edges | Curvature | Source |")
            lines.append("| ---: | --- | ---: | ---: | ---: | --- |")
            for row in rows:
                lines.append(
                    "| {loss:.6g} | {kind} | {faces} | {edges} | {curv:.4g} | `{source}` |".format(
                        loss=float(row["loss"]),
                        kind=row["kind"],
                        faces=int(row["n_faces"]),
                        edges=int(row["n_edges"]),
                        curv=float(row["curvature_score"]),
                        source=row["source_path"],
                    )
                )
            lines.append("")

    lines.extend(["## Recommendation", ""])
    for reason in report["decision"]["reasons"]:
        lines.append(f"- {reason}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide_from_report(
    checkpoint_summaries: dict[str, dict[str, dict[str, Any]]],
    target_loss: float,
    required_buckets: list[str] | None = None,
) -> dict[str, Any]:
    required_buckets = required_buckets or ["all", "complex_source", "curved_patch"]
    reasons: list[str] = []
    promoted = True

    for label, summary in checkpoint_summaries.items():
        for bucket in required_buckets:
            stat = summary.get(bucket)
            if not stat or not stat.get("count"):
                promoted = False
                reasons.append(f"{label}:{bucket} has no samples")
                continue
            mean = _finite_float(stat.get("mean"))
            if mean is None or mean > target_loss:
                promoted = False
                ratio = None if mean is None else mean / float(target_loss)
                suffix = "" if ratio is None else f" ({ratio:.1f}x target)"
                reasons.append(f"{label}:{bucket} mean is above target{suffix}")

    if promoted:
        return {
            "status": "bucket_losses_at_or_below_target",
            "next_action": "run_step_reconstruction_gate",
            "reasons": ["required bucket means are at or below target"],
        }
    return {
        "status": "diagnose_vqvae_before_retraining",
        "next_action": "inspect bucket losses and adjust decoder/loss/sampling before another run",
        "reasons": reasons,
    }


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_items = [parse_checkpoint_arg(item) for item in args.checkpoint]
    split_paths = load_split_paths(args.split, args.split_name)
    records, sampling_summary = collect_records(
        split_paths,
        sample_cap=args.sample_cap,
        seed=args.seed,
        complex_fraction=args.complex_fraction,
        curved_fraction=args.curved_fraction,
        complex_min_faces=args.complex_min_faces,
        complex_min_edges=args.complex_min_edges,
    )
    records, source_cap_filter = filter_records_by_source_caps(
        records,
        max_faces=args.max_source_faces,
        max_edges=args.max_source_edges,
    )
    if not records:
        raise RuntimeError("no diagnostic records were collected")

    checkpoint_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    checkpoint_loss_stats: dict[str, dict[str, Any]] = {}
    losses_by_label: dict[str, list[float]] = {}
    for label, checkpoint in checkpoint_items:
        losses = evaluate_checkpoint_losses(
            checkpoint,
            records,
            batch_size=args.batch_size,
            device=args.device,
        )
        losses_by_label[label] = losses
        checkpoint_summaries[label] = summarize_bucket_losses(
            records,
            losses,
            target_loss=args.target_loss,
            curved_threshold=args.curved_threshold,
        )
        checkpoint_loss_stats[label] = {
            "checkpoint": str(checkpoint),
            "finite_losses": int(sum(np.isfinite(loss) for loss in losses)),
            "loss_count": int(len(losses)),
        }

    reference_label = args.reference_label or checkpoint_items[0][0]
    comparison = compare_bucket_summaries(checkpoint_summaries, reference_label=reference_label)
    decision = decide_from_report(checkpoint_summaries, target_loss=args.target_loss)
    top_losses = top_loss_records(
        records,
        losses_by_label,
        limit=args.top_loss_limit,
        curved_threshold=args.curved_threshold,
    )

    return {
        "created": args.created,
        "split_path": str(args.split),
        "split_name": args.split_name,
        "sample_count": len(records),
        "target_loss": args.target_loss,
        "curved_threshold": args.curved_threshold,
        "sampling_summary": sampling_summary,
        "source_cap_filter": source_cap_filter,
        "loss_stats": checkpoint_loss_stats,
        "checkpoints": checkpoint_summaries,
        "comparison": comparison,
        "top_losses": top_losses,
        "decision": decision,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH; may be repeated")
    parser.add_argument("--reference-label", default="")
    parser.add_argument("--sample-cap", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--complex-fraction", type=float, default=0.5)
    parser.add_argument("--curved-fraction", type=float, default=0.5)
    parser.add_argument("--complex-min-faces", type=int, default=12)
    parser.add_argument("--complex-min-edges", type=int, default=20)
    parser.add_argument("--max-source-faces", type=int, default=0)
    parser.add_argument("--max-source-edges", type=int, default=0)
    parser.add_argument("--curved-threshold", type=float, default=0.02)
    parser.add_argument("--target-loss", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-loss-limit", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--created", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.created = args.created or __import__("time").strftime("%Y-%m-%d %H:%M:%S")
    report = run_diagnostic(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(report, args.markdown_output)
    print(json.dumps({"status": report["decision"]["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
