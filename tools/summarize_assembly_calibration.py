"""Summarize CAD assembly validity as a function of reconstruction error."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence


def _wilson_interval(successes: int, attempts: int, confidence: float = 0.95) -> list[float | None]:
    if attempts <= 0:
        return [None, None]
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / attempts
    denominator = 1.0 + z * z / attempts
    center = (p + z * z / (2.0 * attempts)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * attempts)) / attempts) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for position in order[index:end]:
            ranks[position] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator <= 0:
        return 0.0
    return sum(a * b for a, b in zip(centered_left, centered_right)) / denominator


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = len(rows)
    valid = sum(bool(row.get("brep_valid")) for row in rows)
    step_saved = sum(bool(row.get("step_saved")) for row in rows)
    finite = [
        row for row in rows
        if row.get("curved_mse") is not None and math.isfinite(float(row["curved_mse"]))
    ]
    log_errors = [math.log10(max(float(row["curved_mse"]), 1e-16)) for row in finite]
    invalid_indicator = [0.0 if bool(row.get("brep_valid")) else 1.0 for row in finite]
    association = _pearson(_rankdata(log_errors), _rankdata(invalid_indicator))
    ordered_rows = sorted(finite, key=lambda row: float(row["curved_mse"]))
    errors = [float(row["curved_mse"]) for row in ordered_rows]
    empirical_gate = None
    if ordered_rows:
        minimum_gate_samples = min(10, len(ordered_rows))
        for end in range(max(1, minimum_gate_samples), len(ordered_rows) + 1):
            prefix = ordered_rows[:end]
            if sum(bool(row.get("brep_valid")) for row in prefix) / end >= 0.8:
                empirical_gate = float(prefix[-1]["curved_mse"])
    bins = []
    if errors:
        for bin_index in range(min(4, len(errors))):
            start = bin_index * len(errors) // min(4, len(errors))
            end = (bin_index + 1) * len(errors) // min(4, len(errors))
            selected_rows = ordered_rows[start:end]
            selected = [float(row["curved_mse"]) for row in selected_rows]
            bin_attempts = len(selected_rows)
            bin_valid = sum(bool(row.get("brep_valid")) for row in selected_rows)
            bins.append(
                {
                    "mse_min": min(selected),
                    "mse_max": max(selected),
                    "attempts": bin_attempts,
                    "brep_valid": bin_valid,
                    "valid_rate": bin_valid / bin_attempts if bin_attempts else None,
                }
            )
    return {
        "attempts": attempts,
        "step_saved": step_saved,
        "step_saved_rate": step_saved / attempts if attempts else None,
        "brep_valid": valid,
        "valid_rate": valid / attempts if attempts else None,
        "valid_rate_ci95": _wilson_interval(valid, attempts),
        "finite_curved_mse": len(finite),
        "spearman_error_vs_invalid": association,
        "empirical_curved_mse_gate_at_80pct_valid": empirical_gate,
        "curved_mse_bins": bins,
        "failure_stages": dict(
            sorted(
                {
                    status: sum(str(row.get("status")) == status for row in rows)
                    for status in {str(row.get("status")) for row in rows}
                }.items()
            )
        ),
    }


def summarize_calibration(
    rows: Iterable[Mapping[str, Any]],
    *,
    acceptable_valid_rate: float = 0.8,
    min_cads: int = 100,
    strong_association: float = 0.35,
) -> dict[str, Any]:
    ordered = [dict(row) for row in rows]
    # manifest 是 append-only:断点续跑/换 checkpoint 重跑会追加新行。
    # 同一 (arm, cad_id) 只保留最后一次尝试,旧行不得混入统计。
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ordered:
        latest[(str(row["arm"]), str(row["cad_id"]))] = row
    normalized = list(latest.values())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[str(row["arm"])].append(row)
    for name, group in sorted(grouped.items()):
        checkpoint_shas = {
            str(row.get("checkpoint_sha256"))
            for row in group
            if row.get("checkpoint_sha256")
        }
        if len(checkpoint_shas) > 1:
            raise RuntimeError(
                f"arm {name!r} mixes rows from different checkpoints: "
                f"{sorted(checkpoint_shas)}; re-run the oracle into a fresh "
                "output dir (or prune stale manifest rows) before summarizing"
            )
    arms = {name: _arm_summary(group) for name, group in sorted(grouped.items())}
    by_cad: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in normalized:
        by_cad[str(row["cad_id"])][str(row["arm"])] = row
    paired_against_original = {}
    for model_name in sorted(name for name in arms if name != "original"):
        matched = [
            group for group in by_cad.values()
            if "original" in group and model_name in group
        ]
        original_valid = [group for group in matched if bool(group["original"].get("brep_valid"))]
        both_valid = sum(bool(group[model_name].get("brep_valid")) for group in original_valid)
        model_valid_original_invalid = sum(
            bool(group[model_name].get("brep_valid"))
            for group in matched
            if not bool(group["original"].get("brep_valid"))
        )
        paired_against_original[model_name] = {
            "matched_cads": len(matched),
            "original_valid_cads": len(original_valid),
            "both_valid": both_valid,
            "original_valid_model_invalid": len(original_valid) - both_valid,
            "original_invalid_model_valid": model_valid_original_invalid,
            "model_valid_rate_given_original_valid": (
                both_valid / len(original_valid) if original_valid else None
            ),
        }
    original = arms.get("original")
    model_names = [name for name in arms if name != "original"]
    primary_name = "continuous_bypass_64d" if "continuous_bypass_64d" in arms else (model_names[0] if model_names else None)
    primary = arms.get(primary_name) if primary_name else None

    status = "INSUFFICIENT_EVIDENCE"
    reason = "calibration does not have the required attempt coverage"
    if original and original["attempts"] >= min_cads and float(original["valid_rate"] or 0.0) < acceptable_valid_rate:
        status = "ASSEMBLY_CONTROL_FAILED"
        reason = "original parsed patches do not meet the assembly validity floor"
    elif primary and primary["attempts"] >= min_cads:
        valid_rate = float(primary["valid_rate"] or 0.0)
        association = primary.get("spearman_error_vs_invalid")
        if valid_rate >= acceptable_valid_rate:
            status = "CURRENT_ERROR_ACCEPTABLE"
            reason = "current continuous reconstruction error already assembles at an acceptable rate"
        elif association is None:
            status = "INSUFFICIENT_EVIDENCE"
            reason = "not enough finite curved_mse rows to estimate the error-validity association"
        elif float(association) >= strong_association:
            status = "REPRESENTATION_ERROR_CORRELATED"
            reason = "low validity is strongly associated with higher curved reconstruction error"
        else:
            status = "ASSEMBLY_DOMINATED"
            reason = "low validity is weakly associated with curved reconstruction error"

    return {
        "rows": len(normalized),
        "arms": arms,
        "paired_against_original": paired_against_original,
        "decision": {
            "status": status,
            "reason": reason,
            "primary_arm": primary_name,
            "acceptable_valid_rate": float(acceptable_valid_rate),
            "min_cads": int(min_cads),
            "strong_association": float(strong_association),
            "advance_to_vq_300k": status == "CURRENT_ERROR_ACCEPTABLE",
            "decoder_work_authorized": status == "REPRESENTATION_ERROR_CORRELATED",
            "assembly_repair_required": status in {"ASSEMBLY_DOMINATED", "ASSEMBLY_CONTROL_FAILED"},
            "advance_to_ar": False,
        },
    }


def render_calibration_png(summary: Mapping[str, Any], output_path: Path) -> None:
    """Render validity-versus-curved-MSE calibration without Matplotlib."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1350, 864
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot = (130, 90, 1280, 735)
    left, top, right, bottom = plot
    draw.text((left, 35), "Curved reconstruction MSE vs strict BRep validity", fill="#111111", font=font)
    draw.line((left, top, left, bottom), fill="#222222", width=2)
    draw.line((left, bottom, right, bottom), fill="#222222", width=2)
    for index in range(6):
        y = bottom - index * (bottom - top) / 5
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((55, y - 6), f"{index / 5:.1f}", fill="#333333", font=font)
    draw.text((25, top - 20), "valid rate", fill="#333333", font=font)
    draw.text(((left + right) // 2 - 80, bottom + 55), "curved MSE (log scale)", fill="#333333", font=font)

    colors = {
        "continuous_bypass_64d": "#1f77b4",
        "fsq_8192_4d": "#d62728",
        "original": "#2ca02c",
    }
    arm_bins = {
        name: list((arm or {}).get("curved_mse_bins") or [])
        for name, arm in (summary.get("arms") or {}).items()
        if name != "original"
    }
    positive = [
        float(bucket["mse_max"])
        for bins in arm_bins.values()
        for bucket in bins
        if bucket.get("mse_max") is not None and float(bucket["mse_max"]) > 0
    ]
    minimum = min(positive) if positive else 1e-6
    maximum = max(positive) if positive else 1e-2
    if maximum <= minimum:
        maximum = minimum * 10.0
    log_min, log_max = math.log10(minimum), math.log10(maximum)
    for arm_index, (name, bins) in enumerate(sorted(arm_bins.items())):
        points = []
        for bucket in bins:
            low, high = float(bucket["mse_min"]), float(bucket["mse_max"])
            center = math.sqrt(max(low, 1e-16) * max(high, 1e-16))
            x = left + (math.log10(center) - log_min) / (log_max - log_min) * (right - left)
            y = bottom - float(bucket["valid_rate"] or 0.0) * (bottom - top)
            points.append((x, y))
        color = colors.get(name, ("#9467bd", "#8c564b", "#17becf")[arm_index % 3])
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="white")
        legend_y = top + arm_index * 24
        draw.line((right - 260, legend_y, right - 225, legend_y), fill=color, width=4)
        draw.text((right - 215, legend_y - 7), name, fill="#222222", font=font)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at line {line_number}") from exc
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acceptable-valid-rate", type=float, default=0.8)
    parser.add_argument("--min-cads", type=int, default=100)
    parser.add_argument("--strong-association", type=float, default=0.35)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_calibration(
        _read_jsonl(args.manifest),
        acceptable_valid_rate=args.acceptable_valid_rate,
        min_cads=args.min_cads,
        strong_association=args.strong_association,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "assembly_calibration_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    render_calibration_png(summary, args.output_dir / "assembly_calibration.png")
    print(json.dumps(summary["decision"], indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if summary["decision"]["status"] != "INSUFFICIENT_EVIDENCE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
