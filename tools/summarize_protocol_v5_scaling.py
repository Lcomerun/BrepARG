"""Summarize Protocol V5 reconstruction and code-usage scaling results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping


ARM_CODEBOOKS = {
    "fsq_8192_4d": 8192,
    "fsq_4096_6d": 4096,
    "vq_4096_64d_random": 4096,
}


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metric must be finite: {value!r}")
    return number


def summarize_scaling(
    rows: Iterable[Mapping[str, Any]],
    *,
    target_curved_mse: float = 5e-5,
    projected_full_patches: int = 3_000_000,
) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("at least one scaling result row is required")
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    normalized_rows = []
    for row in rows:
        patches = int(row["patches"])
        codebook = int(row["codebook_size"])
        if patches <= 0 or codebook <= 0:
            raise ValueError("patch and codebook sizes must be positive")
        normalized = {
            **row,
            "patches": patches,
            "seed": int(row["seed"]),
            "codebook_size": codebook,
            "curved_parent_mse": _finite(row["curved_parent_mse"]),
            "perplexity": _finite(row["perplexity"]),
            "usage_fraction": _finite(row["perplexity"]) / codebook,
        }
        normalized_rows.append(normalized)
        grouped[(patches, str(row["arm"]), str(row["protocol_scope"]))].append(normalized)

    points = []
    for (patches, arm, protocol_scope), group in sorted(grouped.items()):
        curved = [item["curved_parent_mse"] for item in group]
        usage = [item["usage_fraction"] for item in group]
        perplexity = [item["perplexity"] for item in group]
        points.append(
            {
                "patches": patches,
                "arm": arm,
                "protocol_scope": protocol_scope,
                "seeds": sorted(item["seed"] for item in group),
                "runs": len(group),
                "curved_parent_mse_mean": mean(curved),
                "curved_parent_mse_sd": stdev(curved) if len(curved) > 1 else 0.0,
                "perplexity_mean": mean(perplexity),
                "usage_fraction_mean": mean(usage),
                "usage_fraction_sd": stdev(usage) if len(usage) > 1 else 0.0,
                "codebook_size": group[0]["codebook_size"],
            }
        )

    point_lookup = {(point["patches"], point["arm"]): point for point in points}
    fsq_60 = point_lookup.get((60000, "fsq_4096_6d"))
    vq_60 = point_lookup.get((60000, "vq_4096_64d_random"))
    vq_control = {
        "available": bool(fsq_60 and vq_60),
        "vq_minus_fsq_curved_mse": (
            vq_60["curved_parent_mse_mean"] - fsq_60["curved_parent_mse_mean"]
            if fsq_60 and vq_60
            else None
        ),
        "interpretation": None,
    }
    if vq_control["available"]:
        vq_control["interpretation"] = (
            "LEARNED_VQ_LOWER_CURVED_MSE"
            if vq_control["vq_minus_fsq_curved_mse"] < 0
            else "FSQ_NOT_WORSE_AT_60K"
        )

    master_b = sorted(
        (
            point
            for point in points
            if point["arm"] == "fsq_4096_6d"
            and point["protocol_scope"] == "master"
            and point["patches"] in {60000, 300000}
        ),
        key=lambda item: item["patches"],
    )
    projection = {
        "method": "two_point_power_law_on_shared_master_protocol",
        "source_points": [point["patches"] for point in master_b],
        "projected_full_patches": int(projected_full_patches),
        "power": None,
        "projected_curved_parent_mse": None,
    }
    if len(master_b) == 2:
        left, right = master_b
        x1, x2 = left["patches"], right["patches"]
        y1 = left["curved_parent_mse_mean"]
        y2 = right["curved_parent_mse_mean"]
        if y1 > 0 and y2 > 0 and x2 > x1:
            power = math.log(y2 / y1) / math.log(x2 / x1)
            projection["power"] = power
            projection["projected_curved_parent_mse"] = y2 * (
                int(projected_full_patches) / x2
            ) ** power

    projected = projection["projected_curved_parent_mse"]
    target_plausible = projected is not None and projected <= float(target_curved_mse)
    decision = {
        "status": "TARGET_PLAUSIBLE" if target_plausible else "CONTINUE_CAPACITY_INVESTIGATION",
        "target_curved_parent_mse": float(target_curved_mse),
        "continuous_bypass_oracle_recommended": not target_plausible,
        "advance_to_ar": False,
        "reason": (
            "shared-protocol 60k-to-300k scaling projects at or below the target"
            if target_plausible
            else "shared-protocol 60k-to-300k scaling does not project to the target; isolate decoder/quantizer capacity before AR"
        ),
    }
    return {
        "schema_version": 1,
        "rows": normalized_rows,
        "points": points,
        "vq_control": vq_control,
        "projection": projection,
        "decision": decision,
        "limitations": [
            "12k is a historical single-chunk reference and is excluded from the projection fit.",
            "The two-point power-law projection is diagnostic, not a guarantee of full-data behavior.",
            "No result in this report authorizes sequence generation or AR training.",
        ],
    }


def collect_rows(v4_summary: Path, rung_root: Path) -> list[dict[str, Any]]:
    v4 = json.loads(Path(v4_summary).read_text(encoding="utf-8"))
    rows = []
    for run in v4.get("runs", []):
        arm = str(run.get("arm"))
        if arm not in {"fsq_8192_4d", "fsq_4096_6d"}:
            continue
        rows.append(
            {
                "patches": 12000,
                "protocol_scope": "historical_reference",
                "seed": int(run["seed"]),
                "arm": arm,
                "codebook_size": ARM_CODEBOOKS[arm],
                "curved_parent_mse": run["checkpoint_curved_parent_mse"],
                "perplexity": run["checkpoint_perplexity"],
            }
        )
    for patches, rung_name in ((60000, "60k"), (300000, "300k")):
        for sweep_path in sorted((Path(rung_root) / rung_name).glob("seed*/vqvae_hp_sweep.json")):
            payload = json.loads(sweep_path.read_text(encoding="utf-8"))
            seed = int(payload["run_manifest"]["experiment"]["seed"])
            for result in payload.get("mse_ranking", []):
                arm = str(result.get("name"))
                if arm not in ARM_CODEBOOKS:
                    continue
                metrics = result.get("best_val_metrics") or {}
                curved = (
                    metrics.get("parent_cluster_reconstruction_mse", {})
                    .get("surface_curved_proxy", {})
                    .get("mse")
                )
                perplexity = (metrics.get("code_usage") or {}).get("entropy_perplexity")
                if curved is None or perplexity is None:
                    raise ValueError(f"missing checkpoint scaling metrics: {sweep_path} {arm}")
                rows.append(
                    {
                        "patches": patches,
                        "protocol_scope": "master",
                        "seed": seed,
                        "arm": arm,
                        "codebook_size": int(result.get("codebook") or ARM_CODEBOOKS[arm]),
                        "curved_parent_mse": curved,
                        "perplexity": perplexity,
                    }
                )
    return rows


def write_outputs(summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scaling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "scaling_points.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "patches",
            "arm",
            "protocol_scope",
            "runs",
            "curved_parent_mse_mean",
            "curved_parent_mse_sd",
            "perplexity_mean",
            "usage_fraction_mean",
            "usage_fraction_sd",
            "codebook_size",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in summary["points"]:
            writer.writerow({name: point[name] for name in fieldnames})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "fsq_8192_4d": ("o", "FSQ 8192/4D"),
        "fsq_4096_6d": ("s", "FSQ 4096/6D"),
        "vq_4096_64d_random": ("^", "VQ 4096/64D random"),
    }
    for metric, ylabel, filename in (
        ("curved_parent_mse_mean", "Curved parent-cluster MSE", "curved_mse_scaling.png"),
        ("usage_fraction_mean", "Perplexity / codebook size", "usage_scaling.png"),
    ):
        figure, axis = plt.subplots(figsize=(7.5, 4.8))
        for arm, (marker, label) in styles.items():
            points = sorted(
                (point for point in summary["points"] if point["arm"] == arm),
                key=lambda item: item["patches"],
            )
            if not points:
                continue
            axis.plot(
                [point["patches"] for point in points],
                [point[metric] for point in points],
                marker=marker,
                label=label,
            )
        axis.set_xscale("log")
        if metric == "curved_parent_mse_mean":
            axis.set_yscale("log")
        axis.set_xlabel("Training patches")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-summary", type=Path, required=True)
    parser.add_argument("--rung-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-curved-mse", type=float, default=5e-5)
    parser.add_argument("--projected-full-patches", type=int, default=3_000_000)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = collect_rows(args.v4_summary, args.rung_root)
    summary = summarize_scaling(
        rows,
        target_curved_mse=args.target_curved_mse,
        projected_full_patches=args.projected_full_patches,
    )
    write_outputs(summary, args.output_dir)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
