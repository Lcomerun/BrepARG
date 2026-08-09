"""Reconstruct a fixed validation CAD cohort from final VQ-VAE checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for value in (str(REPO_ROOT), str(REPO_ROOT / "breparg_improvements")):
    if value not in sys.path:
        sys.path.insert(0, value)

from breparg_improvements.vqvae_metrics import surface_plane_residual  # noqa: E402
try:  # noqa: E402
    from .run_assembly_calibration_oracle import (
        load_checkpoint_model,
        reconstruct_patch_batch,
        select_validation_cads,
        sha256_file,
    )
except ImportError:  # direct script execution
    from run_assembly_calibration_oracle import (
        load_checkpoint_model,
        reconstruct_patch_batch,
        select_validation_cads,
        sha256_file,
    )


ARMS = (
    "fsq_8192_4d",
    "fsq_4096_6d",
    "vq_4096_64d_random",
    "continuous_bypass_64d",
)


def surface_reconstruction_metrics(
    target: np.ndarray, reconstructed: np.ndarray, *, curved_threshold: float = 0.02
) -> dict[str, Any]:
    target = np.asarray(target, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    if target.shape != reconstructed.shape or target.ndim != 4 or target.shape[1:] != (32, 32, 3):
        raise ValueError(f"surface shape mismatch: {target.shape} vs {reconstructed.shape}")
    losses = np.mean((reconstructed - target) ** 2, axis=(1, 2, 3))
    curved = np.asarray(
        [surface_plane_residual(surface) >= curved_threshold for surface in target], dtype=bool
    )
    if not np.all(np.isfinite(losses)):
        raise ValueError("nonfinite surface reconstruction MSE")

    def bucket(mask: np.ndarray) -> float | None:
        values = losses[mask]
        return float(np.mean(values)) if len(values) else None

    return {
        "surface_count": int(len(losses)),
        "curved_surface_count": int(np.count_nonzero(curved)),
        "planar_surface_count": int(np.count_nonzero(~curved)),
        "surface_mse": bucket(np.ones(len(losses), dtype=bool)),
        "curved_mse": bucket(curved),
        "planar_mse": bucket(~curved),
        "max_surface_mse": float(np.max(losses)) if len(losses) else None,
    }


def aggregate_checkpoint_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    successful = [row for row in rows if row.get("status") == "saved"]

    def average(name: str) -> float | None:
        values = [float(row[name]) for row in successful if row.get(name) is not None]
        return mean(values) if values else None

    return {
        "attempts": len(rows),
        "successful_cads": len(successful),
        "failed_cads": len(rows) - len(successful),
        "cad_equal_surface_mse": average("surface_mse"),
        "cad_equal_curved_mse": average("curved_mse"),
        "cad_equal_planar_mse": average("planar_mse"),
        "total_surfaces": sum(int(row.get("surface_count") or 0) for row in successful),
        "total_curved_surfaces": sum(
            int(row.get("curved_surface_count") or 0) for row in successful
        ),
        "total_planar_surfaces": sum(
            int(row.get("planar_surface_count") or 0) for row in successful
        ),
    }


def discover_final_checkpoints(
    training_root: Path, *, seeds: Sequence[int], arms: Sequence[str], expected_epoch: int
) -> list[dict[str, Any]]:
    checkpoints = []
    for seed in seeds:
        sweep_path = Path(training_root) / f"seed{seed}" / "vqvae_hp_sweep.json"
        if not sweep_path.is_file():
            raise FileNotFoundError(f"sweep missing for seed {seed}: {sweep_path}")
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        by_name = {str(row.get("name")): row for row in sweep.get("mse_ranking") or []}
        if set(by_name) != set(arms):
            raise RuntimeError(f"seed {seed} arm set mismatch: {sorted(by_name)}")
        for arm in arms:
            row = by_name[arm]
            path = Path(str(row.get("checkpoint_final") or ""))
            if not path.is_file():
                raise FileNotFoundError(f"final checkpoint missing: seed={seed} arm={arm} path={path}")
            if int(row.get("epochs_ran") or 0) != expected_epoch:
                raise RuntimeError(f"seed {seed} arm {arm} did not run {expected_epoch} epochs")
            checkpoints.append({
                "seed": int(seed), "arm": arm, "path": str(path.resolve()),
                "sha256": sha256_file(path), "bytes": path.stat().st_size,
                "epochs_ran": int(row["epochs_ran"]),
                "final_checkpoint_epoch": int(row.get("final_checkpoint_epoch")),
            })
    return checkpoints


def aggregate_across_seeds(checkpoint_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in sorted({str(row["arm"]) for row in checkpoint_summaries}):
        rows = [row for row in checkpoint_summaries if str(row["arm"]) == arm]
        arm_summary: dict[str, Any] = {"seeds": sorted(int(row["seed"]) for row in rows)}
        for metric in (
            "cad_equal_surface_mse", "cad_equal_curved_mse", "cad_equal_planar_mse"
        ):
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            arm_summary[metric] = {
                "mean": mean(values) if values else None,
                "std": pstdev(values) if len(values) > 1 else (0.0 if values else None),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        result[arm] = arm_summary
    return result


def latest_attempt_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last retry for each checkpoint/CAD attempt key."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row.get("checkpoint_sha256")), str(row.get("cad_id")))] = dict(row)
    return list(latest.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--max-cads", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=20260810)
    parser.add_argument("--expected-epochs", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if str(args.breparg_root.resolve()) not in sys.path:
        sys.path.insert(0, str(args.breparg_root.resolve()))
    import torch

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    checkpoints = discover_final_checkpoints(
        args.training_root, seeds=seeds, arms=arms, expected_epoch=args.expected_epochs
    )
    cads = select_validation_cads(
        args.protocol_dir, max_cads=args.max_cads, seed=args.selection_seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "surface_reconstruction_manifest.jsonl"
    existing = []
    if manifest_path.is_file():
        existing = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = {
        (row.get("checkpoint_sha256"), row.get("cad_id")) for row in existing
        if row.get("status") == "saved" and Path(str(row.get("reconstruction_path"))).is_file()
    }
    rows = list(existing)
    for checkpoint in checkpoints:
        model, metadata = load_checkpoint_model(
            checkpoint["arm"], Path(checkpoint["path"]), device=args.device,
            import_output_root=args.output_dir / "import_state",
        )
        if int(metadata.get("checkpoint_epoch")) != args.expected_epochs - 1:
            raise RuntimeError(
                f"final checkpoint epoch mismatch for {checkpoint['arm']} seed {checkpoint['seed']}: "
                f"{metadata.get('checkpoint_epoch')}"
            )
        for cad in cads:
            key = (checkpoint["sha256"], cad["cad_id"])
            if key in done:
                continue
            row: dict[str, Any] = {
                "seed": checkpoint["seed"], "arm": checkpoint["arm"],
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_epoch": metadata.get("checkpoint_epoch"),
                "cad_id": cad["cad_id"], "parent_id": cad["parent_id"],
                "source_path": cad["path"], "status": "pending",
            }
            try:
                with Path(cad["path"]).open("rb") as handle:
                    parsed = pickle.load(handle)
                target = np.asarray(parsed["surf_ncs"], dtype=np.float32)
                reconstructed = reconstruct_patch_batch(
                    model, target, kind="surface", device=args.device,
                    batch_size=args.batch_size,
                )
                row.update(surface_reconstruction_metrics(target, reconstructed))
                reconstruction_path = (
                    args.output_dir / "arrays" / checkpoint["arm"]
                    / f"seed{checkpoint['seed']}" / f"{cad['cad_id']}.npz"
                )
                reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    reconstruction_path,
                    reconstructed_surfaces=reconstructed.astype(np.float32),
                )
                row.update({
                    "status": "saved", "reconstruction_path": str(reconstruction_path),
                    "reconstruction_bytes": reconstruction_path.stat().st_size,
                    "reconstruction_sha256": sha256_file(reconstruction_path),
                })
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            with manifest_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            rows.append(row); done.add(key)
            print(json.dumps({key: row.get(key) for key in ("seed", "arm", "cad_id", "status", "curved_mse")}), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = latest_attempt_rows(rows)
    checkpoint_summaries = []
    for checkpoint in checkpoints:
        selected = [
            row for row in rows
            if row.get("checkpoint_sha256") == checkpoint["sha256"]
        ]
        summary = aggregate_checkpoint_rows(selected)
        summary.update(checkpoint)
        checkpoint_summaries.append(summary)
    cross_seed = aggregate_across_seeds(checkpoint_summaries)
    final = {
        "status": "COMPLETED" if all(row["failed_cads"] == 0 for row in checkpoint_summaries) else "FAILED",
        "selection_seed": args.selection_seed, "selected_cads": len(cads),
        "cad_ids": [cad["cad_id"] for cad in cads], "checkpoints": checkpoint_summaries,
        "cross_seed": cross_seed, "advance_to_ar": False,
    }
    (args.output_dir / "surface_reconstruction_summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "surface_reconstruction_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "arm", "seed", "checkpoint_sha256", "attempts", "successful_cads",
            "failed_cads", "cad_equal_surface_mse", "cad_equal_curved_mse",
            "cad_equal_planar_mse",
        ])
        writer.writeheader()
        for row in checkpoint_summaries:
            writer.writerow({name: row.get(name) for name in writer.fieldnames})
    return 0 if final["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
