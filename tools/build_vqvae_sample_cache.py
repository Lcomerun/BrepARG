"""Build a reusable FSQ VQ-VAE sample cache from VQ patch shards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BREPARG_IMPROVEMENTS = REPO_ROOT / "breparg_improvements"
if str(BREPARG_IMPROVEMENTS) not in sys.path:
    sys.path.insert(0, str(BREPARG_IMPROVEMENTS))

from vqvae_sample_cache import save_vqvae_sample_cache  # noqa: E402
from vqvae_sampling import (  # noqa: E402
    collect_vqvae_patch_shard_records,
    records_to_chw_array,
    records_to_patch_weights,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def patch_shard_paths(root: Path) -> list[Path]:
    patterns = ("vq_patch_shard_*.pkl.zst", "vq_patch_shard_*.pkl.gz", "vq_patch_shard_*.pkl")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    return sorted(set(paths))


def build_sample_cache(
    *,
    patch_shard_root: Path,
    output: Path,
    summary_output: Path | None,
    samples: int,
    seed: int,
    complex_fraction: float,
    complex_min_faces: int,
    complex_min_edges: int,
    curved_fraction: float,
    max_source_faces: int,
    max_source_edges: int,
    complex_loss_weight: float,
    curved_loss_weight: float,
    curved_loss_threshold: float,
    force: bool = False,
) -> dict[str, Any]:
    patch_shard_root = Path(patch_shard_root)
    output = Path(output)
    if output.exists() and not force:
        raise FileExistsError(f"sample cache already exists: {output}")
    shard_paths = patch_shard_paths(patch_shard_root)
    if not shard_paths:
        raise FileNotFoundError(f"no VQ patch shards under {patch_shard_root}")

    started = time.time()
    records, sampling = collect_vqvae_patch_shard_records(
        shard_paths,
        int(samples),
        seed=int(seed),
        complex_fraction=float(complex_fraction),
        complex_min_faces=int(complex_min_faces),
        complex_min_edges=int(complex_min_edges),
        curved_fraction=float(curved_fraction),
        max_source_faces=int(max_source_faces),
        max_source_edges=int(max_source_edges),
    )
    arrays = records_to_chw_array(records)
    weights = records_to_patch_weights(
        records,
        complex_weight=float(complex_loss_weight),
        curved_weight=float(curved_loss_weight),
        curved_threshold=float(curved_loss_threshold),
    )
    if len(arrays) == 0:
        raise RuntimeError("sample cache build selected zero patches")

    sampling = dict(sampling)
    sampling.update(
        {
            "source": "vq_patch_shards",
            "patch_shards_requested": len(shard_paths),
            "complex_loss_weight": float(complex_loss_weight),
            "curved_loss_weight": float(curved_loss_weight),
            "curved_loss_threshold": float(curved_loss_threshold),
            "weight_mean": float(np.mean(weights)) if len(weights) else None,
            "weight_max": float(np.max(weights)) if len(weights) else None,
        }
    )
    save_vqvae_sample_cache(output, arrays, weights, sampling)
    summary = {
        "status": "BUILT",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round((time.time() - started) / 60, 3),
        "patch_shard_root": str(patch_shard_root),
        "cache": {
            "path": str(output),
            "samples": int(len(arrays)),
            "bytes": output.stat().st_size if output.exists() else 0,
        },
        "sampling": sampling,
    }
    if summary_output:
        write_json(Path(summary_output), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--samples", type=int, default=450000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--complex-fraction", type=float, default=0.50)
    parser.add_argument("--complex-min-faces", type=int, default=12)
    parser.add_argument("--complex-min-edges", type=int, default=20)
    parser.add_argument("--curved-fraction", type=float, default=0.35)
    parser.add_argument("--max-source-faces", type=int, default=50)
    parser.add_argument("--max-source-edges", type=int, default=150)
    parser.add_argument("--complex-loss-weight", type=float, default=1.25)
    parser.add_argument("--curved-loss-weight", type=float, default=2.0)
    parser.add_argument("--curved-loss-threshold", type=float, default=0.02)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_sample_cache(
            patch_shard_root=args.patch_shard_root,
            output=args.output,
            summary_output=args.summary_output,
            samples=args.samples,
            seed=args.seed,
            complex_fraction=args.complex_fraction,
            complex_min_faces=args.complex_min_faces,
            complex_min_edges=args.complex_min_edges,
            curved_fraction=args.curved_fraction,
            max_source_faces=args.max_source_faces,
            max_source_edges=args.max_source_edges,
            complex_loss_weight=args.complex_loss_weight,
            curved_loss_weight=args.curved_loss_weight,
            curved_loss_threshold=args.curved_loss_threshold,
            force=args.force,
        )
    except Exception as exc:
        if args.summary_output:
            write_json(
                args.summary_output,
                {
                    "status": "FAILED",
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "patch_shard_root": str(args.patch_shard_root),
                    "cache": {"path": str(args.output)},
                },
            )
        raise
    print(json.dumps({"status": summary["status"], "cache": summary["cache"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
