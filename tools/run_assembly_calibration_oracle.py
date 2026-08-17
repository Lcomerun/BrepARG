"""Run CAD-level reconstruction-to-assembly calibration on Protocol V5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPROVEMENTS_DIR = REPO_ROOT / "breparg_improvements"
for item in (str(REPO_ROOT), str(IMPROVEMENTS_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

from breparg_improvements.cad_protocol import parent_cad_id  # noqa: E402
from breparg_improvements.vqvae_metrics import surface_plane_residual  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_validation_cads(
    protocol_dir: Path,
    *,
    max_cads: int,
    seed: int,
) -> list[dict[str, str]]:
    """Return a deterministic parent-unique subset from a verified val split."""
    protocol_dir = Path(protocol_dir)
    summary = _read_json(protocol_dir / "protocol_summary.json")
    if summary.get("status") != "VERIFIED":
        raise RuntimeError("calibration requires protocol status VERIFIED")
    overlaps = summary.get("parent_overlap_counts") or {}
    if any(int(value) != 0 for value in overlaps.values()):
        raise RuntimeError("calibration rejects protocol parent overlap")
    with (protocol_dir / "split.pkl").open("rb") as handle:
        split = pickle.load(handle)
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_path in split.get("val", []):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"validation CAD does not exist: {path}")
        parent = parent_cad_id(path.name)
        if parent is None:
            raise RuntimeError(f"validation CAD parent is unresolved: {path.name}")
        if parent in seen:
            raise RuntimeError(f"validation split contains duplicate parent: {parent}")
        seen.add(parent)
        candidates.append({"path": str(path), "parent_id": parent, "cad_id": path.stem})
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    limit = max(0, int(max_cads))
    return candidates[:limit]


def edge_patches_from_model_output(
    output: np.ndarray,
) -> np.ndarray:
    """Collapse edge outputs using BrepARG's width-mean decoding rule."""
    values = np.asarray(output, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (32, 32, 3):
        raise ValueError(f"edge model output must have shape (N,32,32,3), got {values.shape}")
    return np.mean(values, axis=2, dtype=np.float32)


def summarize_cad_reconstruction_error(
    target_surfaces: np.ndarray,
    reconstructed_surfaces: np.ndarray,
    target_edges: np.ndarray,
    reconstructed_edges: np.ndarray,
    *,
    curved_threshold: float = 0.02,
) -> dict[str, Any]:
    target_surfaces = np.asarray(target_surfaces, dtype=np.float32)
    reconstructed_surfaces = np.asarray(reconstructed_surfaces, dtype=np.float32)
    target_edges = np.asarray(target_edges, dtype=np.float32)
    reconstructed_edges = np.asarray(reconstructed_edges, dtype=np.float32)
    if target_surfaces.shape != reconstructed_surfaces.shape:
        raise ValueError("surface reconstruction shape mismatch")
    if target_edges.shape != reconstructed_edges.shape:
        raise ValueError("edge reconstruction shape mismatch")

    surface_losses = np.mean((reconstructed_surfaces - target_surfaces) ** 2, axis=(1, 2, 3))
    edge_losses = np.mean((reconstructed_edges - target_edges) ** 2, axis=(1, 2))
    curved_mask = np.asarray(
        [surface_plane_residual(surface) >= float(curved_threshold) for surface in target_surfaces],
        dtype=bool,
    )
    finite_surfaces = np.isfinite(surface_losses)
    finite_edges = np.isfinite(edge_losses)

    def finite_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        selected = values[mask & np.isfinite(values)]
        return float(np.mean(selected)) if len(selected) else None

    all_values = np.concatenate((surface_losses, edge_losses))
    return {
        "surface_count": int(len(surface_losses)),
        "curved_surface_count": int(np.count_nonzero(curved_mask)),
        "planar_surface_count": int(np.count_nonzero(~curved_mask)),
        "edge_count": int(len(edge_losses)),
        "curved_mse": finite_mean(surface_losses, curved_mask),
        "planar_mse": finite_mean(surface_losses, ~curved_mask),
        "surface_mse": finite_mean(surface_losses, np.ones(len(surface_losses), dtype=bool)),
        "edge_mse": finite_mean(edge_losses, np.ones(len(edge_losses), dtype=bool)),
        "global_patch_mse": float(np.mean(all_values[np.isfinite(all_values)])) if np.any(np.isfinite(all_values)) else None,
        "nonfinite_patches": int(np.count_nonzero(~finite_surfaces) + np.count_nonzero(~finite_edges)),
    }


def evaluate_cad_arm(
    cad: Mapping[str, Any],
    *,
    arm: str,
    checkpoint_sha256: str | None,
    parsed: Mapping[str, Any],
    reconstructed_surfaces: np.ndarray,
    reconstructed_edges: np.ndarray,
    output_dir: Path,
    assembler: Any,
) -> dict[str, Any]:
    """Evaluate one arm/CAD and convert stage-local failures into evidence rows."""
    row: dict[str, Any] = {
        "cad_id": str(cad["cad_id"]),
        "parent_id": str(cad["parent_id"]),
        "source_path": str(cad["path"]),
        "arm": str(arm),
        "checkpoint_sha256": checkpoint_sha256,
        "status": "pending",
        "step_saved": False,
        "brep_valid": False,
    }
    try:
        metrics = summarize_cad_reconstruction_error(
            parsed["surf_ncs"],
            reconstructed_surfaces,
            parsed["edge_ncs"],
            reconstructed_edges,
        )
        row.update(metrics)
        if metrics["nonfinite_patches"]:
            row["status"] = "nonfinite_reconstruction"
            return row
        result = assembler(
            parsed,
            reconstructed_surfaces,
            reconstructed_edges,
            Path(output_dir),
            str(arm),
            str(cad["cad_id"]),
        )
        row.update(dict(result))
        row["step_saved"] = bool(row.get("step_saved"))
        row["brep_valid"] = bool(row.get("brep_valid"))
        row["status"] = str(row.get("status") or "saved")
    except Exception as exc:
        row.update(
            {
                "status": "assembly_error",
                "step_saved": False,
                "brep_valid": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(
    arm: str,
    checkpoint_path: Path,
    *,
    device: str,
    import_output_root: Path,
):
    """Construct the arm recorded by Protocol V5 and load it strictly."""
    import torch

    os.environ.setdefault("NS_OUTBASE", str(Path(import_output_root).resolve()))
    os.environ.setdefault("NS_OUT", "_assembly_calibration_import")
    from breparg_improvements import train as train_mod

    configs = {
        config["name"]: config
        for config in train_mod.quantizer_comparison_configs()
    }
    if arm not in configs:
        raise ValueError(f"unsupported calibration arm: {arm}")
    checkpoint_path = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint_path, map_location=device)
    checkpoint_quantizer = payload.get("quantizer") or {}
    expected_kind = configs[arm]["quantizer"]["kind"]
    if checkpoint_quantizer.get("kind") != expected_kind:
        raise RuntimeError(
            f"checkpoint quantizer kind mismatch for {arm}: "
            f"{checkpoint_quantizer.get('kind')!r} != {expected_kind!r}"
        )
    model = train_mod.build_quantized_vqvae(configs[arm]).to(device).eval()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "checkpoint_epoch": payload.get("checkpoint_epoch"),
        "quantizer": checkpoint_quantizer,
        "validation_metrics": payload.get("validation_metrics"),
    }


def reconstruct_patch_batch(
    model: Any,
    patches: np.ndarray,
    *,
    kind: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Run direct patch autoencoding without sequence or AR variables."""
    import torch

    patches = np.asarray(patches, dtype=np.float32)
    if kind == "surface":
        if patches.ndim != 4 or patches.shape[1:] != (32, 32, 3):
            raise ValueError(f"surface patches must be (N,32,32,3), got {patches.shape}")
        model_input = patches
    elif kind == "edge":
        if patches.ndim != 3 or patches.shape[1:] != (32, 3):
            raise ValueError(f"edge patches must be (N,32,3), got {patches.shape}")
        model_input = np.tile(patches[:, :, None, :], (1, 1, 32, 1))
    else:
        raise ValueError(f"unknown patch kind: {kind}")

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(model_input), max(1, int(batch_size))):
            values = torch.as_tensor(
                model_input[start:start + max(1, int(batch_size))],
                dtype=torch.float32,
                device=device,
            ).permute(0, 3, 1, 2)
            latent = model.quant_conv(model.encoder(values))
            quantized, _, _ = model.quantize(latent)
            reconstructed = model.decoder(model.post_quant_conv(quantized))
            output = reconstructed.permute(0, 2, 3, 1).float().cpu().numpy()
            outputs.append(output)
    stacked = np.concatenate(outputs, axis=0) if outputs else np.zeros_like(model_input)
    return stacked if kind == "surface" else edge_patches_from_model_output(stacked)


def evaluate_cad_model_arm(
    cad: Mapping[str, Any],
    *,
    arm: str,
    checkpoint_sha256: str,
    parsed: Mapping[str, Any],
    model: Any,
    output_dir: Path,
    reconstructor: Any,
    assembler: Any,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Reconstruct one CAD and retain inference failures as attempt rows."""
    try:
        reconstructed_surfaces = reconstructor(
            model,
            np.asarray(parsed["surf_ncs"], dtype=np.float32),
            kind="surface",
            device=device,
            batch_size=batch_size,
        )
        reconstructed_edges = reconstructor(
            model,
            np.asarray(parsed["edge_ncs"], dtype=np.float32),
            kind="edge",
            device=device,
            batch_size=batch_size,
        )
    except Exception as exc:
        return {
            "cad_id": str(cad["cad_id"]),
            "parent_id": str(cad["parent_id"]),
            "source_path": str(cad["path"]),
            "arm": str(arm),
            "checkpoint_sha256": checkpoint_sha256,
            "status": "reconstruction_error",
            "step_saved": False,
            "brep_valid": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return evaluate_cad_arm(
        cad,
        arm=arm,
        checkpoint_sha256=checkpoint_sha256,
        parsed=parsed,
        reconstructed_surfaces=reconstructed_surfaces,
        reconstructed_edges=reconstructed_edges,
        output_dir=output_dir,
        assembler=assembler,
    )


def _bbox_center_and_size(bbox: np.ndarray) -> tuple[np.ndarray, float]:
    bbox = np.asarray(bbox, dtype=np.float32)
    return (bbox[:3] + bbox[3:]) / 2.0, float(np.max(bbox[3:] - bbox[:3]))


def cpu_joint_optimize(
    surf_ncs: np.ndarray,
    edge_ncs: np.ndarray,
    surf_bbox_wcs: np.ndarray,
    unique_vertices: np.ndarray,
    edge_vertex_adj: np.ndarray,
    face_edge_adj: list[list[int]],
    *,
    iterations: int = 200,
    edge_bboxes: np.ndarray | None = None,
    edge_scale_resolver: Callable[..., float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Place reconstructed NCS patches using BrepARG-equivalent CPU fitting.

    The optional resolver lets an isolated production-source evaluation share
    the production closed-edge scaling rule without changing any existing
    calibration path.
    """
    import torch

    edge_ncs = np.asarray(edge_ncs, dtype=np.float32)
    if edge_bboxes is not None:
        edge_bboxes = np.asarray(edge_bboxes, dtype=np.float32)
        if edge_bboxes.shape != (len(edge_ncs), 6):
            raise ValueError(
                f"edge_bboxes must have shape ({len(edge_ncs)}, 6), got {edge_bboxes.shape}"
            )
    edge_vertex_se = np.asarray(unique_vertices, dtype=np.float32)[
        np.asarray(edge_vertex_adj, dtype=np.int64)
    ]
    edge_wcs_chunks: list[np.ndarray] = []
    for edge_index, (curve, vertex_se) in enumerate(zip(edge_ncs, edge_vertex_se)):
        endpoints = curve[[0, -1]]
        target_scale = float(np.linalg.norm(vertex_se[0] - vertex_se[1]))
        ncs_scale = float(np.linalg.norm(endpoints[0] - endpoints[1]))
        if edge_scale_resolver is None:
            edge_scale = target_scale / max(ncs_scale, 1e-8)
        else:
            bbox = edge_bboxes[edge_index] if edge_bboxes is not None else None
            edge_scale = float(
                edge_scale_resolver(target_scale, ncs_scale, curve, bbox)
            )
        if not np.isfinite(edge_scale) or edge_scale < 0:
            raise ValueError(f"edge {edge_index} has invalid scale {edge_scale!r}")
        updated = curve * edge_scale
        scaled_endpoints = endpoints * edge_scale
        offset = vertex_se - scaled_endpoints
        reversed_offset = vertex_se - scaled_endpoints[::-1]
        if np.abs(reversed_offset[0] - reversed_offset[1]).mean() < np.abs(offset[0] - offset[1]).mean():
            updated = updated[::-1]
            offset = reversed_offset
        updated = updated + offset.mean(0)
        start_delta = vertex_se[0] - updated[0]
        end_delta = vertex_se[1] - updated[-1]
        weights = np.linspace(0.0, 1.0, 32, dtype=np.float32)[:, None]
        updated = updated + start_delta * (1.0 - weights) + end_delta * weights
        edge_wcs_chunks.append(updated.astype(np.float32))
    edge_wcs = np.stack(edge_wcs_chunks)

    surfaces: list[np.ndarray] = []
    face_edges: list[torch.Tensor] = []
    for adjacency, surface, bbox in zip(face_edge_adj, surf_ncs, surf_bbox_wcs):
        if not adjacency:
            raise ValueError("surface has no incident edge")
        edges = edge_wcs[np.asarray(adjacency, dtype=np.int64)]
        flat_edges = edges.reshape(-1, 3)
        center, surface_scale = _bbox_center_and_size(bbox)
        edge_scale = float(np.max(np.max(flat_edges, axis=0) - np.min(flat_edges, axis=0)))
        if surface_scale < edge_scale:
            surface_scale = 1.05 * edge_scale
        surfaces.append(np.asarray(surface, dtype=np.float32) * (surface_scale / 2.0) + center)
        face_edges.append(torch.as_tensor(flat_edges, dtype=torch.float32))

    surface_tensor = torch.as_tensor(np.stack(surfaces), dtype=torch.float32)
    offsets = torch.zeros((len(surfaces), 1, 1, 3), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([offsets], lr=1e-3, betas=(0.95, 0.999), weight_decay=1e-6)
    for _ in range(max(0, int(iterations))):
        updated = surface_tensor + offsets
        loss = torch.zeros((), dtype=torch.float32)
        for surface, edges in zip(updated, face_edges):
            distances = torch.cdist(edges[None, ...], surface.reshape(1, -1, 3))
            loss = loss + distances.min(dim=2).values.mean()
        loss = loss / len(surfaces)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (surface_tensor + offsets).detach().numpy(), edge_wcs


def assemble_and_validate(
    parsed: Mapping[str, Any],
    reconstructed_surfaces: np.ndarray,
    reconstructed_edges: np.ndarray,
    output_dir: Path,
    arm: str,
    cad_id: str,
    *,
    breparg_root: Path,
    joint_iterations: int = 200,
) -> dict[str, Any]:
    """Joint-optimize, write STEP, and check strict OCC BRep validity."""
    breparg_root = Path(breparg_root).resolve()
    if not (breparg_root / "utils.py").is_file():
        raise FileNotFoundError(f"BrepARG utils.py is missing: {breparg_root}")
    if str(breparg_root) not in sys.path:
        sys.path.insert(0, str(breparg_root))
    import utils as brep_utils
    from OCC.Extend.DataExchange import write_step_file

    required = ("surf_bbox_wcs", "corner_unique", "edgeCorner_adj", "faceEdge_adj")
    missing = [name for name in required if name not in parsed]
    if missing:
        raise KeyError("assembly fields are missing: " + ", ".join(missing))
    face_edge_adj = [list(map(int, row)) for row in parsed["faceEdge_adj"]]
    edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
    surf_wcs, edge_wcs = cpu_joint_optimize(
        reconstructed_surfaces,
        reconstructed_edges,
        np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
        np.asarray(parsed["corner_unique"], dtype=np.float32),
        edge_vertex_adj,
        face_edge_adj,
        iterations=joint_iterations,
    )
    solid = brep_utils.construct_brep(surf_wcs, edge_wcs, face_edge_adj, edge_vertex_adj)
    if solid is None:
        return {"status": "construct_brep_failed", "step_saved": False, "brep_valid": False}
    step_path = Path(output_dir) / "steps" / str(arm) / f"{cad_id}.step"
    step_path.parent.mkdir(parents=True, exist_ok=True)
    write_step_file(solid, str(step_path))
    saved = step_path.is_file() and step_path.stat().st_size > 0
    valid = bool(brep_utils.check_brep_validity(str(step_path))) if saved else False
    return {
        "status": "saved" if valid else ("brep_invalid" if saved else "step_save_failed"),
        "step_saved": saved,
        "brep_valid": valid,
        "step_path": str(step_path),
        "step_bytes": step_path.stat().st_size if saved else 0,
        "step_sha256": sha256_file(step_path) if saved else None,
    }


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def parse_checkpoint(value: str) -> tuple[str, Path]:
    arm, separator, raw_path = value.partition("=")
    if not separator or not arm.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("checkpoint must use ARM=PATH")
    return arm.strip(), Path(raw_path.strip())


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "calibration_manifest.jsonl"
    selected = select_validation_cads(
        args.protocol_dir,
        max_cads=args.max_cads,
        seed=args.seed,
    )
    checkpoint_args = dict(args.checkpoint or [])
    arms = (["original"] if args.include_original_control else []) + list(checkpoint_args)
    if not arms:
        raise ValueError("at least one checkpoint or original control is required")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    models = {}
    checkpoint_metadata = {}
    for arm, path in checkpoint_args.items():
        models[arm], checkpoint_metadata[arm] = load_checkpoint_model(
            arm,
            path,
            device=device,
            import_output_root=output_dir / "import_state",
        )

    existing = _read_jsonl(manifest_path)
    completed = {
        (row.get("cad_id"), row.get("arm"), row.get("checkpoint_sha256"))
        for row in existing
    }
    started = time.time()
    for cad in selected:
        with Path(cad["path"]).open("rb") as handle:
            parsed = pickle.load(handle)
        target_surfaces = np.asarray(parsed.get("surf_ncs"), dtype=np.float32)
        target_edges = np.asarray(parsed.get("edge_ncs"), dtype=np.float32)
        for arm in arms:
            checkpoint_sha = checkpoint_metadata.get(arm, {}).get("sha256")
            key = (cad["cad_id"], arm, checkpoint_sha)
            if key in completed:
                continue
            assembler = lambda *values, **kwargs: assemble_and_validate(
                *values,
                **kwargs,
                breparg_root=args.breparg_root,
                joint_iterations=args.joint_iterations,
            )
            if arm == "original":
                reconstructed_surfaces = target_surfaces.copy()
                reconstructed_edges = target_edges.copy()
                row = evaluate_cad_arm(
                    cad,
                    arm=arm,
                    checkpoint_sha256=checkpoint_sha,
                    parsed=parsed,
                    reconstructed_surfaces=reconstructed_surfaces,
                    reconstructed_edges=reconstructed_edges,
                    output_dir=output_dir,
                    assembler=assembler,
                )
            else:
                row = evaluate_cad_model_arm(
                    cad,
                    arm=arm,
                    checkpoint_sha256=checkpoint_sha,
                    parsed=parsed,
                    model=models[arm],
                    output_dir=output_dir,
                    reconstructor=reconstruct_patch_batch,
                    assembler=assembler,
                    device=device,
                    batch_size=args.batch_size,
                )
            row.update(
                {
                    "protocol_sha256": _read_json(Path(args.protocol_dir) / "protocol_summary.json").get("protocol_sha256"),
                    "selection_seed": int(args.seed),
                    "device": str(device),
                    "elapsed_seconds": time.time() - started,
                }
            )
            _append_jsonl(manifest_path, row)
            completed.add(key)
            print(
                json.dumps(
                    {
                        "cad_id": row["cad_id"],
                        "arm": arm,
                        "status": row["status"],
                        "curved_mse": row.get("curved_mse"),
                        "brep_valid": row["brep_valid"],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    final_rows = _read_jsonl(manifest_path)
    state = {
        "status": "COMPLETED",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_dir": str(Path(args.protocol_dir).resolve()),
        "protocol_sha256": _read_json(Path(args.protocol_dir) / "protocol_summary.json").get("protocol_sha256"),
        "selected_cads": len(selected),
        "expected_rows": len(selected) * len(arms),
        "manifest_rows": len(final_rows),
        "arms": arms,
        "checkpoints": checkpoint_metadata,
        "manifest": str(manifest_path),
        "advance_to_ar": False,
    }
    if len(final_rows) < state["expected_rows"]:
        state["status"] = "FAILED"
    (output_dir / "calibration_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--max-cads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=parse_checkpoint, action="append", default=[])
    parser.add_argument("--include-original-control", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--joint-iterations", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state = run_calibration(args)
    print(json.dumps(state, indent=2, ensure_ascii=True))
    return 0 if state["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
