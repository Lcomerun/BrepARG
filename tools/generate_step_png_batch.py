"""Generate STEP files and quick PNG previews from a V13 AR/VQ-VAE checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

import sys

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from evaluate_reconstruction_v13 import (
    append_jsonl,
    generate_ar_records,
    infer_device,
    load_fsq_vqvae,
    normalize_vocab_info,
    read_pickle,
    reconstruct_one,
    summarize_manifest_rows,
    write_json,
)


def render_stl_png(stl_path: Path, png_path: Path, title: str = "") -> bool:
    triangles = triangles_from_ascii_stl(stl_path)
    return render_triangles_png(triangles, png_path, title=title)


def triangles_from_ascii_stl(stl_path: Path) -> np.ndarray:
    triangles: list[list[list[float]]] = []
    current: list[list[float]] = []
    with stl_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("vertex "):
                continue
            parts = stripped.split()
            if len(parts) != 4:
                continue
            current.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(current) == 3:
                triangles.append(current)
                current = []
    if not triangles:
        raise RuntimeError(f"no ASCII STL triangles parsed: {stl_path}")
    return np.asarray(triangles, dtype=float)


def render_triangles_png(triangles: Any, png_path: Path, title: str = "") -> bool:
    tri_list = triangles.tolist() if hasattr(triangles, "tolist") else triangles
    points = [point for tri in tri_list for point in tri]
    if not points:
        raise RuntimeError("no points to render")

    mins = [min(float(point[axis]) for point in points) for axis in range(3)]
    maxs = [max(float(point[axis]) for point in points) for axis in range(3)]
    center = [(mins[axis] + maxs[axis]) / 2.0 for axis in range(3)]

    azim = math.radians(-42.0)
    elev = math.radians(24.0)
    # Use scalar math here. On one local Windows/conda combination, numpy
    # matmul exited the process while rendering otherwise valid STL previews.
    cos_azim = math.cos(azim)
    sin_azim = math.sin(azim)
    cos_elev = math.cos(elev)
    sin_elev = math.sin(elev)
    rotated: list[list[list[float]]] = []
    max_abs = 0.0
    for tri in tri_list:
        tri_rotated: list[list[float]] = []
        for point in tri:
            x = float(point[0]) - center[0]
            y = float(point[1]) - center[1]
            z0 = float(point[2]) - center[2]
            x1 = cos_azim * x - sin_azim * y
            y1 = sin_azim * x + cos_azim * y
            z1 = z0
            x2 = x1
            y2 = cos_elev * y1 - sin_elev * z1
            z2 = sin_elev * y1 + cos_elev * z1
            max_abs = max(max_abs, abs(x2), abs(y2))
            tri_rotated.append([x2, y2, z2])
        rotated.append(tri_rotated)
    if not math.isfinite(max_abs) or max_abs <= 0:
        max_abs = 1.0

    size = 900
    margin = 70
    scale = (size - 2 * margin) / (2 * max_abs)
    projected: list[list[tuple[float, float]]] = []
    for tri in rotated:
        projected.append([(size / 2.0 + point[0] * scale, size / 2.0 - point[1] * scale) for point in tri])

    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    order = sorted(range(len(rotated)), key=lambda idx: sum(point[2] for point in rotated[idx]) / 3.0)
    for tri_idx in order:
        tri3 = rotated[tri_idx]
        ux = tri3[1][0] - tri3[0][0]
        uy = tri3[1][1] - tri3[0][1]
        uz = tri3[1][2] - tri3[0][2]
        vx = tri3[2][0] - tri3[0][0]
        vy = tri3[2][1] - tri3[0][1]
        vz = tri3[2][2] - tri3[0][2]
        normal = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
        norm = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
        shade = 0.55
        if norm > 0:
            shade = 0.52 + 0.34 * max(0.0, normal[2] / norm)
        color = tuple(int(max(0, min(255, component * shade + 30))) for component in (118, 165, 195))
        draw.polygon(projected[tri_idx], fill=(*color, 232), outline=(40, 48, 56, 42))
    if title:
        draw.rectangle((8, 8, min(size - 8, 8 + 9 * len(title)), 32), fill=(255, 255, 255, 210))
        draw.text((12, 13), title, fill=(30, 35, 40, 255))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path)
    return png_path.exists() and png_path.stat().st_size > 0


def triangles_from_step(step_path: Path, linear_deflection: float = 0.01) -> np.ndarray:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods_Face

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"failed to read STEP: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, float(linear_deflection))

    triangles: list[np.ndarray] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation:
            transform = loc.Transformation()
            for tri_index in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(tri_index)
                node_indices = triangle.Get()
                pts = []
                for node_index in node_indices:
                    point = triangulation.Node(node_index).Transformed(transform)
                    pts.append([point.X(), point.Y(), point.Z()])
                triangles.append(np.asarray(pts, dtype=float))
        explorer.Next()
    if not triangles:
        raise RuntimeError(f"no triangulated faces in STEP: {step_path}")
    return np.stack(triangles, axis=0)


def render_step_png(step_path: Path, png_path: Path, title: str = "") -> bool:
    triangles = triangles_from_step(step_path)
    return render_triangles_png(triangles, png_path, title=title)


def enrich_png(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if not row.get("stl_saved") or not row.get("stl_path"):
        row["png_saved"] = False
        return row
    stl_path = Path(row["stl_path"])
    png_path = output_dir / "png" / f"{stl_path.stem}.png"
    title = f"f={row.get('grammar_faces', '?')} e={row.get('grammar_edges', '?')} len={row.get('sequence_length', '?')}"
    try:
        row["png_saved"] = render_stl_png(stl_path, png_path, title=title)
        row["png_path"] = str(png_path)
    except Exception as exc:
        row["png_saved"] = False
        row["png_error"] = f"{type(exc).__name__}: {exc}"
    return row


def pick_sampling_profile(success_count: int, target_count: int) -> tuple[float, float]:
    third = max(1, target_count // 3)
    if success_count < third:
        return 0.55, 0.85
    if success_count < third * 2:
        return 0.70, 0.90
    return 0.90, 0.95


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--vqvae-checkpoint", type=Path, required=True)
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-attempts", type=int, default=220)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--validate-step", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    package = read_pickle(args.sequence)
    vocab_info = normalize_vocab_info(package)
    device = infer_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "generation_manifest.jsonl"
    report_path = output_dir / "generation_report.json"

    vqvae_model = load_fsq_vqvae(args.vqvae_checkpoint, device)
    rows: list[dict[str, Any]] = []
    attempts = 0
    batch_index = 0
    t0 = time.time()

    while len([row for row in rows if row.get("step_saved") and row.get("png_saved")]) < args.target_count and attempts < args.max_attempts:
        remaining_attempts = args.max_attempts - attempts
        batch = min(args.batch_size, remaining_attempts)
        temperature, top_p = pick_sampling_profile(len([row for row in rows if row.get("step_saved") and row.get("png_saved")]), args.target_count)
        records = generate_ar_records(
            args.ar_checkpoint,
            vocab_info,
            count=batch,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            constrained=True,
        )
        for record in records:
            attempts += 1
            record["index"] = attempts - 1
            row = reconstruct_one(
                record,
                vocab_info=vocab_info,
                vqvae_model=vqvae_model,
                device=device,
                output_dir=output_dir,
                write_step=True,
                write_stl=True,
                validate_step=args.validate_step,
                scale_factor=1.0,
            )
            row["attempt"] = attempts
            row["batch_index"] = batch_index
            row["temperature"] = temperature
            row["top_p"] = top_p
            row = enrich_png(row, output_dir)
            rows.append(row)
            append_jsonl(manifest_path, row)
            ok = row.get("step_saved") and row.get("png_saved")
            print(
                f"[{len(rows):04d}] ok={int(bool(ok))} "
                f"saved={sum(1 for r in rows if r.get('step_saved') and r.get('png_saved'))}/{args.target_count} "
                f"status={row.get('status')} faces={row.get('grammar_faces')} edges={row.get('grammar_edges')} "
                f"len={row.get('sequence_length')} temp={temperature} top_p={top_p}",
                flush=True,
            )
            if len([r for r in rows if r.get("step_saved") and r.get("png_saved")]) >= args.target_count:
                break
        batch_index += 1

    successful = [row for row in rows if row.get("step_saved") and row.get("png_saved")]
    summary = summarize_manifest_rows(rows)
    summary["png_saved"] = len(successful)
    summary["target_count"] = int(args.target_count)
    summary["attempts"] = int(attempts)
    summary["elapsed_min"] = round((time.time() - t0) / 60.0, 3)
    if successful:
        summary["faces"] = {
            "min": min(int(row.get("grammar_faces", 0)) for row in successful),
            "max": max(int(row.get("grammar_faces", 0)) for row in successful),
        }
        summary["edges"] = {
            "min": min(int(row.get("grammar_edges", 0)) for row in successful),
            "max": max(int(row.get("grammar_edges", 0)) for row in successful),
        }
        summary["length"] = {
            "min": min(int(row.get("sequence_length", 0)) for row in successful),
            "max": max(int(row.get("sequence_length", 0)) for row in successful),
        }
    report = {
        "status": "VERIFIED" if len(successful) >= args.target_count else "INCOMPLETE",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequence": str(args.sequence),
        "ar_checkpoint": str(args.ar_checkpoint),
        "vqvae_checkpoint": str(args.vqvae_checkpoint),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "device": str(device),
        "summary": summary,
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
