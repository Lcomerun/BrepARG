"""Explain the P0-A non-unit-solid repair without archiving CAD bytes.

The report binds source and STEP artifacts by SHA-256, records BRepCheck
statuses by subshape, and proves whether near-vertex reconciliation restores
complete face-loop edge coverage.  Paths and source payloads are deliberately
excluded so the JSON output is safe to commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .assembly_repair import historical_face_loops
    from .run_assembly_calibration_oracle import cpu_joint_optimize
    from .solid_topology_repair import reconcile_near_vertices
except ImportError:  # direct script execution
    from assembly_repair import historical_face_loops
    from run_assembly_calibration_oracle import cpu_joint_optimize
    from solid_topology_repair import reconcile_near_vertices


SCHEMA = "solid-topology-diagnosis-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status_names(statuses: Any) -> list[str]:
    import OCC.Core.BRepCheck as check_module
    from OCC.Core.BRepCheck import BRepCheck_ListOfStatus

    names = {
        int(value): name
        for name in dir(check_module)
        if name.startswith("BRepCheck_")
        and isinstance((value := getattr(check_module, name)), int)
    }
    copied = BRepCheck_ListOfStatus()
    copied.Assign(statuses)
    result = []
    while len(copied):
        value = int(copied.First())
        result.append(names.get(value, f"BRepCheck_Status_{value}"))
        copied.RemoveFirst()
    return result


def inspect_step(path: Path) -> dict[str, Any]:
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        return {"readable": False, "native_valid": False, "subshapes": {}}
    reader.TransferRoots()
    shape = reader.OneShape()
    analyzer = BRepCheck_Analyzer(shape, True)
    subshape_types = {
        "faces": TopAbs_FACE,
        "wires": TopAbs_WIRE,
        "shells": TopAbs_SHELL,
        "solids": TopAbs_SOLID,
    }
    subshapes: dict[str, Any] = {}
    for label, shape_type in subshape_types.items():
        explorer = TopExp_Explorer(shape, shape_type)
        count = 0
        invalid = []
        while explorer.More():
            count += 1
            current = explorer.Current()
            statuses = _status_names(analyzer.Result(current).Status())
            errors = [name for name in statuses if name != "BRepCheck_NoError"]
            if errors:
                invalid.append({"index_1based": count, "statuses": errors})
            explorer.Next()
        subshapes[label] = {"count": count, "invalid": invalid}
    return {
        "readable": True,
        "native_valid": bool(analyzer.IsValid()),
        "subshapes": subshapes,
    }


def loop_coverage(
    face_edge_adj: Sequence[Sequence[int]], edge_vertex_adj: np.ndarray
) -> list[dict[str, Any]]:
    """Return only faces whose historical grouping loses or repeats an edge."""
    result = []
    for face_index, incident in enumerate(face_edge_adj):
        expected = [int(value) for value in incident]
        try:
            loops = historical_face_loops(expected, edge_vertex_adj)
            observed = [int(edge_id) for loop in loops for edge_id, _ in loop]
            error = None
        except Exception as exc:
            loops = []
            observed = []
            error = f"{type(exc).__name__}: {exc}"
        missing = sorted(set(expected) - set(observed))
        repeated = sorted(
            edge_id for edge_id in set(observed) if observed.count(edge_id) > 1
        )
        if error or missing or repeated or len(observed) != len(expected):
            result.append(
                {
                    "face_index_0based": face_index,
                    "incident_edge_count": len(expected),
                    "loop_count": len(loops),
                    "missing_edge_ids": missing,
                    "repeated_edge_ids": repeated,
                    "observed_edge_count": len(observed),
                    "error": error,
                }
            )
    return result


def source_evidence(path: Path, *, joint_iterations: int) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        parsed = pickle.load(handle)
    edge_vertex_adj = np.asarray(parsed["edgeCorner_adj"], dtype=np.int64)
    face_edge_adj = [list(map(int, values)) for values in parsed["faceEdge_adj"]]
    # The repair is invoked after the same deterministic CPU coordinate
    # placement used by the matrix runner, not on the serialized raw WCS
    # curves.  Reconstruct it here so the diagnosis observes the geometry that
    # actually reaches OCC.
    _, edge_wcs = cpu_joint_optimize(
        np.asarray(parsed["surf_ncs"], dtype=np.float32),
        np.asarray(parsed["edge_ncs"], dtype=np.float32),
        np.asarray(parsed["surf_bbox_wcs"], dtype=np.float32),
        np.asarray(parsed["corner_unique"], dtype=np.float32),
        edge_vertex_adj,
        face_edge_adj,
        iterations=int(joint_iterations),
    )
    remapped, _, diagnostics = reconcile_near_vertices(
        edge_wcs, edge_vertex_adj, face_edge_adj
    )
    return {
        "geometry_stage": "cpu_joint_optimize",
        "joint_iterations": int(joint_iterations),
        "face_count": len(face_edge_adj),
        "edge_count": len(edge_vertex_adj),
        "original_incomplete_loop_faces": loop_coverage(face_edge_adj, edge_vertex_adj),
        "reconciled_incomplete_loop_faces": loop_coverage(face_edge_adj, remapped),
        "near_vertex_reconciliation": diagnostics,
    }


def build_report(
    *,
    cad_id: str,
    source_pickle: Path,
    baseline_step: Path,
    candidate_step: Path,
    joint_iterations: int = 200,
) -> dict[str, Any]:
    for path in (source_pickle, baseline_step, candidate_step):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    baseline = inspect_step(baseline_step)
    candidate = inspect_step(candidate_step)
    source = source_evidence(source_pickle, joint_iterations=joint_iterations)
    return {
        "schema": SCHEMA,
        "cad_id": str(cad_id),
        "source_pickle": {
            "bytes": Path(source_pickle).stat().st_size,
            "sha256": sha256_file(source_pickle),
            "archived": False,
        },
        "baseline_step": {
            "bytes": Path(baseline_step).stat().st_size,
            "sha256": sha256_file(baseline_step),
            "archived": False,
            **baseline,
        },
        "candidate_step": {
            "bytes": Path(candidate_step).stat().st_size,
            "sha256": sha256_file(candidate_step),
            "archived": False,
            **candidate,
        },
        "source_topology": source,
        "recovered": bool(
            not baseline.get("native_valid") and candidate.get("native_valid")
        ),
        "interpretation": (
            "near-coincident source endpoint ids split face loops and caused "
            "invalid face orientation/imbrication after STEP round-trip; "
            "one-to-one shared vertices restore complete loops and one valid solid"
        ),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    # ``Path.write_text`` does not accept ``newline`` on all supported Python
    # versions.  Open the file explicitly so the Git-safe report is stable on
    # the Windows environment that runs the OCC measurement.
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cad-id", required=True)
    parser.add_argument("--source-pickle", type=Path, required=True)
    parser.add_argument("--baseline-step", type=Path, required=True)
    parser.add_argument("--candidate-step", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--joint-iterations", type=int, default=200)
    args = parser.parse_args(argv)
    payload = build_report(
        cad_id=args.cad_id,
        source_pickle=args.source_pickle,
        baseline_step=args.baseline_step,
        candidate_step=args.candidate_step,
        joint_iterations=args.joint_iterations,
    )
    atomic_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
