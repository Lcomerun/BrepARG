"""Validate one STEP file and optionally render a PNG preview.

Run this script in a subprocess with a timeout. It uses OCC/occwl operations
that may be slow or unstable for malformed geometry, so callers should isolate
each candidate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from generate_step_png_batch import render_stl_png


ENTITY_NAMES = (
    "ADVANCED_FACE",
    "EDGE_CURVE",
    "MANIFOLD_SOLID_BREP",
    "CLOSED_SHELL",
    "OPEN_SHELL",
)


def count_step_entities(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="ignore").upper()
    return {name: len(re.findall(rf"\b{re.escape(name)}\s*\(", text)) for name in ENTITY_NAMES}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--skip-preview", action="store_true")
    args = parser.parse_args()

    from BrepARG import utils as brep_utils
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Extend.DataExchange import write_stl_file

    entities = count_step_entities(args.step)
    reader = STEPControl_Reader()
    read_status = reader.ReadFile(str(args.step))
    step_read_ok = read_status == IFSelect_RetDone
    stl_saved = False
    png_saved = False

    if step_read_ok and not args.skip_preview:
        reader.TransferRoots()
        shape = reader.OneShape()
        args.stl.parent.mkdir(parents=True, exist_ok=True)
        args.png.parent.mkdir(parents=True, exist_ok=True)
        write_stl_file(shape, str(args.stl), linear_deflection=0.003, angular_deflection=0.5)
        stl_saved = args.stl.exists() and args.stl.stat().st_size > 0
        if stl_saved:
            png_saved = render_stl_png(args.stl, args.png, title=args.title)

    try:
        brep_valid = bool(brep_utils.check_brep_validity(str(args.step)))
    except Exception:
        brep_valid = False

    result = {
        "step": str(args.step),
        "step_read_ok": bool(step_read_ok),
        "brep_valid": bool(brep_valid),
        "solid_closed_no_open_shell": (
            entities["MANIFOLD_SOLID_BREP"] > 0 and entities["CLOSED_SHELL"] > 0 and entities["OPEN_SHELL"] == 0
        ),
        "has_manifold_solid_brep": entities["MANIFOLD_SOLID_BREP"] > 0,
        "has_closed_shell": entities["CLOSED_SHELL"] > 0,
        "has_open_shell": entities["OPEN_SHELL"] > 0,
        "advanced_faces": int(entities["ADVANCED_FACE"]),
        "edge_curves": int(entities["EDGE_CURVE"]),
        "stl_saved": bool(stl_saved),
        "stl_path": str(args.stl),
        "png_saved": bool(png_saved),
        "png_path": str(args.png),
    }
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result["step_read_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
