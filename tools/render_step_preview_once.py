"""Render one STEP file to STL and PNG.

This script is intentionally small so callers can run it in a subprocess with a
timeout. OCC meshing can occasionally hang on malformed reconstructed geometry;
isolating one preview per process keeps long experiments moving.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from generate_step_png_batch import render_stl_png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Extend.DataExchange import write_stl_file

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(args.step))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"failed to read STEP: {args.step}")
    reader.TransferRoots()
    shape = reader.OneShape()

    args.stl.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    write_stl_file(shape, str(args.stl), linear_deflection=0.003, angular_deflection=0.5)
    stl_saved = args.stl.exists() and args.stl.stat().st_size > 0
    png_saved = False
    if stl_saved:
        png_saved = render_stl_png(args.stl, args.png, title=args.title)
    payload = {
        "step": str(args.step),
        "stl": str(args.stl),
        "png": str(args.png),
        "stl_saved": bool(stl_saved),
        "png_saved": bool(png_saved),
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if png_saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
