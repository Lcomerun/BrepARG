from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


def run_probe() -> dict[str, Any]:
    try:
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import (
            STEPControl_AsIs,
            STEPControl_Reader,
            STEPControl_Writer,
        )
    except Exception as exc:
        return {"status": "failed", "stage": "import", "error": repr(exc)}

    try:
        import occwl

        occwl_status: dict[str, Any] = {
            "available": True,
            "version": getattr(occwl, "__version__", None),
        }
    except Exception as exc:
        occwl_status = {"available": False, "error": repr(exc)}

    try:
        shape = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()
        source_valid = bool(BRepCheck_Analyzer(shape).IsValid())
        with tempfile.TemporaryDirectory(prefix="v13-occ-probe-") as temporary:
            step_path = Path(temporary) / "probe.step"
            writer = STEPControl_Writer()
            transfer_status = int(writer.Transfer(shape, STEPControl_AsIs))
            write_status = int(writer.Write(str(step_path)))
            reader = STEPControl_Reader()
            read_status = int(reader.ReadFile(str(step_path)))
            transferred = bool(reader.TransferRoots()) if read_status == int(IFSelect_RetDone) else False
            restored = reader.OneShape() if transferred else None
            restored_valid = bool(
                restored is not None and BRepCheck_Analyzer(restored).IsValid()
            )
            step_bytes = step_path.stat().st_size if step_path.is_file() else 0
    except Exception as exc:
        return {
            "status": "failed",
            "stage": "step_roundtrip",
            "error": repr(exc),
            "occwl": occwl_status,
        }

    passed = source_valid and transferred and restored_valid and step_bytes > 0
    return {
        "status": "ok" if passed else "failed",
        "stage": "complete",
        "source_valid": source_valid,
        "transfer_status": transfer_status,
        "write_status": write_status,
        "read_status": read_status,
        "transferred": transferred,
        "restored_valid": restored_valid,
        "step_bytes": step_bytes,
        "occwl": occwl_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe()
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
