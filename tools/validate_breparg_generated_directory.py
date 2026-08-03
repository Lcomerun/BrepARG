"""Validate and render STEP files from a BrepARG generated directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_last_json(stdout: str) -> dict[str, Any]:
    for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def validate_one(step_path: Path, run_dir: Path, timeout_sec: int, skip_preview: bool) -> dict[str, Any]:
    stl_path = run_dir / "quality_check" / "stl" / f"{step_path.stem}.stl"
    png_path = run_dir / "quality_check" / "png" / f"{step_path.stem}.png"
    cmd = [
        sys.executable,
        str(TOOLS_DIR / "validate_step_quality_once.py"),
        "--step",
        str(step_path),
        "--stl",
        str(stl_path),
        "--png",
        str(png_path),
        "--title",
        step_path.stem,
    ]
    if skip_preview:
        cmd.append("--skip-preview")

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(TOOLS_DIR.parents[0]),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
        )
        payload = parse_last_json(completed.stdout)
        if not payload:
            payload = {
                "step": str(step_path),
                "step_read_ok": False,
                "brep_valid": False,
                "solid_closed_no_open_shell": False,
                "stl_saved": False,
                "stl_path": str(stl_path),
                "png_saved": False,
                "png_path": str(png_path),
                "quality_error": (completed.stderr or completed.stdout or f"exit_{completed.returncode}")[-1000:],
            }
        payload["quality_exit_code"] = int(completed.returncode)
        if completed.stderr.strip():
            payload["quality_stderr_tail"] = completed.stderr.strip()[-1000:]
        return payload
    except subprocess.TimeoutExpired:
        return {
            "step": str(step_path),
            "step_read_ok": False,
            "brep_valid": False,
            "solid_closed_no_open_shell": False,
            "stl_saved": stl_path.exists() and stl_path.stat().st_size > 0,
            "stl_path": str(stl_path),
            "png_saved": png_path.exists() and png_path.stat().st_size > 0,
            "png_path": str(png_path),
            "quality_error": f"quality_timeout_{timeout_sec}s",
            "quality_exit_code": None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--skip-preview", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    step_files = sorted(
        path
        for path in run_dir.rglob("*.step")
        if path.is_file() and "quality_check" not in path.parts
    )
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, step_path in enumerate(step_files, start=1):
        row = validate_one(step_path, run_dir, args.timeout_sec, args.skip_preview)
        rows.append(row)
        print(
            f"[{index:03d}/{len(step_files):03d}] "
            f"read={int(bool(row.get('step_read_ok')))} "
            f"brep={int(bool(row.get('brep_valid')))} "
            f"closed={int(bool(row.get('solid_closed_no_open_shell')))} "
            f"png={int(bool(row.get('png_saved')))} "
            f"faces={row.get('advanced_faces')} edges={row.get('edge_curves')} "
            f"{step_path.name}",
            flush=True,
        )

    summary = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "step_files": len(step_files),
        "rows": len(rows),
        "step_read_ok": sum(1 for row in rows if row.get("step_read_ok")),
        "brep_valid": sum(1 for row in rows if row.get("brep_valid")),
        "solid_closed_no_open_shell": sum(1 for row in rows if row.get("solid_closed_no_open_shell")),
        "stl_saved": sum(1 for row in rows if row.get("stl_saved")),
        "png_saved": sum(1 for row in rows if row.get("png_saved")),
        "timeouts": sum(1 for row in rows if str(row.get("quality_error", "")).startswith("quality_timeout_")),
        "elapsed_min": round((time.time() - started) / 60.0, 3),
    }
    write_jsonl(args.manifest_output, rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
