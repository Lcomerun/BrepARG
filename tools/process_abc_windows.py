import argparse
import concurrent.futures
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path


EXPECTED_FIELDS = {
    "surf_wcs",
    "edge_wcs",
    "surf_ncs",
    "edge_ncs",
    "corner_wcs",
    "edgeFace_adj",
    "edgeCorner_adj",
    "faceEdge_adj",
    "surf_bbox_wcs",
    "edge_bbox_wcs",
    "corner_unique",
}
CHUNK_RE = re.compile(r"^abc_(\d{4})_step_v00$")
MANIFEST_SKIP_STATUSES = {"multi", "filtered", "badfields", "zero_bytes", "timeout", "error"}


def repo_root():
    return Path(__file__).resolve().parents[1]


def output_dir_for_chunk(out_root, chunk):
    return Path(out_root) / f"abc_{int(chunk):04d}"


def report_dir_for_chunk(out_root, chunk):
    return output_dir_for_chunk(out_root, chunk) / "_reports"


def manifest_path_for_chunk(out_root, chunk):
    return report_dir_for_chunk(out_root, chunk) / "manifest.jsonl"


def summary_path_for_chunk(out_root, chunk):
    return report_dir_for_chunk(out_root, chunk) / "summary.json"


def find_chunk_dirs(raw_root):
    rows = []
    raw_root = Path(raw_root)
    if not raw_root.exists():
        return rows
    for item in raw_root.iterdir():
        if not item.is_dir():
            continue
        match = CHUNK_RE.match(item.name)
        if match:
            rows.append({"chunk": int(match.group(1)), "path": item})
    rows.sort(key=lambda row: row["chunk"])
    return rows


def select_chunks(chunks, spec):
    if spec in (None, "", "all"):
        return chunks
    wanted = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            wanted.update(range(int(lo), int(hi) + 1))
        else:
            wanted.add(int(part))
    return [row for row in chunks if row["chunk"] in wanted]


def enumerate_steps(raw_root, chunk_spec, limit=0):
    selected = select_chunks(find_chunk_dirs(raw_root), chunk_spec)
    tasks = []
    for row in selected:
        out_chunk = row["chunk"]
        for step_path in sorted(Path(row["path"]).rglob("*.step")):
            tasks.append({"chunk": out_chunk, "step": step_path})
            if limit and len(tasks) >= limit:
                return selected, tasks
    return selected, tasks


def summarize_data(data):
    fields = {}
    for key, value in data.items():
        if hasattr(value, "shape"):
            fields[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, list):
            fields[key] = {"type": "list", "len": len(value)}
        else:
            fields[key] = {"type": type(value).__name__}
    return fields


def parse_child(step_path, out_pkl):
    start = time.time()
    row = {"step_path": str(step_path), "output_pkl": str(out_pkl)}
    try:
        step_path = Path(step_path)
        out_pkl = Path(out_pkl)
        if out_pkl.exists():
            row["status"] = "skip"
            return row
        if step_path.stat().st_size == 0:
            row["status"] = "zero_bytes"
            return row
        process_data = repo_root() / "BrepARG" / "process_data"
        if str(process_data) not in sys.path:
            sys.path.insert(0, str(process_data))
        import process_brep
        from occwl.io import load_step

        solids = load_step(str(step_path))
        if len(solids) != 1:
            row["status"] = "multi"
            row["solid_count"] = len(solids)
            return row
        data = process_brep.parse_solid(solids[0])
        if data is None:
            row["status"] = "filtered"
            return row
        missing = EXPECTED_FIELDS - set(data.keys())
        if missing:
            row["status"] = "badfields"
            row["missing_fields"] = sorted(missing)
            return row
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_pkl.with_suffix(out_pkl.suffix + f".tmp{os.getpid()}")
        with tmp.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, out_pkl)
        row["status"] = "ok"
        row["pkl_bytes"] = out_pkl.stat().st_size
        row["fields"] = summarize_data(data)
        return row
    except Exception as exc:
        row["status"] = "error"
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)
        return row
    finally:
        row["seconds"] = round(time.time() - start, 3)


def child_main(args):
    row = parse_child(Path(args.child_step), Path(args.child_out))
    print(json.dumps(row, ensure_ascii=False, sort_keys=True))


def out_pkl_for_task(task, out_root):
    return output_dir_for_chunk(out_root, task["chunk"]) / (Path(task["step"]).stem + ".pkl")


def manifest_key(path):
    return os.path.normcase(os.path.abspath(str(path)))


def load_manifest_by_step(paths):
    rows = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step_path = row.get("step_path")
                if step_path:
                    rows[manifest_key(step_path)] = row
    return rows


def previous_terminal_row(task, out_root, manifest_by_step, retry_failures=False):
    if out_pkl_for_task(task, out_root).exists():
        return None
    row = manifest_by_step.get(manifest_key(task["step"]))
    if not row:
        return None
    status = row.get("status")
    if status in {"timeout", "error"} and retry_failures:
        return None
    if status in MANIFEST_SKIP_STATUSES:
        return dict(row)
    return None


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_one_task(task, out_root, timeout):
    out_pkl = out_pkl_for_task(task, out_root)
    if out_pkl.exists():
        return {
            "status": "skip",
            "step_path": str(task["step"]),
            "output_pkl": str(out_pkl),
            "chunk": task["chunk"],
            "seconds": 0.0,
        }
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-step",
        str(task["step"]),
        "--child-out",
        str(out_pkl),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "step_path": str(task["step"]),
            "output_pkl": str(out_pkl),
            "chunk": task["chunk"],
            "seconds": timeout,
        }
    row = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                row = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if row is None:
        row = {
            "status": "error",
            "step_path": str(task["step"]),
            "output_pkl": str(out_pkl),
            "chunk": task["chunk"],
            "error": "child produced no JSON row",
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
            "returncode": proc.returncode,
        }
    row["chunk"] = task["chunk"]
    if proc.returncode != 0 and row.get("status") not in {"ok", "skip", "filtered", "multi", "badfields", "zero_bytes"}:
        row["returncode"] = proc.returncode
    return row


def chunk_has_too_many_failures(counts, total, max_failure_rate):
    if total <= 0:
        return True
    failures = counts.get("error", 0) + counts.get("timeout", 0)
    return (failures / total) > max_failure_rate


def add_count(counts, row):
    status = row.get("status", "unknown")
    counts[status] = counts.get(status, 0) + 1


def write_chunk_summaries(out_root, selected, total_by_chunk, counts_by_chunk, start, args):
    summaries = {}
    for row in selected:
        chunk = row["chunk"]
        counts = counts_by_chunk.get(chunk, {})
        total = total_by_chunk.get(chunk, 0)
        summary = {
            "raw_root": str(Path(args.raw_root)),
            "out": str(Path(out_root)),
            "chunk": chunk,
            "steps": total,
            "counts": counts,
            "failure_rate": (counts.get("error", 0) + counts.get("timeout", 0)) / total if total else 1.0,
            "max_failure_rate": args.max_failure_rate,
            "manifest": str(manifest_path_for_chunk(out_root, chunk)),
            "elapsed_seconds": round(time.time() - start, 3),
        }
        summary["status"] = "failed" if chunk_has_too_many_failures(counts, total, args.max_failure_rate) else "ok"
        path = summary_path_for_chunk(out_root, chunk)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[chunk] = summary
    return summaries


def parent_main(args):
    selected, tasks = enumerate_steps(args.raw_root, args.chunks, args.limit)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_paths = [manifest_path_for_chunk(out_root, row["chunk"]) for row in selected]
    manifest_by_step = load_manifest_by_step(manifest_paths)
    total_by_chunk = {}
    counts_by_chunk = {row["chunk"]: {} for row in selected}
    counts = {}
    pending = []
    start = time.time()
    known_terminal = 0
    for task in tasks:
        chunk = task["chunk"]
        total_by_chunk[chunk] = total_by_chunk.get(chunk, 0) + 1
        previous = previous_terminal_row(task, out_root, manifest_by_step, retry_failures=args.retry_failures)
        if previous:
            previous["chunk"] = chunk
            add_count(counts, previous)
            add_count(counts_by_chunk.setdefault(chunk, {}), previous)
            known_terminal += 1
        else:
            pending.append(task)
    print(
        f"[{time.strftime('%H:%M:%S')}] chunks={len(selected)} steps={len(tasks)} "
        f"pending={len(pending)} known_terminal={known_terminal} "
        f"out={out_root} workers={args.workers} timeout={args.timeout}s",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one_task, task, out_root, args.timeout) for task in pending]
        processed = known_terminal
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            processed += 1
            add_count(counts, row)
            add_count(counts_by_chunk.setdefault(row["chunk"], {}), row)
            if row.get("status") != "skip":
                append_jsonl(manifest_path_for_chunk(out_root, row["chunk"]), row)
            if processed % max(1, args.progress_every) == 0 or processed == len(tasks):
                print(f"[{time.strftime('%H:%M:%S')}] {processed}/{len(tasks)} {counts}", flush=True)
    chunk_summaries = write_chunk_summaries(out_root, selected, total_by_chunk, counts_by_chunk, start, args)
    summary = {
        "raw_root": str(Path(args.raw_root)),
        "out": str(out_root),
        "chunks": [row["chunk"] for row in selected],
        "steps": len(tasks),
        "counts": counts,
        "failure_rate": (counts.get("error", 0) + counts.get("timeout", 0)) / len(tasks) if tasks else 1.0,
        "max_failure_rate": args.max_failure_rate,
        "manifests": [str(path) for path in manifest_paths],
        "known_terminal": known_terminal,
        "pending": len(pending),
        "chunk_summaries": {f"{chunk:04d}": data["status"] for chunk, data in chunk_summaries.items()},
        "elapsed_seconds": round(time.time() - start, 3),
    }
    summary["status"] = "failed" if any(data["status"] == "failed" for data in chunk_summaries.values()) else "ok"
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if summary["status"] == "failed" else 0


def main():
    parser = argparse.ArgumentParser(description="Windows-safe ABC STEP to parsed pkl processor.")
    parser.add_argument("--raw-root", default=r"D:\luolin\V13\ABC")
    parser.add_argument("--chunks", default="4")
    parser.add_argument("--out", default=r"D:\luolin\V13\processed_local\abc_parsed_full")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--child-step", default=None)
    parser.add_argument("--child-out", default=None)
    args = parser.parse_args()

    if args.child_step and args.child_out:
        child_main(args)
        return
    raise SystemExit(parent_main(args))


if __name__ == "__main__":
    main()
