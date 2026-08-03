import argparse
import csv
import gzip
import json
import os
import pickle
import sys
import time
import types
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


def find_step_files(root):
    return sorted(Path(root).rglob("*.step"))


def file_size_stats(files):
    sizes = [p.stat().st_size for p in files]
    if not sizes:
        return {}
    sizes_sorted = sorted(sizes)
    return {
        "count": len(sizes),
        "total_bytes": int(sum(sizes)),
        "min_bytes": int(sizes_sorted[0]),
        "max_bytes": int(sizes_sorted[-1]),
        "mean_bytes": float(sum(sizes) / len(sizes)),
        "p50_bytes": int(sizes_sorted[len(sizes_sorted) // 2]),
        "p90_bytes": int(sizes_sorted[int(len(sizes_sorted) * 0.9)]),
        "p99_bytes": int(sizes_sorted[min(len(sizes_sorted) - 1, int(len(sizes_sorted) * 0.99))]),
        "zero_byte_files": int(sum(1 for value in sizes if value == 0)),
        "over_20mb_files": int(sum(1 for value in sizes if value > 20 * 1024 * 1024)),
        "over_100mb_files": int(sum(1 for value in sizes if value > 100 * 1024 * 1024)),
    }


def summarize_parsed(data):
    summary = {}
    for key, value in data.items():
        if hasattr(value, "shape"):
            summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, list):
            summary[key] = {"type": "list", "len": len(value)}
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def install_noop_shutup():
    if "shutup" in sys.modules:
        return
    module = types.ModuleType("shutup")
    module.please = lambda *args, **kwargs: None
    sys.modules["shutup"] = module


def import_parser(repo_root):
    install_noop_shutup()
    process_data = Path(repo_root) / "BrepARG" / "process_data"
    if str(process_data) not in sys.path:
        sys.path.insert(0, str(process_data))
    import process_brep
    from occwl.io import load_step

    return process_brep, load_step


def select_smoke_files(files, limit, max_step_bytes):
    selected = []
    skipped = []
    for path in files:
        size = path.stat().st_size
        if size <= 0:
            skipped.append({"path": str(path), "reason": "zero_bytes", "bytes": size})
            continue
        if size > max_step_bytes:
            skipped.append({"path": str(path), "reason": "over_max_step_bytes", "bytes": size})
            continue
        selected.append(path)
        if len(selected) >= limit:
            break
    return selected, skipped


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def compare_pickle_compression(pkl_path):
    gz_path = pkl_path.with_suffix(pkl_path.suffix + ".gztest")
    with pkl_path.open("rb") as source, gzip.open(gz_path, "wb", compresslevel=6) as target:
        target.write(source.read())
    raw = pkl_path.stat().st_size
    gz = gz_path.stat().st_size
    gz_path.unlink(missing_ok=True)
    return {"pkl_bytes": raw, "pkl_gzip6_bytes": gz, "gzip6_ratio": gz / raw if raw else None}


def smoke_parse(args, files):
    parsed_dir = Path(args.output) / "parsed_pkl"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    process_brep, load_step = import_parser(args.repo_root)

    selected, pre_skipped = select_smoke_files(files, args.limit, args.max_step_bytes)
    rows = []
    compression_rows = []
    counts = {
        "selected": len(selected),
        "ok": 0,
        "multi": 0,
        "filtered": 0,
        "badfields": 0,
        "error": 0,
        "pre_skipped": len(pre_skipped),
    }

    for step_path in selected:
        start = time.time()
        base = step_path.stem
        out_pkl = parsed_dir / f"{base}.pkl"
        row = {
            "step_path": str(step_path),
            "step_bytes": step_path.stat().st_size,
            "output_pkl": str(out_pkl),
        }
        try:
            solids = load_step(str(step_path))
            if len(solids) != 1:
                row["status"] = "multi"
                row["solid_count"] = len(solids)
                counts["multi"] += 1
            else:
                data = process_brep.parse_solid(solids[0])
                if data is None:
                    row["status"] = "filtered"
                    counts["filtered"] += 1
                elif not EXPECTED_FIELDS.issubset(set(data.keys())):
                    row["status"] = "badfields"
                    row["missing_fields"] = sorted(EXPECTED_FIELDS - set(data.keys()))
                    counts["badfields"] += 1
                else:
                    with out_pkl.open("wb") as handle:
                        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    row["status"] = "ok"
                    row["pkl_bytes"] = out_pkl.stat().st_size
                    row["fields"] = summarize_parsed(data)
                    row.update(compare_pickle_compression(out_pkl))
                    compression_rows.append(
                        {
                            "step_path": str(step_path),
                            "step_bytes": row["step_bytes"],
                            "pkl_bytes": row["pkl_bytes"],
                            "pkl_gzip6_bytes": row["pkl_gzip6_bytes"],
                            "gzip6_ratio": row["gzip6_ratio"],
                        }
                    )
                    counts["ok"] += 1
        except Exception as exc:
            row["status"] = "error"
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            counts["error"] += 1
        row["seconds"] = round(time.time() - start, 3)
        rows.append(row)

    return rows, pre_skipped, compression_rows, counts


def write_size_csv(path, files):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes"])
        writer.writeheader()
        for step in files:
            writer.writerow({"path": str(step), "bytes": step.stat().st_size})


def main():
    parser = argparse.ArgumentParser(description="Audit and smoke-parse local ABC chunk4 data.")
    parser.add_argument("--repo-root", default=r"D:\luolin\V13")
    parser.add_argument("--input", default=r"D:\luolin\V13\ABC\abc_0004_step_v00")
    parser.add_argument("--output", default=r"D:\luolin\V13\processed_local\abc_0004_smoke")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-step-bytes", type=int, default=2_000_000)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    files = find_step_files(args.input)
    stats = file_size_stats(files)
    write_size_csv(output / "step_file_sizes.csv", files)

    parse_rows = []
    pre_skipped = []
    compression_rows = []
    counts = {}
    if not args.audit_only:
        parse_rows, pre_skipped, compression_rows, counts = smoke_parse(args, files)
        write_jsonl(output / "manifest.jsonl", parse_rows)
        write_jsonl(output / "pre_skipped.jsonl", pre_skipped)
        write_jsonl(output / "compression_probe.jsonl", compression_rows)

    ok_rows = [row for row in parse_rows if row.get("status") == "ok"]
    parsed_bytes = sum(row.get("pkl_bytes", 0) for row in ok_rows)
    step_bytes = sum(row.get("step_bytes", 0) for row in ok_rows)
    gzip_bytes = sum(row.get("pkl_gzip6_bytes", 0) for row in ok_rows)

    summary = {
        "input": str(Path(args.input)),
        "output": str(output),
        "repo_root": str(Path(args.repo_root)),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step_stats": stats,
        "smoke_limit": args.limit,
        "max_step_bytes": args.max_step_bytes,
        "audit_only": args.audit_only,
        "parse_counts": counts,
        "ok_step_bytes": step_bytes,
        "ok_parsed_pkl_bytes": parsed_bytes,
        "ok_parsed_pkl_to_step_ratio": parsed_bytes / step_bytes if step_bytes else None,
        "ok_gzip6_bytes": gzip_bytes,
        "ok_gzip6_to_pkl_ratio": gzip_bytes / parsed_bytes if parsed_bytes else None,
        "notes": [
            "Parsed pkl files preserve the existing BrepARG dictionary contract.",
            "Gzip numbers are probe-only; no .pkl.gz files are retained.",
            "Large STEP files are skipped by default during smoke parsing.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
