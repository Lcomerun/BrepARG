import argparse
import json
import re
import subprocess
from pathlib import Path


def tail_lines(path, count):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
        if text.count("\x00") > max(8, len(text) // 20):
            text = raw.decode("utf-16-le", errors="replace")
    lines = text.splitlines()
    return lines[-count:]


def latest_file(root, pattern):
    root = Path(root)
    if not root.exists():
        return None
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def ps_processes(pattern):
    script = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.CommandLine -match {pattern!r} -and $_.CommandLine -notmatch 'Get-CimInstance Win32_Process' }} | "
        "Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Depth 3"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    data = json.loads(result.stdout)
    return data if isinstance(data, list) else [data]


def nvidia_smi():
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}
    fields = [part.strip() for part in result.stdout.strip().split(",")]
    if len(fields) != 6:
        return {"available": True, "raw": result.stdout.strip()}
    return {
        "available": True,
        "name": fields[0],
        "utilization_gpu_percent": int(float(fields[1])),
        "memory_used_mib": int(float(fields[2])),
        "memory_total_mib": int(float(fields[3])),
        "temperature_c": int(float(fields[4])),
        "power_w": float(fields[5]),
    }


def ps_drive_free(names):
    rows = []
    for name in names:
        script = (
            f"Get-PSDrive '{name}' | "
            "Select-Object Name,Used,Free | ConvertTo-Json -Depth 3"
        )
        result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            rows.append({"Name": name, "error": (result.stderr or result.stdout).strip()})
            continue
        data = json.loads(result.stdout)
        rows.extend(data if isinstance(data, list) else [data])
    return rows


def parse_history(path):
    rows = []
    for line in tail_lines(path, 200):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"parse_error": line})
    return rows


def parse_ar_progress(lines):
    batch_re = re.compile(
        r"\[ar\]\s+ep\s+(?P<epoch>\d+)\s+batch\s+(?P<batch>\d+)/(?P<total>\d+)"
        r"\s+train_CE_running=(?P<ce>[0-9.]+)\s+elapsed_min=(?P<elapsed>[0-9.]+)"
    )
    epoch_re = re.compile(
        r"\[ar\]\s+ep\s+(?P<epoch>\d+)\s+train_CE=(?P<train>[0-9.]+)"
        r"\s+val_CE=(?P<val>[0-9.]+)\s+best=(?P<best>[0-9.]+)"
    )
    last_batch = None
    last_epoch = None
    prev_batch_by_epoch = {}
    for line in lines:
        batch_match = batch_re.search(line)
        if batch_match:
            epoch = int(batch_match.group("epoch"))
            batch = int(batch_match.group("batch"))
            total = int(batch_match.group("total"))
            elapsed = float(batch_match.group("elapsed"))
            progress = {
                "type": "batch",
                "epoch": epoch,
                "batch": batch,
                "total_batches": total,
                "train_ce_running": float(batch_match.group("ce")),
                "elapsed_min": elapsed,
                "epoch_progress_percent": round(batch * 100.0 / total, 2) if total else None,
            }
            previous = prev_batch_by_epoch.get(epoch)
            if previous and total >= batch and batch > previous["batch"] and elapsed > previous["elapsed_min"]:
                batches_per_min = (batch - previous["batch"]) / (elapsed - previous["elapsed_min"])
                remaining_batches = total - batch
                progress["recent_batches_per_min"] = round(batches_per_min, 2)
                progress["estimated_epoch_remaining_min"] = round(remaining_batches / batches_per_min, 2)
            prev_batch_by_epoch[epoch] = progress
            last_batch = progress
            continue
        epoch_match = epoch_re.search(line)
        if epoch_match:
            last_epoch = {
                "type": "epoch",
                "epoch": int(epoch_match.group("epoch")),
                "train_ce": float(epoch_match.group("train")),
                "val_ce": float(epoch_match.group("val")),
                "best_val_ce": float(epoch_match.group("best")),
            }
    return last_batch or last_epoch


def build_status(args):
    out_dir = Path(args.ar_out)
    ar_log = latest_file(args.ar_log_dir, getattr(args, "ar_log_pattern", "ar_*.log"))
    archive_log = latest_file(args.archive_log_dir, "archive_parsed_chunks_*.log")
    archive_root = Path(args.archive_root)
    manifest = archive_root / "_manifest.jsonl"
    checkpoints = out_dir / "ar_checkpoints"
    history = out_dir / "ar_history.jsonl"
    periodic = []
    if checkpoints.exists():
        periodic = [
            {"name": p.name, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime}
            for p in sorted(checkpoints.glob("ar_epoch_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        ]

    archive_zip_count = len(list(archive_root.glob("abc_*_parsed.zip"))) if archive_root.exists() else 0
    archive_tmp = [p.name for p in sorted(archive_root.glob("*.tmp"))] if archive_root.exists() else []
    history_rows = parse_history(history)
    latest_epoch = history_rows[-1] if history_rows else None
    ar_log_tail = tail_lines(ar_log, args.tail) if ar_log else []
    archive_log_tail = tail_lines(archive_log, args.tail) if archive_log else []
    return {
        "ar": {
            "processes": ps_processes("run_ar_v13_epoch100|train.py --stage ar"),
            "log": str(ar_log) if ar_log else None,
            "log_tail": ar_log_tail,
            "progress": parse_ar_progress(ar_log_tail),
            "out_dir": str(out_dir),
            "history": str(history),
            "history_rows": len(history_rows),
            "latest_epoch": latest_epoch,
            "best_checkpoint": checkpoint_info(out_dir / "ar_best.pt"),
            "latest_checkpoint": checkpoint_info(out_dir / "ar_latest.pt"),
            "periodic_checkpoints": periodic[:10],
        },
        "archive": {
            "processes": ps_processes("archive_parsed_chunks"),
            "log": str(archive_log) if archive_log else None,
            "log_tail": archive_log_tail,
            "manifest": str(manifest),
            "manifest_lines": len(tail_lines(manifest, 1000000)) if manifest.exists() else 0,
            "zip_count": archive_zip_count,
            "tmp_files": archive_tmp,
        },
        "gpu": nvidia_smi(),
        "drives": ps_drive_free(parse_drive_names(args.drives)),
    }


def parse_drive_names(value):
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def checkpoint_info(path):
    path = Path(path)
    if not path.exists():
        return None
    return {"path": str(path), "bytes": path.stat().st_size, "mtime": path.stat().st_mtime}


def status_json(status):
    return json.dumps(status, ensure_ascii=True, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Read-only status summary for V13 AR training and parsed archives.")
    parser.add_argument("--ar-out", default=r"D:\luolin\V13\local_runs\ar_training\train_outputs\newscheme_full_v13_ar")
    parser.add_argument("--ar-log-dir", default=r"D:\luolin\V13\local_runs\ar_training\logs")
    parser.add_argument("--ar-log-pattern", default="ar_*.log")
    parser.add_argument("--archive-root", default=r"D:\luolin\V13\ABC\processed\abc_parsed_full_archives")
    parser.add_argument("--archive-log-dir", default=r"D:\luolin\V13\ABC\processed\logs")
    parser.add_argument("--drives", default="D", help="Comma-separated drive letters to report, for example D or D,E.")
    parser.add_argument("--tail", type=int, default=30)
    args = parser.parse_args()
    print(status_json(build_status(args)))


if __name__ == "__main__":
    main()
