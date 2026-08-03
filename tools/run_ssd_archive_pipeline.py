import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ARCHIVE_RE = re.compile(r"^abc_(\d{4})_step_v00\.7z$")
EXTRACTED_RE = re.compile(r"^abc_(\d{4})_step_v00$")
ENV_PYTHON = Path(r"C:\Users\YU\.conda\envs\brepgen_env\python.exe")


def repo_root():
    return Path(__file__).resolve().parents[1]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_resolve(path):
    return Path(path).expanduser().resolve()


def is_relative_to(path, parent):
    try:
        safe_resolve(path).relative_to(safe_resolve(parent))
        return True
    except ValueError:
        return False


def assert_within(path, parent, label):
    if not is_relative_to(path, parent):
        raise ValueError(f"{label} must stay under {parent}; got {path}")


def discover_archives(archive_root):
    archive_root = Path(archive_root)
    rows = []
    if not archive_root.exists():
        return rows
    for item in archive_root.iterdir():
        if not item.is_file():
            continue
        match = ARCHIVE_RE.match(item.name)
        if match:
            rows.append({"chunk": int(match.group(1)), "archive": str(item)})
    rows.sort(key=lambda row: row["chunk"])
    return rows


def discover_chunk_inputs(archive_root):
    archive_root = Path(archive_root)
    by_chunk = {}
    for row in discover_archives(archive_root):
        by_chunk.setdefault(row["chunk"], {"chunk": row["chunk"]})["archive"] = row["archive"]
    if archive_root.exists():
        for item in archive_root.iterdir():
            if not item.is_dir():
                continue
            match = EXTRACTED_RE.match(item.name)
            if match:
                chunk = int(match.group(1))
                entry = by_chunk.setdefault(chunk, {"chunk": chunk})
                entry["extract_dir"] = str(item)
    rows = list(by_chunk.values())
    rows.sort(key=lambda row: row["chunk"])
    return rows


def select_archives(archives, spec):
    if spec in (None, "", "all"):
        return archives
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
    return [row for row in archives if row["chunk"] in wanted]


def paths_for_archive(row, parsed_root):
    chunk = int(row["chunk"])
    archive = Path(row["archive"]) if row.get("archive") else None
    extract_dir = Path(row["extract_dir"]) if row.get("extract_dir") else archive.with_suffix("")
    chunk_name = f"abc_{chunk:04d}_step_v00"
    return {
        "archive": archive,
        "extract_dir": extract_dir,
        "parsed_chunk_dir": Path(parsed_root) / f"abc_{chunk:04d}",
        "chunk_name": chunk_name,
    }


def count_steps(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.step"))


def disk_free(path):
    path = Path(path)
    existing = path if path.exists() else path.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return shutil.disk_usage(existing).free


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_log(log_path, message):
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with Path(log_path).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_command(cmd, log_path, cwd=None, env=None):
    append_log(log_path, "RUN " + " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    start = time.time()
    with Path(log_path).open("a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    elapsed = round(time.time() - start, 3)
    append_log(log_path, f"EXIT code={proc.returncode} elapsed_seconds={elapsed}")
    return proc.returncode


def extract_archive(archive, extract_dir, log_path):
    if extract_dir.exists() and count_steps(extract_dir) > 0:
        append_log(log_path, f"EXTRACT skip existing {extract_dir}")
        return True
    if archive is None or not Path(archive).exists():
        append_log(log_path, f"EXTRACT failed missing archive={archive} extract_dir={extract_dir}")
        return False
    extract_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["tar", "-xf", str(archive), "-C", str(extract_dir)]
    code = run_command(cmd, log_path)
    ok = code == 0 and count_steps(extract_dir) > 0
    append_log(log_path, f"EXTRACT {'ok' if ok else 'failed'} archive={archive} steps={count_steps(extract_dir)}")
    return ok


def process_chunk(raw_root, chunk, parsed_root, workers, timeout, max_failure_rate, log_path):
    cmd = [
        str(ENV_PYTHON),
        str(repo_root() / "tools" / "process_abc_windows.py"),
        "--raw-root",
        str(raw_root),
        "--chunks",
        str(chunk),
        "--out",
        str(parsed_root),
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--max-failure-rate",
        str(max_failure_rate),
        "--progress-every",
        "1000",
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return run_command(cmd, log_path, cwd=repo_root(), env=env) == 0


def count_chunk_pkls(parsed_chunk_dir):
    return len(list(Path(parsed_chunk_dir).glob("*.pkl"))) if Path(parsed_chunk_dir).exists() else 0


def load_chunk_summary(parsed_chunk_dir):
    path = Path(parsed_chunk_dir) / "_reports" / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def train_after_processing(
    parsed_root,
    train_out_root,
    run_name,
    log_path,
    ns_n,
    vq_samples,
    vq_epochs,
    vq_bs,
    ar_epochs,
    ar_bs,
    vq_min_epochs,
    vq_patience,
    vq_min_delta,
    vq_max_nonfinite_val_epochs,
    disable_amp_vqvae,
):
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "NS_POOL": str(parsed_root),
            "NS_OUTBASE": str(train_out_root),
            "NS_OUT": run_name,
            "NS_N": str(ns_n),
            "NS_VQ_SAMPLES": str(vq_samples),
            "NS_VQ_EPOCHS": str(vq_epochs),
            "NS_VQ_BS": str(vq_bs),
            "NS_VQ_MIN_EPOCHS": str(vq_min_epochs),
            "NS_VQ_PATIENCE": str(vq_patience),
            "NS_VQ_MIN_DELTA": str(vq_min_delta),
            "NS_VQ_MAX_NONFINITE_VAL_EPOCHS": str(vq_max_nonfinite_val_epochs),
            "NS_DISABLE_AMP_VQVAE": "1" if disable_amp_vqvae else "0",
            "NS_AR_EPOCHS": str(ar_epochs),
            "NS_AR_BS": str(ar_bs),
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    cmd = [str(ENV_PYTHON), str(repo_root() / "breparg_improvements" / "train.py"), "--stage", "all"]
    return run_command(cmd, log_path, cwd=repo_root(), env=env) == 0


def load_state(path):
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created": now(), "chunks": {}, "training": {"status": "not_started"}}


def update_chunk_state(state, state_path, chunk, **updates):
    entry = state["chunks"].setdefault(f"{int(chunk):04d}", {})
    entry.update(updates)
    entry["updated"] = now()
    write_json(state_path, state)


def remove_file(path, allowed_parent, log_path):
    assert_within(path, allowed_parent, "file deletion")
    path = Path(path)
    if path.exists():
        path.unlink()
        append_log(log_path, f"DELETE file {path}")


def remove_dir(path, allowed_parent, log_path):
    assert_within(path, allowed_parent, "directory deletion")
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
        append_log(log_path, f"DELETE dir {path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract, parse, delete, and train ABC chunks on SSD.")
    parser.add_argument("--archive-root", default=r"E:\ABC\step")
    parser.add_argument("--parsed-root", default=r"E:\ABC\processed\abc_parsed_full")
    parser.add_argument("--train-out-root", default=r"E:\ABC\processed\train_outputs")
    parser.add_argument("--chunks", default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--delete-archive-after-extract", action="store_true")
    parser.add_argument("--delete-extracted-after-process", action="store_true")
    parser.add_argument("--train-after", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-name", default="newscheme_full_local")
    parser.add_argument("--ns-n", type=int, default=999999)
    parser.add_argument("--vq-samples", type=int, default=300000)
    parser.add_argument("--vq-epochs", type=int, default=120)
    parser.add_argument("--vq-bs", type=int, default=128)
    parser.add_argument("--vq-min-epochs", type=int, default=12)
    parser.add_argument("--vq-patience", type=int, default=8)
    parser.add_argument("--vq-min-delta", type=float, default=1e-5)
    parser.add_argument("--vq-max-nonfinite-val-epochs", type=int, default=2)
    parser.add_argument("--disable-amp-vqvae", action="store_true")
    parser.add_argument("--ar-epochs", type=int, default=120)
    parser.add_argument("--ar-bs", type=int, default=8)
    return parser


def main():
    args = build_arg_parser().parse_args()
    archive_root = safe_resolve(args.archive_root)
    parsed_root = safe_resolve(args.parsed_root)
    train_out_root = safe_resolve(args.train_out_root)
    processed_root = safe_resolve(parsed_root.parent)
    logs_root = processed_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"archive_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    state_path = logs_root / "archive_pipeline_state.json"

    assert_within(archive_root, r"E:\ABC", "archive_root")
    assert_within(parsed_root, r"E:\ABC", "parsed_root")
    assert_within(train_out_root, r"E:\ABC", "train_out_root")

    archives = select_archives(discover_chunk_inputs(archive_root), args.chunks)
    state = load_state(state_path)
    state["last_started"] = now()
    state["archive_root"] = str(archive_root)
    state["parsed_root"] = str(parsed_root)
    state["train_out_root"] = str(train_out_root)
    state["workers"] = args.workers
    state["timeout"] = args.timeout
    state["max_failure_rate"] = args.max_failure_rate
    write_json(state_path, state)

    append_log(log_path, f"PIPELINE start archives={len(archives)} dry_run={args.dry_run}")
    append_log(log_path, f"STATE {state_path}")
    append_log(log_path, f"FREE archive_root_gb={disk_free(archive_root)/1e9:.1f} parsed_root_gb={disk_free(parsed_root)/1e9:.1f}")

    if args.dry_run:
        payload = {
            "archives": archives,
            "archive_count": len(archives),
            "parsed_root": str(parsed_root),
            "train_out_root": str(train_out_root),
            "delete_archive_after_extract": args.delete_archive_after_extract,
            "delete_extracted_after_process": args.delete_extracted_after_process,
            "train_after": args.train_after,
        }
        append_log(log_path, json.dumps(payload, ensure_ascii=False, indent=2))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for row in archives:
        chunk = row["chunk"]
        paths = paths_for_archive(row, parsed_root)
        archive = paths["archive"]
        extract_dir = paths["extract_dir"]
        parsed_chunk_dir = paths["parsed_chunk_dir"]
        if archive is not None:
            assert_within(archive, archive_root, "archive")
        assert_within(extract_dir, archive_root, "extract_dir")
        assert_within(parsed_chunk_dir, parsed_root, "parsed_chunk_dir")

        if disk_free(parsed_root) < args.min_free_gb * 1e9:
            append_log(log_path, f"STOP free space below {args.min_free_gb} GB")
            return 2

        update_chunk_state(state, state_path, chunk, status="started", archive=str(archive), extract_dir=str(extract_dir), parsed_dir=str(parsed_chunk_dir))
        append_log(log_path, f"CHUNK {chunk:04d} start archive={archive} extract_dir={extract_dir}")

        ok_extract = extract_archive(archive, extract_dir, log_path)
        if not ok_extract:
            update_chunk_state(state, state_path, chunk, status="extract_failed")
            continue
        update_chunk_state(state, state_path, chunk, status="extracted", step_count=count_steps(extract_dir))

        if args.delete_archive_after_extract and archive is not None and archive.exists():
            remove_file(archive, archive_root, log_path)
            update_chunk_state(state, state_path, chunk, archive_deleted=True)

        ok_parse = process_chunk(archive_root, chunk, parsed_root, args.workers, args.timeout, args.max_failure_rate, log_path)
        chunk_summary = load_chunk_summary(parsed_chunk_dir)
        parsed_count = count_chunk_pkls(parsed_chunk_dir)
        update_chunk_state(
            state,
            state_path,
            chunk,
            status="parsed" if ok_parse else "parse_failed",
            parsed_count=parsed_count,
            parse_summary=chunk_summary,
        )
        if not ok_parse:
            append_log(log_path, f"CHUNK {chunk:04d} parse failed; keeping extracted directory for retry")
            continue

        if args.delete_extracted_after_process and extract_dir.exists():
            remove_dir(extract_dir, archive_root, log_path)
            update_chunk_state(state, state_path, chunk, extracted_deleted=True)

        update_chunk_state(state, state_path, chunk, status="done", free_gb=round(disk_free(parsed_root) / 1e9, 3))
        append_log(log_path, f"CHUNK {chunk:04d} done parsed_count={parsed_count}")

    failed = {key: value.get("status") for key, value in state.get("chunks", {}).items() if value.get("status") not in {"done"}}
    if failed:
        append_log(log_path, f"STOP training skipped because chunks are incomplete_or_failed: {failed}")
        state["training"] = {"status": "skipped_due_to_incomplete_chunks", "updated": now(), "chunks": failed}
        write_json(state_path, state)
        return 4

    if args.train_after:
        state["training"] = {"status": "started", "updated": now()}
        write_json(state_path, state)
        append_log(log_path, "TRAIN start")
        ok_train = train_after_processing(
            parsed_root,
            train_out_root,
            args.run_name,
            log_path,
            args.ns_n,
            args.vq_samples,
            args.vq_epochs,
            args.vq_bs,
            args.ar_epochs,
            args.ar_bs,
            args.vq_min_epochs,
            args.vq_patience,
            args.vq_min_delta,
            args.vq_max_nonfinite_val_epochs,
            args.disable_amp_vqvae,
        )
        state["training"] = {"status": "done" if ok_train else "failed", "updated": now()}
        write_json(state_path, state)
        append_log(log_path, f"TRAIN {'done' if ok_train else 'failed'}")
        return 0 if ok_train else 3

    append_log(log_path, "PIPELINE done without training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
