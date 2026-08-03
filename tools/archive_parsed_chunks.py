import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path


def parse_chunks(value):
    if not value or value.lower() == "all":
        return None
    chunks = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            for idx in range(int(start.replace("abc_", "")), int(end.replace("abc_", "")) + 1):
                chunks.add(f"abc_{idx:04d}")
        else:
            chunks.add(f"abc_{int(part.replace('abc_', '')):04d}")
    return chunks


def append_manifest(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def archive_chunk(chunk_dir, archive_path, compression):
    compression_mode = zipfile.ZIP_STORED if compression == "store" else zipfile.ZIP_DEFLATED
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    file_count = 0
    total_bytes = 0
    with zipfile.ZipFile(tmp_path, "w", compression=compression_mode, allowZip64=True) as zf:
        for path in sorted(chunk_dir.rglob("*")):
            if not path.is_file():
                continue
            file_count += 1
            total_bytes += path.stat().st_size
            zf.write(path, path.relative_to(chunk_dir.parent))
    tmp_path.replace(archive_path)
    return file_count, total_bytes


def archive_is_valid(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
        return bad is None
    except zipfile.BadZipFile:
        return False


def main():
    parser = argparse.ArgumentParser(description="Archive parsed ABC chunk directories one zip per chunk.")
    parser.add_argument("--parsed-root", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--chunks", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compression", choices=("deflate", "store"), default="deflate")
    parser.add_argument("--delete-after-verify", action="store_true")
    args = parser.parse_args()

    selected = parse_chunks(args.chunks)
    args.archive_root.mkdir(parents=True, exist_ok=True)
    chunk_dirs = [p for p in sorted(args.parsed_root.glob("abc_*")) if p.is_dir()]
    if selected:
        chunk_dirs = [p for p in chunk_dirs if p.name in selected]

    print(f"chunks={len(chunk_dirs)} parsed_root={args.parsed_root} archive_root={args.archive_root}", flush=True)
    for chunk_dir in chunk_dirs:
        started = time.time()
        archive_path = args.archive_root / f"{chunk_dir.name}_parsed.zip"
        if args.resume and archive_is_valid(archive_path):
            row = {
                "chunk": chunk_dir.name,
                "status": "skipped_existing",
                "archive": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
            }
            append_manifest(args.manifest, row)
            print(f"skip {chunk_dir.name}: existing archive", flush=True)
            continue
        try:
            file_count, source_bytes = archive_chunk(chunk_dir, archive_path, args.compression)
            ok = archive_is_valid(archive_path)
            row = {
                "chunk": chunk_dir.name,
                "status": "archived" if ok else "verify_failed",
                "archive": str(archive_path),
                "files": file_count,
                "source_bytes": source_bytes,
                "archive_bytes": archive_path.stat().st_size if archive_path.exists() else 0,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            if ok and args.delete_after_verify:
                shutil.rmtree(chunk_dir)
                row["deleted_source"] = True
            append_manifest(args.manifest, row)
            print(f"{row['status']} {chunk_dir.name}: files={file_count} archive={archive_path}", flush=True)
            if not ok:
                raise SystemExit(2)
        except Exception as exc:
            row = {
                "chunk": chunk_dir.name,
                "status": "error",
                "archive": str(archive_path),
                "error": repr(exc),
                "elapsed_seconds": round(time.time() - started, 3),
            }
            append_manifest(args.manifest, row)
            print(f"error {chunk_dir.name}: {exc}", flush=True)
            raise


if __name__ == "__main__":
    main()
