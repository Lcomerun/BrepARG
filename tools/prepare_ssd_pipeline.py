import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CHUNK_RE = re.compile(r"^abc_(\d{4})_step_v00$")
DEFAULT_CONFIG = Path(r"D:\luolin\V13\local_training_config.json")
DEFAULT_REPORT = Path(r"D:\luolin\V13\local_reports\ssd_pipeline_readiness.json")
DEFAULT_COMMANDS = Path(r"D:\luolin\V13\local_reports\ssd_pipeline_commands.md")
DEFAULT_ENV_PYTHON = Path(r"C:\Users\YU\.conda\envs\brepgen_env\python.exe")
TRUE_VALUES = {"1", "true", "yes", "on", "y"}
FALSE_VALUES = {"0", "false", "no", "off", "n"}


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    return default


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def find_chunk_dirs(raw_root):
    raw_root = Path(raw_root)
    rows = []
    if not raw_root.exists():
        return rows
    for item in raw_root.iterdir():
        if not item.is_dir():
            continue
        match = CHUNK_RE.match(item.name)
        if not match:
            continue
        rows.append({"chunk": int(match.group(1)), "path": str(item)})
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


def count_step_files(path, limit=None):
    count = 0
    bytes_total = 0
    for step in Path(path).rglob("*.step"):
        count += 1
        try:
            bytes_total += step.stat().st_size
        except OSError:
            pass
        if limit and count >= limit:
            break
    return {"step_count": count, "step_bytes": bytes_total, "limited": bool(limit and count >= limit)}


def detect_gpu():
    result = {
        "python": sys.executable,
        "torch_import": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    try:
        import torch

        result["torch_import"] = True
        result["torch_version"] = torch.__version__
        result["torch_cuda_version"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["device_count"] = int(torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            result["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(props.total_memory),
                }
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def import_status():
    import importlib.util

    modules = ["numpy", "torch", "occwl", "OCC", "diffusers", "transformers", "tqdm", "shutup", "psutil"]
    return {name: bool(importlib.util.find_spec(name)) for name in modules}


def drive_info(paths):
    rows = []
    for path in paths:
        root = Path(path)
        existing = root if root.exists() else root.parent
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        try:
            usage = shutil.disk_usage(existing)
            rows.append(
                {
                    "path": str(path),
                    "checked_path": str(existing),
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                }
            )
        except OSError as exc:
            rows.append({"path": str(path), "error": str(exc)})
    return rows


def build_training_env(config):
    training = config.get("training", {})
    paths = config.get("paths", {})
    mapping = {
        "NS_POOL": paths.get("parsed_root", ""),
        "NS_OUTBASE": paths.get("train_out_root", ""),
        "NS_OUT": training.get("run_name", "newscheme_full_local"),
        "NS_N": training.get("ns_n", 5000),
        "NS_VQ_SAMPLES": training.get("ns_vq_samples", 60000),
        "NS_VQ_EPOCHS": training.get("ns_vq_epochs", 120),
        "NS_VQ_BS": training.get("ns_vq_bs", 128),
        "NS_VQ_MIN_EPOCHS": training.get("ns_vq_min_epochs", 12),
        "NS_VQ_PATIENCE": training.get("ns_vq_patience", 8),
        "NS_VQ_MIN_DELTA": training.get("ns_vq_min_delta", 1e-5),
        "NS_VQ_MAX_NONFINITE_VAL_EPOCHS": training.get("ns_vq_max_nonfinite_val_epochs", 2),
        "NS_DISABLE_AMP_VQVAE": int(parse_bool(training.get("ns_disable_amp_vqvae", False))),
        "NS_VQ_COMPLEX_FRACTION": training.get("ns_vq_complex_fraction", 0),
        "NS_VQ_COMPLEX_MIN_FACES": training.get("ns_vq_complex_min_faces", 12),
        "NS_VQ_COMPLEX_MIN_EDGES": training.get("ns_vq_complex_min_edges", 20),
        "NS_VQ_CURVED_FRACTION": training.get("ns_vq_curved_fraction", 0),
        "NS_AR_EPOCHS": training.get("ns_ar_epochs", 120),
        "NS_AR_BS": training.get("ns_ar_bs", 8),
        "NS_AR_DMODEL": training.get("ns_ar_dmodel", 256),
        "NS_AR_LAYERS": training.get("ns_ar_layers", 8),
        "NS_AR_MAX_SEQ_LEN": training.get("ns_ar_max_seq_len", 1024),
        "CUDA_VISIBLE_DEVICES": training.get("cuda_visible_devices", "0"),
        "PYTORCH_CUDA_ALLOC_CONF": training.get("pytorch_cuda_alloc_conf", "expandable_segments:True"),
    }
    return {key: str(value) for key, value in mapping.items()}


def powershell_env_prefix(env):
    return "; ".join(f"$env:{key}='{value}'" for key, value in env.items())


def render_commands(config, selected_chunks):
    paths = config["paths"]
    processing = config.get("processing", {})
    training_env = build_training_env(config)
    chunks = processing.get("chunks", "all")
    workers = processing.get("workers", 2)
    timeout = processing.get("timeout_seconds", 60)
    limit = processing.get("limit", 0)
    limit_arg = f" --limit {limit}" if int(limit) > 0 else ""
    lines = [
        "# SSD Pipeline Commands",
        "",
        "Run these from `D:\\luolin\\V13` in PowerShell.",
        "These commands call the conda environment's Python directly to avoid Windows `conda run` GBK output issues.",
        "",
        "## Preflight",
        "",
        "```powershell",
        "$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'",
        f"& '{DEFAULT_ENV_PYTHON}' tools/prepare_ssd_pipeline.py --config {DEFAULT_CONFIG} --write-report",
        "```",
        "",
        "## Parse ABC chunks on Windows",
        "",
        "```powershell",
        "$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'",
        f"& '{DEFAULT_ENV_PYTHON}' tools/process_abc_windows.py "
        f"--raw-root \"{paths['raw_root']}\" --chunks {chunks} --out \"{paths['parsed_root']}\" "
        f"--workers {workers} --timeout {timeout}{limit_arg}",
        "```",
        "",
        "## Train the improved FSQ/RCM pipeline",
        "",
        "```powershell",
        "$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'",
        f"{powershell_env_prefix(training_env)}",
        f"& '{DEFAULT_ENV_PYTHON}' breparg_improvements/train.py --stage all",
        "```",
        "",
        "## Discovered chunks",
        "",
    ]
    if selected_chunks:
        for row in selected_chunks:
            lines.append(f"- chunk {row['chunk']:04d}: `{row['path']}`")
    else:
        lines.append("- No selected chunk directories found yet.")
    return "\n".join(lines) + "\n"


def run_project_test():
    cmd = [sys.executable, "breparg_improvements/test_all.py"]
    started = datetime.now()
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "started": started.strftime("%Y-%m-%d %H:%M:%S"),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def build_report(config, args):
    raw_root = Path(args.raw_root or config["paths"]["raw_root"])
    config["paths"]["raw_root"] = str(raw_root)
    chunks = find_chunk_dirs(raw_root)
    selected = select_chunks(chunks, args.chunks or config.get("processing", {}).get("chunks", "all"))
    selected_with_counts = []
    for row in selected:
        enriched = dict(row)
        enriched.update(count_step_files(row["path"], limit=100000 if args.fast_count else None))
        selected_with_counts.append(enriched)
    report = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(args.config),
        "paths": config.get("paths", {}),
        "processing": config.get("processing", {}),
        "training": config.get("training", {}),
        "imports": import_status(),
        "gpu": detect_gpu(),
        "drives": drive_info(
            [
                config["paths"].get("raw_root", ""),
                config["paths"].get("parsed_root", ""),
                config["paths"].get("train_out_root", ""),
            ]
        ),
        "chunks_found": chunks,
        "chunks_selected": selected_with_counts,
        "training_env": build_training_env(config),
    }
    return report, selected


def main():
    parser = argparse.ArgumentParser(description="Prepare and report SSD-based ABC processing/training commands.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--chunks", default=None)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--run-project-test", action="store_true")
    parser.add_argument("--fast-count", action="store_true", help="Stop counting each selected chunk after 100000 STEP files.")
    args = parser.parse_args()

    config = load_config(args.config)
    report, selected = build_report(config, args)
    if args.run_project_test:
        report["project_test"] = run_project_test()

    commands = render_commands(config, selected)
    if args.write_report:
        save_json(DEFAULT_REPORT, report)
        DEFAULT_COMMANDS.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_COMMANDS.write_text(commands, encoding="utf-8")
        print(f"report={DEFAULT_REPORT}")
        print(f"commands={DEFAULT_COMMANDS}")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
