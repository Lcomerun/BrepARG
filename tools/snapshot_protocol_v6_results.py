"""Archive lightweight Protocol V6 evidence without copying model/data artifacts."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ARMS = (
    "fsq_8192_4d",
    "fsq_4096_6d",
    "vq_4096_64d_random",
    "continuous_bypass_64d",
)
SEEDS = (0, 1, 2, 3, 4)
SUMMARY_FIELDS = (
    "seed",
    "arm",
    "status",
    "health",
    "epochs_ran",
    "latest_epoch",
    "best_epoch",
    "best_val_recon",
    "fully_finite_train_epochs",
    "fully_finite_val_epochs",
    "first_nonfinite_epoch",
    "last_finite_train_fraction",
    "last_finite_val_fraction",
    "best_curved_parent_mse",
    "best_curved_parent_epoch",
    "final_curved_parent_mse",
    "final_perplexity",
    "final_checkpoint_epoch",
    "promotion_eligible",
    "source_commit",
)


def process_exists(pid: Any) -> bool:
    try:
        pid_value = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_value <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid_value
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid_value, 0)
        return True
    except OSError:
        return False


def observed_run_state(run_root: Path, recorded: dict[str, Any]) -> dict[str, Any]:
    pid_path = run_root / "launcher.pid"
    pid = pid_path.read_text(encoding="utf-8").strip() if pid_path.is_file() else None
    launcher_alive = process_exists(pid)
    recorded_status = recorded.get("status")
    effective_status = recorded_status
    if recorded_status == "RUNNING" and not launcher_alive:
        effective_status = "INTERRUPTED"
    return {
        "recorded_status": recorded_status,
        "effective_status": effective_status,
        "recorded_active_seed": recorded.get("active_seed"),
        "launcher_pid": int(pid) if pid and pid.isdigit() else None,
        "launcher_alive": launcher_alive,
        "observed_at": now(),
    }


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, attempts: int = 6) -> tuple[dict[str, Any], bytes]:
    """Read JSON only when its size/mtime remain unchanged across the read."""
    error: Exception | None = None
    for _ in range(attempts):
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                time.sleep(0.2)
                continue
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"expected JSON object in {path}")
            return payload, raw
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            error = exc
            time.sleep(0.2)
    raise RuntimeError(f"could not read stable JSON from {path}: {error}")


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    except (TypeError, ValueError):
        return None


def curved_parent_mse(row: dict[str, Any]) -> float | None:
    buckets = row.get("val_parent_cluster_reconstruction_mse") or {}
    value = (buckets.get("surface_curved_proxy") or {}).get("mse")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fully_finite(row: dict[str, Any], prefix: str) -> bool:
    total = row.get(f"{prefix}_batches")
    finite = row.get(f"finite_{prefix}_batches")
    try:
        return int(total) > 0 and int(finite) == int(total)
    except (TypeError, ValueError):
        return False


def summarize_history(
    seed: int,
    arm: str,
    payload: dict[str, Any] | None,
    sweep_row: dict[str, Any] | None,
    source_commit: str | None,
) -> dict[str, Any]:
    if payload is None:
        return {
            **{field: None for field in SUMMARY_FIELDS},
            "seed": seed,
            "arm": arm,
            "status": "pending",
            "health": "PENDING",
            "epochs_ran": 0,
            "fully_finite_train_epochs": 0,
            "fully_finite_val_epochs": 0,
            "source_commit": source_commit,
        }
    rows = list(payload.get("history") or [])
    latest = rows[-1] if rows else {}
    target = int((payload.get("config") or {}).get("target_epoch") or 100)
    completed = len(rows) == target and int(latest.get("epoch", -1)) == target - 1
    finite_train = sum(fully_finite(row, "train") for row in rows)
    finite_val = sum(fully_finite(row, "val") for row in rows)
    nonfinite_rows = [
        row for row in rows if not fully_finite(row, "train") or not fully_finite(row, "val")
    ]
    curved = [
        (int(row.get("epoch", -1)), curved_parent_mse(row))
        for row in rows
        if fully_finite(row, "val") and curved_parent_mse(row) is not None
    ]
    best_curved = min(curved, key=lambda item: item[1]) if curved else (None, None)
    final_curved = curved_parent_mse(latest) if fully_finite(latest, "val") else None
    code_usage = latest.get("val_code_usage") or {}
    final_perplexity = code_usage.get("entropy_perplexity") if fully_finite(latest, "val") else None
    promotion = (sweep_row or {}).get("promotion") or {}
    health = "NUMERICALLY_UNSTABLE" if nonfinite_rows else (
        "HEALTHY_COMPLETE" if completed else "RUNNING_FINITE"
    )
    return {
        "seed": seed,
        "arm": arm,
        "status": "completed" if completed else "running",
        "health": health,
        "epochs_ran": len(rows),
        "latest_epoch": latest.get("epoch"),
        "best_epoch": payload.get("best_epoch", latest.get("best_epoch")),
        "best_val_recon": payload.get("best_val_recon", latest.get("best_val")),
        "fully_finite_train_epochs": finite_train,
        "fully_finite_val_epochs": finite_val,
        "first_nonfinite_epoch": nonfinite_rows[0].get("epoch") if nonfinite_rows else None,
        "last_finite_train_fraction": ratio(
            latest.get("finite_train_batches"), latest.get("train_batches")
        ),
        "last_finite_val_fraction": ratio(
            latest.get("finite_val_batches"), latest.get("val_batches")
        ),
        "best_curved_parent_mse": best_curved[1],
        "best_curved_parent_epoch": best_curved[0],
        "final_curved_parent_mse": final_curved,
        "final_perplexity": final_perplexity,
        "final_checkpoint_epoch": (sweep_row or {}).get("final_checkpoint_epoch"),
        "promotion_eligible": promotion.get("eligible"),
        "source_commit": source_commit,
    }


def archive_json(source: Path, target: Path) -> dict[str, Any]:
    payload, raw = stable_json(source)
    write_bytes(target, raw)
    return payload


def iter_tensorboard_files(seed_dir: Path) -> Iterable[Path]:
    root = seed_dir / "tensorboard"
    if root.is_dir():
        yield from sorted(path for path in root.rglob("events.out.tfevents.*") if path.is_file())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in SUMMARY_FIELDS} for row in rows)


def artifact_manifest(report_dir: Path) -> list[dict[str, Any]]:
    excluded = {"artifact_manifest.json"}
    return [
        {
            "path": path.relative_to(report_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(report_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]


def checkpoint_manifest(run_root: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    """Bind local checkpoints by hash without copying checkpoint bytes."""
    recorded: dict[str, dict[str, Any]] = {}
    for step in state.get("steps") or []:
        validation = step.get("validation") or {}
        for item in validation.get("checkpoints") or []:
            source = Path(str(item.get("path") or ""))
            if not source.name:
                continue
            recorded[source.name + f"@seed{step.get('seed')}"] = item
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_dir = run_root / f"seed{seed}"
        for path in sorted(seed_dir.glob("*.pt")) if seed_dir.is_dir() else []:
            key = path.name + f"@seed{seed}"
            item = recorded.get(key) or {}
            expected_hash = item.get("sha256")
            actual_hash = expected_hash or sha256_file(path)
            rows.append({
                "seed": seed,
                "file": path.name,
                "relative_path": path.relative_to(run_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": actual_hash,
                "hash_source": "cohort_state" if expected_hash else "snapshot_recomputed",
                "checkpoint_bytes_archived": False,
            })
    return rows


def render_readme(
    state: dict[str, Any], observed: dict[str, Any], rows: list[dict[str, Any]], snapshot_at: str
) -> str:
    completed = sorted({row["seed"] for row in rows if row["status"] == "completed"} & set(SEEDS))
    completed_seeds = [
        seed for seed in completed
        if all(row["status"] == "completed" for row in rows if row["seed"] == seed)
    ]
    active = state.get("active_seed")
    unstable = sum(row["health"] == "NUMERICALLY_UNSTABLE" for row in rows)
    healthy = sum(row["health"] == "HEALTHY_COMPLETE" for row in rows)
    interrupted_finite = sum(row["health"] == "INTERRUPTED_FINITE" for row in rows)
    return f"""# Protocol V6: five-seed 100-epoch cohort

This is a lightweight snapshot generated at `{snapshot_at}` from the local
Protocol V6 run. The formal matrix contains four representation arms at seeds
0 through 4, with 300,000 train patches, 12,000 validation patches, batch size
128, learning rate 3e-4, and 100 requested epochs per arm.

## Snapshot status

- Effective launcher status: `{observed.get('effective_status')}`.
- Last recorded launcher status: `{observed.get('recorded_status')}`; launcher PID
  `{observed.get('launcher_pid')}` alive: `{observed.get('launcher_alive')}`.
- Fully completed seeds: `{completed_seeds}`.
- Active seed: `{active}`.
- Numerically healthy completed arm/seed histories: `{healthy}`.
- Histories with at least one incomplete/non-finite train or validation epoch: `{unstable}`.
- Fully finite histories interrupted before their target epoch: `{interrupted_finite}`.
- Surface reconstruction: `{'completed' if state.get('phase') == 'COMPLETED' else 'pending'}`.
- Sequence regeneration and AR: blocked.

`training_health_summary.csv` and `.json` distinguish a fixed 100-epoch loop
from numerical health. `NUMERICALLY_UNSTABLE` means at least one epoch did not
have all expected train and validation batches finite; such a result must not
be promoted even when the launcher accepted checkpoint/cap integrity.

## Tracked evidence

- `cohort_state.json`: launcher state and checkpoint hashes for completed seeds.
- `seedN/`: available per-arm histories and completed sweep manifests.
- `logs/`: stdout/stderr snapshots.
- `tensorboard/`: small TensorBoard event snapshots.
- `checkpoint_manifest.json`: local checkpoint sizes and SHA-256 hashes without
  checkpoint bytes.
- `interruption_evidence.json`: Windows restart and stale-launcher evidence.
- `continuation_assessment.md`: recovery value, constraints, and recommended scope.
- `artifact_manifest.json`: byte size and SHA-256 for every archived artifact.

Model checkpoints (`*.pt`), reconstructed arrays (`*.npz`), raw protocol data,
and PID files are excluded. Surface reconstruction JSON/JSONL/CSV evidence will
be archived after the training matrix finishes and the automatic evaluator runs.
"""


def snapshot(run_root: Path, report_dir: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    report_dir = report_dir.resolve()
    state, _ = stable_json(run_root / "cohort_state.json")
    observed = observed_run_state(run_root, state)
    archive_json(run_root / "cohort_state.json", report_dir / "cohort_state.json")
    legacy_partial = report_dir / "seed2_fsq_8192_4d_history.json"
    if legacy_partial.is_file() and (run_root / "seed2" / "fsq_8192_4d_history.json").is_file():
        legacy_partial.unlink()
    summaries: list[dict[str, Any]] = []
    for log in sorted((run_root / "logs").glob("seed*.log")):
        copy_file(log, report_dir / "logs" / log.name)
    for seed in SEEDS:
        source_seed = run_root / f"seed{seed}"
        target_seed = report_dir / f"seed{seed}"
        sweep_path = source_seed / "vqvae_hp_sweep.json"
        sweep: dict[str, Any] | None = None
        if sweep_path.is_file():
            sweep = archive_json(sweep_path, target_seed / sweep_path.name)
        sweep_rows = {
            str(row.get("name")): row for row in (sweep or {}).get("mse_ranking", [])
        }
        source_commit = ((sweep or {}).get("run_manifest") or {}).get("git", {}).get("commit")
        for arm in ARMS:
            history_path = source_seed / f"{arm}_history.json"
            history = None
            if history_path.is_file():
                history = archive_json(history_path, target_seed / history_path.name)
            summaries.append(
                summarize_history(seed, arm, history, sweep_rows.get(arm), source_commit)
            )
        for event in iter_tensorboard_files(source_seed):
            relative = event.relative_to(source_seed / "tensorboard")
            copy_file(event, report_dir / "tensorboard" / f"seed{seed}" / relative)
    if observed["effective_status"] == "INTERRUPTED":
        for row in summaries:
            if row["status"] == "running":
                row["status"] = "interrupted"
                if row["health"] == "RUNNING_FINITE":
                    row["health"] = "INTERRUPTED_FINITE"
    reconstruction = run_root / "surface_reconstruction"
    if reconstruction.is_dir():
        for source in sorted(reconstruction.rglob("*")):
            if source.is_file() and source.suffix.lower() in {".json", ".jsonl", ".csv", ".log"}:
                copy_file(source, report_dir / "surface_reconstruction" / source.relative_to(reconstruction))
    snapshot_at = now()
    summary_payload = {
        "report": report_dir.name,
        "snapshot_at": snapshot_at,
        "status": observed["effective_status"],
        "recorded_status": state.get("status"),
        "active_seed": state.get("active_seed"),
        "observed_run_state": observed,
        "surface_reconstruction": "completed" if state.get("phase") == "COMPLETED" else "pending",
        "downstream_ar_allowed": False,
        "configuration": state.get("configuration"),
        "rows": summaries,
    }
    write_bytes(
        report_dir / "training_health_summary.json",
        (json.dumps(summary_payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    write_csv(report_dir / "training_health_summary.csv", summaries)
    write_bytes(
        report_dir / "checkpoint_manifest.json",
        (json.dumps({
            "generated_at": snapshot_at,
            "policy": "Checkpoint bytes remain local and are not archived in Git.",
            "checkpoints": checkpoint_manifest(run_root, state),
        }, indent=2) + "\n").encode("utf-8"),
    )
    write_bytes(
        report_dir / "README.md",
        render_readme(state, observed, summaries, snapshot_at).encode("utf-8"),
    )
    manifest = artifact_manifest(report_dir)
    write_bytes(
        report_dir / "artifact_manifest.json",
        (json.dumps({"generated_at": snapshot_at, "artifacts": manifest}, indent=2) + "\n").encode("utf-8"),
    )
    return summary_payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = snapshot(args.run_root, args.report_dir)
    print(json.dumps({
        "status": summary["status"],
        "active_seed": summary["active_seed"],
        "report_dir": str(args.report_dir.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
