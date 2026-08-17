"""Archive lightweight P0-B precision and resume evidence for Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tools.run_p0b_stability_retest import (
        read_json,
        refresh_state,
        validation_summary,
    )
except ModuleNotFoundError:  # Direct `python tools/...py` execution.
    from run_p0b_stability_retest import (
        read_json,
        refresh_state,
        validation_summary,
    )


CHECKPOINT_FIELDS = ("best_checkpoint", "final_checkpoint", "rolling_checkpoint")
NONFINITE_FIELDS = (
    "nonfinite_loss_batches",
    "nonfinite_gradient_batches",
    "nonfinite_state_audits",
    "nonfinite_val_batches",
    "nonfinite_val_samples",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def parse_lock_metadata(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes().lstrip(b"\x00")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("writer lock metadata must be a JSON object")
    return payload


def curved_parent_mse(row: Mapping[str, Any]) -> float | None:
    value = (
        (row.get("val_parent_cluster_reconstruction_mse") or {})
        .get("surface_curved_proxy", {})
        .get("mse")
    )
    return float(value) if value is not None else None


def compact_epoch(row: Mapping[str, Any]) -> dict[str, Any]:
    audit = row.get("finite_state_audit") or {}
    usage = row.get("val_code_usage") or {}
    return {
        "epoch": row.get("epoch"),
        "train_loss": row.get("train_loss"),
        "val_loss": row.get("val_loss"),
        "curved_parent_mse": curved_parent_mse(row),
        "entropy_perplexity": usage.get("entropy_perplexity"),
        "coverage": usage.get("coverage"),
        "preclip_grad_norm": row.get("preclip_grad_norm"),
        "train_batches": row.get("train_batches"),
        "finite_train_batches": row.get("finite_train_batches"),
        "val_batches": row.get("val_batches"),
        "finite_val_batches": row.get("finite_val_batches"),
        **{field: row.get(field) for field in NONFINITE_FIELDS},
        "training_state_finite": row.get("training_state_finite"),
        "finite_state_audit_status": audit.get("status"),
        "finite_state_audit_tensors": audit.get("tensors"),
        "finite_state_audit_elements": audit.get("elements"),
        "resumed": row.get("resumed"),
        "resume_from_epoch": row.get("resume_from_epoch"),
    }


def validated_state(root: Path) -> dict[str, Any]:
    state = read_json(Path(root) / "p0b_state.json")
    refreshed = refresh_state(state)
    summary = validation_summary(refreshed)
    if refreshed.get("status") != "COMPLETED" or not summary.get("valid"):
        raise RuntimeError(
            f"P0-B evidence is not complete and valid at {root}: "
            + "; ".join(summary.get("reasons") or [])
        )
    return refreshed


def task_summary(task: Mapping[str, Any]) -> dict[str, Any]:
    history = read_json(Path(str(task["history"])))
    rows = list(history.get("history") or [])
    return {
        "task_id": task.get("task_id"),
        "arm": task.get("arm"),
        "seed": task.get("seed"),
        "status": task.get("status"),
        "precision": task.get("signature_payload", {}).get("precision"),
        "source_commit": (
            (history.get("config") or {})
            .get("signature_configuration", {})
            .get("git", {})
            .get("commit")
        ),
        "experiment_signature": task.get("signature"),
        "inventory": (task.get("validation") or {}).get("inventory"),
        "epochs": [compact_epoch(row) for row in rows],
        "attempts": [
            {
                key: attempt.get(key)
                for key in ("attempt", "auto_resume", "started_at", "finished_at", "returncode")
            }
            for attempt in task.get("attempts") or []
        ],
    }


def checkpoint_manifest(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for state in states:
        label = state["configuration"]["output_root"]
        root_name = Path(str(label)).name
        for task in state.get("tasks") or []:
            for field in CHECKPOINT_FIELDS:
                path = Path(str(task[field]))
                key = str(path.resolve()).casefold()
                if key in seen:
                    continue
                seen.add(key)
                if not path.is_file():
                    raise FileNotFoundError(path)
                rows.append(
                    {
                        "run": root_name,
                        "task_id": task.get("task_id"),
                        "role": field,
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "checkpoint_bytes_archived": False,
                    }
                )
    return rows


def scan_missing_runtime_logs(state: Mapping[str, Any], label: str) -> list[str]:
    """在写入任何文件之前预检:缺失的 attempt 日志会使快照静默漏采。"""
    missing: list[str] = []
    for task in state.get("tasks") or []:
        task_label = str(task["task_id"]).replace(":", "_")
        for attempt in task.get("attempts") or []:
            attempt_number = int(attempt["attempt"])
            for stream in ("stdout", "stderr"):
                source = Path(str(attempt.get(stream) or ""))
                if not source.is_file():
                    missing.append(
                        f"{label}/{task_label} attempt{attempt_number:03d} {stream}: {source}")
    return missing


def copy_runtime_logs(state: Mapping[str, Any], report_dir: Path, label: str) -> None:
    for task in state.get("tasks") or []:
        task_label = str(task["task_id"]).replace(":", "_")
        for attempt in task.get("attempts") or []:
            attempt_number = int(attempt["attempt"])
            for stream in ("stdout", "stderr"):
                source = Path(str(attempt.get(stream) or ""))
                if source.is_file():
                    target = (
                        report_dir
                        / "logs"
                        / label
                        / f"{task_label}_attempt{attempt_number:03d}.{stream}.log"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
        tensorboard = Path(str(task["task_root"])) / "tensorboard"
        if tensorboard.is_dir():
            for source in sorted(tensorboard.rglob("events.out.tfevents.*")):
                relative = source.relative_to(tensorboard).as_posix().replace("/", "__")
                target = report_dir / "tensorboard" / label / task_label / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)


def artifact_manifest(report_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(report_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(report_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]


def render_readme(payload: Mapping[str, Any]) -> str:
    probes = payload["precision_probes"]
    resume = payload["resume_smoke"]
    return f"""# P0-B runtime evidence

This directory is a lightweight, Git-safe snapshot of the P0-B CUDA preflight.
Checkpoint bytes and protocol data remain local; their local files are bound by
size and SHA-256 in `checkpoint_manifest.json`.

## Decision

- fp32 probe: `{probes['fp32']['status']}` with zero non-finite events.
- bf16 probe: `{probes['bf16']['status']}` with zero non-finite events.
- Selected formal precision: `{payload['selected_precision']}`.
- Both probes used the same ordered and sorted train/validation inventory digests.

## Resume and writer exclusion

The bf16 resume smoke was interrupted immediately after epoch 0 was atomically
saved. The next invocation restored the full rolling state and completed epoch 1.
Final history epochs are `{resume['history_epochs']}` and rolling checkpoint epoch
is `{resume['rolling_epoch']}`. A concurrent second writer exited nonzero, and the
subsequent launcher recorded stale/unreleased-lock recovery before completing.

## Contents

- `runtime_evidence.json`: compact per-epoch metrics, inventory digests, precision
  decision, resume evidence, and lock evidence.
- `checkpoint_manifest.json`: checkpoint sizes and SHA-256 only; no model bytes.
- `logs/`: task stdout/stderr, including the rejected concurrent writer evidence.
- `tensorboard/`: small probe TensorBoard event files.
- `artifact_manifest.json`: size and SHA-256 of every tracked report artifact.

These probes establish numerical execution and restart behavior only. They are
not representation-quality comparisons. Boundary-consistency work remains gated
until all four formal 100-epoch tasks and the learned-VQ 100-CAD assembly measure
are complete.
"""


def snapshot(
    fp32_root: Path,
    bf16_root: Path,
    resume_root: Path,
    report_dir: Path,
    *,
    second_writer_log: Path,
    allow_missing_logs: bool = False,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    if any(report_dir.iterdir()):
        raise RuntimeError(f"report directory must be empty: {report_dir}")
    states = {
        "fp32": validated_state(fp32_root),
        "bf16": validated_state(bf16_root),
        "resume": validated_state(resume_root),
    }
    missing_runtime_logs: list[str] = []
    for label, state in states.items():
        missing_runtime_logs.extend(scan_missing_runtime_logs(state, label))
    if missing_runtime_logs and not allow_missing_logs:
        head = "; ".join(missing_runtime_logs[:8])
        extra = f" (+{len(missing_runtime_logs) - 8} more)" if len(missing_runtime_logs) > 8 else ""
        raise RuntimeError(
            "runtime evidence logs are missing, snapshot would silently under-archive "
            "(rerun with --allow-missing-logs to archive anyway, recorded in payload): "
            + head + extra)
    probes = {}
    for precision in ("fp32", "bf16"):
        state = states[precision]
        tasks = [task_summary(task) for task in state["tasks"]]
        if any(
            epoch.get(field) != 0
            for task in tasks
            for epoch in task["epochs"]
            for field in NONFINITE_FIELDS
        ):
            raise RuntimeError(f"{precision} probe contains a non-finite event")
        probes[precision] = {
            "status": state["status"],
            "inventory_consistent": state.get("inventory_consistent"),
            "tasks": tasks,
        }
    fp_tasks = probes["fp32"]["tasks"]
    bf_tasks = probes["bf16"]["tasks"]
    if len(fp_tasks) != len(bf_tasks):
        raise RuntimeError("fp32/bf16 probes have different task counts")
    for task_index, (fp_task, bf_task) in enumerate(zip(fp_tasks, bf_tasks)):
        if fp_task["inventory"] != bf_task["inventory"]:
            raise RuntimeError(
                "fp32 and bf16 probes used different inventories "
                f"(task index {task_index})")

    resume_state = states["resume"]
    resume_task = task_summary(resume_state["tasks"][0])
    history_epochs = [row["epoch"] for row in resume_task["epochs"]]
    if history_epochs != [0, 1]:
        raise RuntimeError(f"resume history must be [0, 1], got {history_epochs}")
    if resume_task["epochs"][0]["resumed"] is not False:
        raise RuntimeError("resume epoch 0 must be the original epoch")
    if (
        resume_task["epochs"][1]["resumed"] is not True
        or resume_task["epochs"][1]["resume_from_epoch"] != 0
    ):
        raise RuntimeError("resume epoch 1 is not bound to rolling epoch 0")
    rolling_path = Path(str(resume_state["tasks"][0]["rolling_checkpoint"]))
    import torch

    rolling = torch.load(rolling_path, map_location="cpu", weights_only=False)
    if rolling.get("epoch") != 1:
        raise RuntimeError("resume rolling checkpoint did not finish at epoch 1")
    lock = parse_lock_metadata(Path(resume_root) / ".p0b_writer.lock")
    if not all(
        (
            lock.get("stale_lock_recovered") is True,
            lock.get("unreleased_lock_recovered") is True,
            bool(lock.get("released_at")),
        )
    ):
        raise RuntimeError("resume writer lock does not prove stale-lock recovery")
    writer_text = Path(second_writer_log).read_text(encoding="utf-8", errors="replace")
    if "already has an active writer" not in writer_text:
        raise RuntimeError("second-writer rejection evidence is missing")

    payload = {
        "schema": "p0b-runtime-evidence-v1",
        "generated_at": now(),
        "selected_precision": "bf16",
        "precision_probes": probes,
        "resume_smoke": {
            "status": resume_state["status"],
            "task": resume_task,
            "history_epochs": history_epochs,
            "rolling_epoch": rolling.get("epoch"),
            "second_writer_rejected": True,
            "stale_lock_recovered": lock.get("stale_lock_recovered"),
            "unreleased_lock_recovered": lock.get("unreleased_lock_recovered"),
            "lock_released": bool(lock.get("released_at")),
        },
        "formal_training_started": False,
        "boundary_consistency_allowed": False,
    }
    if missing_runtime_logs:
        payload["missing_runtime_logs"] = missing_runtime_logs
    write_json(report_dir / "runtime_evidence.json", payload)
    write_json(
        report_dir / "checkpoint_manifest.json",
        {
            "generated_at": payload["generated_at"],
            "policy": "Checkpoint bytes remain local and are not archived in Git.",
            "checkpoints": checkpoint_manifest(list(states.values())),
        },
    )
    for label, state in states.items():
        copy_runtime_logs(state, report_dir, label)
    writer_target = report_dir / "logs" / "resume" / "second_writer_rejected.stderr.log"
    writer_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(second_writer_log, writer_target)
    (report_dir / "README.md").write_text(render_readme(payload), encoding="utf-8")
    write_json(
        report_dir / "artifact_manifest.json",
        {
            "generated_at": payload["generated_at"],
            "artifacts": artifact_manifest(report_dir),
        },
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-root", type=Path, required=True)
    parser.add_argument("--bf16-root", type=Path, required=True)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--second-writer-log", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--allow-missing-logs", action="store_true")
    args = parser.parse_args(argv)
    payload = snapshot(
        args.fp32_root,
        args.bf16_root,
        args.resume_root,
        args.report_dir,
        second_writer_log=args.second_writer_log,
        allow_missing_logs=args.allow_missing_logs,
    )
    print(
        json.dumps(
            {
                "selected_precision": payload["selected_precision"],
                "report_dir": str(args.report_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
