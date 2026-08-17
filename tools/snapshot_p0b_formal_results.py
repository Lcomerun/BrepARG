"""Archive a Git-safe snapshot of the completed formal P0-B cohort.

The snapshot keeps lightweight training evidence and TensorBoard events while
binding local checkpoints by size and SHA-256. Model bytes and protocol data
are never copied into the report directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from tools.run_p0b_stability_retest import read_json, refresh_state, validation_summary
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    from run_p0b_stability_retest import read_json, refresh_state, validation_summary


ARMS = ("vq_4096_64d_random", "continuous_bypass_64d")
SEEDS = (3, 4)
EXPECTED_TASKS = {(arm, seed) for arm in ARMS for seed in SEEDS}
CHECKPOINT_FIELDS = ("best_checkpoint", "final_checkpoint", "rolling_checkpoint")
NONFINITE_FIELDS = (
    "nonfinite_loss_batches",
    "nonfinite_gradient_batches",
    "nonfinite_state_batches",
    "nonfinite_state_audits",
    "nonfinite_val_batches",
    "nonfinite_val_samples",
)
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
    ".step",
    ".stp",
}
MAX_LIGHTWEIGHT_FILE_BYTES = 8 * 1024 * 1024
MACHINE_LOCAL_PATH = "<MACHINE_LOCAL_PATH>"
WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?:^|[\s\"'`=(:,;\[])(?:[A-Za-z]:[\\/]|\\\\)", re.MULTILINE
)
POSIX_ABSOLUTE_PATH = re.compile(
    r"(?:^|[\s\"'`=(:,;\[])/(?!/)", re.MULTILINE
)
WINDOWS_PATH_TOKEN = re.compile(
    r"(?P<prefix>^|[\s\"'`=(:,;\[])(?P<path>(?:[A-Za-z]:[\\/]|\\\\)"
    r"[^\s\"'`<>|,;)\]]*)",
    re.MULTILINE,
)
POSIX_PATH_TOKEN = re.compile(
    r"(?P<prefix>^|[\s\"'`=(:,;\[])(?P<path>/(?!/)"
    r"[^\s\"'`<>|,;)\]]*)",
    re.MULTILINE,
)
REPORT_TEXT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".log", ".md", ".txt"})
TASK_SUMMARY_FIELDS = (
    "task_id",
    "arm",
    "seed",
    "status",
    "precision",
    "epochs",
    "last_epoch",
    "best_val_mse",
    "best_val_epoch",
    "best_curved_parent_mse",
    "best_curved_parent_epoch",
    "final_val_mse",
    "final_curved_parent_mse",
    "initial_perplexity",
    "final_perplexity",
    "initial_coverage",
    "final_coverage",
    "final_unique_bins",
    "lr_reduction_count",
    "final_lr",
    "grad_clip_effective_epochs",
    "max_preclip_grad_norm",
    "train_batches",
    "val_batches",
    "skipped_train_batches",
    "nonfinite_events",
    "all_epochs_finite",
    "experiment_signature",
    "source_commit",
)
EPOCH_FIELDS = (
    "task_id",
    "arm",
    "seed",
    "epoch",
    "train_loss",
    "val_loss",
    "curved_parent_mse",
    "planar_parent_mse",
    "edge_parent_mse",
    "entropy_perplexity",
    "coverage",
    "unique_bins",
    "lr",
    "lr_after_scheduler",
    "preclip_grad_norm",
    "grad_clip_was_effective",
    "train_batches",
    "finite_train_batches",
    "skipped_train_batches",
    "val_batches",
    "finite_val_batches",
    "finite_val_samples",
    *NONFINITE_FIELDS,
    "training_state_finite",
    "finite_state_audit_status",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = redact_machine_local_paths(payload)
    assert_no_machine_local_paths(payload, path.name)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(payload, indent=2, sort_keys=sort_keys, ensure_ascii=True) + "\n"
        )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(normalized)


def read_log_text(path: Path) -> str:
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # The completed Windows P0-B console logs use the system GBK code
        # page. GB18030 is its strict superset and is available cross-platform.
        return data.decode("gb18030")


def _is_machine_local_absolute_path(value: str) -> bool:
    return bool(
        WINDOWS_ABSOLUTE_PATH.search(value)
        or value.startswith("\\\\")
        or POSIX_ABSOLUTE_PATH.search(value)
    )


def _path_marker(value: str) -> str:
    normalized = value.replace("\\", "/").rstrip("/")
    basename = normalized.rsplit("/", 1)[-1]
    return f"{MACHINE_LOCAL_PATH}/{basename}" if basename else MACHINE_LOCAL_PATH


def redact_machine_local_text(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{_path_marker(match.group('path'))}"

    value = WINDOWS_PATH_TOKEN.sub(replace, value)
    return POSIX_PATH_TOKEN.sub(replace, value)


def redact_machine_local_paths(value: Any) -> Any:
    """Remove absolute machine paths from a JSON-compatible value.

    The basename is retained when available so checkpoint and artifact roles
    remain readable, but no drive, user directory, or local run root survives.
    Source identity is recorded separately by ``copy_lightweight``.
    """
    if isinstance(value, Mapping):
        return {str(key): redact_machine_local_paths(child) for key, child in value.items()}
    if isinstance(value, list):
        return [redact_machine_local_paths(child) for child in value]
    if isinstance(value, tuple):
        return [redact_machine_local_paths(child) for child in value]
    if not isinstance(value, str) or not _is_machine_local_absolute_path(value):
        return value
    return redact_machine_local_text(value)


def assert_no_machine_local_paths(value: Any, location: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_no_machine_local_paths(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_machine_local_paths(child, f"{location}[{index}]")
    elif isinstance(value, str) and _is_machine_local_absolute_path(value):
        raise RuntimeError(f"Git-safe archive contains absolute path at {location}")


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parent_bucket_mse(row: Mapping[str, Any], bucket: str) -> float | None:
    value = (
        (row.get("val_parent_cluster_reconstruction_mse") or {})
        .get(bucket, {})
        .get("mse")
    )
    return as_float(value)


def validated_state(run_root: Path, *, capacity_ab: bool = False) -> dict[str, Any]:
    if capacity_ab:
        try:
            from tools.run_capacity_ab_60k import validation_summary as capacity_validation
            from tools.run_capacity_ab_60k import load_and_refresh as load_capacity_state
        except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
            from run_capacity_ab_60k import validation_summary as capacity_validation
            from run_capacity_ab_60k import load_and_refresh as load_capacity_state

        _path, refreshed = load_capacity_state(run_root)
        validation = capacity_validation(refreshed)
        reasons = list(validation.get("reasons") or [])
        if refreshed.get("status") != "COMPLETED":
            reasons.append(f"state status is {refreshed.get('status')!r}, expected COMPLETED")
        if validation.get("formal_result_eligible") is not True:
            reasons.append("formal_result_eligible is not true")
        if validation.get("inventory_consistent") is not True:
            reasons.append("inventory_consistent is not true")
        if not validation.get("valid") or reasons:
            raise RuntimeError(
                "formal capacity A/B state is not valid: " + "; ".join(reasons)
            )
        refreshed["formal_result_eligible"] = validation["formal_result_eligible"]
        refreshed["inventory_consistent"] = validation["inventory_consistent"]
        return refreshed

    state = read_json(Path(run_root) / "p0b_state.json")
    refreshed = refresh_state(state)
    validation = validation_summary(refreshed)
    reasons = list(validation.get("reasons") or [])
    if refreshed.get("status") != "COMPLETED":
        reasons.append(f"state status is {refreshed.get('status')!r}, expected COMPLETED")
    if refreshed.get("formal_result_eligible") is not True:
        reasons.append("formal_result_eligible is not true")
    if refreshed.get("inventory_consistent") is not True:
        reasons.append("inventory_consistent is not true")
    if not validation.get("valid") or reasons:
        raise RuntimeError("formal P0-B state is not valid: " + "; ".join(reasons))
    return refreshed


def task_key(task: Mapping[str, Any]) -> tuple[str, int]:
    return str(task.get("arm")), int(task.get("seed"))


def task_inventory(task: Mapping[str, Any]) -> Mapping[str, Any]:
    return (task.get("validation") or {}).get("inventory") or {}


def load_and_validate_history(
    task: Mapping[str, Any], expected_epochs: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(Path(str(task["history"])))
    rows = list(payload.get("history") or [])
    observed = [int(row.get("epoch", -1)) for row in rows]
    if observed != list(range(expected_epochs)):
        raise RuntimeError(
            f"{task.get('task_id')} history epochs are not contiguous 0..{expected_epochs - 1}"
        )
    for row in rows:
        epoch = row.get("epoch")
        for field in NONFINITE_FIELDS:
            if int(row.get(field) or 0) != 0:
                raise RuntimeError(f"{task.get('task_id')} epoch {epoch} has {field}")
        if int(row.get("skipped_train_batches") or 0) != 0:
            raise RuntimeError(f"{task.get('task_id')} epoch {epoch} skipped train batches")
        if int(row.get("finite_train_batches") or -1) != int(row.get("train_batches") or 0):
            raise RuntimeError(f"{task.get('task_id')} epoch {epoch} has incomplete train batches")
        if int(row.get("finite_val_batches") or -1) != int(row.get("val_batches") or 0):
            raise RuntimeError(f"{task.get('task_id')} epoch {epoch} has incomplete val batches")
        if row.get("training_state_finite") is not True:
            raise RuntimeError(f"{task.get('task_id')} epoch {epoch} state is not finite")
        if (row.get("finite_state_audit") or {}).get("status") != "finite":
            raise RuntimeError(f"{task.get('task_id')} epoch {epoch} finite audit failed")
    return payload, rows


def epoch_summary(task: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    usage = row.get("val_code_usage") or {}
    has_codebook = str(task.get("arm")) != "continuous_bypass_64d"
    return {
        "task_id": task.get("task_id"),
        "arm": task.get("arm"),
        "seed": task.get("seed"),
        "epoch": row.get("epoch"),
        "train_loss": row.get("train_loss"),
        "val_loss": row.get("val_loss"),
        "curved_parent_mse": parent_bucket_mse(row, "surface_curved_proxy"),
        "planar_parent_mse": parent_bucket_mse(row, "surface_planar_like"),
        "edge_parent_mse": parent_bucket_mse(row, "edge"),
        "entropy_perplexity": usage.get("entropy_perplexity") if has_codebook else None,
        "coverage": usage.get("coverage") if has_codebook else None,
        "unique_bins": usage.get("unique_bins") if has_codebook else None,
        "lr": row.get("lr"),
        "lr_after_scheduler": row.get("lr_after_scheduler"),
        "preclip_grad_norm": row.get("preclip_grad_norm"),
        "grad_clip_was_effective": row.get("grad_clip_was_effective"),
        "train_batches": row.get("train_batches"),
        "finite_train_batches": row.get("finite_train_batches"),
        "skipped_train_batches": row.get("skipped_train_batches"),
        "val_batches": row.get("val_batches"),
        "finite_val_batches": row.get("finite_val_batches"),
        "finite_val_samples": row.get("finite_val_samples"),
        **{field: row.get(field) for field in NONFINITE_FIELDS},
        "training_state_finite": row.get("training_state_finite"),
        "finite_state_audit_status": (row.get("finite_state_audit") or {}).get("status"),
    }


def best_epoch(rows: Sequence[Mapping[str, Any]], value_getter: Any) -> tuple[int, float]:
    candidates = [
        (int(row["epoch"]), value)
        for row in rows
        if (value := as_float(value_getter(row))) is not None
    ]
    if not candidates:
        raise RuntimeError("history has no finite metric candidates")
    return min(candidates, key=lambda item: item[1])


def summarize_task(
    task: Mapping[str, Any], history: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    best_val_epoch, best_val = best_epoch(rows, lambda row: row.get("val_loss"))
    best_curved_epoch, best_curved = best_epoch(
        rows, lambda row: parent_bucket_mse(row, "surface_curved_proxy")
    )
    has_codebook = str(task.get("arm")) != "continuous_bypass_64d"
    initial_usage = rows[0].get("val_code_usage") or {}
    final_usage = rows[-1].get("val_code_usage") or {}
    lr_reductions = sum(
        (as_float(row.get("lr_after_scheduler")) or 0.0)
        < (as_float(row.get("lr")) or 0.0)
        for row in rows
    )
    nonfinite = sum(
        int(row.get(field) or 0) for row in rows for field in NONFINITE_FIELDS
    )
    signature_config = (history.get("config") or {}).get("signature_configuration") or {}
    return {
        "task_id": task.get("task_id"),
        "arm": task.get("arm"),
        "seed": task.get("seed"),
        "status": task.get("status"),
        "precision": (task.get("signature_payload") or {}).get("precision"),
        "epochs": len(rows),
        "last_epoch": rows[-1].get("epoch"),
        "best_val_mse": best_val,
        "best_val_epoch": best_val_epoch,
        "best_curved_parent_mse": best_curved,
        "best_curved_parent_epoch": best_curved_epoch,
        "final_val_mse": rows[-1].get("val_loss"),
        "final_curved_parent_mse": parent_bucket_mse(rows[-1], "surface_curved_proxy"),
        "initial_perplexity": initial_usage.get("entropy_perplexity") if has_codebook else None,
        "final_perplexity": final_usage.get("entropy_perplexity") if has_codebook else None,
        "initial_coverage": initial_usage.get("coverage") if has_codebook else None,
        "final_coverage": final_usage.get("coverage") if has_codebook else None,
        "final_unique_bins": final_usage.get("unique_bins") if has_codebook else None,
        "lr_reduction_count": lr_reductions,
        "final_lr": rows[-1].get("lr_after_scheduler"),
        "grad_clip_effective_epochs": sum(bool(row.get("grad_clip_was_effective")) for row in rows),
        "max_preclip_grad_norm": max(
            as_float(row.get("preclip_grad_norm")) or 0.0 for row in rows
        ),
        "train_batches": sum(int(row.get("train_batches") or 0) for row in rows),
        "val_batches": sum(int(row.get("val_batches") or 0) for row in rows),
        "skipped_train_batches": sum(
            int(row.get("skipped_train_batches") or 0) for row in rows
        ),
        "nonfinite_events": nonfinite,
        "all_epochs_finite": nonfinite == 0,
        "experiment_signature": task.get("signature"),
        "source_commit": (signature_config.get("git") or {}).get("commit"),
    }


def aggregate(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def arm_aggregates(
    rows: Sequence[Mapping[str, Any]], arms: Sequence[str] | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    observed_arms = tuple(arms or ARMS)
    for arm in observed_arms:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        result[arm] = {
            "best_val_mse": aggregate([float(row["best_val_mse"]) for row in arm_rows]),
            "best_curved_parent_mse": aggregate(
                [float(row["best_curved_parent_mse"]) for row in arm_rows]
            ),
        }
        perplexities = [
            float(row["final_perplexity"])
            for row in arm_rows
            if row.get("final_perplexity") is not None
        ]
        coverages = [
            float(row["final_coverage"])
            for row in arm_rows
            if row.get("final_coverage") is not None
        ]
        result[arm]["final_perplexity"] = aggregate(perplexities) if perplexities else None
        result[arm]["final_coverage"] = aggregate(coverages) if coverages else None
    if set(ARMS).issubset(result):
        vq = result["vq_4096_64d_random"]
        bypass = result["continuous_bypass_64d"]
        result["vq_vs_bypass"] = {
            "best_val_mse_ratio": vq["best_val_mse"]["mean"]
            / bypass["best_val_mse"]["mean"],
            "best_val_mse_absolute_gap": vq["best_val_mse"]["mean"]
            - bypass["best_val_mse"]["mean"],
            "best_curved_parent_mse_ratio": vq["best_curved_parent_mse"]["mean"]
            / bypass["best_curved_parent_mse"]["mean"],
            "best_curved_parent_mse_absolute_gap": vq["best_curved_parent_mse"]["mean"]
            - bypass["best_curved_parent_mse"]["mean"],
        }
    elif len(observed_arms) == 2:
        left, right = observed_arms
        result["pairwise"] = {
            "left_arm": left,
            "right_arm": right,
            "best_val_mse_ratio_left_over_right": (
                result[left]["best_val_mse"]["mean"]
                / result[right]["best_val_mse"]["mean"]
            ),
            "best_curved_parent_mse_ratio_left_over_right": (
                result[left]["best_curved_parent_mse"]["mean"]
                / result[right]["best_curved_parent_mse"]["mean"]
            ),
        }
    return result


def copy_lightweight(
    source: Path,
    target: Path,
    run_root: Path,
    report_dir: Path,
    source_manifest: list[dict[str, Any]],
) -> None:
    source = source.resolve()
    try:
        relative_source = source.relative_to(run_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"archive source is outside run root: {source}") from exc
    try:
        relative_target = target.resolve().relative_to(Path(report_dir).resolve())
    except ValueError as exc:
        raise RuntimeError(f"archive target is outside report directory: {target}") from exc
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise RuntimeError(f"refusing to copy forbidden artifact: {source}")
    if source.stat().st_size > MAX_LIGHTWEIGHT_FILE_BYTES:
        raise RuntimeError(f"lightweight artifact exceeds size cap: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    transformation = "identity"
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload = redact_machine_local_paths(payload)
        assert_no_machine_local_paths(payload, relative_source.as_posix())
        # Preserve producer key order so path redaction changes only the
        # machine-local values instead of rewriting every evidence line.
        write_json(target, payload, sort_keys=False)
        transformation = "json_machine_paths_redacted"
    elif source.suffix.lower() == ".log":
        source_text = read_log_text(source)
        archived_text = redact_machine_local_text(source_text)
        assert_no_machine_local_paths(archived_text, relative_source.as_posix())
        write_text_lf(target, archived_text)
        transformation = (
            "text_utf8_lf_machine_paths_redacted"
            if archived_text != source_text
            else "text_utf8_lf"
        )
    else:
        shutil.copy2(source, target)
    target_hash = sha256_file(target)
    if transformation == "identity" and source_hash != target_hash:
        raise RuntimeError(f"copied artifact hash mismatch: {source}")
    source_manifest.append(
        {
            "source_relative_path": relative_source.as_posix(),
            "archive_relative_path": relative_target.as_posix(),
            "bytes": source.stat().st_size,
            "sha256": source_hash,
            "source_bytes": source.stat().st_size,
            "source_sha256": source_hash,
            "archive_bytes": target.stat().st_size,
            "archive_sha256": target_hash,
            "transformation": transformation,
        }
    )


def checkpoint_manifest(state: Mapping[str, Any], run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in sorted(state.get("tasks") or [], key=task_key):
        for role in CHECKPOINT_FIELDS:
            path = Path(str(task[role])).resolve()
            try:
                relative = path.relative_to(run_root.resolve())
            except ValueError as exc:
                raise RuntimeError(f"checkpoint is outside run root: {path}") from exc
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "task_id": task.get("task_id"),
                    "arm": task.get("arm"),
                    "seed": task.get("seed"),
                    "role": role,
                    "relative_path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "checkpoint_bytes_archived": False,
                }
            )
    return rows


def log_summary(path: Path, task: Mapping[str, Any], attempt: int, stream: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    return {
        "task_id": task.get("task_id"),
        "attempt": attempt,
        "stream": stream,
        "bytes": path.stat().st_size,
        "line_count": len(lines),
        "nonempty_line_count": len(nonempty),
        "sha256": sha256_file(path),
        "last_nonempty_line": nonempty[-1] if nonempty else None,
    }


def artifact_manifest(report_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(report_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        data = path.read_bytes()
        if path.suffix.lower() in REPORT_TEXT_SUFFIXES and b"\r" in data:
            raise RuntimeError(f"report text artifact is not canonical LF: {path}")
        artifacts.append(
            {
                "path": path.relative_to(report_dir).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return artifacts


def render_readme(summary: Mapping[str, Any]) -> str:
    tasks = summary["tasks"]
    aggregate_rows = summary["aggregates"]
    table = "\n".join(
        "| {arm} | {seed} | {best_val_mse:.8g} | {best_curved_parent_mse:.8g} | "
        "{ppl} | {coverage} | {nonfinite_events} |".format(
            arm=row["arm"],
            seed=row["seed"],
            best_val_mse=row["best_val_mse"],
            best_curved_parent_mse=row["best_curved_parent_mse"],
            ppl=(
                f"{row['final_perplexity']:.2f}"
                if row.get("final_perplexity") is not None
                else "n/a"
            ),
            coverage=(
                f"{100.0 * row['final_coverage']:.2f}%"
                if row.get("final_coverage") is not None
                else "n/a"
            ),
            nonfinite_events=row["nonfinite_events"],
        )
        for row in tasks
    )
    capacity_ab = summary.get("archive_kind") == "capacity_ab"
    title = (
        "Capacity A/B formal 60k VQ-8192/RVQ stability result"
        if capacity_ab
        else "P0-B formal 60k VQ/bypass stability result"
    )
    if capacity_ab:
        interpretation = (
            "- This archive is reconstruction and codebook-health evidence only. "
            "The fixed 100-CAD unchanged-chain measurement decides the winner and "
            "must include RVQ's downstream sequence cost."
        )
    else:
        gap = aggregate_rows["vq_vs_bypass"]
        interpretation = f"""- Learned VQ final codebook coverage is
  `{100.0 * aggregate_rows['vq_4096_64d_random']['final_coverage']['mean']:.2f}%`
  on average and final perplexity is
  `{aggregate_rows['vq_4096_64d_random']['final_perplexity']['mean']:.2f}`.
- Mean learned-VQ curved parent MSE is
  `{gap['best_curved_parent_mse_ratio']:.3f}x` the continuous-bypass mean.
  This metric is representation evidence, not an assembly-validity result.
- The next decision requires the fixed 100-CAD, same-cohort assembly comparison
  for learned VQ seed 3 and bypass seed 3."""
    return f"""# {title}

This is a lightweight, Git-safe archive of the completed formal training run.
The four tasks used 60,000 train patches, 12,000 validation patches, bf16,
batch size 128, and 100 epochs. The launcher validator reports the run as
formal-result eligible with identical train/validation inventories across all
four tasks.

## Result

| Arm | Seed | Best val MSE | Best curved parent MSE | Final perplexity | Final coverage | Non-finite events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

- All four tasks completed epochs 0 through 99 with zero skipped batches and
  zero non-finite loss, gradient, state-audit, validation-batch, or validation-
  sample events.
{interpretation}

## Evidence

- `training_summary.json` and `training_summary.csv`: four-task metrics,
  inventory binding, finite-state totals, and cross-seed aggregates.
- `epoch_metrics.csv`: compact metrics for all 400 epochs.
- `tasks/`: metric-complete history, task manifest, train report, and sweep
  JSON files with machine-local absolute paths replaced by a stable marker.
- `logs/`: complete stdout/stderr normalized to portable UTF-8/LF text plus
  `log_summary.json`; original and archived hashes are both retained.
- `tensorboard/`: all `{summary['tensorboard_event_count']}` small TensorBoard
  event files. A task may have more than one file after an automatic resume;
  every segment is preserved and hash-bound.
- `checkpoint_manifest.json`: size and SHA-256 for all 12 local checkpoints.
  It does not contain checkpoint bytes.
- `source_archive_manifest.json`: separate source/archive size and SHA-256
  bindings plus the named path-redaction or identity transformation.
- `artifact_manifest.json`: size and SHA-256 for every archived file.

No checkpoint, pickle, NumPy array, raw protocol data, CAD, or STEP file is
present in this directory.
"""


def validate_report(
    report_dir: Path,
    expected_histories: int = 4,
    expected_tensorboard_events: int | None = None,
) -> dict[str, Any]:
    files = [path for path in report_dir.rglob("*") if path.is_file()]
    forbidden = [
        path.relative_to(report_dir).as_posix()
        for path in files
        if path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    histories = list((report_dir / "tasks").rglob("*_history.json"))
    events = list((report_dir / "tensorboard").rglob("events.out.tfevents.*"))
    if forbidden:
        raise RuntimeError(f"forbidden artifacts entered report: {forbidden}")
    if len(histories) != expected_histories:
        raise RuntimeError(f"expected {expected_histories} histories, found {len(histories)}")
    expected_events = (
        expected_histories
        if expected_tensorboard_events is None
        else expected_tensorboard_events
    )
    if len(events) != expected_events:
        raise RuntimeError(
            f"expected {expected_events} TensorBoard events, found {len(events)}"
        )
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in REPORT_TEXT_SUFFIXES:
            continue
        relative = path.relative_to(report_dir).as_posix()
        data = path.read_bytes()
        if b"\r" in data:
            raise RuntimeError(f"report text artifact is not canonical LF: {relative}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"report text artifact is not UTF-8: {relative}") from exc
        if suffix == ".json":
            payload = json.loads(text)
            assert_no_machine_local_paths(payload, relative)
        else:
            assert_no_machine_local_paths(text, relative)
    return {
        "valid": True,
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "histories": len(histories),
        "tensorboard_events": len(events),
        "forbidden_artifacts": forbidden,
        "machine_absolute_paths": False,
    }


def snapshot_from_state(
    run_root: Path,
    report_dir: Path,
    state: Mapping[str, Any],
    *,
    refresh_existing: bool = False,
    capacity_ab: bool = False,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [path for path in report_dir.rglob("*") if path.is_file()]
    if existing_files and not refresh_existing:
        raise RuntimeError(f"report directory must be empty: {report_dir}")
    if refresh_existing:
        unexpected = [
            path for path in existing_files if path.suffix.lower() in FORBIDDEN_SUFFIXES
        ]
        if unexpected:
            raise RuntimeError(f"refusing to refresh report with forbidden files: {unexpected}")
        for path in existing_files:
            path.unlink()
        for directory in sorted(
            (path for path in report_dir.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            os.rmdir(directory)
    tasks = list(state.get("tasks") or [])
    configuration = state.get("configuration") or {}
    configured_arms = tuple(configuration.get("arms") or ())
    configured_seeds = tuple((state.get("configuration") or {}).get("seeds") or ())
    expected_tasks = (
        {(str(arm), int(seed)) for arm in configured_arms for seed in configured_seeds}
        if capacity_ab
        else EXPECTED_TASKS
    )
    observed_tasks = {task_key(task) for task in tasks}
    if observed_tasks != expected_tasks or len(tasks) != len(expected_tasks):
        raise RuntimeError(
            f"formal task matrix mismatch: expected {sorted(expected_tasks)}, "
            f"observed {sorted(observed_tasks)}"
        )
    expected_epochs = int(configuration.get("epochs") or 0)
    if expected_epochs <= 0:
        raise RuntimeError("formal state has no positive epoch target")
    inventories = [task_inventory(task) for task in tasks]
    if not inventories[0] or any(item != inventories[0] for item in inventories[1:]):
        raise RuntimeError("formal tasks do not share one exact inventory")

    source_manifest: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    tensorboard_event_counts: dict[str, int] = {}
    for task in sorted(tasks, key=task_key):
        history, history_rows = load_and_validate_history(task, expected_epochs)
        task_rows.append(summarize_task(task, history, history_rows))
        epoch_rows.extend(epoch_summary(task, row) for row in history_rows)
        target_task = report_dir / "tasks" / str(task["arm"]) / f"seed{task['seed']}"
        sources = {
            "history": Path(str(task["history"])),
            "task_manifest": Path(str(task["manifest"])),
            "train_report": Path(str(task["task_root"])) / "train_report.json",
            "sweep": Path(str(task["sweep"])),
        }
        for source in sources.values():
            copy_lightweight(
                source, target_task / source.name, run_root, report_dir, source_manifest
            )
        for attempt in task.get("attempts") or []:
            number = int(attempt["attempt"])
            for stream in ("stdout", "stderr"):
                source = Path(str(attempt[stream]))
                target = (
                    report_dir
                    / "logs"
                    / str(task["arm"])
                    / f"seed{task['seed']}"
                    / f"attempt_{number:03d}.{stream}.log"
                )
                copy_lightweight(source, target, run_root, report_dir, source_manifest)
                logs.append(log_summary(target, task, number, stream))
        tensorboard_root = Path(str(task["task_root"])) / "tensorboard"
        events = sorted(tensorboard_root.rglob("events.out.tfevents.*"))
        if not events:
            raise RuntimeError(f"{task.get('task_id')} has no TensorBoard events")
        tensorboard_event_counts[str(task["task_id"])] = len(events)
        for source in events:
            target = (
                report_dir
                / "tensorboard"
                / str(task["arm"])
                / f"seed{task['seed']}"
                / source.name
            )
            copy_lightweight(source, target, run_root, report_dir, source_manifest)

    generated_at = now()
    aggregates = arm_aggregates(task_rows, configured_arms if capacity_ab else ARMS)
    summary = {
        "schema": "capacity-ab-formal-results-v1" if capacity_ab else "p0b-formal-results-v1",
        "archive_kind": "capacity_ab" if capacity_ab else "p0b",
        "generated_at": generated_at,
        "source_run": run_root.name,
        "status": state.get("status"),
        "formal_result_eligible": state.get("formal_result_eligible"),
        "inventory_consistent": state.get("inventory_consistent"),
        "configuration_signature": state.get("configuration_signature"),
        "configuration": {
            key: configuration.get(key)
            for key in (
                "arms",
                "seeds",
                "epochs",
                "train_cap",
                "val_cap",
                "batch_size",
                "learning_rate",
                "precision",
                "smoke",
            )
        },
        "protocol": state.get("protocol") if capacity_ab else None,
        "inventory": inventories[0],
        "tasks": task_rows,
        "aggregates": aggregates,
        "tensorboard_event_count": sum(tensorboard_event_counts.values()),
        "tensorboard_event_counts_by_task": dict(
            sorted(tensorboard_event_counts.items())
        ),
        "gate": {
            "all_four_tasks_completed": all(row["status"] == "COMPLETED" for row in task_rows),
            "all_400_epochs_finite": all(row["all_epochs_finite"] for row in task_rows)
            and sum(int(row["epochs"]) for row in task_rows) == 400,
            "nonfinite_events": sum(int(row["nonfinite_events"]) for row in task_rows),
            "skipped_train_batches": sum(
                int(row["skipped_train_batches"]) for row in task_rows
            ),
            "fixed_100cad_assembly_required": True,
            "boundary_consistency_allowed": False,
            "sequence_or_ar_allowed": False,
        },
    }
    write_json(report_dir / "training_summary.json", summary)
    write_csv(report_dir / "training_summary.csv", task_rows, TASK_SUMMARY_FIELDS)
    write_csv(report_dir / "epoch_metrics.csv", epoch_rows, EPOCH_FIELDS)
    write_json(report_dir / "logs" / "log_summary.json", {"logs": logs})
    write_json(
        report_dir / "formal_run_state.json",
        {
            "schema": state.get("schema"),
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            "status": state.get("status"),
            "mode": state.get("mode"),
            "active_task": state.get("active_task"),
            "formal_result_eligible": state.get("formal_result_eligible"),
            "inventory_consistent": state.get("inventory_consistent"),
            "configuration_signature": state.get("configuration_signature"),
            "task_validations": [
                {
                    "task_id": task.get("task_id"),
                    "signature": task.get("signature"),
                    "attempts": [
                        {
                            key: attempt.get(key)
                            for key in (
                                "attempt",
                                "auto_resume",
                                "started_at",
                                "finished_at",
                                "returncode",
                            )
                        }
                        for attempt in task.get("attempts") or []
                    ],
                    "validation": task.get("validation"),
                }
                for task in sorted(tasks, key=task_key)
            ],
        },
    )
    checkpoints = checkpoint_manifest(state, run_root)
    write_json(
        report_dir / "checkpoint_manifest.json",
        {
            "generated_at": generated_at,
            "policy": "Checkpoint bytes remain local and are not archived in Git.",
            "total_checkpoint_bytes": sum(row["bytes"] for row in checkpoints),
            "checkpoints": checkpoints,
        },
    )
    write_json(
        report_dir / "source_archive_manifest.json",
        {"generated_at": generated_at, "artifacts": source_manifest},
    )
    with (report_dir / "README.md").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(render_readme(summary))
    validation = validate_report(
        report_dir,
        expected_histories=len(tasks),
        expected_tensorboard_events=sum(tensorboard_event_counts.values()),
    )
    write_json(report_dir / "archive_validation.json", validation)
    write_json(
        report_dir / "artifact_manifest.json",
        {"generated_at": generated_at, "artifacts": artifact_manifest(report_dir)},
    )
    return {**summary, "archive_validation": validation}


def snapshot(
    run_root: Path, report_dir: Path, *, refresh_existing: bool = False,
    capacity_ab: bool = False,
) -> dict[str, Any]:
    return snapshot_from_state(
        run_root,
        report_dir,
        validated_state(run_root, capacity_ab=capacity_ab),
        refresh_existing=refresh_existing,
        capacity_ab=capacity_ab,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Replace only files inside an existing Git-safe report directory.",
    )
    parser.add_argument(
        "--capacity-ab",
        action="store_true",
        help="Archive a completed run_capacity_ab_60k.py state instead of P0-B.",
    )
    args = parser.parse_args(argv)
    result = snapshot(
        args.run_root, args.report_dir, refresh_existing=args.refresh_existing,
        capacity_ab=args.capacity_ab,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "formal_result_eligible": result["formal_result_eligible"],
                "all_400_epochs_finite": result["gate"]["all_400_epochs_finite"],
                "report_dir": str(args.report_dir.resolve()),
                "archive_validation": result["archive_validation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
