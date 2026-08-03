"""Read-only monitor for the rented-server VQ-VAE recovery gate.

The monitor never starts or stops training. It inspects the recovery run
directory, optional benchmark summary, and optional copy-back manifest so the
server session can decide whether to keep waiting, promote the checkpoint for a
source-path sequence rebuild, or stop for failure analysis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path("/workspace/ABC/processed/train_outputs/newscheme_full_vqvae_complex_recovery")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        return {"_json_error": str(exc)}


def file_status(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "bytes": int(path.stat().st_size) if exists and path.is_file() else 0,
    }


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return float(numerator) / denominator_float
    except (TypeError, ValueError):
        return None


def summarize_history(run_dir: Path, explicit_target_epoch: int | None = None) -> dict[str, Any]:
    path = run_dir / "vqvae_history.json"
    payload = load_json(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": payload is not None,
        "valid_json": payload is not None and "_json_error" not in payload,
    }
    if payload is None:
        summary.update(
            {
                "history_count": 0,
                "latest_epoch": None,
                "target_epoch": explicit_target_epoch,
                "best_epoch": None,
                "best_val_recon": None,
                "stop_reason": None,
            }
        )
        return summary
    if "_json_error" in payload:
        summary["json_error"] = payload["_json_error"]
        return summary

    rows = list(payload.get("history") or [])
    latest = rows[-1] if rows else {}
    config = payload.get("config") or {}
    target_epoch = explicit_target_epoch
    if target_epoch is None:
        target_epoch = safe_int(config.get("target_epoch"))
    if target_epoch is None and config.get("start_epoch") is not None and config.get("epochs_requested") is not None:
        target_epoch = safe_int(int(config["start_epoch"]) + int(config["epochs_requested"]))

    summary.update(
        {
            "history_count": len(rows),
            "latest_epoch": safe_int(latest.get("epoch")),
            "target_epoch": target_epoch,
            "best_epoch": safe_int(payload.get("best_epoch", latest.get("best_epoch"))),
            "best_val_recon": payload.get("best_val_recon", latest.get("best_val")),
            "last_train_loss": latest.get("train_loss"),
            "last_val_loss": latest.get("val_loss"),
            "stop_reason": payload.get("stop_reason") or latest.get("stop_reason") or None,
            "finite_train_fraction": safe_ratio(latest.get("finite_train_batches"), latest.get("train_batches")),
            "finite_val_fraction": safe_ratio(latest.get("finite_val_batches"), latest.get("val_batches")),
            "epochs_without_improvement": latest.get("epochs_without_improvement"),
            "consecutive_nonfinite_val_epochs": latest.get("consecutive_nonfinite_val_epochs"),
        }
    )
    return summary


def resolve_benchmark_summary(
    benchmark_summary: str | Path | None,
    copy_back_manifest: str | Path | None,
) -> Path | None:
    if benchmark_summary is not None:
        return Path(benchmark_summary)
    if copy_back_manifest is None:
        return None
    manifest = load_json(Path(copy_back_manifest))
    if not manifest or "_json_error" in manifest:
        return None
    value = manifest.get("benchmark_summary")
    return Path(value) if value else None


def summarize_benchmark(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "valid_json": False, "promotion_decision": None, "reasons": []}
    payload = load_json(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": payload is not None,
        "valid_json": payload is not None and "_json_error" not in payload,
        "promotion_decision": None,
        "reasons": [],
    }
    if payload is None:
        return summary
    if "_json_error" in payload:
        summary["json_error"] = payload["_json_error"]
        return summary
    gate = payload.get("promotion_gate") or {}
    summary.update(
        {
            "promotion_decision": gate.get("decision"),
            "promote": bool(gate.get("promote", gate.get("decision") == "promote_for_ar_rebuild")),
            "reasons": list(gate.get("reasons") or []),
            "requirements": gate.get("requirements") or {},
        }
    )
    return summary


def summarize_copy_back(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "valid_json": False, "complete": False, "missing_required": []}
    payload = load_json(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": payload is not None,
        "valid_json": payload is not None and "_json_error" not in payload,
        "complete": False,
        "missing_required": [],
    }
    if payload is None:
        return summary
    if "_json_error" in payload:
        summary["json_error"] = payload["_json_error"]
        return summary
    summary.update(
        {
            "complete": bool(payload.get("complete")),
            "missing_required": list(payload.get("missing_required") or []),
            "promotion_decision": payload.get("promotion_decision"),
        }
    )
    return summary


def status_payload(
    *,
    run_dir: Path,
    state: str,
    ready: bool,
    terminal: bool,
    exit_code: int,
    reason: str,
    history: dict[str, Any],
    checkpoints: dict[str, Any],
    benchmark: dict[str, Any],
    copy_back: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "state": state,
        "ready": bool(ready),
        "terminal": bool(terminal),
        "exit_code": int(exit_code),
        "reason": reason,
        "history": history,
        "checkpoints": checkpoints,
        "benchmark": benchmark,
        "copy_back_manifest": copy_back,
    }


def evaluate_vqvae_recovery_gate(
    run_dir: str | Path,
    *,
    benchmark_summary: str | Path | None = None,
    copy_back_manifest: str | Path | None = None,
    target_epoch: int | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    default_copy_back = run_dir / "copy_back_manifest.json"
    copy_back_path = Path(copy_back_manifest) if copy_back_manifest is not None else default_copy_back
    benchmark_path = resolve_benchmark_summary(benchmark_summary, copy_back_path if copy_back_path.exists() else None)

    history = summarize_history(run_dir, explicit_target_epoch=target_epoch)
    checkpoints = {
        "best": file_status(run_dir / "fsq_vqvae_best.pt"),
        "final": file_status(run_dir / "fsq_vqvae_final.pt"),
        "history": file_status(run_dir / "vqvae_history.json"),
        "ledger": file_status(run_dir / "server_run_ledger.txt"),
    }
    benchmark = summarize_benchmark(benchmark_path)
    copy_back = summarize_copy_back(copy_back_path if copy_back_path.exists() else None)

    if not history.get("exists"):
        return status_payload(
            run_dir=run_dir,
            state="waiting_for_history",
            ready=False,
            terminal=False,
            exit_code=1,
            reason="vqvae_history.json is not present yet",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )
    if not history.get("valid_json"):
        return status_payload(
            run_dir=run_dir,
            state="history_invalid_json",
            ready=False,
            terminal=True,
            exit_code=2,
            reason="vqvae_history.json is not valid JSON",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )
    if not checkpoints["best"]["exists"]:
        return status_payload(
            run_dir=run_dir,
            state="waiting_for_best_checkpoint",
            ready=False,
            terminal=False,
            exit_code=1,
            reason="fsq_vqvae_best.pt is not present yet",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )

    latest_epoch = history.get("latest_epoch")
    expected_epoch = history.get("target_epoch")
    stop_reason = history.get("stop_reason")
    if expected_epoch is not None and latest_epoch is not None and latest_epoch < expected_epoch and not stop_reason:
        return status_payload(
            run_dir=run_dir,
            state=f"waiting_for_epoch_{expected_epoch}",
            ready=False,
            terminal=False,
            exit_code=1,
            reason=f"latest VQ-VAE epoch is {latest_epoch}, waiting for {expected_epoch}",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )

    if not benchmark.get("exists"):
        return status_payload(
            run_dir=run_dir,
            state="waiting_for_benchmark_summary",
            ready=False,
            terminal=False,
            exit_code=1,
            reason="benchmark summary is not present yet",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )
    if not benchmark.get("valid_json"):
        return status_payload(
            run_dir=run_dir,
            state="benchmark_invalid_json",
            ready=False,
            terminal=True,
            exit_code=2,
            reason="benchmark summary is not valid JSON",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )

    decision = benchmark.get("promotion_decision")
    if decision == "hold_vqvae_checkpoint":
        reasons = benchmark.get("reasons") or []
        return status_payload(
            run_dir=run_dir,
            state="hold_vqvae_checkpoint",
            ready=False,
            terminal=True,
            exit_code=2,
            reason="; ".join(reasons) if reasons else "benchmark gate held the VQ-VAE checkpoint",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )
    if decision != "promote_for_ar_rebuild":
        return status_payload(
            run_dir=run_dir,
            state="waiting_for_benchmark_decision",
            ready=False,
            terminal=False,
            exit_code=1,
            reason="benchmark summary has no promotion decision yet",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )

    if not copy_back.get("exists"):
        return status_payload(
            run_dir=run_dir,
            state="waiting_for_copy_back_manifest",
            ready=False,
            terminal=False,
            exit_code=1,
            reason="copy_back_manifest.json is not present yet",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )
    if not copy_back.get("valid_json"):
        return status_payload(
            run_dir=run_dir,
            state="copy_back_manifest_invalid_json",
            ready=False,
            terminal=True,
            exit_code=2,
            reason="copy_back_manifest.json is not valid JSON",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )
    if not copy_back.get("complete"):
        return status_payload(
            run_dir=run_dir,
            state="copy_back_incomplete",
            ready=False,
            terminal=True,
            exit_code=2,
            reason="copy-back manifest is missing required artifacts",
            history=history,
            checkpoints=checkpoints,
            benchmark=benchmark,
            copy_back=copy_back,
        )

    return status_payload(
        run_dir=run_dir,
        state="ready_for_sequence_rebuild",
        ready=True,
        terminal=True,
        exit_code=0,
        reason="VQ-VAE benchmark promoted the checkpoint and copy-back manifest is complete",
        history=history,
        checkpoints=checkpoints,
        benchmark=benchmark,
        copy_back=copy_back,
    )


def append_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(status, ensure_ascii=True) + "\n")


def monitor(
    *,
    run_dir: Path,
    benchmark_summary: Path | None,
    copy_back_manifest: Path | None,
    target_epoch: int | None,
    interval_seconds: int,
    once: bool,
    status_log: Path | None,
) -> int:
    while True:
        status = evaluate_vqvae_recovery_gate(
            run_dir,
            benchmark_summary=benchmark_summary,
            copy_back_manifest=copy_back_manifest,
            target_epoch=target_epoch,
        )
        print(json.dumps(status, indent=2, ensure_ascii=True), flush=True)
        if status_log is not None:
            append_status(status_log, status)
        if once or status["terminal"]:
            return int(status["exit_code"])
        time.sleep(interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only monitor for the V13 VQ-VAE recovery gate.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--benchmark-summary", type=Path)
    parser.add_argument("--copy-back-manifest", type=Path)
    parser.add_argument("--target-epoch", type=int)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status-log", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return monitor(
        run_dir=args.run_dir,
        benchmark_summary=args.benchmark_summary,
        copy_back_manifest=args.copy_back_manifest,
        target_epoch=args.target_epoch,
        interval_seconds=args.interval_seconds,
        once=args.once,
        status_log=args.status_log,
    )


if __name__ == "__main__":
    raise SystemExit(main())
