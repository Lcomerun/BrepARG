"""Run original BrepARG generation in resumable, timeout-bounded batches."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def remaining_batch_size(current: int, target: int, batch_size: int) -> int:
    return max(0, min(int(batch_size), int(target) - int(current)))


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_command_with_timeout(
    command: Sequence[str],
    *,
    timeout_sec: float,
    stdout_path: Path,
    stderr_path: Path,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr_handle:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(cwd),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            **popen_kwargs,
        )
        try:
            returncode = process.wait(timeout=max(0.05, float(timeout_sec)))
            timed_out = False
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            returncode = None
            timed_out = True

    return {
        "command": [str(item) for item in command],
        "pid": process.pid,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def count_steps(output_dir: Path) -> int:
    return sum(1 for path in output_dir.glob("*.step") if path.is_file())


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--generator", type=Path, default=REPO_ROOT / "BrepARG" / "generate_brep.py")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "BrepARG" / "config.json")
    parser.add_argument("--ar-model", type=Path, required=True)
    parser.add_argument("--se-vqvae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-timeout-sec", type=float, default=180)
    parser.add_argument("--max-attempts-per-batch", type=int, default=80)
    parser.add_argument("--start-seed", type=int, default=43)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--write-timeout", type=int, default=120)
    parser.add_argument("--filename-prefix", default="breparg_same_data_resume_best_20260726")
    parser.add_argument("--state", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "batch_logs"
    state_path = args.state or output_dir / "batch_generation_state.json"

    for required in (args.python, args.generator, args.config, args.ar_model, args.se_vqvae):
        if not required.exists():
            raise FileNotFoundError(f"Missing required generation input: {required}")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "BREPARG_SERIAL_WRITE": "1",
            "BREPARG_JOINT_OPTIMIZE_DEVICE": "cpu",
        }
    )

    state: dict[str, Any] = {
        "status": "RUNNING",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output_dir),
        "target_count": int(args.target_count),
        "batch_size": int(args.batch_size),
        "batch_timeout_sec": float(args.batch_timeout_sec),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "initial_step_count": count_steps(output_dir),
        "batches": [],
    }

    for batch_index in range(max(0, int(args.max_batches))):
        before = count_steps(output_dir)
        requested = remaining_batch_size(before, args.target_count, args.batch_size)
        if requested <= 0:
            break
        seed = int(args.start_seed) + batch_index
        command = [
            str(args.python),
            str(args.generator),
            "--dataset_type",
            "abc",
            "--config",
            str(args.config),
            "--ar_model",
            str(args.ar_model),
            "--se_vqvae",
            str(args.se_vqvae),
            "--num_samples",
            str(requested),
            "--max_attempts",
            str(args.max_attempts_per_batch),
            "--mode",
            "batch",
            "--max_length",
            str(args.max_length),
            "--temperature",
            str(args.temperature),
            "--top_p",
            str(args.top_p),
            "--output_dir",
            str(output_dir),
            "--filename_prefix",
            f"{args.filename_prefix}_seed{seed:04d}",
            "--device",
            "cuda",
            "--gpu",
            str(args.gpu),
            "--seed",
            str(seed),
            "--write_timeout",
            str(args.write_timeout),
        ]
        result = run_command_with_timeout(
            command,
            timeout_sec=args.batch_timeout_sec,
            stdout_path=log_dir / f"batch_{batch_index:03d}_seed_{seed}.stdout.log",
            stderr_path=log_dir / f"batch_{batch_index:03d}_seed_{seed}.stderr.log",
            cwd=REPO_ROOT,
            env=env,
        )
        after = count_steps(output_dir)
        result.update(
            {
                "batch_index": batch_index,
                "seed": seed,
                "requested_successes": requested,
                "step_count_before": before,
                "step_count_after": after,
                "new_steps": after - before,
            }
        )
        state["batches"].append(result)
        state["current_step_count"] = after
        write_state(state_path, state)
        print(
            f"batch={batch_index} seed={seed} requested={requested} new={after - before} "
            f"total={after}/{args.target_count} timeout={int(result['timed_out'])} rc={result['returncode']}",
            flush=True,
        )

    final_count = count_steps(output_dir)
    state["current_step_count"] = final_count
    state["status"] = "COMPLETE" if final_count >= int(args.target_count) else "INCOMPLETE"
    state["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_state(state_path, state)
    print(json.dumps({"status": state["status"], "step_count": final_count, "state": str(state_path)}), flush=True)
    return 0 if state["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
