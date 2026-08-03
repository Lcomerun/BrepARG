from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCE = (
    REPO_ROOT
    / "local_runs"
    / "ar_training"
    / "train_outputs"
    / "newscheme_full_v13_ar_lr5e6"
    / "sequences_fsq_rcm.pkl"
)
DEFAULT_VQVAE = (
    REPO_ROOT
    / "ABC"
    / "processed"
    / "train_outputs"
    / "newscheme_full_vqvae_epoch100"
    / "fsq_vqvae_best.pt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "local_runs" / "reconstruction_eval"
EVALUATOR = REPO_ROOT / "tools" / "evaluate_reconstruction_v13.py"
RENDERER = REPO_ROOT / "papers" / "aaai_v13" / "render_step_directory.py"
DEFAULT_ORDERS = ("shortest", "random", "longest", "most_faces")
VALID_ORDERS = DEFAULT_ORDERS + ("most_curved",)
DEFAULT_BASELINE_BREP_VALID = {"shortest": 8, "random": 6, "longest": 3, "most_faces": 5}
PROMOTION_REQUIRED_ORDERS = ("shortest", "random", "longest", "most_faces")
COMPLEX_PROMOTION_ORDERS = ("longest", "most_faces")
EASY_PROMOTION_ORDERS = ("shortest", "random")


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def parse_orders(value: str) -> list[str]:
    orders = [item.strip() for item in value.split(",") if item.strip()]
    valid = set(VALID_ORDERS)
    unknown = [item for item in orders if item not in valid]
    if unknown:
        raise ValueError(f"Unknown VQ-VAE benchmark order(s): {', '.join(unknown)}")
    return orders


def run_name_for(prefix: str, order: str, max_samples: int, seed: int) -> str:
    labels = {"most_faces": "mostfaces", "most_curved": "mostcurved"}
    label = labels.get(order, order)
    if order == "random":
        return f"{prefix}_{label}{int(max_samples)}_seed{int(seed)}"
    return f"{prefix}_{label}{int(max_samples)}"


def build_benchmark_plan(
    *,
    python: str,
    sequence: Path,
    vqvae_checkpoint: Path,
    output_root: Path,
    run_prefix: str,
    max_samples: int,
    max_seq_len: int,
    device: str,
    seed: int,
    orders: Iterable[str],
    render: bool,
    cols: int,
) -> list[dict]:
    plan = []
    for order in orders:
        run_name = run_name_for(run_prefix, order, max_samples, seed)
        run_dir = Path(output_root) / run_name
        evaluate_command = [
            str(python),
            str(EVALUATOR),
            "--sequence",
            str(sequence),
            "--vqvae-checkpoint",
            str(vqvae_checkpoint),
            "--output-root",
            str(output_root),
            "--source",
            "validation",
            "--max-samples",
            str(int(max_samples)),
            "--order",
            order,
            "--max-seq-len",
            str(int(max_seq_len)),
            "--device",
            str(device),
            "--write-step",
            "--validate-step",
            "--run-name",
            run_name,
        ]
        if order == "random":
            evaluate_command.extend(["--seed", str(int(seed))])

        render_command = None
        if render:
            render_command = [
                str(python),
                str(RENDERER),
                str(run_dir),
                "--cols",
                str(int(cols)),
            ]

        plan.append(
            {
                "order": order,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "evaluate_command": evaluate_command,
                "render_command": render_command,
            }
        )
    return plan


def write_plan(output_root: Path, run_prefix: str, plan: list[dict]) -> Path:
    path = Path(output_root) / f"{run_prefix}_benchmark_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "plan": plan}, indent=2), encoding="utf-8")
    return path


def safe_rate(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if denominator in (None, 0):
        return None
    return float(numerator or 0) / float(denominator)


def load_report_summary(run_dir: Path) -> tuple[dict | None, Path]:
    report_path = Path(run_dir) / "reconstruction_report.json"
    if not report_path.exists():
        return None, report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get("summary", {}), report_path


def summarize_benchmark_plan(
    plan: list[dict],
    *,
    baseline_brep_valid: dict[str, int] | None = None,
    require_renders: bool = True,
) -> dict:
    baseline = dict(DEFAULT_BASELINE_BREP_VALID if baseline_brep_valid is None else baseline_brep_valid)
    slices = {}

    for item in plan:
        order = item["order"]
        run_dir = Path(item["run_dir"])
        report_summary, report_path = load_report_summary(run_dir)
        contact_sheet = run_dir / "renders" / "contact_sheet.png"
        report_found = report_summary is not None
        row = {
            "order": order,
            "run_name": item["run_name"],
            "run_dir": str(run_dir),
            "report_path": str(report_path),
            "report_found": report_found,
            "contact_sheet": str(contact_sheet),
            "contact_sheet_found": contact_sheet.exists(),
        }

        if report_summary is None:
            row.update(
                {
                    "attempted": 0,
                    "grammar_valid": 0,
                    "reconstruct_success": 0,
                    "step_saved": 0,
                    "stl_saved": 0,
                    "brep_valid": 0,
                    "errors": 0,
                    "strict_valid_rate": None,
                    "step_saved_rate": None,
                }
            )
        else:
            attempted = int(report_summary.get("attempted", 0) or 0)
            step_saved = int(report_summary.get("step_saved", 0) or 0)
            brep_valid = int(report_summary.get("brep_valid", 0) or 0)
            row.update(
                {
                    "attempted": attempted,
                    "grammar_valid": int(report_summary.get("grammar_valid", 0) or 0),
                    "reconstruct_success": int(report_summary.get("reconstruct_success", 0) or 0),
                    "step_saved": step_saved,
                    "stl_saved": int(report_summary.get("stl_saved", 0) or 0),
                    "brep_valid": brep_valid,
                    "errors": int(report_summary.get("errors", 0) or 0),
                    "strict_valid_rate": safe_rate(brep_valid, attempted),
                    "step_saved_rate": safe_rate(step_saved, attempted),
                }
            )

        baseline_count = baseline.get(order)
        row["baseline_brep_valid"] = baseline_count
        row["delta_brep_valid"] = None if baseline_count is None else row["brep_valid"] - int(baseline_count)
        slices[order] = row

    requirements = {
        "reports_complete": all(slices.get(order, {}).get("report_found", False) for order in PROMOTION_REQUIRED_ORDERS),
        "renders_complete": (not require_renders)
        or all(slices.get(order, {}).get("contact_sheet_found", False) for order in PROMOTION_REQUIRED_ORDERS),
        "longest_improved": slices.get("longest", {}).get("delta_brep_valid", -999) is not None
        and slices.get("longest", {}).get("delta_brep_valid", -999) > 0,
        "most_faces_improved": slices.get("most_faces", {}).get("delta_brep_valid", -999) is not None
        and slices.get("most_faces", {}).get("delta_brep_valid", -999) > 0,
        "easy_slices_not_regressed": all(
            slices.get(order, {}).get("delta_brep_valid", -999) is not None
            and slices.get(order, {}).get("delta_brep_valid", -999) >= 0
            for order in EASY_PROMOTION_ORDERS
        ),
    }

    reasons = []
    if not requirements["reports_complete"]:
        reasons.append("reconstruction reports are incomplete")
    if not requirements["renders_complete"]:
        reasons.append("rendered contact sheets are incomplete")
    if not requirements["longest_improved"]:
        reasons.append("longest strict-valid count did not improve over baseline")
    if not requirements["most_faces_improved"]:
        reasons.append("most_faces strict-valid count did not improve over baseline")
    if not requirements["easy_slices_not_regressed"]:
        reasons.append("shortest or random slice regressed below baseline")

    promote = all(requirements.values())
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_brep_valid": baseline,
        "require_renders": require_renders,
        "slices": slices,
        "promotion_gate": {
            "promote": promote,
            "decision": "promote_for_ar_rebuild" if promote else "hold_vqvae_checkpoint",
            "requirements": requirements,
            "reasons": reasons,
        },
    }


def write_summary(output_root: Path, run_prefix: str, summary: dict) -> Path:
    path = Path(output_root) / f"{run_prefix}_benchmark_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def run_command(command: list[str], *, continue_on_failure: bool) -> int:
    print("+ " + " ".join(command), flush=True)
    proc = subprocess.run(command)
    if proc.returncode and not continue_on_failure:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four-slice VQ-VAE-only promotion benchmark, with optional most_curved diagnostics."
    )
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--vqvae-checkpoint", type=Path, default=DEFAULT_VQVAE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-prefix", default=f"vqvae_slice_benchmark_{now_tag()}")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--orders", default=",".join(DEFAULT_ORDERS))
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        orders = parse_orders(args.orders)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    plan = build_benchmark_plan(
        python=args.python,
        sequence=args.sequence,
        vqvae_checkpoint=args.vqvae_checkpoint,
        output_root=args.output_root,
        run_prefix=args.run_prefix,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        device=args.device,
        seed=args.seed,
        orders=orders,
        render=not args.no_render,
        cols=args.cols,
    )
    plan_path = write_plan(args.output_root, args.run_prefix, plan)

    if args.dry_run:
        print(json.dumps({"plan_path": str(plan_path), "plan": plan}, indent=2))
        return 0

    failures = 0
    for item in plan:
        failures += 1 if run_command(item["evaluate_command"], continue_on_failure=args.continue_on_failure) else 0
        if item["render_command"] is not None:
            failures += 1 if run_command(item["render_command"], continue_on_failure=args.continue_on_failure) else 0

    summary = summarize_benchmark_plan(plan, require_renders=not args.no_render)
    summary_path = write_summary(args.output_root, args.run_prefix, summary)
    print(f"Benchmark plan written to {plan_path}")
    print(f"Benchmark summary written to {summary_path}")
    print(f"Promotion decision: {summary['promotion_gate']['decision']}")
    for reason in summary["promotion_gate"]["reasons"]:
        print(f"- {reason}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
