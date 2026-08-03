from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def maybe_relative(path: Path, repo_root: Path | None) -> str:
    if repo_root is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def artifact(label: str, path: Path, *, kind: str, repo_root: Path | None = None, required: bool = True) -> dict[str, Any]:
    return {
        "label": label,
        "kind": kind,
        "path": str(path),
        "relative_path": maybe_relative(path, repo_root),
        "required": bool(required),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def infer_plan_path(summary_path: Path) -> Path:
    suffix = "_benchmark_summary.json"
    text = str(summary_path)
    if text.endswith(suffix):
        return Path(text[: -len(suffix)] + "_benchmark_plan.json")
    return summary_path.with_name(summary_path.stem + "_plan.json")


def build_handoff_manifest(
    *,
    run_dir: Path,
    benchmark_summary: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    repo_root = Path(repo_root) if repo_root is not None else None
    artifacts = [
        artifact("vqvae_best_checkpoint", run_dir / "fsq_vqvae_best.pt", kind="checkpoint", repo_root=repo_root),
        artifact("vqvae_final_checkpoint", run_dir / "fsq_vqvae_final.pt", kind="checkpoint", repo_root=repo_root),
        artifact("vqvae_history", run_dir / "vqvae_history.json", kind="training_history", repo_root=repo_root),
        artifact("server_run_ledger", run_dir / "server_run_ledger.txt", kind="ledger", repo_root=repo_root),
    ]

    summary_data: dict[str, Any] = {}
    if benchmark_summary is not None:
        benchmark_summary = Path(benchmark_summary)
        summary_data = load_json(benchmark_summary)
        artifacts.append(artifact("benchmark_summary", benchmark_summary, kind="benchmark_summary", repo_root=repo_root))
        benchmark_plan = infer_plan_path(benchmark_summary)
        if benchmark_plan.exists():
            artifacts.append(
                artifact(
                    "benchmark_plan",
                    benchmark_plan,
                    kind="benchmark_plan",
                    repo_root=repo_root,
                    required=False,
                )
            )
        for order, row in sorted(summary_data.get("slices", {}).items()):
            run_path = Path(row.get("run_dir", ""))
            report_path = Path(row.get("report_path", run_path / "reconstruction_report.json"))
            contact_sheet_path = Path(row.get("contact_sheet", run_path / "renders" / "contact_sheet.png"))
            artifacts.append(artifact(f"{order}_run_dir", run_path, kind="benchmark_run_dir", repo_root=repo_root))
            artifacts.append(artifact(f"{order}_report", report_path, kind="reconstruction_report", repo_root=repo_root))
            artifacts.append(
                artifact(
                    f"{order}_manifest",
                    run_path / "reconstruction_manifest.jsonl",
                    kind="reconstruction_manifest",
                    repo_root=repo_root,
                )
            )
            artifacts.append(artifact(f"{order}_contact_sheet", contact_sheet_path, kind="contact_sheet", repo_root=repo_root))

    missing_required = [item for item in artifacts if item["required"] and not item["exists"]]
    promotion_gate = summary_data.get("promotion_gate", {})
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "repo_root": str(repo_root) if repo_root is not None else None,
        "benchmark_summary": str(benchmark_summary) if benchmark_summary is not None else None,
        "promotion_decision": promotion_gate.get("decision"),
        "promotion_reasons": promotion_gate.get("reasons", []),
        "complete": not missing_required,
        "missing_required": missing_required,
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a copy-back manifest for a rented-server VQ-VAE run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-summary", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or (args.run_dir / "copy_back_manifest.json")
    manifest = build_handoff_manifest(
        run_dir=args.run_dir,
        benchmark_summary=args.benchmark_summary,
        repo_root=args.repo_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Copy-back manifest written to {output}")
    print(f"Complete: {manifest['complete']}")
    if manifest["promotion_decision"]:
        print(f"Promotion decision: {manifest['promotion_decision']}")
    if manifest["missing_required"]:
        print("Missing required artifacts:")
        for item in manifest["missing_required"]:
            print(f"- {item['label']}: {item['path']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
