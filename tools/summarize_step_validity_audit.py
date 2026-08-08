"""Create a compact attempts-based report from a STEP validity audit JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .audit_assembly_step_validity import summarize_validity_rows
except ImportError:  # direct script execution
    from audit_assembly_step_validity import summarize_validity_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = summarize_validity_rows(rows)
    return {
        "attempts_denominator": summary["attempts"],
        "overall": summary,
        "arms": summary.get("by_arm", {}),
        "interpretation": {
            "strict_is_project_metric": True,
            "native_and_strict_must_both_be_reported": True,
            "no_step_remains_in_denominator": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(read_jsonl(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
