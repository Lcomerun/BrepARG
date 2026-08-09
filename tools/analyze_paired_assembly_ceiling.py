"""Separate assembly-floor failures from representation-induced validity loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def paired_ceiling(rows: Sequence[Mapping[str, Any]], *, reference_arm: str) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row.get("arm")), {})[str(row.get("cad_id"))] = row
    if reference_arm not in by_arm:
        raise ValueError(f"reference arm missing: {reference_arm}")
    reference = by_arm[reference_arm]
    result: dict[str, Any] = {
        "reference_arm": reference_arm,
        "reference_attempts": len(reference),
        "reference_strict_valid": sum(row.get("strict_brep_valid") is True for row in reference.values()),
        "reference_both_valid": sum(
            row.get("strict_brep_valid") is True and row.get("native_brep_valid") is True
            for row in reference.values()
        ),
        "arms": {},
    }
    for arm, arm_rows in sorted(by_arm.items()):
        if arm == reference_arm:
            continue
        common = sorted(set(reference) & set(arm_rows))
        ref_valid = [cad for cad in common if reference[cad].get("strict_brep_valid") is True]
        ref_invalid = [cad for cad in common if reference[cad].get("strict_brep_valid") is not True]
        retained = sum(arm_rows[cad].get("strict_brep_valid") is True for cad in ref_valid)
        recovered = sum(arm_rows[cad].get("strict_brep_valid") is True for cad in ref_invalid)
        both_ref_valid = [
            cad for cad in common
            if reference[cad].get("strict_brep_valid") is True
            and reference[cad].get("native_brep_valid") is True
        ]
        both_retained = sum(
            arm_rows[cad].get("strict_brep_valid") is True
            and arm_rows[cad].get("native_brep_valid") is True
            for cad in both_ref_valid
        )
        result["arms"][arm] = {
            "paired_attempts": len(common),
            "reference_strict_valid_attempts": len(ref_valid),
            "strict_valid_retained": retained,
            "strict_retention_rate": retained / len(ref_valid) if ref_valid else None,
            "reference_strict_invalid_attempts": len(ref_invalid),
            "strict_invalid_recovered": recovered,
            "strict_recovery_rate": recovered / len(ref_invalid) if ref_invalid else None,
            "reference_both_valid_attempts": len(both_ref_valid),
            "both_valid_retained": both_retained,
            "both_valid_retention_rate": both_retained / len(both_ref_valid) if both_ref_valid else None,
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--reference-arm", default="original")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.audit_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = paired_ceiling(rows, reference_arm=args.reference_arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
