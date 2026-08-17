"""Archive lightweight, path-normalized P0-A assembly-chain evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ATTEMPT_FIELDS = (
    "cad_id",
    "joint_iterations",
    "sewing_tolerance",
    "status",
    "failure_stage",
    "failure_entity_kind",
    "failure_entity_index",
    "error_type",
    "step_saved",
    "construction_native_brep_valid",
    "native_brep_valid",
    "strict_brep_valid",
    "both_valid",
    "wire_count",
    "wire_order_failures",
    "wire_self_intersections",
    "shell_count",
    "shells_with_bad_edges",
    "free_edges",
    "solid_count",
    "curve_fit_failures",
    "selected_curve_tolerances",
    "elapsed_seconds",
    "step_relative_path",
    "step_bytes",
    "step_sha256",
)

STEP_FIELDS = (
    "cad_id",
    "joint_iterations",
    "sewing_tolerance",
    "step_relative_path",
    "bytes",
    "sha256",
    "native_brep_valid",
    "strict_brep_valid",
    "both_valid",
    "step_bytes_archived",
)

BASELINE_JOINT_ITERATIONS = 200
BASELINE_SEWING_TOLERANCE = 1e-4
REPORT_TEXT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".md"})


def write_text_lf(path: Path, text: str) -> None:
    """Write UTF-8 text with stable LF bytes on every platform."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(normalized)


def normalize_report_text_files(report_dir: Path) -> None:
    """Normalize existing Git-safe report text files before hashing them.

    A Windows checkout may materialize an already archived report as CRLF.
    Rewriting only the report's text artifacts keeps their content unchanged
    while making the bytes hashed into ``artifact_manifest.json`` identical to
    the LF bytes stored by Git under the report's ``eol=lf`` attributes.
    """
    for path in sorted(Path(report_dir).rglob("*")):
        if (
            path.is_file()
            and path.name != "artifact_manifest.json"
            and path.suffix.lower() in REPORT_TEXT_SUFFIXES
        ):
            write_text_lf(path, path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    write_text_lf(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_text_lf(
        path,
        "".join(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
    )


def _normalized_source_manifest(raw: Any, *, repo_root: Path) -> str | None:
    if not raw:
        return None
    path = Path(str(raw))
    try:
        return path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        parts = list(path.parts)
        if "reports" in parts:
            return Path(*parts[parts.index("reports") :]).as_posix()
        return path.name


def normalize_summary(summary: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    normalized = dict(summary)
    normalized["source_manifest"] = _normalized_source_manifest(
        summary.get("source_manifest"), repo_root=repo_root
    )
    normalized["heavy_artifacts_archived"] = False
    normalized["step_bytes_archived"] = False
    normalized["raw_pickle_archived"] = False
    return normalized


def validate_evidence(
    summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any]]
) -> None:
    errors = []
    expected_cases = int(summary.get("expected_cases", -1))
    expected_attempts = int(summary.get("expected_attempts", -1))
    if expected_cases != 16 or int(summary.get("cases", -1)) != 16:
        errors.append("P0-A frozen cohort must contain exactly 16 cases")
    if expected_attempts != 96 or int(summary.get("attempts", -1)) != 96:
        errors.append("P0-A matrix must contain exactly 96 attempts")
    if len(cases) != 16 or len({str(case.get("cad_id")) for case in cases}) != 16:
        errors.append("case JSON must contain 16 unique CAD identities")
    if len(attempts) != 96:
        errors.append("attempt JSONL must contain 96 rows")
    keys = {
        (
            str(row.get("cad_id")),
            int(row.get("joint_iterations", -1)),
            format(float(row.get("sewing_tolerance", -1)), ".12g"),
        )
        for row in attempts
    }
    if len(keys) != 96:
        errors.append("attempt matrix contains duplicate or missing CAD/variant keys")
    if int(summary.get("attributed_cases", -1)) != sum(
        bool(case.get("attributed")) for case in cases
    ):
        errors.append("attributed case count does not reconcile")
    if sum(int(value) for value in (summary.get("primary_cause_counts") or {}).values()) != 16:
        errors.append("primary cause counts do not sum to 16")
    case_ids = {str(case.get("cad_id")) for case in cases}
    if case_ids != {str(row.get("cad_id")) for row in attempts}:
        errors.append("attempt CAD identities do not match case identities")
    if not summary.get("matrix_complete") or not summary.get("gate_passed"):
        errors.append("P0-A summary is not a complete passing attribution matrix")
    if errors:
        raise RuntimeError("; ".join(errors))


def compact_attempt(row: Mapping[str, Any], *, run_root: Path) -> dict[str, Any]:
    components = dict(row.get("validity_components") or {})
    curve_attempts = list(row.get("curve_fit_attempts") or [])
    selected = sorted(
        {
            float(item["tolerance"])
            for item in curve_attempts
            if item.get("status") == "succeeded" and item.get("tolerance") is not None
        }
    )
    step_path = Path(str(row.get("step_path"))) if row.get("step_path") else None
    relative = None
    if step_path is not None:
        try:
            relative = step_path.resolve().relative_to(Path(run_root).resolve()).as_posix()
        except ValueError:
            relative = Path("steps") / step_path.parent.name / step_path.name
            relative = relative.as_posix()
    return {
        "cad_id": row.get("cad_id"),
        "joint_iterations": row.get("joint_iterations"),
        "sewing_tolerance": row.get("sewing_tolerance"),
        "status": row.get("status"),
        "failure_stage": row.get("failure_stage"),
        "failure_entity_kind": row.get("failure_entity_kind"),
        "failure_entity_index": row.get("failure_entity_index"),
        "error_type": row.get("error_type"),
        "step_saved": bool(row.get("step_saved")),
        "construction_native_brep_valid": row.get("construction_native_brep_valid"),
        "native_brep_valid": row.get("native_brep_valid"),
        "strict_brep_valid": bool(row.get("strict_brep_valid")),
        "both_valid": bool(row.get("both_valid")),
        "wire_count": components.get("wire_count"),
        "wire_order_failures": components.get("wire_order_failures"),
        "wire_self_intersections": components.get("wire_self_intersections"),
        "shell_count": components.get("shell_count"),
        "shells_with_bad_edges": components.get("shells_with_bad_edges"),
        "free_edges": components.get("free_edges"),
        "solid_count": components.get("solid_count"),
        "curve_fit_failures": sum(item.get("status") == "failed" for item in curve_attempts),
        "selected_curve_tolerances": ";".join(format(value, ".12g") for value in selected),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "step_relative_path": relative,
        "step_bytes": row.get("step_bytes"),
        "step_sha256": row.get("step_sha256"),
    }


def detailed_attempt(row: Mapping[str, Any], *, run_root: Path) -> dict[str, Any]:
    """Retain stage evidence while removing machine-local source and STEP paths."""
    archived = dict(row)
    step_path = archived.pop("step_path", None)
    archived.pop("source_path", None)
    archived.pop("source_manifest", None)
    if step_path:
        path = Path(str(step_path))
        try:
            relative = path.resolve().relative_to(Path(run_root).resolve()).as_posix()
        except ValueError:
            relative = (Path("steps") / path.parent.name / path.name).as_posix()
        archived["step_relative_path"] = relative
    archived["source_path_archived"] = False
    archived["step_bytes_archived"] = False
    return archived


def _outcome_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    components = dict(row.get("validity_components") or {})
    return (
        row.get("failure_stage"), bool(row.get("step_saved")),
        row.get("construction_native_brep_valid"), row.get("native_brep_valid"),
        bool(row.get("strict_brep_valid")), bool(row.get("both_valid")),
        components.get("wire_order_failures"), components.get("wire_self_intersections"),
        components.get("shells_with_bad_edges"), components.get("free_edges"),
        components.get("shell_count"), components.get("solid_count"),
    )


def _aggregate_attempts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    stages: dict[str, int] = {}
    for row in rows:
        if row.get("failure_stage"):
            stage = str(row["failure_stage"])
            stages[stage] = stages.get(stage, 0) + 1
    return {
        "attempts": len(rows),
        "step_saved": sum(bool(row.get("step_saved")) for row in rows),
        "construction_native_brep_valid": sum(row.get("construction_native_brep_valid") is True for row in rows),
        "native_brep_valid": sum(row.get("native_brep_valid") is True for row in rows),
        "strict_brep_valid": sum(bool(row.get("strict_brep_valid")) for row in rows),
        "both_valid": sum(bool(row.get("both_valid")) for row in rows),
        "failure_stage_counts": dict(sorted(stages.items())),
    }


def build_ablation_summary(attempts: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(str(row.get("cad_id")), int(row.get("joint_iterations", -1)), format(float(row.get("sewing_tolerance", -1)), ".12g")): dict(row) for row in attempts}
    cad_ids = sorted(str(case.get("cad_id")) for case in cases)
    tolerances = (1e-4, 1e-3, 1e-2)
    joint_rows = []
    any_joint_changed: set[str] = set()
    recovered_with_joint_disabled: set[str] = set()
    for tolerance in tolerances:
        changed, arm_200, arm_0 = [], [], []
        for cad_id in cad_ids:
            row_200 = by_key[(cad_id, 200, format(tolerance, ".12g"))]
            row_0 = by_key[(cad_id, 0, format(tolerance, ".12g"))]
            arm_200.append(row_200); arm_0.append(row_0)
            if _outcome_signature(row_200) != _outcome_signature(row_0):
                changed.append(cad_id); any_joint_changed.add(cad_id)
            if bool(row_0.get("both_valid")) and not bool(row_200.get("both_valid")):
                recovered_with_joint_disabled.add(cad_id)
        joint_rows.append({
            "sewing_tolerance": tolerance, "compared_cases": len(cad_ids),
            "signature_changed_cases": len(changed), "signature_changed_case_ids": changed,
            "joint_200": _aggregate_attempts(arm_200), "joint_0": _aggregate_attempts(arm_0),
        })
    tolerance_rows = []
    for joint_iterations in (200, 0):
        baseline = {cad_id: by_key[(cad_id, joint_iterations, format(BASELINE_SEWING_TOLERANCE, ".12g"))] for cad_id in cad_ids}
        for tolerance in tolerances:
            rows = [by_key[(cad_id, joint_iterations, format(tolerance, ".12g"))] for cad_id in cad_ids]
            changed = [cad_id for cad_id, row in zip(cad_ids, rows) if _outcome_signature(baseline[cad_id]) != _outcome_signature(row)]
            tolerance_rows.append({
                "joint_iterations": joint_iterations, "sewing_tolerance": tolerance,
                "compared_to_tolerance": BASELINE_SEWING_TOLERANCE,
                "signature_changed_cases": len(changed), "signature_changed_case_ids": changed,
                "outcomes": _aggregate_attempts(rows),
            })
    return {
        "joint_optimize_ablation": {
            "baseline_joint_iterations": 200, "ablation_joint_iterations": 0,
            "comparison_definition": "Paired by CAD and sewing tolerance; signature includes failure stage, STEP/native/strict/both flags, and decomposed wire/shell/solid counts.",
            "cases_with_any_signature_change_across_tolerances": len(any_joint_changed),
            "case_ids_with_any_signature_change_across_tolerances": sorted(any_joint_changed),
            "cases_recovered_to_both_valid_only_with_joint_disabled": sorted(recovered_with_joint_disabled),
            "by_tolerance": joint_rows,
        },
        "tolerance_scan": {
            "baseline_tolerance": BASELINE_SEWING_TOLERANCE, "scan_tolerances": list(tolerances),
            "comparison_definition": "Within each joint arm, paired by CAD against sewing tolerance 1e-4 using the same outcome signature.",
            "cases_marked_sensitive_at_joint_200": sorted(str(case.get("cad_id")) for case in cases if bool(case.get("tolerance_sensitive"))),
            "by_joint_and_tolerance": tolerance_rows,
        },
    }


def build_failure_taxonomy(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {}
    for case in cases:
        grouped.setdefault(str(case.get("primary_cause")), []).append(str(case.get("cad_id")))
    return {
        "cases": len(cases),
        "attributed_cases": sum(bool(case.get("attributed")) for case in cases),
        "taxonomy": [{"primary_cause": cause, "count": len(ids), "cad_ids": sorted(ids)} for cause, ids in sorted(grouped.items())],
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)


def build_step_manifest(
    attempts: Sequence[Mapping[str, Any]], *, run_root: Path
) -> list[dict[str, Any]]:
    rows = []
    for attempt in attempts:
        if not attempt.get("step_saved"):
            continue
        raw = attempt.get("step_path")
        if not raw:
            raise RuntimeError(f"saved STEP row has no path: {attempt.get('cad_id')}")
        path = Path(str(raw))
        if not path.is_file():
            raise FileNotFoundError(f"saved STEP file is missing: {path}")
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if int(attempt.get("step_bytes") or -1) != actual_bytes:
            raise RuntimeError(f"STEP byte count mismatch: {path}")
        if str(attempt.get("step_sha256")) != actual_sha256:
            raise RuntimeError(f"STEP SHA-256 mismatch: {path}")
        relative = path.resolve().relative_to(Path(run_root).resolve()).as_posix()
        rows.append(
            {
                "cad_id": attempt.get("cad_id"),
                "joint_iterations": attempt.get("joint_iterations"),
                "sewing_tolerance": attempt.get("sewing_tolerance"),
                "step_relative_path": relative,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
                "native_brep_valid": attempt.get("native_brep_valid"),
                "strict_brep_valid": bool(attempt.get("strict_brep_valid")),
                "both_valid": bool(attempt.get("both_valid")),
                "step_bytes_archived": False,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["cad_id"]),
            int(row["joint_iterations"]),
            float(row["sewing_tolerance"]),
        ),
    )


def render_readme(summary: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> str:
    causes = dict(summary.get("primary_cause_counts") or {})
    recovered_cases = [str(case["cad_id"]) for case in cases if case.get("any_variant_both_valid")]
    recovered_label = "、".join(f"`{cad_id}`" for cad_id in recovered_cases) or "无"
    return f"""# P0-A：100-CAD 原始控制组装配链诊断

## 一页结论

本实验冻结既有 100-CAD assembly calibration 中的 **16 个 original/GT strict-invalid CAD**，没有重新抽样。每个 CAD 运行 `joint_iterations ∈ {{200, 0}}` 与 sewing tolerance `{{1e-4, 1e-3, 1e-2}}` 的笛卡尔积，共 **{summary.get('attempts')}/96 attempts**。结果为 **{summary.get('attributed_cases')}/16 明确归因，attribution rate={float(summary.get('attribution_rate', 0)):.1%}**，P0-A 归因门（≥80%）通过。

主因分布：

- wire self-intersection：**{causes.get('wire_self_intersection', 0)}** 个；
- curve B-spline fit 在三档 fallback 后仍失败：**{causes.get('pre_step:curve_fit', 0)}** 个；
- OCC wire build 失败：**{causes.get('pre_step:wire_build', 0)}** 个；
- non-unit/empty solid：**{causes.get('nonunit_solid_count', 0)}** 个。

这证明 16 个 GT-invalid 不是同一种问题，其中 trim/wire self-intersection 是首要族群。现有聚合结论 `ASSEMBLY_DOMINATED` 得到了逐例、逐阶段证据支持。

## 消融解释

- `joint_iterations=0` 使 **{summary.get('joint_sensitive_cases')}** 个 CAD 的完整 outcome signature 发生变化；
- sewing tolerance 三档扫描使 **{summary.get('tolerance_sensitive_cases')}** 个 CAD 的完整 signature 发生变化；
- 但只有 **{summary.get('cases_with_any_both_valid_variant')}/16** 个 CAD 在任一变体达到 both-valid：{recovered_label}；该 CAD 的三个 `joint=0` tolerance 变体均有效。

因此，**“签名敏感”不等于“修复成功”**。它可能只是 native/strict、wire self-intersection 数量、shell/solid 计数或失败 stage 改变。三档 tolerance 没有形成可推广的恢复证据，不能据此全局放宽容差；关掉 joint optimize 也不是通用修复。

## 修复优先级

1. **P0-A1：trim/wire self-intersection（10/16）**。逐 face/wire 记录自交实体，检查 edge orientation、pcurve 与 outer/inner loop 语义；不得先用宽松 ShapeFix 掩盖根因。
2. **P0-A2：退化 curve fit（3/16）**。利用已记录的 edge index 与 `5e-3 → 8e-3 → 5e-2` fallback 证据，区分重复点/零长度/病态曲线，再加入有界 lower-degree 或 degenerate-edge 策略。
3. **P0-A3：wire build（2/16）**。在送入 OCC 前验证端点连续性、拓扑次序和方向，并把 builder error 与 face/loop 对齐。
4. **P0-A4：shell→solid cardinality（1/16）**。sewing 后显式枚举 shell，只允许单一闭合 shell 进入 `MakeSolid`，empty/compound/multi-shell 分开报告。
5. **横向项：joint offset**。对 4 个签名敏感 CAD 比较 surface-edge residual 与偏移量，将它视为放大器而不是已证实主因。

## 门控结论

P0-A 的“≥80% 可归因”验收已经通过（实际 100%），但这只关闭了诊断门，不代表装配已修复。`advance_to_boundary_consistency` 仍为 `{str(summary.get('advance_to_boundary_consistency')).lower()}`：必须先完成 P0-B 的 0-nonfinite 稳定性重测和健康 VQ 的固定 100-CAD 装配测量，再决定是否启动 boundary-consistency loss。序列重生成和 AR 继续阻塞。

## 证据索引

- `assembly_chain_summary.json`：规范化总体 gate、cause 和 sensitivity 计数；
- `assembly_chain_cases.json`：16 个 CAD 的 baseline signature、主因和二级证据；
- `attempts_compact.csv`：96 个 CAD×variant 的阶段、validity 组件和 STEP 绑定；
- `repair_checklist.md`：由 case 分类生成的修复清单；
- `step_sha256.csv`：本地保存的 STEP 大小和 SHA-256；**不包含 STEP 文件本体**；
- `artifact_manifest.json`：Git 归档内所有轻量文件的大小和 SHA-256。

未上传内容包括 STEP、原始 pickle、模型 checkpoint、重建数组和整个 upstream `BrepARG/`。
"""


def artifact_manifest(report_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(Path(report_dir).rglob("*")):
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


def snapshot(run_root: Path, report_dir: Path, *, repo_root: Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(run_root / "assembly_chain_summary.json")
    cases = read_json(run_root / "assembly_chain_cases.json")
    attempts = read_jsonl(run_root / "assembly_chain_attempts.jsonl")
    validate_evidence(summary, cases, attempts)
    normalized_summary = normalize_summary(summary, repo_root=repo_root)
    write_json(report_dir / "assembly_chain_summary.json", normalized_summary)
    write_json(report_dir / "assembly_chain_cases.json", cases)
    checklist = (run_root / "repair_checklist.md").read_text(encoding="utf-8")
    write_text_lf(report_dir / "repair_checklist.md", checklist.rstrip() + "\n")
    compact = [compact_attempt(row, run_root=run_root) for row in attempts]
    compact.sort(
        key=lambda row: (
            str(row["cad_id"]),
            int(row["joint_iterations"]),
            float(row["sewing_tolerance"]),
        )
    )
    write_csv(report_dir / "attempts_compact.csv", ATTEMPT_FIELDS, compact)
    steps = build_step_manifest(attempts, run_root=run_root)
    write_csv(report_dir / "step_sha256.csv", STEP_FIELDS, steps)
    write_text_lf(report_dir / "README.md", render_readme(normalized_summary, cases))
    normalize_report_text_files(report_dir)
    write_json(
        report_dir / "artifact_manifest.json",
        {
            "policy": "STEP, pickle, checkpoint, and reconstructed-array bytes remain local.",
            "artifacts": artifact_manifest(report_dir),
        },
    )
    return {
        "cases": len(cases),
        "attempts": len(attempts),
        "saved_steps_bound_by_sha256": len(steps),
        "attribution_rate": normalized_summary.get("attribution_rate"),
        "gate_passed": normalized_summary.get("gate_passed"),
        "report_dir": str(report_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    result = snapshot(args.run_root, args.report_dir, repo_root=args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
