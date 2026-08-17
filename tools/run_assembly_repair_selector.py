"""Run the pre-registered failure-triggered CAD assembly selector.

The selector retains the current guarded directed/local-topology result whenever
it is project-strict-valid.  Only a primary failure may try near-vertex
reconciliation, curve interpolation, high-precision surface fitting with local
topology repair, and finally high-precision surface fitting with curve
interpolation.  A fallback must additionally be native-valid, both-valid, and
geometry/topology preserving before it may replace the primary output.  The
runner keeps all failed candidates in a per-CAD selection record while retaining
exactly one final denominator row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .assembly_repair import RepairProfile, parse_profiles
    from .assembly_selector_geometry import (
        GEOMETRY_GATE_SCHEMA,
        MAX_BBOX_RELATIVE_DELTA,
        MAX_EDGE_LENGTH_RELATIVE_DELTA,
        MAX_EDGE_SAMPLE_MAX_NORMALIZED,
        MAX_EDGE_SAMPLE_RMS_NORMALIZED,
        validate_accepted_geometry_gate,
    )
    from .run_assembly_repair_matrix import (
        EXPECTED_BASELINE_VALID,
        EXPECTED_CADS,
        RUN_MANIFEST_NAME,
        RUN_SCHEMA,
        append_jsonl,
        bind_run_manifest,
        cohort_signature,
        frozen_original_rows,
        git_identity,
        historical_strict_map,
        output_root_writer_lock,
        profile_kwargs,
        read_jsonl,
        run_one_isolated,
        sha256_file,
        validate_attempt_row,
    )
    from .run_p0b_stability_retest import atomic_json
except ImportError:  # direct script execution
    from assembly_repair import RepairProfile, parse_profiles
    from assembly_selector_geometry import (
        GEOMETRY_GATE_SCHEMA,
        MAX_BBOX_RELATIVE_DELTA,
        MAX_EDGE_LENGTH_RELATIVE_DELTA,
        MAX_EDGE_SAMPLE_MAX_NORMALIZED,
        MAX_EDGE_SAMPLE_RMS_NORMALIZED,
        validate_accepted_geometry_gate,
    )
    from run_assembly_repair_matrix import (
        EXPECTED_BASELINE_VALID,
        EXPECTED_CADS,
        RUN_MANIFEST_NAME,
        RUN_SCHEMA,
        append_jsonl,
        bind_run_manifest,
        cohort_signature,
        frozen_original_rows,
        git_identity,
        historical_strict_map,
        output_root_writer_lock,
        profile_kwargs,
        read_jsonl,
        run_one_isolated,
        sha256_file,
        validate_attempt_row,
    )
    from run_p0b_stability_retest import atomic_json


SELECTOR_SCHEMA = "assembly-repair-selector-v1"
CANDIDATE_SCHEMA = "assembly-selector-candidate-v1"
SELECTOR_PROFILE_NAME = "failure_triggered_selector"
SELECTOR_SWITCHES = ("failure_triggered_selector",)
PRIMARY_PROFILE_NAME = "directed_trim_local_intersection_topology"
NEAR_VERTEX_PROFILE_NAME = "near_vertex_reconciliation"
INTERPOLATE_PROFILE_NAME = "directed_trim_curve_interpolate_local_intersection_topology"
SURFACE_PRECISION_PROFILE_NAME = (
    "directed_trim_surface_precision_local_intersection_topology"
)
SURFACE_PRECISION_CURVE_INTERPOLATE_PROFILE_NAME = (
    "directed_trim_surface_precision_curve_interpolate"
)
FALLBACK_PROFILE_NAMES = (
    NEAR_VERTEX_PROFILE_NAME,
    INTERPOLATE_PROFILE_NAME,
    SURFACE_PRECISION_PROFILE_NAME,
    SURFACE_PRECISION_CURVE_INTERPOLATE_PROFILE_NAME,
)

# This identity set is registered before executing the selector.  It comes
# from independently measured no-regression primitives, not from selector
# routing or run-time CAD-specific conditions.
EXPECTED_BOTH_VALID_RESTORATIONS = frozenset(
    {
        "00000444_4ed4c78d6d754aac90876fc2_step_003",
        "00002441_179a8df075c94d0596b1f20d_step_010",
        "00016845_33dbc9e6ea684970be61af74_step_000",
        "00029780_a4788d5955c04fe1886666b7_step_000",
        "00032004_b91639b3cc4b41c7bf6a854d_step_000",
        "00051587_446e8810d6884cae80689579_step_000",
        "00008763_affd199038524a1c8e3b7ad4_step_001",
    }
)
EXPECTED_FALLBACK_ACCEPTED_IDS = frozenset(
    {
        "00000444_4ed4c78d6d754aac90876fc2_step_003",
        "00002441_179a8df075c94d0596b1f20d_step_010",
        "00051587_446e8810d6884cae80689579_step_000",
        "00008763_affd199038524a1c8e3b7ad4_step_001",
    }
)


def canonical_result_sha256(result: Mapping[str, Any]) -> str:
    """Fingerprint one persisted worker result without relying on its profile name."""
    payload = json.dumps(
        dict(result),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selector_profiles() -> tuple[RepairProfile, tuple[RepairProfile, ...]]:
    (
        primary,
        near_vertex,
        interpolate,
        surface_precision,
        surface_precision_curve_interpolate,
    ) = parse_profiles(
        [
            PRIMARY_PROFILE_NAME,
            NEAR_VERTEX_PROFILE_NAME,
            INTERPOLATE_PROFILE_NAME,
            SURFACE_PRECISION_PROFILE_NAME,
            SURFACE_PRECISION_CURVE_INTERPOLATE_PROFILE_NAME,
        ]
    )
    return primary, (
        near_vertex,
        interpolate,
        surface_precision,
        surface_precision_curve_interpolate,
    )


def _validity_components(row: Mapping[str, Any]) -> dict[str, Any]:
    components = row.get("validity_components")
    if not isinstance(components, Mapping):
        return {}
    return {
        key: components.get(key)
        for key in (
            "status",
            "wire_count",
            "wire_order_failures",
            "wire_self_intersections",
            "free_edges",
            "shell_count",
            "shells_with_bad_edges",
            "solid_count",
        )
    }


def compact_candidate(
    row: Mapping[str, Any], *, geometry_gate: Mapping[str, Any] | None = None,
    rejection_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Return candidate evidence without source, STEP, log, or host paths."""
    result = {
        key: row.get(key)
        for key in (
            "profile",
            "switches",
            "status",
            "step_saved",
            "native_brep_valid",
            "strict_brep_valid",
            "both_valid",
            "step_bytes",
            "step_sha256",
            "error_type",
            "elapsed_seconds",
            "worker_returncode",
        )
    }
    result["candidate_result_sha256"] = canonical_result_sha256(row)
    result["validity_components"] = _validity_components(row)
    result["rejection_reasons"] = list(rejection_reasons)
    if geometry_gate is not None:
        result["geometry_topology_gate"] = dict(geometry_gate)
    return result


def candidate_rejection_reasons(
    row: Mapping[str, Any], geometry_gate: Mapping[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if row.get("status") != "both_valid":
        reasons.append("candidate_status_not_both_valid")
    if not bool(row.get("step_saved")):
        reasons.append("candidate_step_missing")
    if not bool(row.get("native_brep_valid")):
        reasons.append("candidate_native_invalid")
    if not bool(row.get("strict_brep_valid")):
        reasons.append("candidate_strict_invalid")
    if not bool(row.get("both_valid")):
        reasons.append("candidate_not_both_valid")
    if geometry_gate is not None and not bool(geometry_gate.get("accepted")):
        reasons.extend(str(value) for value in geometry_gate.get("rejection_reasons") or ())
        if not geometry_gate.get("rejection_reasons"):
            reasons.append("geometry_gate_rejected")
    return reasons


def fallback_eligible(
    row: Mapping[str, Any], geometry_gate: Mapping[str, Any] | None
) -> tuple[bool, list[str]]:
    """Apply the auditable fallback predicate without trusting one status field."""
    reasons = candidate_rejection_reasons(row, geometry_gate)
    return not reasons, reasons


def first_eligible_compact_fallback(
    candidates: Sequence[Mapping[str, Any]],
) -> str | None:
    """Return the first semantically valid fallback in persisted route order."""
    for candidate in candidates[1:]:
        gate = candidate.get("geometry_topology_gate")
        gate_valid, _ = validate_accepted_geometry_gate(gate)
        if not gate_valid:
            continue
        eligible, _ = fallback_eligible(candidate, gate)
        if eligible:
            return str(candidate.get("profile"))
    return None


def expected_selected_profile(
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    """Recompute the fixed route decision from compact persisted evidence."""
    if bool(candidates[0].get("strict_brep_valid")):
        return PRIMARY_PROFILE_NAME
    return first_eligible_compact_fallback(candidates) or PRIMARY_PROFILE_NAME


def route_selector_candidates(
    *,
    run_candidate: Callable[[RepairProfile], Mapping[str, Any]],
    geometry_gate_for: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Run only the candidates permitted by the fixed failure-triggered policy."""
    primary, fallbacks = selector_profiles()
    primary_row = dict(run_candidate(primary))
    candidates: list[dict[str, Any]] = [
        compact_candidate(
            primary_row,
            rejection_reasons=(
                ()
                if bool(primary_row.get("strict_brep_valid"))
                else ("primary_strict_invalid",)
            ),
        )
    ]
    if bool(primary_row.get("strict_brep_valid")):
        return primary_row, {
            "schema": SELECTOR_SCHEMA,
            "primary_profile": primary.name,
            "fallback_order": [profile.name for profile in fallbacks],
            "attempted_profiles": [primary.name],
            "selected_profile": primary.name,
            "selected_reason": "primary_strict_valid",
            "fallback_accepted": False,
            "candidates": candidates,
        }

    selected = primary_row
    selected_reason = "no_fallback_passed"
    for profile in fallbacks:
        candidate = dict(run_candidate(profile))
        gate: Mapping[str, Any] | None = None
        validity_ready = bool(
            candidate.get("step_saved") is True
            and candidate.get("status") == "both_valid"
            and candidate.get("native_brep_valid") is True
            and candidate.get("strict_brep_valid") is True
            and candidate.get("both_valid") is True
        )
        if validity_ready:
            gate = geometry_gate_for(candidate)
        eligible, reasons = fallback_eligible(candidate, gate)
        candidates.append(
            compact_candidate(
                candidate,
                geometry_gate=gate,
                rejection_reasons=reasons,
            )
        )
        if eligible:
            selected = candidate
            selected_reason = "fallback_native_strict_geometry_passed"
            return selected, {
                "schema": SELECTOR_SCHEMA,
                "primary_profile": primary.name,
                "fallback_order": [fallback.name for fallback in fallbacks],
                "attempted_profiles": [
                    str(item["profile"]) for item in candidates
                ],
                "selected_profile": profile.name,
                "selected_reason": selected_reason,
                "fallback_accepted": True,
                "candidates": candidates,
            }
    return selected, {
        "schema": SELECTOR_SCHEMA,
        "primary_profile": primary.name,
        "fallback_order": [profile.name for profile in fallbacks],
        "attempted_profiles": [str(item["profile"]) for item in candidates],
        "selected_profile": primary.name,
        "selected_reason": selected_reason,
        "fallback_accepted": False,
        "candidates": candidates,
    }


def _rejected_accepted_geometry_gate(
    gate: Mapping[str, Any], validation_reasons: Sequence[str]
) -> dict[str, Any]:
    """Fail closed while retaining the worker's path-free explanation."""
    normalized = dict(gate)
    original_reasons = gate.get("rejection_reasons")
    if isinstance(original_reasons, Sequence) and not isinstance(
        original_reasons, (str, bytes, bytearray)
    ):
        reasons = [str(reason) for reason in original_reasons]
    elif original_reasons is None:
        reasons = []
    else:
        reasons = ["accepted_gate_rejection_reasons_malformed"]
    reasons.extend(str(reason) for reason in validation_reasons)
    normalized["accepted"] = False
    normalized["rejection_reasons"] = list(dict.fromkeys(reasons))
    return normalized


def geometry_gate_for_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Consume only a complete, path-free gate from the isolated worker."""
    gate = candidate.get("selector_geometry_topology_gate")
    if isinstance(gate, Mapping) and gate.get("schema") == GEOMETRY_GATE_SCHEMA:
        normalized = dict(gate)
        if normalized.get("accepted") is True:
            gate_valid, validation_reasons = validate_accepted_geometry_gate(
                normalized
            )
            if gate_valid:
                return normalized
            return _rejected_accepted_geometry_gate(
                normalized, validation_reasons
            )
        if normalized.get("accepted") is not False:
            return _rejected_accepted_geometry_gate(
                normalized, ["accepted_gate_flag_not_boolean"]
            )
        return normalized
    return {
        "schema": GEOMETRY_GATE_SCHEMA,
        "accepted": False,
        "checks": {"isolated_geometry_gate_present": False},
        "rejection_reasons": ["isolated_geometry_gate_missing"],
    }


def _materialize_selected_step(
    selected: Mapping[str, Any], *, output_dir: Path, cad_id: str
) -> tuple[bool, str | None, int | None, str | None]:
    if not bool(selected.get("step_saved")):
        return False, None, None, None
    source = Path(str(selected.get("step_path")))
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError("selected candidate STEP is missing or empty")
    destination = (
        Path(output_dir) / "steps" / SELECTOR_PROFILE_NAME / f"{cad_id}.step"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_hash = str(selected.get("step_sha256") or "")
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise RuntimeError("selected candidate STEP hash changed while materializing")
    return True, str(destination), int(destination.stat().st_size), destination_hash


def final_selector_row(
    source: Mapping[str, Any],
    selected: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the one attempts-denominator row retained by the selector."""
    cad_id = str(source["cad_id"])
    step_saved, step_path, step_bytes, step_sha256 = _materialize_selected_step(
        selected, output_dir=output_dir, cad_id=cad_id
    )
    candidates = list(selection.get("candidates") or ())
    elapsed = sum(
        float(candidate.get("elapsed_seconds") or 0.0)
        for candidate in candidates
    )
    result: dict[str, Any] = {
        "schema": SELECTOR_SCHEMA,
        "cad_id": cad_id,
        "parent_id": source.get("parent_id"),
        "profile": SELECTOR_PROFILE_NAME,
        "switches": list(SELECTOR_SWITCHES),
        "historical_strict_valid": bool(source.get("brep_valid")),
        "source_path": str(source["source_path"]),
        "status": str(selected.get("status")),
        "step_saved": bool(step_saved),
        "native_brep_valid": bool(selected.get("native_brep_valid")),
        "strict_brep_valid": bool(selected.get("strict_brep_valid")),
        "both_valid": bool(selected.get("both_valid")),
        "validity_components": _validity_components(selected),
        "elapsed_seconds": elapsed,
        "selection": dict(selection),
    }
    if step_saved:
        result.update(
            step_path=step_path,
            step_bytes=step_bytes,
            step_sha256=step_sha256,
        )
    else:
        for key in ("error_type",):
            if selected.get(key) is not None:
                result[key] = selected.get(key)
    return result


def _candidate_profile_names() -> tuple[str, ...]:
    return (PRIMARY_PROFILE_NAME, *FALLBACK_PROFILE_NAMES)


def candidate_ledger_entry(
    source: Mapping[str, Any],
    profile: RepairProfile,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    order = _candidate_profile_names().index(profile.name)
    return {
        "schema": CANDIDATE_SCHEMA,
        "cad_id": str(source["cad_id"]),
        "profile": profile.name,
        "attempt_order": order,
        "result": dict(result),
        "result_sha256": canonical_result_sha256(result),
    }


def validate_candidate_ledger(
    entries: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
) -> None:
    """Validate a resumable dynamic candidate prefix for every CAD."""
    by_cad = {str(source["cad_id"]): source for source in sources}
    primary, fallbacks = selector_profiles()
    profiles = {profile.name: profile for profile in (primary, *fallbacks)}
    expected_order = list(_candidate_profile_names())
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        if entry.get("schema") != CANDIDATE_SCHEMA:
            raise RuntimeError("selector candidate ledger schema mismatch")
        cad_id = str(entry.get("cad_id"))
        profile_name = str(entry.get("profile"))
        if cad_id not in by_cad or profile_name not in profiles:
            raise RuntimeError(
                f"selector candidate escapes signed cohort or profiles: "
                f"{cad_id}:{profile_name}"
            )
        key = (cad_id, profile_name)
        if key in seen:
            raise RuntimeError(f"duplicate selector candidate attempt: {key}")
        seen.add(key)
        expected_attempt_order = expected_order.index(profile_name)
        if entry.get("attempt_order") != expected_attempt_order:
            raise RuntimeError(f"selector candidate attempt order mismatch: {key}")
        result = entry.get("result")
        if not isinstance(result, Mapping):
            raise RuntimeError(f"selector candidate result is missing: {key}")
        if entry.get("result_sha256") != canonical_result_sha256(result):
            raise RuntimeError(f"selector candidate result hash mismatch: {key}")
        validate_attempt_row(result, by_cad[cad_id], profiles[profile_name])
        if profile_name != PRIMARY_PROFILE_NAME:
            raw_gate = result.get("selector_geometry_topology_gate")
            if isinstance(raw_gate, Mapping) and bool(raw_gate.get("accepted")):
                gate_valid, gate_reasons = validate_accepted_geometry_gate(raw_gate)
                if not gate_valid:
                    raise RuntimeError(
                        "selector candidate ledger has invalid accepted geometry "
                        f"gate: {cad_id}:{profile_name}: {', '.join(gate_reasons)}"
                    )
        grouped.setdefault(cad_id, []).append(entry)
    for cad_id, cad_entries in grouped.items():
        ordered = sorted(cad_entries, key=lambda item: int(item["attempt_order"]))
        names = [str(item["profile"]) for item in ordered]
        if names != expected_order[: len(names)]:
            raise RuntimeError(f"selector candidate sequence is not a prefix: {cad_id}")
        primary_result = ordered[0]["result"]
        if bool(primary_result.get("strict_brep_valid")) and len(ordered) != 1:
            raise RuntimeError(
                f"selector ledger ran fallback after strict primary: {cad_id}"
            )
        for earlier_entry in ordered[1:-1]:
            earlier_result = earlier_entry["result"]
            earlier_gate = geometry_gate_for_candidate(earlier_result)
            earlier_eligible, _ = fallback_eligible(
                earlier_result, earlier_gate
            )
            if earlier_eligible:
                raise RuntimeError(
                    "selector ledger ran a later fallback after an accepted "
                    f"fallback: {cad_id}:{earlier_entry['profile']}"
                )


def candidate_result_map(
    entries: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(entry["cad_id"]), str(entry["profile"])): entry["result"]
        for entry in entries
    }


def validate_final_candidate_bindings(
    rows: Sequence[Mapping[str, Any]],
    candidate_entries: Sequence[Mapping[str, Any]],
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for entry in candidate_entries:
        grouped.setdefault(str(entry["cad_id"]), []).append(entry)
    for row in rows:
        cad_id = str(row["cad_id"])
        ordered_entries = sorted(
            grouped.get(cad_id, ()), key=lambda item: int(item["attempt_order"])
        )
        names = [str(entry["profile"]) for entry in ordered_entries]
        selection = row.get("selection") or {}
        attempted = list(selection.get("attempted_profiles") or ())
        if names != attempted:
            raise RuntimeError(
                f"selector final row and candidate ledger differ: {cad_id}"
            )
        compact_candidates = selection.get("candidates")
        if not isinstance(compact_candidates, list) or len(compact_candidates) != len(
            ordered_entries
        ):
            raise RuntimeError(
                f"selector final candidate evidence count differs from ledger: {cad_id}"
            )
        for compact, entry in zip(compact_candidates, ordered_entries):
            result = entry["result"]
            profile_name = str(entry["profile"])
            if profile_name == PRIMARY_PROFILE_NAME:
                gate = None
                reasons: Sequence[str] = (
                    ()
                    if bool(result.get("strict_brep_valid"))
                    else ("primary_strict_invalid",)
                )
            else:
                validity_ready = bool(
                    result.get("step_saved") is True
                    and result.get("status") == "both_valid"
                    and result.get("native_brep_valid") is True
                    and result.get("strict_brep_valid") is True
                    and result.get("both_valid") is True
                )
                gate = geometry_gate_for_candidate(result) if validity_ready else None
                _, reasons = fallback_eligible(result, gate)
            expected_compact = compact_candidate(
                result, geometry_gate=gate, rejection_reasons=reasons
            )
            if compact != expected_compact:
                raise RuntimeError(
                    f"selector final candidate evidence differs from ledger: "
                    f"{cad_id}:{profile_name}"
                )
            if compact.get("candidate_result_sha256") != entry.get("result_sha256"):
                raise RuntimeError(
                    f"selector final candidate fingerprint differs from ledger: "
                    f"{cad_id}:{profile_name}"
                )
        selected_profile = str(selection.get("selected_profile"))
        expected_profile = expected_selected_profile(compact_candidates)
        if selected_profile != expected_profile:
            raise RuntimeError(
                "selector final row did not select the first eligible candidate: "
                f"{cad_id}: expected {expected_profile}, got {selected_profile}"
            )
        selected_entries = [
            entry for entry in ordered_entries if str(entry["profile"]) == selected_profile
        ]
        if len(selected_entries) != 1:
            raise RuntimeError(
                f"selector final selected candidate is not unique in ledger: {cad_id}"
            )
        selected_result = selected_entries[0]["result"]
        for key in (
            "status",
            "step_saved",
            "native_brep_valid",
            "strict_brep_valid",
            "both_valid",
        ):
            if row.get(key) != selected_result.get(key):
                raise RuntimeError(
                    f"selector final {key} differs from selected ledger result: {cad_id}"
                )
        if bool(row.get("step_saved")):
            for key in ("step_bytes", "step_sha256"):
                if row.get(key) != selected_result.get(key):
                    raise RuntimeError(
                        f"selector final {key} differs from selected ledger result: "
                        f"{cad_id}"
                    )


def validate_selector_row(row: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    """Validate fixed routing and result semantics before a row is persisted."""
    expected = {
        "schema": SELECTOR_SCHEMA,
        "cad_id": str(source["cad_id"]),
        "parent_id": source.get("parent_id"),
        "profile": SELECTOR_PROFILE_NAME,
        "switches": list(SELECTOR_SWITCHES),
        "historical_strict_valid": bool(source.get("brep_valid")),
        "source_path": str(source["source_path"]),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f"selector result {key} mismatch: expected {value!r}, got {row.get(key)!r}"
            )
    for key in (
        "step_saved",
        "native_brep_valid",
        "strict_brep_valid",
        "both_valid",
    ):
        if type(row.get(key)) is not bool:
            raise ValueError(f"selector result {key} must be a boolean")
    if row["both_valid"] is not bool(
        row["native_brep_valid"] and row["strict_brep_valid"]
    ):
        raise ValueError("selector result both_valid is inconsistent")
    selection = row.get("selection")
    if not isinstance(selection, Mapping) or selection.get("schema") != SELECTOR_SCHEMA:
        raise ValueError("selector result is missing a valid selection record")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("selector result has no candidate evidence")
    if candidates[0].get("profile") != PRIMARY_PROFILE_NAME:
        raise ValueError("selector did not run the primary path first")
    profiles = [str(candidate.get("profile")) for candidate in candidates]
    if any(profile not in _candidate_profile_names() for profile in profiles):
        raise ValueError("selector candidate uses an unregistered profile")
    if profiles != [PRIMARY_PROFILE_NAME, *profiles[1:]]:
        raise ValueError("selector primary profile ordering is invalid")
    if profiles[1:] != list(FALLBACK_PROFILE_NAMES[: len(profiles) - 1]):
        raise ValueError("selector fallback ordering is invalid")
    primary_strict = bool(candidates[0].get("strict_brep_valid"))
    selected_profile = str(selection.get("selected_profile"))
    if selected_profile not in profiles:
        raise ValueError("selector selected an unattempted profile")
    if primary_strict:
        if len(candidates) != 1 or selected_profile != PRIMARY_PROFILE_NAME:
            raise ValueError("selector ran a fallback after a strict-valid primary")
    elif selected_profile != PRIMARY_PROFILE_NAME:
        selected_candidates = [
            candidate for candidate in candidates
            if candidate.get("profile") == selected_profile
        ]
        if len(selected_candidates) != 1:
            raise ValueError("selector fallback selection is ambiguous")
        selected_candidate = selected_candidates[0]
        gate = selected_candidate.get("geometry_topology_gate")
        gate_valid, gate_reasons = validate_accepted_geometry_gate(gate)
        if not gate_valid:
            raise ValueError(
                "selector accepted a fallback with invalid geometry gate: "
                + ", ".join(gate_reasons)
            )
        if not (
            selected_candidate.get("native_brep_valid") is True
            and selected_candidate.get("strict_brep_valid") is True
            and selected_candidate.get("both_valid") is True
        ):
            raise ValueError("selector accepted a fallback without all required gates")
    expected_profile = expected_selected_profile(candidates)
    if selected_profile != expected_profile:
        raise ValueError(
            "selector did not select the first eligible candidate: "
            f"expected {expected_profile}, got {selected_profile}"
        )
    selected_candidates = [
        candidate for candidate in candidates
        if candidate.get("profile") == selected_profile
    ]
    if len(selected_candidates) != 1:
        raise ValueError("selector selected candidate is not unique")
    selected = selected_candidates[0]
    for key in (
        "native_brep_valid",
        "strict_brep_valid",
        "both_valid",
    ):
        if row[key] is not bool(selected.get(key)):
            raise ValueError(f"selector final {key} differs from selected candidate")


def validate_selector_step(row: Mapping[str, Any], *, output_dir: Path) -> None:
    """Bind a resumed selector row to its selected local STEP artifact."""
    if not bool(row.get("step_saved")):
        return
    expected = (
        Path(output_dir) / "steps" / SELECTOR_PROFILE_NAME / f"{row['cad_id']}.step"
    )
    if not expected.is_file() or expected.stat().st_size <= 0:
        raise RuntimeError(f"selector saved STEP is missing: {row['cad_id']}")
    if row.get("step_path") != str(expected):
        raise RuntimeError(f"selector STEP path mismatch: {row['cad_id']}")
    if int(row.get("step_bytes") or -1) != expected.stat().st_size:
        raise RuntimeError(f"selector STEP size mismatch: {row['cad_id']}")
    if row.get("step_sha256") != sha256_file(expected):
        raise RuntimeError(f"selector STEP hash mismatch: {row['cad_id']}")


def validate_existing_selector_rows(
    rows: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
) -> None:
    by_cad = {str(source["cad_id"]): source for source in sources}
    seen: set[str] = set()
    for row in rows:
        cad_id = str(row.get("cad_id"))
        if cad_id in seen:
            raise RuntimeError(f"duplicate selector attempt: {cad_id}")
        seen.add(cad_id)
        if cad_id not in by_cad:
            raise RuntimeError(f"selector attempt escapes signed cohort: {cad_id}")
        validate_selector_row(row, by_cad[cad_id])
        validate_selector_step(row, output_dir=output_dir)


def selector_source_hashes(repo_root: Path) -> dict[str, str]:
    relative_paths = (
        "tools/assembly_repair.py",
        "tools/assembly_selector_geometry.py",
        "tools/directed_trim_assembly.py",
        "tools/local_wire_topology_repair.py",
        "tools/solid_topology_repair.py",
        "tools/run_assembly_repair_matrix.py",
        "tools/run_assembly_repair_selector.py",
        "tools/run_assembly_calibration_oracle.py",
        "tools/diagnose_step_validity_components.py",
        "tools/run_p0b_stability_retest.py",
    )
    root = Path(repo_root).resolve()
    return {
        relative: sha256_file(root / relative)
        for relative in relative_paths
    }


def selected_source_pickle_bindings(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bind every executed CAD input without serializing local source paths."""
    bindings: dict[str, dict[str, Any]] = {}
    for row in rows:
        cad_id = str(row["cad_id"])
        source = Path(str(row["source_path"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"selector source pickle is missing for {cad_id}")
        if cad_id in bindings:
            raise RuntimeError(f"duplicate selector source CAD id: {cad_id}")
        bindings[cad_id] = {
            "bytes": int(source.stat().st_size),
            "sha256": sha256_file(source),
        }
    return dict(sorted(bindings.items()))


def build_selector_run_payload(
    *,
    args: argparse.Namespace,
    full_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    breparg_utils = Path(args.breparg_root).resolve() / "utils.py"
    if not breparg_utils.is_file():
        raise FileNotFoundError(breparg_utils)
    primary, fallbacks = selector_profiles()
    return {
        "schema": RUN_SCHEMA,
        "run_kind": SELECTOR_SCHEMA,
        "matrix_schema": SELECTOR_SCHEMA,
        "candidate_schema": CANDIDATE_SCHEMA,
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "full_cohort_count": len(full_rows),
        "full_cohort_signature": cohort_signature(full_rows),
        "selected_cohort_count": len(selected_rows),
        "selected_cohort_signature": cohort_signature(selected_rows),
        "selected_source_pickles": selected_source_pickle_bindings(selected_rows),
        "historical_invalid_only": bool(args.historical_invalid_only),
        "joint_iterations": int(args.joint_iterations),
        "worker_timeout_seconds": float(args.worker_timeout_seconds),
        "selector": {
            "profile": SELECTOR_PROFILE_NAME,
            "primary": {
                "name": primary.name,
                "switches": list(primary.switches),
                "effective_constructor_kwargs": profile_kwargs(primary),
            },
            "fallbacks": [
                {
                    "name": profile.name,
                    "switches": list(profile.switches),
                    "effective_constructor_kwargs": profile_kwargs(profile),
                }
                for profile in fallbacks
            ],
            "expected_both_valid_restorations": sorted(
                EXPECTED_BOTH_VALID_RESTORATIONS
            ),
            "expected_fallback_accepted_ids": sorted(
                EXPECTED_FALLBACK_ACCEPTED_IDS
            ),
            "geometry_gate_schema": GEOMETRY_GATE_SCHEMA,
            "geometry_gate_thresholds": {
                "max_bbox_relative_delta": MAX_BBOX_RELATIVE_DELTA,
                "max_edge_length_relative_delta": MAX_EDGE_LENGTH_RELATIVE_DELTA,
                "max_edge_sample_rms_normalized": (
                    MAX_EDGE_SAMPLE_RMS_NORMALIZED
                ),
                "max_edge_sample_max_normalized": (
                    MAX_EDGE_SAMPLE_MAX_NORMALIZED
                ),
            },
            "fallback_acceptance_predicate": (
                "step_saved && status=both_valid && native && strict && both "
                "&& isolated_geometry_topology_gate"
            ),
            "all_candidates_isolated": True,
        },
        "repository": {
            **git_identity(repo_root),
            "source_sha256": selector_source_hashes(repo_root),
        },
        "breparg_runtime": {"utils_sha256": sha256_file(breparg_utils)},
    }


def verify_selector_payload_integrity(
    payload: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    selected_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed when a selector dependency or input changes during a run.

    The run manifest binds content hashes rather than local paths.  This check
    is made immediately after the writer lock is acquired and again before a
    completed manifest is written, which prevents a resumed directory from
    mixing candidates made from different source pickle or assembly code.
    """
    repo_root = Path(__file__).resolve().parents[1]
    repository = payload.get("repository")
    expected_sources = (
        repository.get("source_sha256") if isinstance(repository, Mapping) else None
    )
    if expected_sources != selector_source_hashes(repo_root):
        raise RuntimeError("selector execution source hashes changed during run")
    runtime = payload.get("breparg_runtime")
    expected_utils = (
        runtime.get("utils_sha256") if isinstance(runtime, Mapping) else None
    )
    utils_path = Path(args.breparg_root).resolve() / "utils.py"
    if not utils_path.is_file() or expected_utils != sha256_file(utils_path):
        raise RuntimeError("BrepARG utils hash changed during selector run")
    expected_pickles = payload.get("selected_source_pickles")
    if expected_pickles != selected_source_pickle_bindings(selected_rows):
        raise RuntimeError("selector source pickle bindings changed during run")


def summarize_selector(
    rows: Sequence[Mapping[str, Any]],
    historical: Mapping[str, bool],
    *,
    historical_invalid_only: bool,
) -> dict[str, Any]:
    if len(rows) != len(historical) or {
        str(row["cad_id"]) for row in rows
    } != set(historical):
        raise ValueError("selector rows do not cover the signed cohort exactly")
    observed_strict = {
        str(row["cad_id"]): bool(row.get("strict_brep_valid"))
        for row in rows
    }
    observed_both = {
        str(row["cad_id"]): bool(row.get("both_valid"))
        for row in rows
    }
    restored = sorted(
        cad_id
        for cad_id, baseline in historical.items()
        if not baseline and observed_strict[cad_id]
    )
    restored_both = sorted(
        cad_id
        for cad_id, baseline in historical.items()
        if not baseline and observed_both[cad_id]
    )
    regressed = sorted(
        cad_id
        for cad_id, baseline in historical.items()
        if baseline and not observed_strict[cad_id]
    )
    candidate_rows = [
        candidate
        for row in rows
        for candidate in (row.get("selection") or {}).get("candidates") or ()
    ]
    selected_profile_counts = Counter(
        str((row.get("selection") or {}).get("selected_profile"))
        for row in rows
    )
    accepted_fallback_ids = sorted(
        str(row["cad_id"])
        for row in rows
        if (row.get("selection") or {}).get("fallback_accepted") is True
    )
    worker_failures = sum(
        str(candidate.get("status")).startswith("worker_")
        for candidate in candidate_rows
    )
    primary_fast_path_violations = sum(
        bool((row.get("selection") or {}).get("candidates", [])[0].get(
            "strict_brep_valid"
        ))
        and len((row.get("selection") or {}).get("candidates") or ()) != 1
        for row in rows
    )
    selected_native_invalid_fallbacks = sum(
        (row.get("selection") or {}).get("selected_profile") != PRIMARY_PROFILE_NAME
        and not bool(row.get("native_brep_valid"))
        for row in rows
    )
    expected_restorations_match = set(restored_both) == EXPECTED_BOTH_VALID_RESTORATIONS
    expected_fallbacks_match = (
        set(accepted_fallback_ids) == EXPECTED_FALLBACK_ACCEPTED_IDS
    )
    preserve_controls = not regressed
    strict_count = sum(observed_strict.values())
    both_count = sum(observed_both.values())
    if historical_invalid_only:
        selector_protocol_passed = bool(
            strict_count == len(EXPECTED_BOTH_VALID_RESTORATIONS)
            and both_count == len(EXPECTED_BOTH_VALID_RESTORATIONS)
            and expected_restorations_match
            and expected_fallbacks_match
            and worker_failures == 0
            and primary_fast_path_violations == 0
            and selected_native_invalid_fallbacks == 0
        )
    else:
        selector_protocol_passed = bool(
            strict_count >= 90
            and both_count >= 87
            and preserve_controls
            and expected_restorations_match
            and expected_fallbacks_match
            and worker_failures == 0
            and primary_fast_path_violations == 0
            and selected_native_invalid_fallbacks == 0
        )
    profile_summary = {
        "profile": SELECTOR_PROFILE_NAME,
        "attempts": len(rows),
        "step_readable": sum(bool(row.get("step_saved")) for row in rows),
        "native_valid": sum(bool(row.get("native_brep_valid")) for row in rows),
        "strict_valid": strict_count,
        "both_valid": both_count,
        "restored_cad_ids": restored,
        "restored_both_valid_cad_ids": restored_both,
        "regressed_cad_ids": regressed,
        "unchanged_cad_ids": sorted(
            cad_id
            for cad_id in historical
            if historical[cad_id] == observed_strict[cad_id]
        ),
        "status_counts": dict(
            sorted(Counter(str(row.get("status")) for row in rows).items())
        ),
        "preserves_original_84": preserve_controls,
        "meets_95_gate": bool(strict_count >= 95 and preserve_controls),
    }
    return {
        "schema": SELECTOR_SCHEMA,
        "cohort_size": len(historical),
        "historical_strict_valid": sum(historical.values()),
        "historical_invalid_only": bool(historical_invalid_only),
        "profiles": [profile_summary],
        "selector": {
            "primary_profile": PRIMARY_PROFILE_NAME,
            "fallback_order": list(FALLBACK_PROFILE_NAMES),
            "expected_both_valid_restorations": sorted(
                EXPECTED_BOTH_VALID_RESTORATIONS
            ),
            "observed_both_valid_restorations": restored_both,
            "expected_fallback_accepted_ids": sorted(
                EXPECTED_FALLBACK_ACCEPTED_IDS
            ),
            "accepted_fallback_ids": accepted_fallback_ids,
            "selected_profile_counts": dict(sorted(selected_profile_counts.items())),
            "candidate_attempt_count": len(candidate_rows),
            "worker_or_protocol_failure_count": worker_failures,
            "primary_fast_path_violations": primary_fast_path_violations,
            "selected_native_invalid_fallbacks": selected_native_invalid_fallbacks,
            "expected_restorations_match": expected_restorations_match,
            "expected_fallbacks_match": expected_fallbacks_match,
            "protocol_passed": selector_protocol_passed,
        },
        # This is intentionally distinct from the P0-A release gate.  Passing
        # the selector's registered proof does not waive the 95/100
        # assembly-release requirement.
        "selector_protocol_passed": selector_protocol_passed,
        "accepted_profiles": [],
        "gate_passed": False,
        "assembly_release_gate_passed": profile_summary["meets_95_gate"],
        "advance_to_boundary_consistency": False,
        "advance_to_sequence_or_ar": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--breparg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-iterations", type=int, default=200)
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--historical-invalid-only",
        action="store_true",
        help="Run only the frozen 16 original failures before the full matrix.",
    )
    args = parser.parse_args(argv)
    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    full_rows = frozen_original_rows(args.calibration_manifest)
    historical = historical_strict_map(full_rows)
    source_rows = list(full_rows)
    if args.historical_invalid_only:
        source_rows = [row for row in source_rows if not bool(row.get("brep_valid"))]
        if len(source_rows) != EXPECTED_CADS - EXPECTED_BASELINE_VALID:
            raise RuntimeError("frozen historical-invalid pilot is not 16 CADs")
        historical = {str(row["cad_id"]): False for row in source_rows}
    payload = build_selector_run_payload(
        args=args,
        full_rows=full_rows,
        selected_rows=source_rows,
    )
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    with output_root_writer_lock(args.output_dir, command=command):
        run_record = bind_run_manifest(args.output_dir, payload)
        try:
            verify_selector_payload_integrity(
                payload, args=args, selected_rows=source_rows
            )
            matrix_path = args.output_dir / "assembly_repair_matrix.jsonl"
            candidate_path = args.output_dir / "assembly_selector_candidates.jsonl"
            rows = (
                read_jsonl(matrix_path, recover_truncated_tail=True)
                if matrix_path.is_file()
                else []
            )
            candidate_entries = (
                read_jsonl(candidate_path, recover_truncated_tail=True)
                if candidate_path.is_file()
                else []
            )
            validate_candidate_ledger(candidate_entries, source_rows)
            validate_existing_selector_rows(
                rows, source_rows, output_dir=args.output_dir
            )
            validate_final_candidate_bindings(rows, candidate_entries)
            completed = {str(row["cad_id"]) for row in rows}
            existing_candidates = candidate_result_map(candidate_entries)
            for source in source_rows:
                cad_id = str(source["cad_id"])
                if cad_id in completed:
                    continue

                def run_candidate(profile: RepairProfile) -> Mapping[str, Any]:
                    key = (cad_id, profile.name)
                    if key in existing_candidates:
                        return existing_candidates[key]
                    result = run_one_isolated(
                        source,
                        profile,
                        calibration_manifest=args.calibration_manifest,
                        output_dir=args.output_dir,
                        breparg_root=args.breparg_root,
                        joint_iterations=args.joint_iterations,
                        timeout_seconds=args.worker_timeout_seconds,
                        selector_geometry_gate=(
                            profile.name in FALLBACK_PROFILE_NAMES
                        ),
                        expected_source_binding=payload[
                            "selected_source_pickles"
                        ][cad_id],
                    )
                    validate_attempt_row(result, source, profile)
                    entry = candidate_ledger_entry(source, profile, result)
                    append_jsonl(candidate_path, entry)
                    candidate_entries.append(entry)
                    existing_candidates[key] = result
                    return result

                def candidate_gate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
                    return geometry_gate_for_candidate(candidate)

                selected, selection = route_selector_candidates(
                    run_candidate=run_candidate,
                    geometry_gate_for=candidate_gate,
                )
                row = final_selector_row(
                    source, selected, selection, output_dir=args.output_dir
                )
                validate_selector_row(row, source)
                validate_selector_step(row, output_dir=args.output_dir)
                append_jsonl(matrix_path, row)
                rows.append(row)
                completed.add(cad_id)
                print(
                    json.dumps(
                        {
                            "cad_id": cad_id,
                            "status": row["status"],
                            "strict_brep_valid": row["strict_brep_valid"],
                            "selected_profile": row["selection"]["selected_profile"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            summary = summarize_selector(
                rows,
                historical,
                historical_invalid_only=bool(args.historical_invalid_only),
            )
            summary_path = args.output_dir / "assembly_repair_summary.json"
            atomic_json(summary_path, summary)
            run_record.update(
                status=(
                    "COMPLETED_PARTIAL"
                    if args.historical_invalid_only
                    else "COMPLETED"
                ),
                attempts=len(rows),
                candidate_attempts=int(
                        summary["selector"]["candidate_attempt_count"]
                ),
                candidate_manifest_sha256=sha256_file(candidate_path),
                final_matrix_sha256=sha256_file(matrix_path),
                summary_sha256=sha256_file(summary_path),
            )
            validate_candidate_ledger(candidate_entries, source_rows)
            validate_final_candidate_bindings(rows, candidate_entries)
            verify_selector_payload_integrity(
                payload, args=args, selected_rows=source_rows
            )
            if len(candidate_entries) != run_record["candidate_attempts"]:
                raise RuntimeError(
                    "selector candidate ledger count differs from final summary"
                )
            atomic_json(args.output_dir / RUN_MANIFEST_NAME, run_record)
        except Exception as exc:
            run_record.update(
                status="FAILED",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            atomic_json(args.output_dir / RUN_MANIFEST_NAME, run_record)
            raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["selector_protocol_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
