from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.assembly_repair import parse_profiles
from tools.assembly_selector_geometry import (
    GEOMETRY_GATE_SCHEMA,
    geometry_topology_gate,
    input_geometry_signature,
    validate_accepted_geometry_gate,
)
from tools.run_assembly_repair_matrix import (
    SCHEMA as CANDIDATE_RESULT_SCHEMA,
    sha256_file,
)
from tools.run_assembly_repair_selector import (
    CANDIDATE_SCHEMA,
    EXPECTED_BOTH_VALID_RESTORATIONS,
    EXPECTED_FALLBACK_ACCEPTED_IDS,
    FALLBACK_PROFILE_NAMES,
    INTERPOLATE_PROFILE_NAME,
    NEAR_VERTEX_PROFILE_NAME,
    PRIMARY_PROFILE_NAME,
    SELECTOR_PROFILE_NAME,
    SELECTOR_SCHEMA,
    SURFACE_PRECISION_PROFILE_NAME,
    canonical_result_sha256,
    candidate_ledger_entry,
    compact_candidate,
    fallback_eligible,
    geometry_gate_for_candidate,
    route_selector_candidates,
    selected_source_pickle_bindings,
    selector_source_hashes,
    summarize_selector,
    validate_candidate_ledger,
    validate_final_candidate_bindings,
    validate_selector_row,
    verify_selector_payload_integrity,
)


def _complete_accepted_geometry_gate():
    surfaces = np.asarray(
        [
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ]
        ]
    )
    edges = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]
    )
    signature = input_geometry_signature(
        surfaces,
        edges,
        [[0, 1]],
        np.asarray([[0, 1], [1, 2]]),
    )
    input_projection = {
        "sample_count": signature["projection_sample_count"],
        "projected_sample_count": signature["projection_sample_count"],
        "projection_failure_count": 0,
        "rms_normalized": 0.0,
        "max_normalized": 0.0,
    }
    candidate_sample_count = signature["edge_count"] * 8
    gate = geometry_topology_gate(
        signature,
        {
            "face_count": signature["face_count"],
            "edge_count": signature["edge_count"],
            "vertex_count": signature["vertex_count"],
            "face_edge_occurrences": signature["face_edge_occurrences"],
            "face_edge_incidence_counts": list(
                signature["face_edge_incidence_counts"]
            ),
            "edge_face_incidence_counts": list(
                signature["edge_face_incidence_counts"]
            ),
            "vertex_edge_incidence_counts": list(
                signature["vertex_edge_incidence_counts"]
            ),
            "solid_count": 1,
            "free_edges": 0,
            "wire_order_failures": 0,
            "wire_self_intersections": 0,
            "bbox": list(signature["bbox"]),
            "edge_length": signature["edge_polyline_length"],
            "projectable_edge_count": signature["edge_count"],
            "unprojectable_edge_count": 0,
            "input_to_candidate_projection": input_projection,
            "candidate_to_input_projection": {
                "sample_count": candidate_sample_count,
                "projected_sample_count": candidate_sample_count,
                "projection_failure_count": 0,
                "rms_normalized": 0.0,
                "max_normalized": 0.0,
            },
            "candidate_curve_sampling": {
                "requested_sample_count": candidate_sample_count,
                "successful_sample_count": candidate_sample_count,
                "sampling_failure_count": 0,
            },
        },
    )
    assert gate["accepted"] is True
    return gate


def _weak_accepted_geometry_gate():
    return {
        "schema": GEOMETRY_GATE_SCHEMA,
        "accepted": True,
    }


def _candidate(
    profile_name,
    *,
    strict=False,
    native=None,
    step_saved=True,
    status=None,
    gate=True,
):
    if native is None:
        native = strict
    both = bool(strict and native)
    profile = parse_profiles([profile_name])[0]
    row = {
        "profile": profile.name,
        "switches": list(profile.switches),
        "status": status or ("both_valid" if both else "step_invalid"),
        "step_saved": bool(step_saved),
        "native_brep_valid": bool(native),
        "strict_brep_valid": bool(strict),
        "both_valid": both,
        "step_bytes": 123 if step_saved else None,
        "step_sha256": "a" * 64 if step_saved else None,
        "validity_components": {
            "wire_order_failures": 0,
            "wire_self_intersections": 0,
            "free_edges": 0,
            "solid_count": 1,
        },
    }
    if both and profile_name != PRIMARY_PROFILE_NAME:
        row["selector_geometry_topology_gate"] = _complete_accepted_geometry_gate()
        if not gate:
            row["selector_geometry_topology_gate"]["accepted"] = False
            row["selector_geometry_topology_gate"]["rejection_reasons"] = [
                "geometry_gate:synthetic"
            ]
    return row


def test_primary_strict_fast_path_never_runs_fallback():
    calls = []

    def run(profile):
        calls.append(profile.name)
        if profile.name != PRIMARY_PROFILE_NAME:
            raise AssertionError("fallback must not run")
        return _candidate(profile.name, strict=True, native=False)

    selected, selection = route_selector_candidates(
        run_candidate=run,
        geometry_gate_for=geometry_gate_for_candidate,
    )

    assert calls == [PRIMARY_PROFILE_NAME]
    assert selected["strict_brep_valid"] is True
    assert selected["both_valid"] is False
    assert selection["selected_profile"] == PRIMARY_PROFILE_NAME
    assert selection["fallback_accepted"] is False


def test_near_vertex_pass_stops_before_interpolation():
    candidates = {
        PRIMARY_PROFILE_NAME: _candidate(PRIMARY_PROFILE_NAME),
        NEAR_VERTEX_PROFILE_NAME: _candidate(
            NEAR_VERTEX_PROFILE_NAME, strict=True
        ),
    }
    calls = []

    selected, selection = route_selector_candidates(
        run_candidate=lambda profile: (
            calls.append(profile.name) or candidates[profile.name]
        ),
        geometry_gate_for=geometry_gate_for_candidate,
    )

    assert calls == [PRIMARY_PROFILE_NAME, NEAR_VERTEX_PROFILE_NAME]
    assert selected["profile"] == NEAR_VERTEX_PROFILE_NAME
    assert selection["selected_reason"] == "fallback_native_strict_geometry_passed"


def test_strict_only_near_is_rejected_before_interpolation_passes():
    candidates = {
        PRIMARY_PROFILE_NAME: _candidate(PRIMARY_PROFILE_NAME),
        NEAR_VERTEX_PROFILE_NAME: _candidate(
            NEAR_VERTEX_PROFILE_NAME, strict=True, native=False
        ),
        INTERPOLATE_PROFILE_NAME: _candidate(
            INTERPOLATE_PROFILE_NAME, strict=True
        ),
    }

    selected, selection = route_selector_candidates(
        run_candidate=lambda profile: candidates[profile.name],
        geometry_gate_for=geometry_gate_for_candidate,
    )

    assert selected["profile"] == INTERPOLATE_PROFILE_NAME
    near = selection["candidates"][1]
    assert "candidate_native_invalid" in near["rejection_reasons"]
    assert "geometry_topology_gate" not in near


def test_surface_precision_runs_only_after_earlier_fallbacks_fail():
    candidates = {
        PRIMARY_PROFILE_NAME: _candidate(PRIMARY_PROFILE_NAME),
        NEAR_VERTEX_PROFILE_NAME: _candidate(NEAR_VERTEX_PROFILE_NAME),
        INTERPOLATE_PROFILE_NAME: _candidate(INTERPOLATE_PROFILE_NAME),
        SURFACE_PRECISION_PROFILE_NAME: _candidate(
            SURFACE_PRECISION_PROFILE_NAME, strict=True
        ),
    }
    calls = []

    selected, selection = route_selector_candidates(
        run_candidate=lambda profile: (
            calls.append(profile.name) or candidates[profile.name]
        ),
        geometry_gate_for=geometry_gate_for_candidate,
    )

    assert calls == [PRIMARY_PROFILE_NAME, *FALLBACK_PROFILE_NAMES]
    assert selected["profile"] == SURFACE_PRECISION_PROFILE_NAME
    assert selection["fallback_accepted"] is True


def test_gate_failure_retains_primary_when_no_fallback_passes():
    candidates = {
        PRIMARY_PROFILE_NAME: _candidate(PRIMARY_PROFILE_NAME),
        NEAR_VERTEX_PROFILE_NAME: _candidate(
            NEAR_VERTEX_PROFILE_NAME, strict=True, gate=False
        ),
        INTERPOLATE_PROFILE_NAME: _candidate(INTERPOLATE_PROFILE_NAME),
        SURFACE_PRECISION_PROFILE_NAME: _candidate(
            SURFACE_PRECISION_PROFILE_NAME
        ),
    }

    selected, selection = route_selector_candidates(
        run_candidate=lambda profile: candidates[profile.name],
        geometry_gate_for=geometry_gate_for_candidate,
    )

    assert selected["profile"] == PRIMARY_PROFILE_NAME
    assert selection["selected_reason"] == "no_fallback_passed"
    assert selection["attempted_profiles"] == [
        PRIMARY_PROFILE_NAME,
        *FALLBACK_PROFILE_NAMES,
    ]


def test_fallback_eligibility_requires_every_validity_and_gate_condition():
    healthy = _candidate(NEAR_VERTEX_PROFILE_NAME, strict=True)
    gate = healthy["selector_geometry_topology_gate"]
    assert fallback_eligible(healthy, gate) == (True, [])

    bad_status = dict(healthy, status="step_invalid")
    eligible, reasons = fallback_eligible(bad_status, gate)
    assert eligible is False
    assert "candidate_status_not_both_valid" in reasons

    missing_step = dict(healthy, step_saved=False)
    assert fallback_eligible(missing_step, gate)[0] is False
    assert fallback_eligible(healthy, dict(gate, accepted=False))[0] is False


def test_complete_generated_v2_gate_passes_selector_normalization():
    gate = _complete_accepted_geometry_gate()

    assert validate_accepted_geometry_gate(gate) == (True, [])
    assert geometry_gate_for_candidate(
        {"selector_geometry_topology_gate": gate}
    ) == gate


def test_weak_accepted_gate_is_rejected_by_route():
    candidates = {
        PRIMARY_PROFILE_NAME: _candidate(PRIMARY_PROFILE_NAME),
        NEAR_VERTEX_PROFILE_NAME: _candidate(
            NEAR_VERTEX_PROFILE_NAME, strict=True
        ),
        INTERPOLATE_PROFILE_NAME: _candidate(INTERPOLATE_PROFILE_NAME),
        SURFACE_PRECISION_PROFILE_NAME: _candidate(
            SURFACE_PRECISION_PROFILE_NAME
        ),
    }
    candidates[NEAR_VERTEX_PROFILE_NAME][
        "selector_geometry_topology_gate"
    ] = _weak_accepted_geometry_gate()

    selected, selection = route_selector_candidates(
        run_candidate=lambda profile: candidates[profile.name],
        geometry_gate_for=geometry_gate_for_candidate,
    )

    assert selected["profile"] == PRIMARY_PROFILE_NAME
    near_gate = selection["candidates"][1]["geometry_topology_gate"]
    assert near_gate["accepted"] is False
    assert "accepted_gate_checks_missing" in near_gate["rejection_reasons"]


def _selection(primary_strict, selected_profile, selected_valid, *, worker_failure=False):
    primary = _candidate(
        PRIMARY_PROFILE_NAME,
        strict=primary_strict,
        native=primary_strict,
    )
    candidates = [primary]
    if selected_profile == NEAR_VERTEX_PROFILE_NAME:
        candidates.append(_candidate(NEAR_VERTEX_PROFILE_NAME, strict=selected_valid))
    elif selected_profile == INTERPOLATE_PROFILE_NAME:
        near = _candidate(NEAR_VERTEX_PROFILE_NAME)
        if worker_failure:
            near["status"] = "worker_process_exit"
        candidates.extend(
            [near, _candidate(INTERPOLATE_PROFILE_NAME, strict=selected_valid)]
        )
    elif selected_profile == SURFACE_PRECISION_PROFILE_NAME:
        near = _candidate(NEAR_VERTEX_PROFILE_NAME)
        interpolate = _candidate(INTERPOLATE_PROFILE_NAME)
        if worker_failure:
            near["status"] = "worker_process_exit"
        candidates.extend(
            [
                near,
                interpolate,
                _candidate(SURFACE_PRECISION_PROFILE_NAME, strict=selected_valid),
            ]
        )
    return {
        "schema": SELECTOR_SCHEMA,
        "selected_profile": selected_profile,
        "fallback_accepted": selected_profile != PRIMARY_PROFILE_NAME,
        "candidates": candidates,
    }


def _summary_rows():
    controls = [f"control-{index:03d}" for index in range(84)]
    remaining = [f"remaining-{index:02d}" for index in range(10)]
    historical = OrderedDict(
        [(cad_id, True) for cad_id in controls]
        + [(cad_id, False) for cad_id in sorted(EXPECTED_BOTH_VALID_RESTORATIONS)]
        + [(cad_id, False) for cad_id in remaining]
    )
    rows = []
    for cad_id, baseline in historical.items():
        restored = cad_id in EXPECTED_BOTH_VALID_RESTORATIONS
        if cad_id in EXPECTED_FALLBACK_ACCEPTED_IDS:
            selected_profile = (
                NEAR_VERTEX_PROFILE_NAME
                if cad_id.startswith("00000444")
                else (
                    SURFACE_PRECISION_PROFILE_NAME
                    if cad_id.startswith("00051587")
                    else INTERPOLATE_PROFILE_NAME
                )
            )
            selection = _selection(False, selected_profile, True)
            valid = True
        elif restored:
            selected_profile = PRIMARY_PROFILE_NAME
            selection = _selection(True, selected_profile, True)
            valid = True
        else:
            selected_profile = PRIMARY_PROFILE_NAME
            selection = _selection(baseline, selected_profile, baseline)
            valid = baseline
        rows.append(
            {
                "cad_id": cad_id,
                "profile": SELECTOR_PROFILE_NAME,
                "status": "both_valid" if valid else "step_invalid",
                "step_saved": True,
                "native_brep_valid": valid,
                "strict_brep_valid": valid,
                "both_valid": valid,
                "selection": selection,
            }
        )
    return rows, historical


def test_selector_summary_passes_only_the_registered_90_87_protocol():
    rows, historical = _summary_rows()

    summary = summarize_selector(
        rows, historical, historical_invalid_only=False
    )

    assert summary["selector_protocol_passed"] is True
    assert summary["profiles"][0]["strict_valid"] == 90
    assert summary["profiles"][0]["both_valid"] == 90
    assert summary["assembly_release_gate_passed"] is False
    assert summary["gate_passed"] is False

    rows[0]["strict_brep_valid"] = False
    rows[0]["both_valid"] = False
    rows[0]["native_brep_valid"] = False
    regressed = summarize_selector(
        rows, historical, historical_invalid_only=False
    )
    assert regressed["selector_protocol_passed"] is False
    assert regressed["profiles"][0]["regressed_cad_ids"] == ["control-000"]


def _ledger_result(source, profile_name, *, strict):
    profile = parse_profiles([profile_name])[0]
    both = bool(strict)
    result = {
        "schema": CANDIDATE_RESULT_SCHEMA,
        "cad_id": source["cad_id"],
        "parent_id": source["parent_id"],
        "profile": profile.name,
        "switches": list(profile.switches),
        "historical_strict_valid": source["brep_valid"],
        "source_path": source["source_path"],
        "status": "both_valid" if both else "step_invalid",
        "step_saved": True,
        "native_brep_valid": both,
        "strict_brep_valid": both,
        "both_valid": both,
    }
    if profile_name != PRIMARY_PROFILE_NAME and both:
        result["selector_geometry_topology_gate"] = _complete_accepted_geometry_gate()
    return result


def test_candidate_ledger_rejects_fallback_after_strict_primary():
    source = {
        "cad_id": "cad-1",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": False,
    }
    primary = parse_profiles([PRIMARY_PROFILE_NAME])[0]
    near = parse_profiles([NEAR_VERTEX_PROFILE_NAME])[0]
    entries = [
        candidate_ledger_entry(
            source, primary, _ledger_result(source, primary.name, strict=True)
        ),
        candidate_ledger_entry(
            source, near, _ledger_result(source, near.name, strict=True)
        ),
    ]

    with pytest.raises(RuntimeError, match="fallback after strict primary"):
        validate_candidate_ledger(entries, [source])


def test_candidate_ledger_rejects_surface_after_eligible_interpolation():
    source = {
        "cad_id": "cad-1",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": False,
    }
    primary, near, interpolate, surface_precision = parse_profiles(
        [PRIMARY_PROFILE_NAME, *FALLBACK_PROFILE_NAMES]
    )
    entries = [
        candidate_ledger_entry(
            source, primary, _ledger_result(source, primary.name, strict=False)
        ),
        candidate_ledger_entry(
            source, near, _ledger_result(source, near.name, strict=False)
        ),
        candidate_ledger_entry(
            source,
            interpolate,
            _ledger_result(source, interpolate.name, strict=True),
        ),
        candidate_ledger_entry(
            source,
            surface_precision,
            _ledger_result(source, surface_precision.name, strict=True),
        ),
    ]

    with pytest.raises(RuntimeError, match="later fallback after an accepted"):
        validate_candidate_ledger(entries, [source])


def test_candidate_ledger_rejects_tampered_result_after_hash_binding():
    source = {
        "cad_id": "cad-1",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": False,
    }
    primary = parse_profiles([PRIMARY_PROFILE_NAME])[0]
    entry = candidate_ledger_entry(
        source, primary, _ledger_result(source, primary.name, strict=False)
    )
    entry["result"]["status"] = "both_valid"

    with pytest.raises(RuntimeError, match="result hash mismatch"):
        validate_candidate_ledger([entry], [source])


def test_candidate_ledger_rejects_weak_accepted_gate_with_matching_hash():
    source = {
        "cad_id": "cad-1",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": False,
    }
    primary, near = parse_profiles(
        [PRIMARY_PROFILE_NAME, NEAR_VERTEX_PROFILE_NAME]
    )
    near_result = _ledger_result(source, near.name, strict=True)
    near_result["selector_geometry_topology_gate"] = _weak_accepted_geometry_gate()
    entries = [
        candidate_ledger_entry(
            source, primary, _ledger_result(source, primary.name, strict=False)
        ),
        # Bind the deliberately weak gate into the hash to prove validation is
        # semantic, not merely a stale-ledger fingerprint check.
        candidate_ledger_entry(source, near, near_result),
    ]

    with pytest.raises(RuntimeError, match="invalid accepted geometry gate"):
        validate_candidate_ledger(entries, [source])


def test_selector_row_rejects_weak_selected_fallback_gate():
    source = {
        "cad_id": "cad-1",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": False,
    }
    candidates = {
        PRIMARY_PROFILE_NAME: _candidate(PRIMARY_PROFILE_NAME),
        NEAR_VERTEX_PROFILE_NAME: _candidate(
            NEAR_VERTEX_PROFILE_NAME, strict=True
        ),
    }
    candidates[NEAR_VERTEX_PROFILE_NAME][
        "selector_geometry_topology_gate"
    ] = _weak_accepted_geometry_gate()
    selected, selection = route_selector_candidates(
        run_candidate=lambda profile: candidates[profile.name],
        # This models a forged persisted selection record that bypassed the
        # normal worker gate normalizer.  Row validation must still fail closed.
        geometry_gate_for=lambda row: row["selector_geometry_topology_gate"],
    )
    row = {
        "schema": SELECTOR_SCHEMA,
        "cad_id": source["cad_id"],
        "parent_id": source["parent_id"],
        "profile": SELECTOR_PROFILE_NAME,
        "switches": ["failure_triggered_selector"],
        "historical_strict_valid": source["brep_valid"],
        "source_path": source["source_path"],
        "status": selected["status"],
        "step_saved": selected["step_saved"],
        "native_brep_valid": selected["native_brep_valid"],
        "strict_brep_valid": selected["strict_brep_valid"],
        "both_valid": selected["both_valid"],
        "selection": selection,
    }

    with pytest.raises(ValueError, match="invalid geometry gate"):
        validate_selector_row(row, source)


def test_selector_row_rejects_skipping_eligible_interpolation():
    source = {
        "cad_id": "cad-1",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": False,
    }
    candidates = {
        PRIMARY_PROFILE_NAME: _candidate(PRIMARY_PROFILE_NAME),
        NEAR_VERTEX_PROFILE_NAME: _candidate(NEAR_VERTEX_PROFILE_NAME),
        INTERPOLATE_PROFILE_NAME: _candidate(INTERPOLATE_PROFILE_NAME),
        SURFACE_PRECISION_PROFILE_NAME: _candidate(
            SURFACE_PRECISION_PROFILE_NAME, strict=True
        ),
    }
    selected, selection = route_selector_candidates(
        run_candidate=lambda profile: candidates[profile.name],
        geometry_gate_for=geometry_gate_for_candidate,
    )
    skipped = selection["candidates"][2]
    skipped.update(
        status="both_valid",
        step_saved=True,
        native_brep_valid=True,
        strict_brep_valid=True,
        both_valid=True,
        geometry_topology_gate=_complete_accepted_geometry_gate(),
        rejection_reasons=[],
    )
    row = {
        "schema": SELECTOR_SCHEMA,
        "cad_id": source["cad_id"],
        "parent_id": source["parent_id"],
        "profile": SELECTOR_PROFILE_NAME,
        "switches": ["failure_triggered_selector"],
        "historical_strict_valid": source["brep_valid"],
        "source_path": source["source_path"],
        "status": selected["status"],
        "step_saved": selected["step_saved"],
        "native_brep_valid": selected["native_brep_valid"],
        "strict_brep_valid": selected["strict_brep_valid"],
        "both_valid": selected["both_valid"],
        "selection": selection,
    }

    with pytest.raises(ValueError, match="first eligible candidate"):
        validate_selector_row(row, source)


def test_final_bindings_reject_skipping_eligible_interpolation():
    source = {
        "cad_id": "cad-1",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": False,
    }
    primary, near, interpolate, surface_precision = parse_profiles(
        [PRIMARY_PROFILE_NAME, *FALLBACK_PROFILE_NAMES]
    )
    raw = {
        profile.name: _ledger_result(source, profile.name, strict=False)
        for profile in (primary, near, interpolate)
    }
    raw[surface_precision.name] = _ledger_result(
        source, surface_precision.name, strict=True
    )
    _, selection = route_selector_candidates(
        run_candidate=lambda profile: raw[profile.name],
        geometry_gate_for=geometry_gate_for_candidate,
    )
    interpolation = raw[interpolate.name]
    interpolation.update(
        status="both_valid",
        native_brep_valid=True,
        strict_brep_valid=True,
        both_valid=True,
        selector_geometry_topology_gate=_complete_accepted_geometry_gate(),
    )
    selection["candidates"][2] = compact_candidate(
        interpolation,
        geometry_gate=interpolation["selector_geometry_topology_gate"],
        rejection_reasons=[],
    )
    entries = []
    for order, profile in enumerate(
        (primary, near, interpolate, surface_precision)
    ):
        result = raw[profile.name]
        entries.append(
            {
                "schema": CANDIDATE_SCHEMA,
                "cad_id": source["cad_id"],
                "profile": profile.name,
                "attempt_order": order,
                "result": result,
                "result_sha256": canonical_result_sha256(result),
            }
        )
    selected = raw[surface_precision.name]
    row = {
        "cad_id": source["cad_id"],
        "status": selected["status"],
        "step_saved": selected["step_saved"],
        "native_brep_valid": selected["native_brep_valid"],
        "strict_brep_valid": selected["strict_brep_valid"],
        "both_valid": selected["both_valid"],
        "selection": selection,
    }

    with pytest.raises(RuntimeError, match="first eligible candidate"):
        validate_final_candidate_bindings([row], entries)


def test_final_selector_evidence_must_match_candidate_ledger_fingerprint():
    source = {
        "cad_id": "cad-1",
        "parent_id": "parent",
        "source_path": "input.pkl",
        "brep_valid": True,
    }
    primary = parse_profiles([PRIMARY_PROFILE_NAME])[0]
    result = _ledger_result(source, primary.name, strict=True)
    entry = candidate_ledger_entry(source, primary, result)
    _, selection = route_selector_candidates(
        run_candidate=lambda profile: result,
        geometry_gate_for=geometry_gate_for_candidate,
    )
    row = {
        "cad_id": source["cad_id"],
        "status": result["status"],
        "step_saved": result["step_saved"],
        "native_brep_valid": result["native_brep_valid"],
        "strict_brep_valid": result["strict_brep_valid"],
        "both_valid": result["both_valid"],
        "selection": selection,
    }
    validate_final_candidate_bindings([row], [entry])

    row["selection"]["candidates"][0]["candidate_result_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="differs from ledger"):
        validate_final_candidate_bindings([row], [entry])


def test_selected_source_pickle_bindings_do_not_serialize_paths(tmp_path):
    first = tmp_path / "first.pkl"
    second = tmp_path / "second.pkl"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    rows = [
        {"cad_id": "cad-1", "source_path": str(first)},
        {"cad_id": "cad-2", "source_path": str(second)},
    ]

    bindings = selected_source_pickle_bindings(rows)

    assert set(bindings) == {"cad-1", "cad-2"}
    assert all("path" not in binding for binding in bindings.values())
    assert bindings["cad-1"]["bytes"] == 5


def test_selector_payload_integrity_rechecks_sources_runtime_and_pickles(tmp_path):
    source = tmp_path / "source.pkl"
    source.write_bytes(b"source-v1")
    breparg_root = tmp_path / "BrepARG"
    breparg_root.mkdir()
    utils_path = breparg_root / "utils.py"
    utils_path.write_text("# runtime-v1\n", encoding="utf-8")
    rows = [{"cad_id": "cad-1", "source_path": str(source)}]
    repo_root = Path(__file__).resolve().parents[1]
    payload = {
        "repository": {"source_sha256": selector_source_hashes(repo_root)},
        "breparg_runtime": {"utils_sha256": sha256_file(utils_path)},
        "selected_source_pickles": selected_source_pickle_bindings(rows),
    }
    args = SimpleNamespace(breparg_root=breparg_root)

    verify_selector_payload_integrity(payload, args=args, selected_rows=rows)

    source.write_bytes(b"source-v2")
    with pytest.raises(RuntimeError, match="source pickle bindings changed"):
        verify_selector_payload_integrity(payload, args=args, selected_rows=rows)

    source.write_bytes(b"source-v1")
    utils_path.write_text("# runtime-v2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="utils hash changed"):
        verify_selector_payload_integrity(payload, args=args, selected_rows=rows)

    bad_payload = dict(payload)
    bad_payload["repository"] = {"source_sha256": {}}
    with pytest.raises(RuntimeError, match="execution source hashes changed"):
        verify_selector_payload_integrity(
            bad_payload, args=args, selected_rows=rows
        )
