import json
import pickle
from pathlib import Path

import numpy as np
import pytest


from tools.run_assembly_calibration_oracle import (
    cpu_joint_optimize,
    edge_patches_from_model_output,
    evaluate_cad_arm,
    evaluate_cad_model_arm,
    select_validation_cads,
    summarize_cad_reconstruction_error,
)
from tools.summarize_assembly_calibration import (
    render_calibration_png,
    summarize_calibration,
)


def _write_verified_protocol(root: Path, val_paths: list[Path]) -> Path:
    protocol = root / "protocol"
    protocol.mkdir()
    (protocol / "protocol_summary.json").write_text(
        json.dumps(
            {
                "status": "VERIFIED",
                "protocol_sha256": "protocol-hash",
                "parent_overlap_counts": {
                    "train__val": 0,
                    "train__test": 0,
                    "val__test": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    with (protocol / "split.pkl").open("wb") as handle:
        pickle.dump({"train": [], "val": [str(path) for path in val_paths], "test": []}, handle)
    return protocol


def test_select_validation_cads_is_deterministic_and_parent_unique(tmp_path):
    paths = []
    for index, parent in enumerate(("a" * 24, "b" * 24, "c" * 24)):
        path = tmp_path / f"{index:08d}_{parent}_step_000.pkl"
        path.write_bytes(b"pickle")
        paths.append(path)
    protocol = _write_verified_protocol(tmp_path, paths)

    first = select_validation_cads(protocol, max_cads=2, seed=17)
    second = select_validation_cads(protocol, max_cads=2, seed=17)

    assert first == second
    assert len(first) == 2
    assert len({row["parent_id"] for row in first}) == 2
    assert all(Path(row["path"]).is_file() for row in first)


def test_select_validation_cads_rejects_unverified_or_overlapping_protocol(tmp_path):
    parent = "a" * 24
    path = tmp_path / f"00000000_{parent}_step_000.pkl"
    path.write_bytes(b"pickle")
    protocol = _write_verified_protocol(tmp_path, [path])
    summary_path = protocol / "protocol_summary.json"

    summary = json.loads(summary_path.read_text())
    summary["parent_overlap_counts"]["train__val"] = 1
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="parent overlap"):
        select_validation_cads(protocol, max_cads=1, seed=0)

    summary["parent_overlap_counts"]["train__val"] = 0
    summary["status"] = "FAILED"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="VERIFIED"):
        select_validation_cads(protocol, max_cads=1, seed=0)


def test_edge_model_output_matches_breparg_width_mean_semantics():
    edges = np.arange(2 * 32 * 3, dtype=np.float32).reshape(2, 32, 3)
    tiled = np.tile(edges[:, :, None, :], (1, 1, 32, 1))

    recovered = edge_patches_from_model_output(tiled)

    np.testing.assert_array_equal(recovered, edges)
    tiled[0, 0, 1, 0] += 1.0
    recovered = edge_patches_from_model_output(tiled)
    assert recovered[0, 0, 0] == pytest.approx(edges[0, 0, 0] + 1.0 / 32.0)


def test_cpu_joint_can_delegate_closed_edge_scaling_to_production_rule():
    surface = np.zeros((1, 32, 32, 3), dtype=np.float32)
    edge = np.zeros((1, 32, 3), dtype=np.float32)
    edge[0, :, 0] = np.linspace(0.0, 1.0, 32)
    observed = {}

    def resolver(target_scale, ncs_scale, curve, bbox):
        observed.update(
            target_scale=target_scale,
            ncs_scale=ncs_scale,
            curve_shape=curve.shape,
            bbox=np.asarray(bbox).tolist(),
        )
        return 1.0

    surfaces, edges = cpu_joint_optimize(
        surface,
        edge,
        np.asarray([[0, 0, 0, 2, 2, 2]], dtype=np.float32),
        np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.asarray([[0, 1]], dtype=np.int64),
        [[0]],
        iterations=0,
        edge_bboxes=np.asarray([[0, 0, 0, 1, 0, 0]], dtype=np.float32),
        edge_scale_resolver=resolver,
    )

    assert surfaces.shape == (1, 32, 32, 3)
    assert edges.shape == (1, 32, 3)
    assert observed == {
        "target_scale": pytest.approx(1.0),
        "ncs_scale": pytest.approx(1.0),
        "curve_shape": (32, 3),
        "bbox": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }


def test_cad_error_summary_uses_curved_surfaces_and_counts_nonfinite():
    planar = np.zeros((32, 32, 3), dtype=np.float32)
    planar[..., 0] = np.linspace(-1, 1, 32)[:, None]
    planar[..., 1] = np.linspace(-1, 1, 32)[None, :]
    curved = planar.copy()
    curved[..., 2] = 0.2 * (curved[..., 0] ** 2 + curved[..., 1] ** 2)
    target_surfaces = np.stack([planar, curved])
    reconstructed = target_surfaces.copy()
    reconstructed[0] += 0.5
    reconstructed[1] += 0.1
    target_edges = np.zeros((1, 32, 3), dtype=np.float32)
    reconstructed_edges = target_edges + 0.2

    summary = summarize_cad_reconstruction_error(
        target_surfaces,
        reconstructed,
        target_edges,
        reconstructed_edges,
    )

    assert summary["curved_surface_count"] == 1
    assert summary["planar_surface_count"] == 1
    assert summary["curved_mse"] == pytest.approx(0.01)
    assert summary["planar_mse"] == pytest.approx(0.25)
    assert summary["edge_mse"] == pytest.approx(0.04)
    assert summary["nonfinite_patches"] == 0


def _rows(valid_by_error, *, original_valid_rate=1.0):
    rows = []
    for index, (mse, valid) in enumerate(valid_by_error):
        rows.append(
            {
                "cad_id": f"cad-{index}",
                "arm": "continuous_bypass_64d",
                "curved_mse": mse,
                "brep_valid": valid,
                "step_saved": valid,
                "status": "saved" if valid else "brep_invalid",
            }
        )
        rows.append(
            {
                "cad_id": f"cad-{index}",
                "arm": "original",
                "curved_mse": 0.0,
                "brep_valid": index < round(len(valid_by_error) * original_valid_rate),
                "step_saved": True,
                "status": "saved",
            }
        )
    return rows


def test_calibration_decision_accepts_current_error_when_validity_is_high():
    rows = _rows([(1e-4 + index * 1e-5, index < 18) for index in range(20)])
    summary = summarize_calibration(rows, acceptable_valid_rate=0.8, min_cads=20)
    assert summary["decision"]["status"] == "CURRENT_ERROR_ACCEPTABLE"
    assert summary["decision"]["advance_to_vq_300k"] is True
    assert summary["decision"]["advance_to_ar"] is False


def test_calibration_decision_identifies_error_correlated_failure():
    values = [(1e-5 * (index + 1), index < 5) for index in range(20)]
    summary = summarize_calibration(
        _rows(values),
        acceptable_valid_rate=0.8,
        min_cads=20,
        strong_association=0.35,
    )
    assert summary["decision"]["status"] == "REPRESENTATION_ERROR_CORRELATED"
    assert summary["decision"]["decoder_work_authorized"] is True
    assert summary["decision"]["advance_to_ar"] is False


def test_calibration_decision_identifies_assembly_dominated_failure():
    values = [(1e-4, index % 4 == 0) for index in range(20)]
    summary = summarize_calibration(
        _rows(values),
        acceptable_valid_rate=0.8,
        min_cads=20,
        strong_association=0.35,
    )
    assert summary["decision"]["status"] == "ASSEMBLY_DOMINATED"
    assert summary["decision"]["assembly_repair_required"] is True
    assert summary["decision"]["advance_to_ar"] is False


def test_calibration_fails_closed_when_original_control_is_poor():
    values = [(1e-4, True) for _ in range(20)]
    summary = summarize_calibration(
        _rows(values, original_valid_rate=0.5),
        acceptable_valid_rate=0.8,
        min_cads=20,
    )
    assert summary["decision"]["status"] == "ASSEMBLY_CONTROL_FAILED"
    assert summary["decision"]["advance_to_ar"] is False


def test_calibration_bins_count_each_attempt_once_with_duplicate_mse():
    values = [(1e-4, index % 2 == 0) for index in range(20)]
    summary = summarize_calibration(_rows(values), min_cads=20)
    bins = summary["arms"]["continuous_bypass_64d"]["curved_mse_bins"]
    assert sum(bucket["attempts"] for bucket in bins) == 20


def test_cad_arm_records_assembly_failure_instead_of_raising(tmp_path):
    surface = np.zeros((1, 32, 32, 3), dtype=np.float32)
    edge = np.zeros((1, 32, 3), dtype=np.float32)
    parsed = {"surf_ncs": surface, "edge_ncs": edge}

    def failing_assembler(*args, **kwargs):
        raise ValueError("broken topology")

    row = evaluate_cad_arm(
        {"cad_id": "cad-1", "parent_id": "a" * 24, "path": "cad.pkl"},
        arm="original",
        checkpoint_sha256=None,
        parsed=parsed,
        reconstructed_surfaces=surface,
        reconstructed_edges=edge,
        output_dir=tmp_path,
        assembler=failing_assembler,
    )

    assert row["status"] == "assembly_error"
    assert row["brep_valid"] is False
    assert row["step_saved"] is False
    assert row["error_type"] == "ValueError"


def test_model_arm_records_reconstruction_failure_instead_of_raising(tmp_path):
    surface = np.zeros((1, 32, 32, 3), dtype=np.float32)
    edge = np.zeros((1, 32, 3), dtype=np.float32)
    parsed = {"surf_ncs": surface, "edge_ncs": edge}

    def failing_reconstructor(*args, **kwargs):
        raise RuntimeError("CUDA failure")

    row = evaluate_cad_model_arm(
        {"cad_id": "cad-1", "parent_id": "a" * 24, "path": "cad.pkl"},
        arm="continuous_bypass_64d",
        checkpoint_sha256="hash",
        parsed=parsed,
        model=object(),
        output_dir=tmp_path,
        reconstructor=failing_reconstructor,
        assembler=lambda *args: {},
        device="cuda",
        batch_size=64,
    )

    assert row["status"] == "reconstruction_error"
    assert row["error_type"] == "RuntimeError"
    assert row["brep_valid"] is False


def test_calibration_renderer_writes_png_without_matplotlib(tmp_path, monkeypatch):
    import builtins
    from PIL import Image

    summary = summarize_calibration(
        _rows([(1e-5 * (index + 1), index < 15) for index in range(20)]),
        min_cads=20,
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise AssertionError("Matplotlib must not be imported")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    output = tmp_path / "calibration.png"
    render_calibration_png(summary, output)

    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.size == (1350, 864)


def test_calibration_reports_matched_model_degradation_from_original_control():
    values = [(1e-5 * (index + 1), index < 15) for index in range(20)]
    summary = summarize_calibration(_rows(values), min_cads=20)
    paired = summary["paired_against_original"]["continuous_bypass_64d"]
    assert paired["matched_cads"] == 20
    assert paired["original_valid_cads"] == 20
    assert paired["both_valid"] == 15
    assert paired["original_valid_model_invalid"] == 5
    assert paired["model_valid_rate_given_original_valid"] == pytest.approx(0.75)
