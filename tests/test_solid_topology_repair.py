import numpy as np

from tools import diagnose_solid_topology_repair as diagnosis
from tools.solid_topology_repair import reconcile_near_vertices


def _curves(points, edges):
    values = []
    for left, right in edges:
        start = np.asarray(points[left], dtype=float)
        end = np.asarray(points[right], dtype=float)
        values.append(np.stack([start, (start + end) / 2.0, end]))
    return np.asarray(values)


def test_reconcile_merges_only_mutual_near_vertices_on_a_common_face():
    points = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (1.0, 1.0, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (1.0 + 5e-5, 1.0, 0.0),
    }
    edge_pairs = [(0, 1), (1, 2), (4, 3), (3, 0)]
    edge_wcs = _curves(points, edge_pairs)
    adjacency = np.asarray(edge_pairs, dtype=np.int64)

    remapped, shared, diagnostics = reconcile_near_vertices(
        edge_wcs, adjacency, [[0, 1, 2, 3]], tolerance=2e-4
    )

    assert diagnostics["applied"] is True
    assert len(diagnostics["merged_pairs"]) == 1
    assert diagnostics["merged_pairs"][0]["left"] == 2
    assert diagnostics["merged_pairs"][0]["right"] == 4
    np.testing.assert_allclose(diagnostics["merged_pairs"][0]["distance"], 5e-5)
    assert remapped.tolist() == [[0, 1], [1, 2], [2, 3], [3, 0]]
    assert sorted(shared) == [0, 1, 2, 3]


def test_reconcile_fails_closed_on_ambiguous_candidates():
    points = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (1.0, 1.0, 0.0),
        3: (1.0 + 5e-5, 1.0, 0.0),
        4: (1.0 - 5e-5, 1.0, 0.0),
    }
    edge_pairs = [(0, 1), (1, 2), (3, 4), (4, 0)]
    edge_wcs = _curves(points, edge_pairs)
    adjacency = np.asarray(edge_pairs, dtype=np.int64)

    remapped, shared, diagnostics = reconcile_near_vertices(
        edge_wcs, adjacency, [[0, 1, 2, 3]], tolerance=2e-4
    )

    assert diagnostics["applied"] is False
    assert diagnostics["reason"] == "no_unambiguous_pair"
    np.testing.assert_array_equal(remapped, adjacency)
    assert shared == {}


def test_reconcile_does_not_merge_close_vertices_on_different_faces():
    points = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (1.0, 1.0, 0.0),
        3: (1.0 + 5e-5, 1.0, 0.0),
    }
    edge_pairs = [(0, 1), (1, 2), (3, 0)]
    edge_wcs = _curves(points, edge_pairs)
    adjacency = np.asarray(edge_pairs, dtype=np.int64)

    remapped, shared, diagnostics = reconcile_near_vertices(
        edge_wcs, adjacency, [[0, 1], [2]], tolerance=2e-4
    )

    assert diagnostics["applied"] is False
    np.testing.assert_array_equal(remapped, adjacency)
    assert shared == {}


def test_loop_coverage_records_a_historical_grouping_that_drops_an_edge(monkeypatch):
    adjacency = np.asarray([[0, 1], [2, 3]], dtype=np.int64)
    monkeypatch.setattr(
        diagnosis,
        "historical_face_loops",
        lambda _incident, _adjacency: [[(0, False)]],
    )

    coverage = diagnosis.loop_coverage([[0, 1]], adjacency)

    assert coverage == [
        {
            "face_index_0based": 0,
            "incident_edge_count": 2,
            "loop_count": 1,
            "missing_edge_ids": [1],
            "repeated_edge_ids": [],
            "observed_edge_count": 1,
            "error": None,
        }
    ]


def test_build_report_binds_artifacts_without_serializing_local_paths(tmp_path, monkeypatch):
    source = tmp_path / "source.pkl"
    baseline = tmp_path / "baseline.step"
    candidate = tmp_path / "candidate.step"
    source.write_bytes(b"source-bytes")
    baseline.write_bytes(b"baseline-bytes")
    candidate.write_bytes(b"candidate-bytes")
    monkeypatch.setattr(
        diagnosis,
        "inspect_step",
        lambda _path: {"readable": True, "native_valid": False, "subshapes": {}},
    )
    monkeypatch.setattr(
        diagnosis,
        "source_evidence",
        lambda _path, *, joint_iterations: {
            "face_count": 1,
            "edge_count": 1,
            "joint_iterations": joint_iterations,
        },
    )

    report = diagnosis.build_report(
        cad_id="cad-safe",
        source_pickle=source,
        baseline_step=baseline,
        candidate_step=candidate,
        joint_iterations=37,
    )

    encoded = __import__("json").dumps(report, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "source-bytes" not in encoded
    assert report["source_pickle"]["archived"] is False
    assert report["source_topology"]["joint_iterations"] == 37
