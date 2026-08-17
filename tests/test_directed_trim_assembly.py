import numpy as np
import pytest

from tools.directed_trim_assembly import (
    directed_face_loops,
    effective_topology_summary,
    loop_bbox_diagonal,
)


def test_directed_face_loop_orients_edges_by_vertex_walk():
    adjacency = np.asarray([[10, 11], [12, 11], [10, 12]], dtype=np.int64)
    assert directed_face_loops([0, 1, 2], adjacency) == [
        [(0, False), (1, True), (2, True)]
    ]


def test_directed_face_loops_reject_open_topology():
    adjacency = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="open or branching"):
        directed_face_loops([0, 1], adjacency)


def test_loop_bbox_uses_global_edge_ids():
    edges = np.zeros((8, 32, 3), dtype=np.float32)
    edges[7, :, 0] = np.linspace(0, 10, 32)
    assert loop_bbox_diagonal([(7, False)], edges) == pytest.approx(10.0)


def test_effective_topology_summary_records_endpoint_incidence_after_remap():
    summary = effective_topology_summary(
        np.asarray([[7, 8], [8, 9], [9, 7]], dtype=np.int64)
    )

    assert summary == {
        "edge_count": 3,
        "vertex_count": 3,
        "vertex_edge_incidence_counts": [2, 2, 2],
    }


@pytest.mark.parametrize(
    "adjacency",
    [np.empty((0, 2), dtype=np.int64), np.asarray([[0, -1]], dtype=np.int64)],
)
def test_effective_topology_summary_rejects_invalid_adjacency(adjacency):
    with pytest.raises(ValueError):
        effective_topology_summary(adjacency)
