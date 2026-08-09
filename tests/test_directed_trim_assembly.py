import numpy as np
import pytest

from tools.directed_trim_assembly import directed_face_loops, loop_bbox_diagonal


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
