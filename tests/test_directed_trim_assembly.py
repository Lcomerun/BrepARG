from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

import numpy as np
import pytest

from tools.directed_trim_assembly import (
    construct_brep_directed,
    directed_face_loops,
    effective_topology_summary,
    loop_bbox_diagonal,
)


OBSERVER_METADATA_KEYS = {
    "phase",
    "loop_count",
    "outer_loop_index",
    "loop_3d_endpoint_max_gaps",
    "face_3d_endpoint_max_gap",
}


def _construct_brep_ast() -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(construct_brep_directed)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


def _post_pcurve_observer_call(
    function: ast.FunctionDef | None = None,
) -> ast.Call:
    if function is None:
        function = _construct_brep_ast()
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "post_pcurve_face_observer"
    ]
    assert len(calls) == 1
    return calls[0]


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


def test_post_pcurve_observer_is_optional_keyword_only_and_none_is_bindable():
    signature = inspect.signature(construct_brep_directed)
    observer = signature.parameters["post_pcurve_face_observer"]

    assert observer.kind is inspect.Parameter.KEYWORD_ONLY
    assert observer.default is None

    positional = (
        np.empty((0, 32, 32, 3), dtype=np.float64),
        np.empty((0, 32, 3), dtype=np.float64),
        [],
        np.empty((0, 2), dtype=np.int64),
    )
    keyword = {"breparg_root": Path("unused-for-signature-check")}
    assert (
        signature.bind(*positional, **keyword).arguments.get(
            "post_pcurve_face_observer"
        )
        is None
    )
    assert (
        signature.bind(
            *positional, **keyword, post_pcurve_face_observer=None
        ).arguments["post_pcurve_face_observer"]
        is None
    )


def test_post_pcurve_observer_call_is_guarded_when_default_is_none():
    function = _construct_brep_ast()
    observer_call = _post_pcurve_observer_call(function)
    guards = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and any(
            descendant is observer_call
            for statement in node.body
            for descendant in ast.walk(statement)
        )
    ]

    assert len(guards) == 1
    guard = guards[0].test
    assert isinstance(guard, ast.Compare)
    assert isinstance(guard.left, ast.Name)
    assert guard.left.id == "post_pcurve_face_observer"
    assert len(guard.ops) == 1 and isinstance(guard.ops[0], ast.IsNot)
    assert len(guard.comparators) == 1
    assert isinstance(guard.comparators[0], ast.Constant)
    assert guard.comparators[0].value is None


def test_post_pcurve_observer_metadata_and_phase_contract():
    observer_call = _post_pcurve_observer_call()

    assert len(observer_call.args) == 3
    assert not observer_call.keywords
    metadata = observer_call.args[2]
    assert isinstance(metadata, ast.Dict)

    keys = []
    values = {}
    for key, value in zip(metadata.keys, metadata.values):
        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
        keys.append(key.value)
        values[key.value] = value

    assert len(keys) == len(set(keys))
    assert set(keys) == OBSERVER_METADATA_KEYS
    phase = values["phase"]
    assert isinstance(phase, ast.Constant)
    assert phase.value == "post_add_pcurves_pre_repair"
