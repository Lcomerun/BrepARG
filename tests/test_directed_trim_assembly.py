from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

import numpy as np
import pytest

from tools.directed_trim_assembly import (
    _invalidate_colliding_sewing_targets,
    _unique_face_local_geometry_target,
    _unique_sewing_history_target,
    construct_brep_directed,
    directed_face_loops,
    effective_topology_summary,
    loop_bbox_diagonal,
)


def test_face_local_geometry_target_requires_exactly_one_edge_multiset(monkeypatch):
    import tools.directed_trim_assembly as module

    class Explorer:
        def __init__(self, face, ignore_orientation=False):
            self.face = face

        def edges(self):
            return iter(self.face)

    monkeypatch.setattr(module, "TopologyExplorer", Explorer, raising=False)
    monkeypatch.setattr(
        module,
        "_identity_or_geometry_edge_assignment",
        lambda observed, candidates: (
            (list(range(len(candidates))), ["geometry_fingerprint"] * len(candidates), [])
            if observed == ["unique_a", "unique_b"]
            else (None, [], ["edge_unique_perfect_assignment_failed"])
        ),
    )
    target = ["unique_a", "unique_b"]
    matched, proof = _unique_face_local_geometry_target(
        [(10, object()), (11, object())], [["other"], target]
    )
    assert matched is target
    assert proof["status"] == "mapped"
    assert proof["failure_codes"] == []

    matched, proof = _unique_face_local_geometry_target(
        [(10, object()), (11, object())], [target, list(target)]
    )
    assert matched is None
    assert proof["failure_codes"] == ["sewn_face_geometry_match_not_unique"]


OBSERVER_METADATA_KEYS = {
    "phase",
    "loop_count",
    "outer_loop_index",
    "loop_3d_endpoint_max_gaps",
    "face_3d_endpoint_max_gap",
}

ASSEMBLY_STAGE_PHASES = {
    "post_add_pcurves_pre_repair",
    "post_optional_face_repair_pre_sewing",
    "post_sewing_pre_step",
}

ASSEMBLY_SOURCE_METADATA_KEYS = {
    "entity_kind",
    "source_face_index",
    "source_loop_edge_uses",
    "outer_loop_index",
    "loop_3d_endpoint_gaps",
    "loop_3d_endpoint_max_gaps",
    "face_3d_endpoint_max_gap",
    "source_mapping",
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


def _assembly_stage_observer_calls(
    function: ast.FunctionDef | None = None,
) -> list[ast.Call]:
    if function is None:
        function = _construct_brep_ast()
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assembly_stage_face_observer"
    ]


def _literal_phase(call: ast.Call) -> str:
    assert len(call.args) == 3
    metadata = call.args[2]
    assert isinstance(metadata, ast.Dict)
    for key, value in zip(metadata.keys, metadata.values):
        if isinstance(key, ast.Constant) and key.value == "phase":
            assert isinstance(value, ast.Constant)
            assert isinstance(value.value, str)
            return value.value
    raise AssertionError("observer call has no literal phase")


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


def test_assembly_stage_observer_is_optional_keyword_only_and_none_is_bindable():
    signature = inspect.signature(construct_brep_directed)
    observer = signature.parameters["assembly_stage_face_observer"]

    assert observer.kind is inspect.Parameter.KEYWORD_ONLY
    assert observer.default is None
    positional = (
        np.empty((0, 32, 32, 3), dtype=np.float64),
        np.empty((0, 32, 3), dtype=np.float64),
        [],
        np.empty((0, 2), dtype=np.int64),
    )
    bound = signature.bind(
        *positional, breparg_root=Path("unused-for-signature-check")
    )
    assert bound.arguments.get("assembly_stage_face_observer") is None


@pytest.mark.parametrize(
    "name",
    ("post_pcurve_face_mutator", "post_sewing_shape_mutator"),
)
def test_experimental_mutation_hooks_are_keyword_only_and_default_off(name):
    signature = inspect.signature(construct_brep_directed)
    parameter = signature.parameters[name]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
    positional = (
        np.empty((0, 32, 32, 3), dtype=np.float64),
        np.empty((0, 32, 3), dtype=np.float64),
        [],
        np.empty((0, 2), dtype=np.int64),
    )
    bound = signature.bind(
        *positional, breparg_root=Path("unused-for-signature-check")
    )
    assert bound.arguments.get(name) is None


def test_sewing_tolerance_is_keyword_only_and_preserves_historical_default():
    parameter = inspect.signature(construct_brep_directed).parameters[
        "sewing_tolerance"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == pytest.approx(1e-3)


def test_assembly_stage_observer_emits_exact_three_phase_contract():
    calls = _assembly_stage_observer_calls()

    assert len(calls) == 3
    assert {_literal_phase(call) for call in calls} == ASSEMBLY_STAGE_PHASES
    for call in calls:
        assert len(call.args) == 3
        assert not call.keywords
        first_argument = call.args[0]
        if isinstance(first_argument, ast.Call):
            assert isinstance(first_argument.func, ast.Name)
            assert first_argument.func.id == "int"
            assert len(first_argument.args) == 1
            first_argument = first_argument.args[0]
        assert isinstance(first_argument, ast.Name)
        assert first_argument.id in {"face_index", "source_face_index"}


def test_assembly_face_stage_metadata_has_source_topology_and_identity_map():
    calls = {
        _literal_phase(call): call for call in _assembly_stage_observer_calls()
    }

    for phase in (
        "post_add_pcurves_pre_repair",
        "post_optional_face_repair_pre_sewing",
    ):
        metadata = calls[phase].args[2]
        assert isinstance(metadata, ast.Dict)
        literal_keys = {
            key.value
            for key in metadata.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert {"phase", "source_mapping"} <= literal_keys
        assert any(key is None for key in metadata.keys)

    function_source = ast.unparse(_construct_brep_ast())
    for key in ASSEMBLY_SOURCE_METADATA_KEYS - {"source_mapping"}:
        assert repr(key) in function_source
    assert "source_edge_id" in function_source
    assert "reversed" in function_source
    assert "endpoint_gap_to_next_3d" in function_source


def test_post_sewing_observer_uses_history_and_never_explorer_ordinal():
    function_source = ast.unparse(_construct_brep_ast())

    assert "_unique_sewing_history_target" in function_source
    assert "ModifiedSubShape" not in function_source
    assert "sewing_lineage" in function_source
    assert "mapping_failed" in function_source
    assert "_invalidate_colliding_sewing_targets" in function_source
    assert "exact_sewing_history" in function_source


class _IdentityShape:
    def __init__(self, identity, *, fail=False):
        self.identity = identity
        self.fail = fail

    def IsNull(self):
        return False

    def IsSame(self, other):
        if self.fail:
            raise RuntimeError("identity unavailable")
        return self.identity == other.identity


def test_sewing_history_requires_both_apis_unique_and_consistent():
    output = _IdentityShape("output")

    class Consistent:
        def ModifiedSubShape(self, _source):
            return _IdentityShape("output")

        def Modified(self, _source):
            return _IdentityShape("output")

    target, evidence = _unique_sewing_history_target(
        Consistent(), _IdentityShape("source"), [output]
    )
    assert target is output
    assert evidence["status"] == "mapped"
    assert evidence["mapping_methods"] == ["Modified", "ModifiedSubShape"]

    class OneFails(Consistent):
        def Modified(self, _source):
            raise RuntimeError("history unavailable")

    target, evidence = _unique_sewing_history_target(
        OneFails(), _IdentityShape("source"), [output]
    )
    assert target is None
    assert evidence["status"] == "mapping_failed"
    assert evidence["failure_codes"] == [
        "sewing_history_methods_not_both_unique"
    ]

    class Disagrees(Consistent):
        def Modified(self, _source):
            return _IdentityShape("other")

    target, evidence = _unique_sewing_history_target(
        Disagrees(),
        _IdentityShape("source"),
        [output, _IdentityShape("other")],
    )
    assert target is None
    assert evidence["failure_codes"] == ["sewing_history_methods_disagree"]


def test_collision_invalidation_does_not_cascade_after_mutation():
    rows = [
        {"status": "mapped", "shape": _IdentityShape("merged"), "failure_codes": []},
        {"status": "mapped", "shape": _IdentityShape("merged"), "failure_codes": []},
        {"status": "mapped", "shape": _IdentityShape("independent"), "failure_codes": []},
    ]
    _invalidate_colliding_sewing_targets(rows)

    assert [row["status"] for row in rows] == [
        "mapping_failed", "mapping_failed", "mapped"
    ]
    assert rows[2]["shape"].identity == "independent"


def test_existing_post_pcurve_hook_remains_one_call_with_frozen_metadata():
    function = _construct_brep_ast()
    call = _post_pcurve_observer_call(function)

    assert len(
        [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "post_pcurve_face_observer"
        ]
    ) == 1
    metadata = call.args[2]
    assert isinstance(metadata, ast.Dict)
    assert {
        key.value for key in metadata.keys if isinstance(key, ast.Constant)
    } == OBSERVER_METADATA_KEYS
