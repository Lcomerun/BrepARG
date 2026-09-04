from __future__ import annotations

import ast
import inspect
from pathlib import Path
import textwrap

import numpy as np
import pytest

from tools.directed_trim_assembly import (
    ASSEMBLY_STAGE_PHASES,
    _emit_assembly_stage_observation,
    _emit_assembly_stage_failure,
    construct_brep_directed,
)


EXPECTED_STAGE_PHASES = (
    ("S1", "post_surface_curve_fit_pre_edge_build"),
    ("S2", "post_edge_build_pre_face_build"),
    ("S3", "post_add_pcurves_pre_optional_face_repair"),
    ("S4", "post_optional_face_repair_pre_sewing"),
    ("S5", "post_sewing_pre_solid"),
    ("S6", "post_solid_pre_step"),
)


def _construct_brep_ast() -> ast.FunctionDef:
    tree = ast.parse(textwrap.dedent(inspect.getsource(construct_brep_directed)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


def _stage_emitter_calls() -> list[ast.Call]:
    return [
        node
        for node in ast.walk(_construct_brep_ast())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_emit_assembly_stage_observation"
    ]


def _literal_keyword(call: ast.Call, name: str) -> str:
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    assert len(matches) == 1
    value = matches[0]
    assert isinstance(value, ast.Constant)
    assert isinstance(value.value, str)
    return value.value


def _ancestor_calls(function: ast.FunctionDef, target: ast.Call) -> list[ast.Call]:
    ancestors = []

    def visit(node: ast.AST, parents: tuple[ast.AST, ...] = ()) -> None:
        if node is target:
            ancestors.extend(parent for parent in parents if isinstance(parent, ast.Call))
            return
        for child in ast.iter_child_nodes(node):
            visit(child, (*parents, node))

    visit(function)
    return ancestors


def test_six_constructor_stage_names_are_frozen_and_s7_is_runner_only():
    assert ASSEMBLY_STAGE_PHASES == EXPECTED_STAGE_PHASES

    calls = _stage_emitter_calls()
    emitted = {
        (_literal_keyword(call, "stage"), _literal_keyword(call, "phase"))
        for call in calls
    }
    assert len(calls) == 6
    assert emitted == set(EXPECTED_STAGE_PHASES)
    assert all(stage != "S7" for stage, _phase in emitted)
    assert "post_step_roundtrip_strict" not in inspect.getsource(
        construct_brep_directed
    )


def test_stage_observer_is_optional_keyword_only_and_default_off():
    signature = inspect.signature(construct_brep_directed)
    parameter = signature.parameters["assembly_stage_observer"]

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
    assert bound.arguments.get("assembly_stage_observer") is None


def test_emitter_is_observation_only_and_ignores_return_value():
    target = object()
    received = []

    def observer(observed, metadata):
        received.append((observed, metadata))
        return object()

    result = _emit_assembly_stage_observation(
        observer,
        target,
        stage="S1",
        phase="post_surface_curve_fit_pre_edge_build",
        metadata={"entity_kind": "geometry_set", "stage": "forged"},
    )

    assert result is None
    assert received == [
        (
            target,
            {
                "entity_kind": "geometry_set",
                "stage": "S1",
                "phase": "post_surface_curve_fit_pre_edge_build",
            },
        )
    ]


def test_emitter_failure_preserves_stage_type_text_and_cause():
    def observer(_target, _metadata):
        raise ValueError("curve_fit_not_done edge=17")

    with pytest.raises(
        RuntimeError,
        match=(
            r"assembly_stage_observer_failed stage=S2 "
            r"phase=post_edge_build_pre_face_build "
            r"error_type=ValueError: curve_fit_not_done edge=17"
        ),
    ) as caught:
        _emit_assembly_stage_observation(
            observer,
            object(),
            stage="S2",
            phase="post_edge_build_pre_face_build",
            metadata={"entity_kind": "edge_set"},
        )

    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == "curve_fit_not_done edge=17"


def test_emitter_rejects_mismatched_stage_phase_before_callback():
    called = False

    def observer(_target, _metadata):
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="invalid assembly observation boundary"):
        _emit_assembly_stage_observation(
            observer,
            object(),
            stage="S1",
            phase="post_edge_build_pre_face_build",
            metadata={},
        )
    assert called is False


def test_terminal_failure_event_is_path_free_and_keeps_original_exception_type():
    received = []
    failure = RuntimeError("edge_builder_not_done edge=4")

    result = _emit_assembly_stage_failure(
        lambda target, metadata: received.append((target, metadata)),
        stage="S2",
        phase="post_edge_build_pre_face_build",
        failure_code="edge_builder_not_done",
        failure=failure,
        metadata={"source_edge_id": 4},
    )

    assert result is None
    assert received == [
        (
            None,
            {
                "source_edge_id": 4,
                "boundary_event": "terminal_failure",
                "failure_code": "edge_builder_not_done",
                "failure_type": "RuntimeError",
                "stage": "S2",
                "phase": "post_edge_build_pre_face_build",
            },
        )
    ]


def test_terminal_failure_observer_error_keeps_both_boundary_and_native_causes():
    native_failure = RuntimeError("edge_builder_not_done edge=4")

    def broken_observer(_target, _metadata):
        raise ValueError("observer rejected evidence")

    with pytest.raises(RuntimeError, match="assembly_stage_observer_failed") as caught:
        try:
            raise native_failure
        except RuntimeError as failure:
            _emit_assembly_stage_failure(
                broken_observer,
                stage="S2",
                phase="post_edge_build_pre_face_build",
                failure_code="edge_builder_not_done",
                failure=failure,
                metadata={"source_edge_id": 4},
            )

    assert isinstance(caught.value.__cause__, ValueError)
    # Python first chains the observer ValueError to the native exception and
    # then wraps that observer error with the stage-aware RuntimeError.
    assert caught.value.__cause__.__context__ is native_failure


def test_stage_payloads_have_identity_lineage_and_population_metadata():
    calls = {}
    for call in _stage_emitter_calls():
        stage = _literal_keyword(call, "stage")
        metadata = next(
            keyword.value for keyword in call.keywords if keyword.arg == "metadata"
        )
        if isinstance(metadata, ast.Dict) and (
            stage in {"S5", "S6"}
            or "'boundary_event': 'completed'" in ast.unparse(metadata)
        ):
            calls[stage] = call
    required_literal_keys = {
        "S1": {
            "entity_kind",
            "boundary_event",
            "observation_scope",
            "source_edge_id",
            "event_sequence_position",
            "expected_source_face_count",
            "expected_source_edge_count",
            "fitted_surface_count",
            "fitted_curve_prefix_count",
            "built_edge_prefix_count",
            "surface_fit_tolerance",
            "source_surface_bindings",
            "source_vertex_ids",
            "effective_vertex_ids",
        },
        "S2": {
            "entity_kind",
            "boundary_event",
            "observation_scope",
            "source_edge_id",
            "event_sequence_position",
            "expected_source_face_count",
            "expected_source_edge_count",
            "fitted_curve_prefix_count",
            "built_edge_prefix_count",
            "source_vertex_ids",
            "effective_vertex_ids",
        },
        "S3": {
            "boundary_event",
            "observation_scope",
            "event_sequence_position",
            "constructed_face_prefix_count",
            "post_repair_face_prefix_count",
            "expected_source_face_count",
            "expected_source_edge_count",
            "source_mapping",
        },
        "S4": {
            "boundary_event",
            "observation_scope",
            "event_sequence_position",
            "constructed_face_prefix_count",
            "post_repair_face_prefix_count",
            "expected_source_face_count",
            "expected_source_edge_count",
            "source_mapping",
        },
        "S5": {
            "entity_kind",
            "expected_source_face_count",
            "expected_source_edge_count",
            "sewn_face_count",
            "sewing_tolerance",
            "source_vertex_lineage",
            "source_face_bindings",
        },
        "S6": {
            "entity_kind",
            "expected_source_face_count",
            "expected_source_edge_count",
            "shell_count",
            "solid_count",
            "sewing_tolerance",
            "post_sewing_mutation_enabled",
            "effective_input_topology",
            "source_vertex_lineage",
            "source_face_bindings",
        },
    }

    for stage, required in required_literal_keys.items():
        metadata_nodes = [
            keyword.value for keyword in calls[stage].keywords if keyword.arg == "metadata"
        ]
        assert len(metadata_nodes) == 1
        metadata = metadata_nodes[0]
        assert isinstance(metadata, ast.Dict)
        literal_keys = {
            key.value
            for key in metadata.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert required <= literal_keys

    source = ast.unparse(_construct_brep_ast())
    for required_source_key in (
        "source_face_index",
        "source_edge_id",
        "source_vertex_ids",
        "effective_vertex_ids",
        "source_loop_edge_uses",
        "outer_loop_index",
        "loop_3d_endpoint_gaps",
    ):
        assert repr(required_source_key) in source


def test_stage_boundaries_precede_mutation_and_follow_native_builders():
    source = inspect.getsource(construct_brep_directed)

    assert source.index('if curve is None:') < source.index('stage="S1"')
    assert source.index('stage="S1"') < source.index(
        "BRepBuilderAPI_MakeEdge(", source.index('stage="S1"')
    )
    completed_s2 = source.index('"boundary_event": "completed"', source.index("edges.append(edge)"))
    assert source.index("edges.append(edge)") < completed_s2
    assert source.index('stage="S2"') < source.index("\n    faces = []")
    assert source.index("brep_utils.add_pcurves_to_edges(face)") < source.index(
        'stage="S3"'
    )
    assert source.index('stage="S3"') < source.index(
        "post_pcurve_face_mutator(", source.index('stage="S3"')
    )
    assert source.index('stage="S4"') < source.index("faces.append(face)")
    assert source.index("sewn = sewing.SewedShape()") < source.index('stage="S5"')
    assert source.index('stage="S5"') < source.index("post_sewing_shape_mutator(")
    assert source.index("solid = maker.Solid()") < source.index('stage="S6"')


def test_default_path_keeps_historical_interleaved_curve_and_edge_execution():
    source = inspect.getsource(construct_brep_directed)

    # Without an observer, every fitted curve is immediately turned into its
    # source edge.  Thus an edge-builder failure still stops before fitting
    # later curves, exactly as the pre-census constructor did.  Only the
    # observer branch delays all edge builds to expose the registered S1
    # boundary after every curve fit and before every MakeEdge call.
    fit_loop = source.index("for edge_index, points in enumerate(edge_wcs):")
    s1 = source.index('stage="S1"', fit_loop)
    make_edge = source.index("BRepBuilderAPI_MakeEdge(", s1)
    appended = source.index("edges.append(edge)", make_edge)
    s2 = source.index('stage="S2"', appended)
    assert fit_loop < s1 < make_edge < appended < s2
    assert "for edge_index, curve in enumerate(curves):" not in source
    assert "build_source_edge" not in source


def test_curve_fit_and_edge_build_split_preserves_historical_fit_recipe():
    source = inspect.getsource(construct_brep_directed)

    fit_loop = source.index("for edge_index, points in enumerate(edge_wcs):")
    fitted_guard = source.index('if curve is None:', fit_loop)
    s1 = source.index('stage="S1"')
    make_edge = source.index("BRepBuilderAPI_MakeEdge(", s1)
    s2 = source.index('stage="S2"', make_edge)

    assert fit_loop < fitted_guard < s1 < make_edge < s2
    assert source.count("historical_fit_attempts = ((0, 8, 5e-3), (0, 8, 8e-3), (0, 8, 5e-2))") == 1
    assert source.count("curve_fit_attempts()") == 2
    assert source.count("GeomAPI_PointsToBSpline(") == 2
    assert source.count("GeomAPI_Interpolate(") == 1
    assert source.count("edges.append(edge)") == 1


def test_s1_s2_distributed_events_keep_exact_source_edge_order():
    calls = {
        _literal_keyword(call, "stage"): call for call in _stage_emitter_calls()
    }
    for stage in ("S1", "S2"):
        metadata = next(
            keyword.value
            for keyword in calls[stage].keywords
            if keyword.arg == "metadata"
        )
        assert isinstance(metadata, ast.Dict)
        source = ast.unparse(metadata)
        assert "'observation_scope': 'distributed_source_edge_event'" in source
        assert "'source_edge_id': int(edge_index)" in source

    s1_metadata = next(
        keyword.value for keyword in calls["S1"].keywords if keyword.arg == "metadata"
    )
    s2_metadata = next(
        keyword.value for keyword in calls["S2"].keywords if keyword.arg == "metadata"
    )
    assert "int(2 * edge_index)" in ast.unparse(s1_metadata)
    assert "int(2 * edge_index + 1)" in ast.unparse(s2_metadata)


def test_s3_s4_distributed_events_keep_exact_source_face_order():
    calls = {
        _literal_keyword(call, "stage"): call for call in _stage_emitter_calls()
    }
    metadata_by_stage = {}
    for stage in ("S3", "S4"):
        metadata = next(
            keyword.value
            for keyword in calls[stage].keywords
            if keyword.arg == "metadata"
        )
        assert isinstance(metadata, ast.Dict)
        source = ast.unparse(metadata)
        assert "'observation_scope': 'distributed_source_face_event'" in source
        assert "'source_face_index': int(face_index)" in ast.unparse(
            _construct_brep_ast()
        )
        metadata_by_stage[stage] = source
    assert "int(2 * face_index)" in metadata_by_stage["S3"]
    assert "int(2 * face_index + 1)" in metadata_by_stage["S4"]


def test_existing_face_observer_contract_is_not_routed_through_new_emitter():
    function = _construct_brep_ast()
    legacy_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assembly_stage_face_observer"
    ]

    assert len(legacy_calls) == 3
    assert all(not _ancestor_calls(function, call) for call in legacy_calls)
    source = ast.unparse(function)
    for legacy_phase in (
        "post_add_pcurves_pre_repair",
        "post_optional_face_repair_pre_sewing",
        "post_sewing_pre_step",
    ):
        assert repr(legacy_phase) in source
