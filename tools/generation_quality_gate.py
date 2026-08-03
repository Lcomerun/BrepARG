"""Quality gates for retaining generated STEP candidates."""

from __future__ import annotations

from typing import Any


PRIMITIVE_LIKE_TOPOLOGIES = {
    (2, 2),
    (3, 3),
    (4, 6),
    (4, 8),
    (5, 8),
    (5, 9),
    (6, 12),
    (7, 12),
    (8, 18),
    (9, 18),
    (9, 20),
    (10, 22),
    (10, 24),
    (11, 24),
    (12, 20),
    (12, 22),
    (12, 24),
}


def is_primitive_like(faces: int, edges: int) -> bool:
    return (int(faces), int(edges)) in PRIMITIVE_LIKE_TOPOLOGIES or (int(faces) <= 12 and int(edges) <= 24)


def quality_gate_decision(
    row: dict[str, Any],
    quality: dict[str, Any],
    *,
    min_faces: int,
    min_edges: int,
    max_faces: int = 45,
    max_edges: int = 120,
    require_brep_valid: bool = True,
    require_closed_solid: bool = True,
    require_preview: bool = True,
    reject_primitive_like: bool = True,
    require_both_min_topology: bool = False,
) -> dict[str, Any]:
    """Return an accept/reject decision for one generated candidate."""
    reasons: list[str] = []
    grammar_ok = bool(row.get("grammar_ok"))
    step_saved = bool(row.get("step_saved"))
    faces = int(row.get("grammar_faces", 0) or 0)
    edges = int(row.get("grammar_edges", 0) or 0)
    advanced_faces = int(quality.get("advanced_faces", 0) or 0)
    edge_curves = int(quality.get("edge_curves", 0) or 0)

    if not grammar_ok:
        reasons.append("grammar_not_ok")
    if not step_saved:
        reasons.append("step_not_saved")
    if require_both_min_topology:
        if faces < int(min_faces):
            reasons.append("too_few_faces")
        if edges < int(min_edges):
            reasons.append("too_few_edges")
    elif faces < int(min_faces) and edges < int(min_edges):
        reasons.append("too_simple")
    if faces > int(max_faces):
        reasons.append("too_many_faces")
    if edges > int(max_edges):
        reasons.append("too_many_edges")
    if require_brep_valid and not bool(quality.get("brep_valid")):
        reasons.append("brep_not_valid")
    if require_closed_solid and not bool(quality.get("solid_closed_no_open_shell")):
        reasons.append("not_solid_closed")
    if require_preview and not bool(quality.get("png_saved")):
        reasons.append("preview_missing")
    if reject_primitive_like and is_primitive_like(faces, edges):
        reasons.append("primitive_like")
    if advanced_faces and advanced_faces < int(min_faces) and edge_curves < int(min_edges):
        reasons.append("step_entities_too_simple")

    return {
        "accept": not reasons,
        "reasons": reasons,
        "faces": faces,
        "edges": edges,
        "advanced_faces": advanced_faces,
        "edge_curves": edge_curves,
    }
