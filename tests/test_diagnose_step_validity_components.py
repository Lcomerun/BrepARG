from tools.diagnose_step_validity_components import summarize


def test_summarize_counts_component_failures():
    result = summarize([{
        "arm": "original", "status": "diagnosed", "free_edges": 2,
        "wire_order_failures": 1, "wire_self_intersections": 0,
        "shells_with_bad_edges": 1, "solid_count": 2,
    }])
    assert result["attempts"] == 1
    assert result["with_free_edges"] == 1
    assert result["with_wire_order_failures"] == 1
    assert result["with_bad_shell_edges"] == 1
    assert result["with_nonunit_solid_count"] == 1
