from tools.summarize_ar_length_coverage import render_markdown, summarize_length_coverage, summarize_records


def test_summarize_records_reports_valid_topology_distributions():
    records = [
        {"length": 100, "grammar_ok": True, "faces": 2, "edges": 3, "complex": False},
        {"length": 200, "grammar_ok": True, "faces": 12, "edges": 20, "complex": True},
        {"length": 300, "grammar_ok": False, "faces": 99, "edges": 99, "complex": False},
    ]

    summary = summarize_records(records, limits=[1024])

    assert summary["lengths"]["p25"] == 150.0
    assert summary["lengths"]["p99"] == 298.0
    assert summary["faces"]["count"] == 2
    assert summary["faces"]["min"] == 2
    assert summary["faces"]["median"] == 7.0
    assert summary["faces"]["max"] == 12
    assert summary["edges"]["count"] == 2
    assert summary["edges"]["median"] == 11.5
    assert summary["edges"]["max"] == 20
    assert summary["complex_total"] == 1
    assert summary["complex_fraction_of_grammar_ok"] == 0.5


def test_render_markdown_includes_split_topology_distributions():
    package = {
        "train": [],
        "val": [],
        "test": [],
        "face_index_size": 50,
        "se_codebook_size": 8192,
        "bbox_index_size": 2048,
    }
    summary = summarize_length_coverage(package, limits=[1024])

    markdown = render_markdown(summary)

    assert "## Split Distributions" in markdown
    assert "| Split | Metric | Count | Min | P25 | Median | P75 | P95 | P99 | Max |" in markdown
    assert "| train | faces | 0 | NA | NA | NA | NA | NA | NA | NA |" in markdown
    assert "Complex fraction" in markdown
