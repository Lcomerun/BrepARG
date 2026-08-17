from __future__ import annotations

import numpy as np

from tools.probe_graph_preserving_trim import (
    DEFAULT_VARIANTS,
    _samples_to_polyline_rms,
)


def test_probe_exposes_only_isolated_non_crashing_policies() -> None:
    modes = {item[1] for item in DEFAULT_VARIANTS}
    labels = {item[0] for item in DEFAULT_VARIANTS}

    assert modes == {"historical", "minimal_no_topology"}
    assert labels == {
        "historical_1e3",
        "minimal_no_topology_1e3",
        "minimal_no_topology_1e4",
        "minimal_no_topology_1e5",
    }


def test_samples_to_polyline_rms_is_zero_on_source_segments() -> None:
    polyline = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    samples = np.asarray(
        [[0.125, 0.0, 0.0], [0.75, 0.0, 0.0]],
        dtype=np.float64,
    )

    assert _samples_to_polyline_rms(samples, polyline) == 0.0

