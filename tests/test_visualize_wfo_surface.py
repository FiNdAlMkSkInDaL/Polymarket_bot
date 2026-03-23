from __future__ import annotations

import numpy as np

from scripts.visualize_wfo_surface import build_grid


def test_build_grid_explicit_bins_aggregate_nearby_float_params():
    points = [
        (0.10, 0.20, 1.0),
        (0.12, 0.22, 5.0),
        (0.11, 0.80, 2.0),
        (0.90, 0.21, 3.0),
        (0.91, 0.81, 4.0),
    ]

    x_values, y_values, x_labels, y_labels, grid, counts, _ = build_grid(
        points,
        "mean",
        x_bins=2,
        y_bins=2,
    )

    assert len(x_values) == 2
    assert len(y_values) == 2
    assert all("-" in label for label in x_labels)
    assert all("-" in label for label in y_labels)
    assert counts.shape == (2, 2)
    assert counts[0, 0] == 2
    assert np.isclose(grid[0, 0], 3.0)
    assert counts[1, 0] == 1
    assert np.isclose(grid[1, 0], 2.0)
    assert counts[0, 1] == 1
    assert np.isclose(grid[0, 1], 3.0)
    assert counts[1, 1] == 1
    assert np.isclose(grid[1, 1], 4.0)


def test_build_grid_auto_bins_dense_float_sweeps():
    points = [(0.10 + i * 0.01, 0.20 + i * 0.01, float(i)) for i in range(16)]

    x_values, y_values, x_labels, y_labels, grid, counts, _ = build_grid(points, "mean")

    assert len(x_values) < 16
    assert len(y_values) < 16
    assert len(x_labels) == len(x_values)
    assert len(y_labels) == len(y_values)
    assert int(np.nansum(counts)) == 16
    assert np.isfinite(grid).any()