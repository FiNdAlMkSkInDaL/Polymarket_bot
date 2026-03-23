#!/usr/bin/env python3
"""Visualize WFO parameter sweeps for plateau hunting.

This script is intentionally standalone: it does not import any repo internals.
It accepts JSON or CSV exports that contain, either directly or through nested
`params` / `best_params` dicts, the following fields:

- pure_mm_wide_spread_pct
- pure_mm_toxic_ofi_ratio
- a value field such as pnl / expected value / objective score

Examples
--------
python scripts/visualize_wfo_surface.py data/wfo_results.csv
python scripts/visualize_wfo_surface.py data/wfo_results.json --plot surface
python scripts/visualize_wfo_surface.py data/wfo_results.json --plot both --agg median
python scripts/visualize_wfo_surface.py data/wfo_results.json --value-column oos_total_pnl
python scripts/visualize_wfo_surface.py data/wfo_results.json --plot both --export-html wfo_surface.html
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors
except ImportError as exc:  # pragma: no cover - import guard only
    raise SystemExit(
        "matplotlib is required for this script. Install it with `pip install matplotlib`."
    ) from exc

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    pio = None
    make_subplots = None


X_FIELD = "pure_mm_wide_spread_pct"
Y_FIELD = "pure_mm_toxic_ofi_ratio"
COMMON_ROW_KEYS = ("results", "rows", "records", "trials", "data", "items", "folds")
VALUE_FIELD_CANDIDATES = (
    "pnl",
    "oos_total_pnl",
    "is_total_pnl",
    "aggregate_oos_total_pnl",
    "total_pnl",
    "profit",
    "expected_value",
    "ev",
    "objective",
    "objective_value",
    "score",
    "best_trial_score",
    "best_value",
)


@dataclass(frozen=True)
class AxisSpec:
    centers: np.ndarray
    labels: list[str]
    edges: np.ndarray | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 2D heatmap or 3D surface for pure_mm_wide_spread_pct vs "
            "pure_mm_toxic_ofi_ratio from JSON/CSV WFO results."
        )
    )
    parser.add_argument("input_path", help="Path to a JSON or CSV WFO results file.")
    parser.add_argument(
        "--plot",
        choices=("heatmap", "surface", "both"),
        default="heatmap",
        help="Plot type to render (default: heatmap).",
    )
    parser.add_argument(
        "--agg",
        choices=("mean", "median", "max", "min"),
        default="mean",
        help="Aggregation used when multiple rows share the same parameter pair.",
    )
    parser.add_argument(
        "--value-column",
        default=None,
        help="Explicit profit/objective field to visualize. If omitted, the script auto-detects one.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output image path. If --plot both is used, this becomes the file stem and the script "
            "writes *_heatmap.png and *_surface.png."
        ),
    )
    parser.add_argument(
        "--export-html",
        default=None,
        help=(
            "Optional output path for an interactive standalone Plotly HTML export. "
            "When used with --plot both, the HTML includes both the heatmap view and the 3D surface."
        ),
    )
    parser.add_argument(
        "--plateau-radius",
        type=int,
        default=1,
        help="Neighborhood radius used for plateau scoring (default: 1).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of plateau candidates to print (default: 5).",
    )
    parser.add_argument(
        "--x-bins",
        type=int,
        default=None,
        help=(
            "Optional number of uniform bins for pure_mm_wide_spread_pct. "
            "If omitted, the script keeps exact coordinates for small discrete grids "
            "and auto-bins dense float sweeps."
        ),
    )
    parser.add_argument(
        "--y-bins",
        type=int,
        default=None,
        help=(
            "Optional number of uniform bins for pure_mm_toxic_ofi_ratio. "
            "If omitted, the script keeps exact coordinates for small discrete grids "
            "and auto-bins dense float sweeps."
        ),
    )
    return parser.parse_args(argv)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        cleaned = text.replace(",", "").replace("$", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        try:
            numeric = float(cleaned)
        except ValueError:
            return None
        return numeric if math.isfinite(numeric) else None
    return None


def _flatten_row(raw_row: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(raw_row)
    for nested_key in ("params", "best_params", "metrics", "meta"):
        nested = raw_row.get(nested_key)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                row.setdefault(key, value)
                row[f"{nested_key}.{key}"] = value
    return row


def _extract_rows_from_json(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [_flatten_row(item) for item in data if isinstance(item, Mapping)]

    if not isinstance(data, Mapping):
        raise ValueError("JSON root must be an object or array of objects.")

    for key in COMMON_ROW_KEYS:
        candidate = data.get(key)
        if isinstance(candidate, list) and candidate and all(isinstance(item, Mapping) for item in candidate):
            return [_flatten_row(item) for item in candidate]

    fold_like_keys = [key for key in data if str(key).startswith("polymarket_wfo_fold_")]
    if fold_like_keys:
        return [_flatten_row(data[key]) for key in sorted(fold_like_keys)]

    dict_values = list(data.values())
    if dict_values and all(isinstance(item, Mapping) for item in dict_values):
        return [_flatten_row(item) for item in dict_values]

    return [_flatten_row(data)]


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [_flatten_row(row) for row in csv.DictReader(handle)]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return _extract_rows_from_json(json.load(handle))
    raise ValueError(f"Unsupported file type: {path.suffix}. Expected .json or .csv.")


def _resolve_value_field(rows: Iterable[Mapping[str, Any]], explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    for candidate in VALUE_FIELD_CANDIDATES:
        for row in rows:
            if candidate in row and _coerce_float(row.get(candidate)) is not None:
                return candidate
    raise ValueError(
        "Could not infer a value column. Pass --value-column with the PnL/EV/objective field name."
    )


def _collect_points(rows: Iterable[Mapping[str, Any]], value_field: str) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for row in rows:
        x_value = _coerce_float(row.get(X_FIELD))
        y_value = _coerce_float(row.get(Y_FIELD))
        z_value = _coerce_float(row.get(value_field))
        if x_value is None or y_value is None or z_value is None:
            continue
        points.append((x_value, y_value, z_value))
    return points


def _aggregate(values: list[float], method: str) -> float:
    if method == "mean":
        return float(sum(values) / len(values))
    if method == "median":
        return float(statistics.median(values))
    if method == "max":
        return float(max(values))
    if method == "min":
        return float(min(values))
    raise ValueError(f"Unsupported aggregation method: {method}")


def _auto_bin_count(unique_count: int) -> int:
    if unique_count <= 12:
        return unique_count
    return min(12, max(4, int(math.ceil(math.sqrt(unique_count)))))


def _format_bin_label(lower: float, upper: float) -> str:
    return f"{lower:.4f}-{upper:.4f}"


def _build_axis_spec(values: list[float], requested_bins: int | None) -> AxisSpec:
    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError("Cannot build a grid axis from an empty value set.")

    if len(unique_values) == 1:
        only_value = float(unique_values[0])
        return AxisSpec(
            centers=np.array([only_value], dtype=float),
            labels=[f"{only_value:.4f}"],
            edges=None,
        )

    n_bins = requested_bins if requested_bins is not None else _auto_bin_count(len(unique_values))
    n_bins = max(1, min(int(n_bins), len(unique_values)))
    if n_bins >= len(unique_values):
        centers = np.array(unique_values, dtype=float)
        return AxisSpec(
            centers=centers,
            labels=[f"{value:.4f}" for value in centers],
            edges=None,
        )

    data_min = float(unique_values[0])
    data_max = float(unique_values[-1])
    edges = np.linspace(data_min, data_max, n_bins + 1, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    labels = [_format_bin_label(float(edges[i]), float(edges[i + 1])) for i in range(n_bins)]
    return AxisSpec(centers=centers, labels=labels, edges=edges)


def _locate_axis_index(axis: AxisSpec, value: float) -> int:
    if axis.edges is None:
        return int(np.searchsorted(axis.centers, value))
    index = int(np.searchsorted(axis.edges, value, side="right") - 1)
    return min(max(index, 0), len(axis.centers) - 1)


def build_grid(
    points: Iterable[tuple[float, float, float]],
    agg: str,
    x_bins: int | None = None,
    y_bins: int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    dict[tuple[float, float], list[float]],
]:
    point_list = list(points)
    x_axis = _build_axis_spec([x_value for x_value, _, _ in point_list], x_bins)
    y_axis = _build_axis_spec([y_value for _, y_value, _ in point_list], y_bins)

    buckets: dict[tuple[float, float], list[float]] = defaultdict(list)
    for x_value, y_value, z_value in point_list:
        x_index = _locate_axis_index(x_axis, x_value)
        y_index = _locate_axis_index(y_axis, y_value)
        buckets[(float(x_axis.centers[x_index]), float(y_axis.centers[y_index]))].append(z_value)

    if not buckets:
        raise ValueError(
            "No usable rows found. The input must contain pure_mm_wide_spread_pct, "
            "pure_mm_toxic_ofi_ratio, and a numeric value column."
        )

    x_values = x_axis.centers
    y_values = y_axis.centers

    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    counts = np.zeros_like(grid)
    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            values = buckets.get((x_value, y_value))
            if not values:
                continue
            grid[y_index, x_index] = _aggregate(values, agg)
            counts[y_index, x_index] = len(values)
    return x_values, y_values, x_axis.labels, y_axis.labels, grid, counts, buckets


def compute_plateau_scores(grid: np.ndarray, radius: int) -> np.ndarray:
    scores = np.full_like(grid, np.nan, dtype=float)
    for row_index in range(grid.shape[0]):
        for col_index in range(grid.shape[1]):
            center = grid[row_index, col_index]
            if np.isnan(center):
                continue
            row_start = max(0, row_index - radius)
            row_end = min(grid.shape[0], row_index + radius + 1)
            col_start = max(0, col_index - radius)
            col_end = min(grid.shape[1], col_index + radius + 1)
            neighborhood = grid[row_start:row_end, col_start:col_end]
            valid = neighborhood[~np.isnan(neighborhood)]
            if valid.size == 0:
                continue
            scores[row_index, col_index] = float(np.mean(valid) - np.std(valid))
    return scores


def summarize_plateaus(
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: np.ndarray,
    counts: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for row_index in range(grid.shape[0]):
        for col_index in range(grid.shape[1]):
            value = grid[row_index, col_index]
            score = scores[row_index, col_index]
            if np.isnan(value) or np.isnan(score):
                continue
            candidates.append(
                {
                    X_FIELD: float(x_values[col_index]),
                    Y_FIELD: float(y_values[row_index]),
                    "value": float(value),
                    "plateau_score": float(score),
                    "samples": float(counts[row_index, col_index]),
                }
            )
    candidates.sort(key=lambda item: (item["plateau_score"], item["value"]), reverse=True)
    return candidates[:top_k]


def _build_norm(grid: np.ndarray) -> colors.Normalize:
    finite_values = grid[np.isfinite(grid)]
    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))
    if data_min < 0 < data_max:
        return colors.TwoSlopeNorm(vmin=data_min, vcenter=0.0, vmax=data_max)
    return colors.Normalize(vmin=data_min, vmax=data_max)


def plot_heatmap(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    grid: np.ndarray,
    plateau_scores: np.ndarray,
    value_field: str,
    output_path: Path,
) -> None:
    norm = _build_norm(grid)
    plateau_norm = _build_norm(plateau_scores[np.isfinite(plateau_scores)])
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    heatmap = axes[0].imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn", norm=norm)
    axes[0].set_title(f"Aggregated {value_field}")
    axes[0].set_xlabel(X_FIELD)
    axes[0].set_ylabel(Y_FIELD)
    axes[0].set_xticks(np.arange(len(x_values)))
    axes[0].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[0].set_yticks(np.arange(len(y_values)))
    axes[0].set_yticklabels(y_labels)
    figure.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04, label=value_field)

    plateau_map = axes[1].imshow(
        plateau_scores,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=plateau_norm,
    )
    axes[1].set_title("Plateau score (neighborhood mean - std)")
    axes[1].set_xlabel(X_FIELD)
    axes[1].set_ylabel(Y_FIELD)
    axes[1].set_xticks(np.arange(len(x_values)))
    axes[1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[1].set_yticks(np.arange(len(y_values)))
    axes[1].set_yticklabels(y_labels)
    figure.colorbar(plateau_map, ax=axes[1], fraction=0.046, pad=0.04, label="plateau score")

    figure.suptitle("WFO parameter surface: broad profitable plateaus beat narrow peaks", fontsize=13)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_surface(
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: np.ndarray,
    value_field: str,
    output_path: Path,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    norm = _build_norm(grid)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    masked = np.ma.masked_invalid(grid)

    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        x_mesh,
        y_mesh,
        masked,
        cmap="RdYlGn",
        norm=norm,
        linewidth=0.25,
        edgecolor="black",
        antialiased=True,
        alpha=0.92,
    )
    axis.set_title("WFO profitability surface")
    axis.set_xlabel(X_FIELD)
    axis.set_ylabel(Y_FIELD)
    axis.set_zlabel(value_field)
    axis.view_init(elev=28, azim=-135)
    figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.08, label=value_field)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def resolve_output_paths(input_path: Path, output_arg: str | None, plot_kind: str) -> list[Path]:
    if plot_kind == "both":
        if output_arg:
            stem = Path(output_arg)
            parent = stem.parent if stem.parent != Path("") else Path.cwd()
            base_name = stem.stem or stem.name
        else:
            parent = input_path.parent
            base_name = f"{input_path.stem}_wfo"
        return [parent / f"{base_name}_heatmap.png", parent / f"{base_name}_surface.png"]

    if output_arg:
        return [Path(output_arg)]
    return [input_path.with_name(f"{input_path.stem}_{plot_kind}.png")]


def _ensure_plotly_available() -> None:
    if go is None or pio is None or make_subplots is None:
        raise SystemExit(
            "Plotly is required for --export-html. Install it with `pip install plotly`."
        )


def _plotly_colorscale() -> list[list[float | str]]:
    return [
        [0.0, "#a50026"],
        [0.1, "#d73027"],
        [0.2, "#f46d43"],
        [0.35, "#fdae61"],
        [0.5, "#ffffbf"],
        [0.65, "#d9ef8b"],
        [0.8, "#66bd63"],
        [0.9, "#1a9850"],
        [1.0, "#006837"],
    ]


def _plotly_surface_color_axis(grid: np.ndarray) -> dict[str, float | list[list[float | str]]]:
    finite_values = grid[np.isfinite(grid)]
    axis: dict[str, float | list[list[float | str]]] = {
        "colorscale": _plotly_colorscale(),
        "cmin": float(np.min(finite_values)),
        "cmax": float(np.max(finite_values)),
    }
    if axis["cmin"] < 0 < axis["cmax"]:
        axis["cmid"] = 0.0
    return axis


def _plotly_heatmap_color_axis(grid: np.ndarray) -> dict[str, float | list[list[float | str]]]:
    finite_values = grid[np.isfinite(grid)]
    axis: dict[str, float | list[list[float | str]]] = {
        "colorscale": _plotly_colorscale(),
        "zmin": float(np.min(finite_values)),
        "zmax": float(np.max(finite_values)),
    }
    if axis["zmin"] < 0 < axis["zmax"]:
        axis["zmid"] = 0.0
    return axis


def _string_grid(values: np.ndarray) -> list[list[str | None]]:
    output: list[list[str | None]] = []
    for row in values:
        formatted_row: list[str | None] = []
        for value in row:
            formatted_row.append(None if np.isnan(value) else f"{float(value):.4f}")
        output.append(formatted_row)
    return output


def build_plotly_heatmap_figure(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    grid: np.ndarray,
    plateau_scores: np.ndarray,
    counts: np.ndarray,
    value_field: str,
):
    _ensure_plotly_available()
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Aggregated {value_field}", "Plateau score (neighborhood mean - std)"),
        horizontal_spacing=0.12,
    )
    value_text = _string_grid(grid)
    sample_text = _string_grid(counts)
    x_label_grid = np.array([x_labels for _ in y_values], dtype=object)
    y_label_grid = np.array([[label] * len(x_values) for label in y_labels], dtype=object)
    value_customdata = np.empty((len(y_values), len(x_values), 3), dtype=object)
    value_customdata[:, :, 0] = x_label_grid
    value_customdata[:, :, 1] = y_label_grid
    value_customdata[:, :, 2] = sample_text
    plateau_customdata = np.empty((len(y_values), len(x_values), 4), dtype=object)
    plateau_customdata[:, :, 0] = x_label_grid
    plateau_customdata[:, :, 1] = y_label_grid
    plateau_customdata[:, :, 2] = value_text
    plateau_customdata[:, :, 3] = sample_text

    figure.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=grid,
            colorbar={"title": value_field, "x": 0.44},
            hovertemplate=(
                f"{X_FIELD}: %{{customdata[0]}}<br>"
                f"{Y_FIELD}: %{{customdata[1]}}<br>"
                f"{value_field}: %{{z:.4f}}<br>"
                "samples: %{customdata[2]}<extra></extra>"
            ),
            customdata=value_customdata,
            **_plotly_heatmap_color_axis(grid),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=plateau_scores,
            colorscale="Viridis",
            colorbar={"title": "plateau score", "x": 1.02},
            hovertemplate=(
                f"{X_FIELD}: %{{customdata[0]}}<br>"
                f"{Y_FIELD}: %{{customdata[1]}}<br>"
                "plateau_score: %{z:.4f}<br>"
                f"{value_field}: %{{customdata[2]}}<br>"
                "samples: %{customdata[3]}<extra></extra>"
            ),
            customdata=plateau_customdata,
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text=X_FIELD, row=1, col=1)
    figure.update_xaxes(title_text=X_FIELD, row=1, col=2)
    figure.update_yaxes(title_text=Y_FIELD, row=1, col=1)
    figure.update_yaxes(title_text=Y_FIELD, row=1, col=2)
    figure.update_layout(
        title="WFO parameter heatmap: broad profitable plateaus beat narrow peaks",
        template="plotly_white",
        width=1400,
        height=600,
        margin={"l": 60, "r": 60, "t": 80, "b": 60},
    )
    return figure


def build_plotly_surface_figure(
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: np.ndarray,
    counts: np.ndarray,
    value_field: str,
):
    _ensure_plotly_available()
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    surface_kwargs = _plotly_surface_color_axis(grid)
    customdata = np.dstack((x_mesh, y_mesh, counts))
    figure = go.Figure(
        data=[
            go.Surface(
                x=x_mesh,
                y=y_mesh,
                z=grid,
                customdata=customdata,
                hovertemplate=(
                    f"{X_FIELD}: %{{customdata[0]:.4f}}<br>"
                    f"{Y_FIELD}: %{{customdata[1]:.4f}}<br>"
                    f"{value_field}: %{{z:.4f}}<br>"
                    "samples: %{customdata[2]:.0f}<extra></extra>"
                ),
                colorbar={"title": value_field},
                **surface_kwargs,
            )
        ]
    )
    figure.update_layout(
        title="WFO profitability surface",
        template="plotly_white",
        width=1100,
        height=800,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        scene={
            "xaxis_title": X_FIELD,
            "yaxis_title": Y_FIELD,
            "zaxis_title": value_field,
            "camera": {"eye": {"x": -1.5, "y": -1.7, "z": 0.9}},
        },
    )
    return figure


def export_plotly_html(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    grid: np.ndarray,
    plateau_scores: np.ndarray,
    counts: np.ndarray,
    value_field: str,
    plot_kind: str,
    output_path: Path,
) -> None:
    _ensure_plotly_available()
    sections: list[str] = []

    if plot_kind in {"heatmap", "both"}:
        heatmap_figure = build_plotly_heatmap_figure(
            x_values=x_values,
            y_values=y_values,
            x_labels=x_labels,
            y_labels=y_labels,
            grid=grid,
            plateau_scores=plateau_scores,
            counts=counts,
            value_field=value_field,
        )
        sections.append(pio.to_html(heatmap_figure, include_plotlyjs=True, full_html=False))

    if plot_kind in {"surface", "both"}:
        surface_figure = build_plotly_surface_figure(
            x_values=x_values,
            y_values=y_values,
            grid=grid,
            counts=counts,
            value_field=value_field,
        )
        sections.append(
            pio.to_html(
                surface_figure,
                include_plotlyjs=False if sections else True,
                full_html=False,
            )
        )

    html = "".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            "<title>WFO Surface Visualizer</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 0; background: #f6f7f9; color: #17202a; }",
            "main { max-width: 1440px; margin: 0 auto; padding: 24px; }",
            "h1 { margin: 0 0 8px; font-size: 28px; }",
            "p { margin: 0 0 20px; line-height: 1.5; }",
            ".chart { background: white; border-radius: 12px; padding: 12px; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }",
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<h1>WFO Surface Visualizer</h1>",
            f"<p>Interactive inspection for {X_FIELD}, {Y_FIELD}, and {value_field}. Rotate the 3D surface, zoom, and hover to inspect parameter plateaus.</p>",
        ]
    )
    for section in sections:
        html += f"<section class=\"chart\">{section}</section>"
    html += "</main></body></html>"
    output_path.write_text(html, encoding="utf-8")


def print_summary(
    input_path: Path,
    rows: list[dict[str, Any]],
    points: list[tuple[float, float, float]],
    x_values: np.ndarray,
    y_values: np.ndarray,
    value_field: str,
    agg: str,
    plateau_candidates: list[dict[str, float]],
) -> None:
    unique_pairs = {(x_value, y_value) for x_value, y_value, _ in points}
    occupied_cells = len(plateau_candidates)
    print(f"Loaded {len(rows)} raw rows from {input_path}")
    print(f"Usable rows: {len(points)}")
    print(f"Unique parameter pairs: {len(unique_pairs)}")
    print(f"Grid bins: {len(x_values)} x {len(y_values)}")
    print(f"Occupied grid cells: {occupied_cells}")
    print(f"Value field: {value_field}")
    print(f"Aggregation: {agg}")
    print()
    print("Top plateau candidates:")
    if not plateau_candidates:
        print("  No plateau candidates found.")
        return
    for index, candidate in enumerate(plateau_candidates, start=1):
        print(
            "  "
            f"{index}. {X_FIELD}={candidate[X_FIELD]:.4f}, "
            f"{Y_FIELD}={candidate[Y_FIELD]:.4f}, "
            f"value={candidate['value']:.4f}, "
            f"plateau_score={candidate['plateau_score']:.4f}, "
            f"samples={candidate['samples']:.0f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    rows = load_rows(input_path)
    value_field = _resolve_value_field(rows, args.value_column)
    points = _collect_points(rows, value_field)
    x_values, y_values, x_labels, y_labels, grid, counts, _ = build_grid(
        points,
        args.agg,
        x_bins=args.x_bins,
        y_bins=args.y_bins,
    )
    plateau_scores = compute_plateau_scores(grid, radius=max(0, args.plateau_radius))
    plateau_candidates = summarize_plateaus(
        x_values=x_values,
        y_values=y_values,
        grid=grid,
        counts=counts,
        scores=plateau_scores,
        top_k=max(1, args.top_k),
    )
    output_paths = resolve_output_paths(input_path, args.output, args.plot)

    if args.plot == "heatmap":
        plot_heatmap(
            x_values,
            y_values,
            x_labels,
            y_labels,
            grid,
            plateau_scores,
            value_field,
            output_paths[0],
        )
    elif args.plot == "surface":
        plot_surface(x_values, y_values, grid, value_field, output_paths[0])
    else:
        plot_heatmap(
            x_values,
            y_values,
            x_labels,
            y_labels,
            grid,
            plateau_scores,
            value_field,
            output_paths[0],
        )
        plot_surface(x_values, y_values, grid, value_field, output_paths[1])

    if args.export_html:
        export_plotly_html(
            x_values=x_values,
            y_values=y_values,
            x_labels=x_labels,
            y_labels=y_labels,
            grid=grid,
            plateau_scores=plateau_scores,
            counts=counts,
            value_field=value_field,
            plot_kind=args.plot,
            output_path=Path(args.export_html),
        )

    print_summary(
        input_path=input_path,
        rows=rows,
        points=points,
        x_values=x_values,
        y_values=y_values,
        value_field=value_field,
        agg=args.agg,
        plateau_candidates=plateau_candidates,
    )

    for output_path in output_paths:
        print(f"Saved plot to {output_path}")
    if args.export_html:
        print(f"Saved interactive HTML to {Path(args.export_html)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())