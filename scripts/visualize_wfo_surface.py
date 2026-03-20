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
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
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


def build_grid(
    points: Iterable[tuple[float, float, float]],
    agg: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[tuple[float, float], list[float]]]:
    buckets: dict[tuple[float, float], list[float]] = defaultdict(list)
    for x_value, y_value, z_value in points:
        buckets[(x_value, y_value)].append(z_value)

    if not buckets:
        raise ValueError(
            "No usable rows found. The input must contain pure_mm_wide_spread_pct, "
            "pure_mm_toxic_ofi_ratio, and a numeric value column."
        )

    x_values = np.array(sorted({pair[0] for pair in buckets}), dtype=float)
    y_values = np.array(sorted({pair[1] for pair in buckets}), dtype=float)

    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    counts = np.zeros_like(grid)
    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            values = buckets.get((x_value, y_value))
            if not values:
                continue
            grid[y_index, x_index] = _aggregate(values, agg)
            counts[y_index, x_index] = len(values)
    return x_values, y_values, grid, counts, buckets


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


def _format_tick_labels(values: np.ndarray) -> list[str]:
    return [f"{value:.3f}" for value in values]


def plot_heatmap(
    x_values: np.ndarray,
    y_values: np.ndarray,
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
    axes[0].set_xticklabels(_format_tick_labels(x_values), rotation=45, ha="right")
    axes[0].set_yticks(np.arange(len(y_values)))
    axes[0].set_yticklabels(_format_tick_labels(y_values))
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
    axes[1].set_xticklabels(_format_tick_labels(x_values), rotation=45, ha="right")
    axes[1].set_yticks(np.arange(len(y_values)))
    axes[1].set_yticklabels(_format_tick_labels(y_values))
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


def print_summary(
    input_path: Path,
    rows: list[dict[str, Any]],
    points: list[tuple[float, float, float]],
    value_field: str,
    agg: str,
    plateau_candidates: list[dict[str, float]],
) -> None:
    unique_pairs = {(x_value, y_value) for x_value, y_value, _ in points}
    print(f"Loaded {len(rows)} raw rows from {input_path}")
    print(f"Usable rows: {len(points)}")
    print(f"Unique parameter pairs: {len(unique_pairs)}")
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
    x_values, y_values, grid, counts, _ = build_grid(points, args.agg)
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
        plot_heatmap(x_values, y_values, grid, plateau_scores, value_field, output_paths[0])
    elif args.plot == "surface":
        plot_surface(x_values, y_values, grid, value_field, output_paths[0])
    else:
        plot_heatmap(x_values, y_values, grid, plateau_scores, value_field, output_paths[0])
        plot_surface(x_values, y_values, grid, value_field, output_paths[1])

    print_summary(
        input_path=input_path,
        rows=rows,
        points=points,
        value_field=value_field,
        agg=args.agg,
        plateau_candidates=plateau_candidates,
    )

    for output_path in output_paths:
        print(f"Saved plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())