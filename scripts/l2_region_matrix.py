#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MIN_FRAMES_FOR_RANKING = 25


@dataclass(frozen=True)
class RankedRegion:
    label: str
    sufficient_sample: bool
    disconnect_count: int
    silence_gap_count: int
    lag_stdev_ms: float | None
    lag_p95_ms: float | None
    lag_mean_ms: float | None
    frame_gap_stdev_ms: float | None
    total_frames: int
    total_events: int
    score: tuple[Any, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate regional L2 probe summaries into a comparison matrix."
    )
    parser.add_argument("summaries", nargs="+", help="JSON summary files produced by l2_region_probe.py")
    parser.add_argument("--output", required=True, help="Path to write the aggregated JSON matrix.")
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional path to also write a markdown table.",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"summary file {path} did not decode to an object")
    return data


def _stats_value(summary: dict[str, Any], section: str, key: str) -> float | None:
    bucket = summary.get(section) or {}
    value = bucket.get(key)
    if value is None:
        return None
    return float(value)


def _region_score(summary: dict[str, Any]) -> tuple[Any, ...]:
    lag_stdev = _stats_value(summary, "exchange_lag_ms", "stdev")
    lag_p95 = _stats_value(summary, "exchange_lag_ms", "p95")
    lag_mean = _stats_value(summary, "exchange_lag_ms", "mean")
    frame_gap_stdev = _stats_value(summary, "frame_gap_ms", "stdev")
    total_frames = int(summary.get("total_frames", 0))
    return (
        1 if total_frames < MIN_FRAMES_FOR_RANKING else 0,
        int(summary.get("disconnect_count", 0)),
        int(summary.get("silence_gap_count", 0)),
        float("inf") if frame_gap_stdev is None else frame_gap_stdev,
        float("inf") if lag_stdev is None else lag_stdev,
        float("inf") if lag_p95 is None else lag_p95,
        float("inf") if lag_mean is None else lag_mean,
        -total_frames,
    )


def rank_summaries(summaries: list[dict[str, Any]]) -> list[RankedRegion]:
    ranked: list[RankedRegion] = []
    for summary in summaries:
        ranked.append(
            RankedRegion(
                label=str(summary.get("label", "unknown")),
                sufficient_sample=int(summary.get("total_frames", 0)) >= MIN_FRAMES_FOR_RANKING,
                disconnect_count=int(summary.get("disconnect_count", 0)),
                silence_gap_count=int(summary.get("silence_gap_count", 0)),
                lag_stdev_ms=_stats_value(summary, "exchange_lag_ms", "stdev"),
                lag_p95_ms=_stats_value(summary, "exchange_lag_ms", "p95"),
                lag_mean_ms=_stats_value(summary, "exchange_lag_ms", "mean"),
                frame_gap_stdev_ms=_stats_value(summary, "frame_gap_ms", "stdev"),
                total_frames=int(summary.get("total_frames", 0)),
                total_events=int(summary.get("total_events", 0)),
                score=_region_score(summary),
            )
        )
    return sorted(ranked, key=lambda item: item.score)


def build_matrix_document(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = rank_summaries(summaries)
    rows = []
    for row in ranked:
        rows.append(
            {
                "label": row.label,
                "sufficient_sample": row.sufficient_sample,
                "disconnect_count": row.disconnect_count,
                "silence_gap_count": row.silence_gap_count,
                "exchange_lag_stdev_ms": row.lag_stdev_ms,
                "exchange_lag_p95_ms": row.lag_p95_ms,
                "exchange_lag_mean_ms": row.lag_mean_ms,
                "frame_gap_stdev_ms": row.frame_gap_stdev_ms,
                "total_frames": row.total_frames,
                "total_events": row.total_events,
            }
        )
    best = rows[0] if rows else None
    return {
        "recommended_region": best,
        "ranking_method": (
            f"Require at least {MIN_FRAMES_FOR_RANKING} frames for a sufficient sample, then sort by "
            "disconnect_count, silence_gap_count, frame_gap_stdev_ms, "
            "exchange_lag_stdev_ms, exchange_lag_p95_ms, exchange_lag_mean_ms, then total_frames descending."
        ),
        "rows": rows,
    }


def render_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "| Region | Sample OK | Disconnects | Silence Gaps | Lag StdDev (ms) | Lag P95 (ms) | Lag Mean (ms) | Frame Gap StdDev (ms) | Frames | Events |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in matrix.get("rows", []):
        lines.append(
            "| {label} | {sufficient_sample} | {disconnect_count} | {silence_gap_count} | {exchange_lag_stdev_ms} | {exchange_lag_p95_ms} | {exchange_lag_mean_ms} | {frame_gap_stdev_ms} | {total_frames} | {total_events} |".format(
                **{
                    key: ("n/a" if value is None else value)
                    for key, value in row.items()
                }
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    summaries = [_load_summary(Path(path)) for path in args.summaries]
    matrix = build_matrix_document(summaries)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(matrix), encoding="utf-8")

    print(json.dumps(matrix, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())