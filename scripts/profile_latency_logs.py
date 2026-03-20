#!/usr/bin/env python3
"""Offline latency histogram profiler for exported journald logs.

This script is intentionally standalone. It parses raw journald text exports,
extracts latency-like processing and heartbeat deltas with permissive regexes,
prints a compact summary, and renders a histogram with a red block-threshold
marker at 5000ms by default.

The regex set is deliberately broad because the exact production log export
shape may vary depending on how journald formatting and structlog fields were
captured. Tighten the patterns later once a representative sample is locked.

Examples
--------
python scripts/profile_latency_logs.py exported-journal.txt
python scripts/profile_latency_logs.py exported-journal.txt --output latency_hist.png
python scripts/profile_latency_logs.py exported-journal.txt --threshold-ms 5000 --bins 60
python scripts/profile_latency_logs.py exported-journal.txt --extra-regex "custom_latency.*?duration=(?P<ms>\\d+(?:\\.\\d+)?)"
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - import guard only
    raise SystemExit(
        "matplotlib is required for this script. Install it with `pip install matplotlib`."
    ) from exc


DEFAULT_THRESHOLD_MS = 5000.0
DEFAULT_BINS = 50
TIMESTAMP_RE = re.compile(
    r"(?P<timestamp>"
    r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+(?:Z|[+-]\d{2}:?\d{2})?"
    r"|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
    r")"
)


@dataclass(frozen=True)
class PatternSpec:
    name: str
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class LatencySample:
    line_no: int
    timestamp: str | None
    label: str
    latency_ms: float
    raw_line: str


PATTERN_SPECS: tuple[PatternSpec, ...] = (
    PatternSpec(
        "latency_blocked",
        re.compile(
            r"\blatency_BLOCKED\b.*?\bdelta_ms\s*[=:]\s*(?P<ms>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
    PatternSpec(
        "latency_degraded",
        re.compile(
            r"\blatency_degraded\b.*?\bdelta_ms\s*[=:]\s*(?P<ms>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
    PatternSpec(
        "latency_recovered",
        re.compile(
            r"\blatency_recovered\b.*?\bdelta_ms\s*[=:]\s*(?P<ms>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
    PatternSpec(
        "latency_delta_text",
        re.compile(r"\bDelta:\s*(?P<ms>\d+(?:\.\d+)?)\s*ms\b", re.IGNORECASE),
    ),
    PatternSpec(
        "heartbeat_gap",
        re.compile(
            r"\bheartbeat(?:_[a-z]+)*\b.*?"
            r"\b(?:gap_ms|lag_ms|delta_ms)\s*[=:]\s*(?P<ms>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
    PatternSpec(
        "processing_latency",
        re.compile(
            r"\b(?:fast_strike_latency|processing(?:_[a-z]+)*|trade_processing_[a-z_]+)\b.*?"
            r"\b(?:signal_to_ack_ms|latency_ms|processing_ms|duration_ms)\s*[=:]\s*"
            r"(?P<ms>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
    PatternSpec(
        "generic_latency_metric",
        re.compile(
            r"\b(?:latency|heartbeat|processing)[\w:-]*\b.*?"
            r"\b(?:delta_ms|gap_ms|lag_ms|latency_ms|signal_to_ack_ms|processing_ms|duration_ms)"
            r"\s*[=:]\s*(?P<ms>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse exported journald text logs, extract latency-like events, and "
            "plot a latency histogram."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to a raw journald export text file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Defaults to <input_stem>_latency_hist.png.",
    )
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=DEFAULT_THRESHOLD_MS,
        help=f"Block threshold marker in milliseconds (default: {DEFAULT_THRESHOLD_MS:.0f}).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"Histogram bin count (default: {DEFAULT_BINS}).",
    )
    parser.add_argument(
        "--title",
        default="Offline Latency Profile",
        help="Chart title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the chart interactively in addition to saving it.",
    )
    parser.add_argument(
        "--extra-regex",
        action="append",
        default=[],
        help=(
            "Optional additional regex with a required named capture group `ms` and an "
            "optional `label` group. May be passed multiple times."
        ),
    )
    return parser.parse_args(argv)


def build_patterns(extra_patterns: Sequence[str]) -> list[PatternSpec]:
    pattern_specs = list(PATTERN_SPECS)
    for index, pattern_text in enumerate(extra_patterns, start=1):
        compiled = re.compile(pattern_text, re.IGNORECASE)
        if "ms" not in compiled.groupindex:
            raise ValueError(
                "Each --extra-regex value must define a named capture group `ms`."
            )
        pattern_specs.append(PatternSpec(f"custom_{index}", compiled))
    return pattern_specs


def extract_latency_samples(
    lines: Iterable[str],
    pattern_specs: Sequence[PatternSpec],
) -> list[LatencySample]:
    samples: list[LatencySample] = []
    for line_no, raw_line in enumerate(lines, start=1):
        timestamp_match = TIMESTAMP_RE.search(raw_line)
        timestamp = timestamp_match.group("timestamp") if timestamp_match else None

        for spec in pattern_specs:
            match = spec.pattern.search(raw_line)
            if match is None:
                continue
            latency_ms = float(match.group("ms"))
            if not math.isfinite(latency_ms):
                break
            label = match.groupdict().get("label") or spec.name
            samples.append(
                LatencySample(
                    line_no=line_no,
                    timestamp=timestamp,
                    label=label,
                    latency_ms=latency_ms,
                    raw_line=raw_line.rstrip("\n"),
                )
            )
            break
    return samples


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty sequence.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * q
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    weight = rank - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return float(lower + (upper - lower) * weight)


def print_summary(samples: Sequence[LatencySample], threshold_ms: float) -> None:
    latencies = sorted(sample.latency_ms for sample in samples)
    counts = Counter(sample.label for sample in samples)
    blocked_count = sum(sample.latency_ms >= threshold_ms for sample in samples)
    first_ts = next((sample.timestamp for sample in samples if sample.timestamp), None)
    last_ts = next((sample.timestamp for sample in reversed(samples) if sample.timestamp), None)

    print()
    print("=" * 72)
    print("OFFLINE LATENCY PROFILE")
    print("=" * 72)
    print(f"Parsed samples        : {len(samples)}")
    print(f"Threshold             : {threshold_ms:.1f} ms")
    if first_ts or last_ts:
        print(f"Observed window       : {first_ts or 'unknown'} -> {last_ts or 'unknown'}")
    print(f"Mean                  : {sum(latencies) / len(latencies):.1f} ms")
    print(f"Median                : {percentile(latencies, 0.50):.1f} ms")
    print(f"P95                   : {percentile(latencies, 0.95):.1f} ms")
    print(f"P99                   : {percentile(latencies, 0.99):.1f} ms")
    print(f"Max                   : {latencies[-1]:.1f} ms")
    print(
        f"At or above threshold : {blocked_count} / {len(samples)} "
        f"({blocked_count / len(samples) * 100:.1f}%)"
    )
    print()
    print("Event breakdown:")
    for label, count in counts.most_common():
        print(f"  {label:<24} {count:>6}")


def plot_histogram(
    samples: Sequence[LatencySample],
    output_path: Path,
    threshold_ms: float,
    bins: int,
    title: str,
    show: bool,
) -> None:
    latencies = [sample.latency_ms for sample in samples]
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(latencies, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.85)
    ax.axvline(
        threshold_ms,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Block threshold ({threshold_ms:.0f} ms)",
    )

    median_ms = percentile(sorted(latencies), 0.50)
    p95_ms = percentile(sorted(latencies), 0.95)
    ax.axvline(median_ms, color="#2E8B57", linestyle=":", linewidth=1.8, label=f"Median ({median_ms:.0f} ms)")
    ax.axvline(p95_ms, color="#F28E2B", linestyle=":", linewidth=1.8, label=f"P95 ({p95_ms:.0f} ms)")

    ax.set_title(title)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Event count")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print()
    print(f"Saved histogram to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input_path)
    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_latency_hist.png"
    )
    pattern_specs = build_patterns(args.extra_regex)

    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        samples = extract_latency_samples(handle, pattern_specs)

    if not samples:
        raise SystemExit(
            "No latency-like samples were parsed from the log file. "
            "Tighten or extend the regex set with --extra-regex once the exact log format is known."
        )

    print_summary(samples, threshold_ms=args.threshold_ms)
    plot_histogram(
        samples,
        output_path=output_path,
        threshold_ms=args.threshold_ms,
        bins=max(1, args.bins),
        title=args.title,
        show=args.show,
    )


if __name__ == "__main__":
    main()