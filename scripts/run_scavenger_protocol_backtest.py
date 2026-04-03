#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.scavenger_protocol import ScavengerConfig, run_scavenger_backtest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorized Polars backtest for Strategy 1: The Scavenger Protocol.",
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Parquet path, glob pattern, or list of Parquet files matching the scavenger schema.",
    )
    parser.add_argument(
        "--fills-output",
        type=Path,
        default=None,
        help="Optional output path for fill-level results (.parquet, .csv, or .json).",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional output path for the one-row summary (.parquet, .csv, or .json).",
    )
    parser.add_argument("--resolution-window-hours", type=int, default=72)
    parser.add_argument("--signal-best-ask-min", type=float, default=0.99)
    parser.add_argument("--signal-best-bid-max", type=float, default=0.96)
    parser.add_argument("--maker-bid-price", type=float, default=0.95)
    parser.add_argument(
        "--head",
        type=int,
        default=20,
        help="Number of fill rows to print after the summary.",
    )
    return parser.parse_args()


def _write_frame(frame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        frame.write_parquet(output_path)
        return
    if suffix == ".csv":
        frame.write_csv(output_path)
        return
    if suffix == ".json":
        output_path.write_text(frame.write_json(), encoding="utf-8")
        return
    raise ValueError(f"Unsupported output format: {output_path}")


def main() -> None:
    args = _parse_args()
    config = ScavengerConfig(
        resolution_window_hours=args.resolution_window_hours,
        signal_best_ask_min=args.signal_best_ask_min,
        signal_best_bid_max=args.signal_best_bid_max,
        maker_bid_price=args.maker_bid_price,
    )
    source = args.input[0] if len(args.input) == 1 else args.input
    result = run_scavenger_backtest(source, config=config)

    print("Scavenger Protocol summary")
    print(result.summary)

    if result.fills.height:
        print("\nFill sample")
        print(result.fills.head(args.head))
    else:
        print("\nNo fills matched the maker touch-through rules.")

    if args.fills_output is not None:
        _write_frame(result.fills, args.fills_output)

    if args.summary_output is not None:
        _write_frame(result.summary, args.summary_output)


if __name__ == "__main__":
    main()