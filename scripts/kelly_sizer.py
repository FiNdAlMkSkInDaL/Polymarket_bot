#!/usr/bin/env python3
"""Standalone Kelly Criterion position sizing calculator.

Usage:
    python scripts/kelly_sizer.py \
        --win-rate 58 \
        --avg-win 12.50 \
        --avg-loss 8.00 \
        --bankroll 5000

Win rate accepts either a decimal probability in [0, 1] or a percentage in
[0, 100]. The script prints both full-Kelly and half-Kelly recommended
pure_mm_wide_size_usd values.
"""

from __future__ import annotations

import argparse


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _win_rate(value: str) -> float:
    parsed = float(value)
    if 0.0 <= parsed <= 1.0:
        return parsed
    if 1.0 < parsed <= 100.0:
        return parsed / 100.0
    raise argparse.ArgumentTypeError("win rate must be in [0, 1] or [0, 100]")


def _clamp_fraction(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate full-Kelly and half-Kelly position sizes from WFO stats.",
    )
    parser.add_argument(
        "--win-rate",
        required=True,
        type=_win_rate,
        help="Win probability from WFO. Accepts 0-1 or 0-100.",
    )
    parser.add_argument(
        "--avg-win",
        required=True,
        type=_positive_float,
        help="Average win in USD.",
    )
    parser.add_argument(
        "--avg-loss",
        required=True,
        type=_positive_float,
        help="Average loss in USD as a positive magnitude.",
    )
    parser.add_argument(
        "--bankroll",
        required=True,
        type=_positive_float,
        help="Total bankroll in USD.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    p = args.win_rate
    q = 1.0 - p
    b = args.avg_win / args.avg_loss

    raw_kelly_fraction = p - (q / b)
    full_kelly_fraction = _clamp_fraction(raw_kelly_fraction)
    half_kelly_fraction = _clamp_fraction(raw_kelly_fraction / 2.0)

    full_kelly_size_usd = args.bankroll * full_kelly_fraction
    half_kelly_size_usd = args.bankroll * half_kelly_fraction

    print("Kelly sizing inputs")
    print("-" * 40)
    print(f"win_probability_p      : {p:.4f} ({p * 100:.2f}%)")
    print(f"loss_probability_q     : {q:.4f} ({q * 100:.2f}%)")
    print(f"avg_win_usd            : ${args.avg_win:,.2f}")
    print(f"avg_loss_usd           : ${args.avg_loss:,.2f}")
    print(f"win_loss_ratio_b       : {b:.4f}")
    print(f"bankroll_usd           : ${args.bankroll:,.2f}")
    print()
    print("Kelly outputs")
    print("-" * 40)
    print(f"raw_kelly_fraction_f*  : {raw_kelly_fraction:.4f} ({raw_kelly_fraction * 100:.2f}%)")
    if raw_kelly_fraction < 0.0:
        print("note                   : negative Kelly edge, recommended size is zero")
    elif raw_kelly_fraction > 1.0:
        print("note                   : Kelly exceeds 100% of bankroll, output is capped at bankroll")
    print(f"full_kelly_fraction    : {full_kelly_fraction:.4f} ({full_kelly_fraction * 100:.2f}%)")
    print(f"half_kelly_fraction    : {half_kelly_fraction:.4f} ({half_kelly_fraction * 100:.2f}%)")
    print()
    print("Config-ready sizing")
    print("-" * 40)
    print(f"pure_mm_wide_size_usd (full Kelly): ${full_kelly_size_usd:,.2f}")
    print(f"pure_mm_wide_size_usd (half Kelly): ${half_kelly_size_usd:,.2f}")
    print()
    print(f"PURE_MM_WIDE_SIZE_USD={half_kelly_size_usd:.2f}    # half-Kelly default")


if __name__ == "__main__":
    main()