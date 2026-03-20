#!/usr/bin/env python3
"""Offline SI-7 pricing calibrator using Black-Scholes ITM probability.

This research-only helper plots the risk-neutral probability of expiring
in-the-money, $N(d_2)$, across a range of spot prices near a fixed strike.
It is intended for offline SI-7 calibration work and does not import or modify
the live engine.

Example:
    python scripts/calibrate_si7_pricing.py --strike 100 --days-to-expiry 7 \
        --ivs 0.50 0.75 1.00 --output si7_pricing_calibration.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _probability_or_percent(value: str) -> float:
    parsed = float(value)
    if 0.0 <= parsed <= 1.0:
        return parsed
    if 1.0 < parsed <= 100.0:
        return parsed / 100.0
    raise argparse.ArgumentTypeError("value must be in [0, 1] or [0, 100]")


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes_itm_probability(
    spot: float,
    strike: float,
    time_to_expiry_years: float,
    implied_volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    if time_to_expiry_years <= 0.0:
        if spot > strike:
            return 1.0
        if spot < strike:
            return 0.0
        return 0.5

    if implied_volatility <= 0.0:
        forward = spot * math.exp(risk_free_rate * time_to_expiry_years)
        if forward > strike:
            return 1.0
        if forward < strike:
            return 0.0
        return 0.5

    sigma_sqrt_t = implied_volatility * math.sqrt(time_to_expiry_years)
    d2 = (
        math.log(spot / strike)
        + (risk_free_rate - 0.5 * implied_volatility * implied_volatility) * time_to_expiry_years
    ) / sigma_sqrt_t
    return normal_cdf(d2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Black-Scholes ITM probability curves for offline SI-7 calibration.",
    )
    parser.add_argument(
        "--strike",
        type=_positive_float,
        required=True,
        help="Option strike price.",
    )
    parser.add_argument(
        "--days-to-expiry",
        type=_positive_float,
        default=7.0,
        help="Time to expiry in calendar days. Default: 7.",
    )
    parser.add_argument(
        "--ivs",
        type=_probability_or_percent,
        nargs="+",
        default=[0.50, 0.75, 1.00],
        help="IV scenarios as decimals or percentages. Default: 0.50 0.75 1.00.",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annualized risk-free rate as a decimal. Default: 0.0.",
    )
    parser.add_argument(
        "--spot-band",
        type=_non_negative_float,
        default=0.25,
        help="Fractional distance around strike for the spot grid. Default: 0.25 for +/-25%%.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=401,
        help="Number of spot points to sample. Default: 401.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("si7_pricing_calibration.png"),
        help="Output PNG path. Default: si7_pricing_calibration.png.",
    )
    return parser


def plot_curves(
    strike: float,
    days_to_expiry: float,
    ivs: list[float],
    risk_free_rate: float,
    spot_band: float,
    points: int,
    output_path: Path,
) -> None:
    if points < 2:
        raise SystemExit("--points must be at least 2")

    time_to_expiry_years = days_to_expiry / 365.0
    min_spot = strike * (1.0 - spot_band)
    max_spot = strike * (1.0 + spot_band)
    if min_spot <= 0.0:
        raise SystemExit("spot range must stay above zero; reduce --spot-band or raise --strike")

    spot_grid = np.linspace(min_spot, max_spot, points)

    figure, axis = plt.subplots(figsize=(11, 6.5))
    for implied_volatility in ivs:
        probabilities = [
            black_scholes_itm_probability(
                spot=float(spot),
                strike=strike,
                time_to_expiry_years=time_to_expiry_years,
                implied_volatility=implied_volatility,
                risk_free_rate=risk_free_rate,
            )
            for spot in spot_grid
        ]
        axis.plot(
            spot_grid,
            probabilities,
            linewidth=2.2,
            label=f"IV {implied_volatility * 100:.0f}%",
        )

    axis.axvline(strike, color="black", linestyle="--", linewidth=1.2, label="Strike")
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
    axis.set_title("SI-7 offline calibration: Black-Scholes ITM probability vs. spot")
    axis.set_xlabel("Spot price")
    axis.set_ylabel("Polymarket-style probability")
    axis.set_xlim(min_spot, max_spot)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    axis.legend()

    subtitle = (
        f"Strike={strike:.2f} | DTE={days_to_expiry:.2f} days | "
        f"r={risk_free_rate * 100:.2f}%"
    )
    figure.text(0.5, 0.94, subtitle, ha="center", va="center")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    plot_curves(
        strike=args.strike,
        days_to_expiry=args.days_to_expiry,
        ivs=args.ivs,
        risk_free_rate=args.risk_free_rate,
        spot_band=args.spot_band,
        points=args.points,
        output_path=args.output,
    )
    print("SI-7 offline calibration plot saved")
    print(f"output_path           : {args.output}")
    print(f"strike                : {args.strike:.4f}")
    print(f"days_to_expiry        : {args.days_to_expiry:.4f}")
    print(f"risk_free_rate        : {args.risk_free_rate:.6f}")
    print(f"iv_scenarios          : {', '.join(f'{value * 100:.0f}%' for value in args.ivs)}")


if __name__ == "__main__":
    main()