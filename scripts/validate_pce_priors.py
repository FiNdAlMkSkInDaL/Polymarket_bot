#!/usr/bin/env python3
"""Validate PCE structural priors against recorded tick data.

Replays tick data from ``data/vps_march2026/ticks/``, builds 1-minute
bars per market (using YES-side midpoint prices), computes realised
pairwise Pearson correlations, and compares against the structural
priors (same_event=0.85, same_tag=0.30, baseline=0.05).

Usage:
    python scripts/validate_pce_priors.py [--tick-dir data/vps_march2026/ticks]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────

STRUCTURAL_PRIORS = {
    "same_event": 0.85,
    "same_tag": 0.30,
    "baseline": 0.05,
}

MIN_OVERLAP_BARS = 30  # need at least 30 concurrent 1-min bars


# ── Math helpers ───────────────────────────────────────────────────────────

def _log_returns(closes: list[float]) -> list[float]:
    """Compute 1-period log returns from a price series."""
    rets: list[float] = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0:
            rets.append(math.log(closes[i] / closes[i - 1]))
    return rets


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pure-Python Pearson correlation; returns None if insufficient data."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy)


# ── Data loading ───────────────────────────────────────────────────────────

def load_market_map(project_root: Path) -> dict:
    """Load market_map.json and build lookup tables.

    Returns
    -------
    dict with keys:
      - ``token_to_market``: token_id → market_id
      - ``yes_tokens``:      set of YES-side token ids
      - ``no_tokens``:       set of NO-side token ids
      - ``market_to_yes``:   market_id → yes_token_id
      - ``market_yes_no_pairs``: set of (yes_token, no_token) tuples
    """
    mm_path = project_root / "data" / "market_map.json"
    if not mm_path.exists():
        print(f"ERROR: market_map.json not found at {mm_path}", file=sys.stderr)
        sys.exit(1)

    with open(mm_path) as f:
        entries = json.load(f)

    token_to_market: dict[str, str] = {}
    yes_tokens: set[str] = set()
    no_tokens: set[str] = set()
    market_to_yes: dict[str, str] = {}
    complement_pairs: set[tuple[str, str]] = set()

    for e in entries:
        mid = e["market_id"]
        yid = e["yes_id"]
        nid = e["no_id"]
        token_to_market[yid] = mid
        token_to_market[nid] = mid
        yes_tokens.add(yid)
        no_tokens.add(nid)
        market_to_yes[mid] = yid
        complement_pairs.add(tuple(sorted([yid, nid])))

    return {
        "token_to_market": token_to_market,
        "yes_tokens": yes_tokens,
        "no_tokens": no_tokens,
        "market_to_yes": market_to_yes,
        "complement_pairs": complement_pairs,
    }


def build_1min_bars(tick_dir: Path, yes_tokens: set[str]) -> dict[str, dict[int, float]]:
    """Read JSONL tick files and build 1-min close bars per YES token.

    We extract the YES-side midpoint (best_bid + best_ask) / 2 from
    ``price_change`` events.  When midpoint isn't available we fall
    back to the trade ``price``.

    Returns
    -------
    dict: market_id → {bucket_minute: close_price}
    """
    # market-level files contain price_changes for both YES and NO tokens
    market_bars: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    if not tick_dir.exists():
        print(f"ERROR: tick directory not found: {tick_dir}", file=sys.stderr)
        sys.exit(1)

    date_dirs = sorted(d for d in tick_dir.iterdir() if d.is_dir())
    print(f"Found {len(date_dirs)} date directories in {tick_dir}")

    for date_dir in date_dirs:
        for tick_file in sorted(date_dir.glob("*.jsonl")):
            asset_id = tick_file.stem

            with open(tick_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts = rec.get("local_ts", 0.0)
                    if ts <= 0:
                        continue

                    payload = rec.get("payload", {})
                    price_changes = payload.get("price_changes", [])
                    bucket = int(ts // 60)

                    for pc in price_changes:
                        tok_id = pc.get("asset_id", "")
                        if tok_id not in yes_tokens:
                            continue

                        # Prefer midpoint from best_bid/best_ask
                        try:
                            bid = float(pc.get("best_bid", 0))
                            ask = float(pc.get("best_ask", 0))
                            if bid > 0 and ask > 0:
                                mid = (bid + ask) / 2
                            else:
                                mid = float(pc.get("price", 0))
                        except (ValueError, TypeError):
                            continue

                        if mid <= 0:
                            continue

                        # Use the market_id from the payload
                        mkt_id = payload.get("market", asset_id)
                        market_bars[mkt_id][bucket].append((ts, mid))

    # Collapse to 1-min close (last price in bucket)
    result: dict[str, dict[int, float]] = {}
    for mkt_id, buckets in market_bars.items():
        closes: dict[int, float] = {}
        for bucket in sorted(buckets):
            ticks_in_bucket = buckets[bucket]
            ticks_in_bucket.sort(key=lambda t: t[0])
            closes[bucket] = ticks_in_bucket[-1][1]
        result[mkt_id] = closes

    return result


def compute_pairwise_correlations(
    market_bars: dict[str, dict[int, float]],
    min_overlap: int = MIN_OVERLAP_BARS,
) -> list[dict]:
    """Compute Pearson correlation between all market pairs.

    Aligns on common 1-minute buckets, computes log returns, then Pearson r.

    Returns list of dicts with keys: market_a, market_b, correlation,
    overlap_bars.
    """
    market_ids = sorted(market_bars.keys())
    results: list[dict] = []
    n = len(market_ids)
    print(f"\nComputing pairwise correlations for {n} markets "
          f"({n * (n - 1) // 2} pairs) …")

    for i in range(n):
        for j in range(i + 1, n):
            a, b = market_ids[i], market_ids[j]
            bars_a = market_bars[a]
            bars_b = market_bars[b]

            # Intersection of buckets
            common = sorted(set(bars_a) & set(bars_b))
            if len(common) < min_overlap + 1:
                continue

            closes_a = [bars_a[t] for t in common]
            closes_b = [bars_b[t] for t in common]

            rets_a = _log_returns(closes_a)
            rets_b = _log_returns(closes_b)

            corr = _pearson(rets_a, rets_b)
            if corr is not None:
                results.append({
                    "market_a": a[:16] + "…",
                    "market_a_full": a,
                    "market_b": b[:16] + "…",
                    "market_b_full": b,
                    "correlation": round(corr, 4),
                    "overlap_bars": len(common),
                })

    return results


# ── Reporting ──────────────────────────────────────────────────────────────

def percentile(values: list[float], p: float) -> float:
    """Simple percentile (nearest-rank method)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return s[k]


def report(pairs: list[dict]) -> None:
    """Print summary statistics and compare against structural priors."""
    if not pairs:
        print("\nNo pairs with sufficient overlap — cannot validate priors.")
        return

    corrs = [p["correlation"] for p in pairs]
    abs_corrs = [abs(c) for c in corrs]

    print(f"\n{'=' * 64}")
    print("PCE Structural Prior Validation Report")
    print(f"{'=' * 64}")
    print(f"  Total pairs analysed  : {len(pairs)}")
    print(f"  Min overlap (bars)    : {min(p['overlap_bars'] for p in pairs)}")
    print(f"  Max overlap (bars)    : {max(p['overlap_bars'] for p in pairs)}")
    print()
    print("Cross-market correlation distribution (log-return Pearson r):")
    print(f"  Mean       : {sum(corrs) / len(corrs):+.4f}")
    print(f"  Median     : {percentile(corrs, 50):+.4f}")
    print(f"  Std Dev    : {(sum((c - sum(corrs)/len(corrs))**2 for c in corrs) / len(corrs))**0.5:.4f}")
    print(f"  5th pctile : {percentile(corrs, 5):+.4f}")
    print(f"  25th pctile: {percentile(corrs, 25):+.4f}")
    print(f"  75th pctile: {percentile(corrs, 75):+.4f}")
    print(f"  95th pctile: {percentile(corrs, 95):+.4f}")
    print(f"  |r| mean   : {sum(abs_corrs) / len(abs_corrs):.4f}")

    print(f"\n{'─' * 64}")
    print("Comparison against structural priors:")
    baseline_prior = STRUCTURAL_PRIORS["baseline"]
    mean_corr = sum(corrs) / len(corrs)
    print(f"  Baseline prior           : {baseline_prior:+.4f}")
    print(f"  Observed mean            : {mean_corr:+.4f}")
    deviation = abs(mean_corr - baseline_prior)
    verdict = "OK" if deviation < 0.10 else "REVIEW NEEDED"
    print(f"  Deviation                : {deviation:.4f}  [{verdict}]")

    # Fraction of pairs with |r| > 0.30 (would be misclassified as same-tag)
    high = sum(1 for c in abs_corrs if c > 0.30)
    print(f"\n  Pairs with |r| > 0.30    : {high}/{len(pairs)} "
          f"({100 * high / len(pairs):.1f}%)")
    high85 = sum(1 for c in abs_corrs if c > 0.50)
    print(f"  Pairs with |r| > 0.50    : {high85}/{len(pairs)} "
          f"({100 * high85 / len(pairs):.1f}%)")

    print(f"\n{'─' * 64}")
    print("Top-10 most correlated pairs:")
    ranked = sorted(pairs, key=lambda p: abs(p["correlation"]), reverse=True)
    for k, p in enumerate(ranked[:10], 1):
        print(f"  {k:2d}. r={p['correlation']:+.4f}  "
              f"overlap={p['overlap_bars']:4d}  "
              f"{p['market_a']}  ↔  {p['market_b']}")

    print(f"\n{'─' * 64}")
    print("Bottom-5 (least correlated pairs):")
    for k, p in enumerate(ranked[-5:], 1):
        print(f"  {k:2d}. r={p['correlation']:+.4f}  "
              f"overlap={p['overlap_bars']:4d}  "
              f"{p['market_a']}  ↔  {p['market_b']}")

    # Recommendation
    print(f"\n{'=' * 64}")
    print("Recommendations:")
    if deviation < 0.05:
        print("  ✓ Baseline prior (0.05) aligns well with observed data.")
    elif deviation < 0.15:
        print(f"  ⚠ Consider adjusting baseline prior to ~{mean_corr:.2f}")
    else:
        print(f"  ✗ Baseline prior ({baseline_prior}) significantly misaligned "
              f"(observed {mean_corr:+.3f}). Immediate recalibration recommended.")

    if high / len(pairs) > 0.10:
        print("  ⚠ >10% of cross-market pairs show |r| > 0.30 — "
              "same-tag tagging quality or prior value may need revision.")
    else:
        print("  ✓ <10% of pairs above same-tag threshold — tag structure adequate.")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate PCE structural priors against tick data"
    )
    parser.add_argument(
        "--tick-dir",
        default="data/vps_march2026/ticks",
        help="Path to tick data directory (default: data/vps_march2026/ticks)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    tick_dir = Path(args.tick_dir)
    if not tick_dir.is_absolute():
        tick_dir = project_root / tick_dir

    print("PCE Structural Prior Validator")
    print(f"Tick directory: {tick_dir}")

    # Load market metadata
    mm = load_market_map(project_root)
    print(f"Loaded {len(mm['market_to_yes'])} markets from market_map.json")

    # Build 1-minute bars
    market_bars = build_1min_bars(tick_dir, mm["yes_tokens"])
    print(f"Built bars for {len(market_bars)} markets")

    if len(market_bars) < 2:
        print("ERROR: need at least 2 markets with bars to compute correlations",
              file=sys.stderr)
        sys.exit(1)

    # Compute pairwise correlations
    pairs = compute_pairwise_correlations(market_bars)
    print(f"Computed {len(pairs)} pairs with ≥{MIN_OVERLAP_BARS} overlap bars")

    # Report
    report(pairs)


if __name__ == "__main__":
    main()
