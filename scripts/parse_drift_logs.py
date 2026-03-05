#!/usr/bin/env python3
"""
Drift Signal Diagnostic Parser — ``parse_drift_logs.py``
=========================================================

Parses structlog JSONL output from the Polymarket bot production logs
and aggregates exactly *which* of the 6 MeanReversionDrift gates
blocked the signal from firing.

Requires the instrumented ``drift_eval`` debug events added to
``src/signals/drift_signal.py``.  Also scans for the legacy
``drift_signal_fired`` events emitted on successful fires.

Usage
-----
    # Parse the latest log file (auto-detects logs/ directory):
    python scripts/parse_drift_logs.py

    # Parse a specific log file:
    python scripts/parse_drift_logs.py logs/bot.jsonl

    # Parse rotated logs (VPS data drop):
    python scripts/parse_drift_logs.py data/vps_march2026/logs/bot.jsonl.*

    # Read from stdin (e.g. live tail):
    python scripts/parse_drift_logs.py -

    # Time-window filter (ISO-8601 start/end):
    python scripts/parse_drift_logs.py --after 2026-03-04T18:00:00Z --before 2026-03-05T05:00:00Z

    # Include parameter-calibration recommendations:
    python scripts/parse_drift_logs.py --recommend
"""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator

# ─── Ensure stdout can handle Unicode on Windows ─────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )

# ─── Gate definitions ────────────────────────────────────────────────
# Maps the ``gate`` field in the ``drift_eval`` log to a human label.

GATE_ORDER = [
    "regime",
    "l2_unreliable",
    "insufficient_history",
    "ewma_vol_bounds",
    "high_volume_bar",
    "sigma_vwap_invalid",         # sub-gate between 5 and 6
    "displacement_below_threshold",
    "all_passed",
]

GATE_LABELS = {
    "regime":                       "Gate 1 — Regime ≠ mean-revert",
    "l2_unreliable":                "Gate 2 — L2 book unreliable",
    "insufficient_history":         "Gate 3 — Insufficient bar history",
    "ewma_vol_bounds":              "Gate 4 — EWMA vol out of bounds",
    "high_volume_bar":              "Gate 5 — High-volume bar exclusion",
    "sigma_vwap_invalid":           "Gate 4b — σ/VWAP ≤ 0 (data quality)",
    "displacement_below_threshold": "Gate 6 — |displacement| < z_threshold",
    "all_passed":                   "✓ All gates passed → signal fired",
}

# ─── ANSI colour helpers ─────────────────────────────────────────────
_NO_COLOUR = os.environ.get("NO_COLOR") is not None

def _c(code: str, text: str) -> str:
    if _NO_COLOUR or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def _red(t: str) -> str:    return _c("31", t)
def _green(t: str) -> str:  return _c("32", t)
def _yellow(t: str) -> str: return _c("33", t)
def _cyan(t: str) -> str:   return _c("36", t)
def _bold(t: str) -> str:   return _c("1", t)
def _dim(t: str) -> str:    return _c("2", t)

# ─── Time parsing ────────────────────────────────────────────────────

def _parse_ts(ts_str: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp from structlog."""
    if not ts_str:
        return None
    try:
        # Handle both 'Z' suffix and '+00:00'
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None


def _parse_bound(s: str | None) -> datetime | None:
    """Parse a CLI --after/--before timestamp."""
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        print(f"WARNING: Could not parse time bound '{s}', ignoring.", file=sys.stderr)
        return None


# ─── Log file discovery ──────────────────────────────────────────────

def _discover_log_files() -> list[str]:
    """Find JSONL log files in priority order."""
    candidates: list[str] = []
    base = Path(__file__).resolve().parent.parent

    # 1. Primary log directory
    log_dir = base / "logs"
    if log_dir.is_dir():
        for f in sorted(log_dir.glob("bot.jsonl*")):
            candidates.append(str(f))

    # 2. VPS data drops
    vps_logs = base / "data" / "vps_march2026" / "logs"
    if vps_logs.is_dir():
        for f in sorted(vps_logs.glob("bot.jsonl*")):
            candidates.append(str(f))

    return candidates


# ─── JSONL streaming reader ──────────────────────────────────────────

def _iter_jsonl(stream: IO[str]) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL stream, skipping bad lines."""
    for line_no, raw in enumerate(stream, 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            # Truncated log rotation artefact — skip silently.
            pass


# ─── Core aggregation ────────────────────────────────────────────────

class DriftDiagnostics:
    """Accumulates drift evaluation telemetry from parsed log entries."""

    def __init__(self):
        # Per-gate rejection counts
        self.gate_counts: dict[str, int] = collections.Counter()
        # Successful fires
        self.fires: list[dict] = []
        # EWMA vol values seen at gate-4 rejection
        self.ewma_vol_samples: list[float] = []
        # EWMA vol == 0 count
        self.ewma_vol_zero_count: int = 0
        # Displacement values seen at gate-6 near-miss
        self.displacement_samples: list[float] = []
        # Per-market breakdown
        self.per_market: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        # Total evaluations
        self.total_evals: int = 0
        # Timestamps for session boundary
        self.first_ts: datetime | None = None
        self.last_ts: datetime | None = None
        # Bar count samples at gate-3 rejection
        self.bars_available_samples: list[int] = []
        # Hourly gate breakdown
        self.hourly_gates: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

    def ingest(self, entry: dict) -> None:
        """Process a single log entry."""
        event = entry.get("event", "")

        if event == "drift_eval":
            self._ingest_eval(entry)
        elif event == "drift_signal_fired":
            # Legacy fire event (pre-instrumentation logs)
            self.fires.append(entry)

    def _ingest_eval(self, entry: dict) -> None:
        gate = entry.get("gate", "unknown")
        market = entry.get("market_id", "unknown")
        ts = _parse_ts(entry.get("timestamp"))

        self.total_evals += 1
        self.gate_counts[gate] += 1
        self.per_market[market][gate] += 1

        # Track session boundaries
        if ts:
            if self.first_ts is None or ts < self.first_ts:
                self.first_ts = ts
            if self.last_ts is None or ts > self.last_ts:
                self.last_ts = ts
            self.hourly_gates[ts.hour][gate] += 1

        # Collect diagnostic samples
        if gate == "ewma_vol_bounds":
            vol = entry.get("ewma_vol", None)
            if vol is not None:
                self.ewma_vol_samples.append(float(vol))
                if float(vol) <= 0:
                    self.ewma_vol_zero_count += 1

        if gate == "displacement_below_threshold":
            disp = entry.get("displacement", None)
            if disp is not None:
                self.displacement_samples.append(float(disp))

        if gate == "insufficient_history":
            bars = entry.get("bars_available", None)
            if bars is not None:
                self.bars_available_samples.append(int(bars))

        if gate == "all_passed":
            self.fires.append(entry)

    # ── Reporting ─────────────────────────────────────────────────

    def print_report(self, *, recommend: bool = False) -> None:
        """Print the full survival funnel report to stdout."""
        w = 72
        print()
        print(_bold("=" * w))
        print(_bold(" DRIFT SIGNAL DIAGNOSTIC — SURVIVAL FUNNEL"))
        print(_bold("=" * w))

        # Session info
        if self.first_ts and self.last_ts:
            duration = self.last_ts - self.first_ts
            hours = duration.total_seconds() / 3600
            print(f"  Session window : {self.first_ts.isoformat()} → {self.last_ts.isoformat()}")
            print(f"  Duration       : {hours:.1f} hours")
        print(f"  Total evals    : {_bold(str(self.total_evals))}")
        print(f"  Signals fired  : {_green(str(len(self.fires))) if self.fires else _red('0')}")
        print()

        if self.total_evals == 0:
            print(_yellow("  ⚠  No drift_eval events found in logs."))
            print(_dim("     Ensure the bot is running with the instrumented"))
            print(_dim("     drift_signal.py and LOG_LEVEL=debug."))
            print()
            if self.fires:
                print(f"  Found {len(self.fires)} legacy drift_signal_fired events.")
                for f in self.fires[:5]:
                    print(f"    {f.get('timestamp', '?')}  disp={f.get('displacement')}  "
                          f"vol={f.get('ewma_vol')}  dir={f.get('direction')}")
            print()
            return

        self._print_funnel(w)
        self._print_ewma_vol_analysis()
        self._print_displacement_analysis()
        self._print_hourly_breakdown()
        self._print_per_market_summary()

        if recommend:
            self._print_recommendations()

        print(_bold("=" * w))
        print()

    def _print_funnel(self, w: int) -> None:
        """Print the sequential gate survival funnel."""
        print(_bold("─" * w))
        print(_bold(" SURVIVAL FUNNEL"))
        print(_bold("─" * w))

        remaining = self.total_evals
        print(f"  {'Evaluations entering pipeline':<45} {remaining:>6}  (100.0%)")
        print()

        for gate_key in GATE_ORDER:
            count = self.gate_counts.get(gate_key, 0)
            if gate_key == "all_passed":
                pct_total = (count / self.total_evals * 100) if self.total_evals else 0
                colour = _green if count > 0 else _red
                print(f"  {colour(GATE_LABELS[gate_key]):<55} {colour(f'{count:>6}')}  "
                      f"({pct_total:>5.1f}% of total)")
            else:
                pct_total = (count / self.total_evals * 100) if self.total_evals else 0
                pct_stage = (count / remaining * 100) if remaining else 0
                label = GATE_LABELS.get(gate_key, gate_key)

                bar_len = int(pct_total / 2)
                bar = "█" * bar_len

                colour_fn = _red if pct_total > 40 else (_yellow if pct_total > 15 else _dim)
                print(f"  {label:<45} {_red(f'-{count:>5}')}  "
                      f"({pct_total:>5.1f}% total, {pct_stage:>5.1f}% of stage)")
                if bar:
                    print(f"  {'':45} {colour_fn(bar)}")
                remaining -= count

        print()

    def _print_ewma_vol_analysis(self) -> None:
        """Check for EWMA vol collapsing to 0.0 (zero-volume overnight bars)."""
        if not self.ewma_vol_samples:
            return

        print(_bold("─" * 72))
        print(_bold(" EWMA VOLATILITY ANALYSIS (Gate 4 rejections)"))
        print(_bold("─" * 72))

        total_g4 = len(self.ewma_vol_samples)
        zero_count = self.ewma_vol_zero_count
        zero_pct = (zero_count / total_g4 * 100) if total_g4 else 0

        print(f"  Gate-4 rejections          : {total_g4}")
        print(f"  ewma_vol == 0.0 (dead bars): {_red(str(zero_count))} ({zero_pct:.1f}%)")

        nonzero = [v for v in self.ewma_vol_samples if v > 0]
        if nonzero:
            print(f"  ewma_vol > ceiling (≥0.015): {total_g4 - zero_count}")
            print(f"  Non-zero vol stats:")
            print(f"    min    = {min(nonzero):.8f}")
            print(f"    median = {statistics.median(nonzero):.8f}")
            print(f"    max    = {max(nonzero):.8f}")
            if len(nonzero) > 1:
                print(f"    stdev  = {statistics.stdev(nonzero):.8f}")

        if zero_pct > 50:
            print()
            print(_red("  ⚠  CRITICAL: >50% of Gate-4 blocks are from ewma_vol == 0.0"))
            print(_red("     This means overnight zero-volume bars are collapsing the"))
            print(_red("     EWMA to zero, which silently kills displacement computation"))
            print(_red("     (ZeroDivisionError avoided but signal is dead)."))
            print()
            print(_yellow("     FIX: Inject a vol floor: max(ewma_vol, 1e-6) or skip"))
            print(_yellow("          bars with volume == 0 from the EWMA window."))
        elif zero_pct > 10:
            print()
            print(_yellow(f"  ⚠  WARNING: {zero_pct:.0f}% of Gate-4 blocks from vol == 0.0"))
            print(_yellow("     Consider adding a minimum vol floor."))
        print()

    def _print_displacement_analysis(self) -> None:
        """Analyse near-miss displacements at Gate 6."""
        if not self.displacement_samples:
            return

        print(_bold("─" * 72))
        print(_bold(" DISPLACEMENT NEAR-MISS ANALYSIS (Gate 6 rejections)"))
        print(_bold("─" * 72))

        abs_disps = [abs(d) for d in self.displacement_samples]
        total = len(abs_disps)

        print(f"  Gate-6 rejections : {total}")
        print(f"  |displacement| stats:")
        print(f"    min    = {min(abs_disps):.4f}")
        print(f"    median = {statistics.median(abs_disps):.4f}")
        print(f"    p75    = {_percentile(abs_disps, 75):.4f}")
        print(f"    p90    = {_percentile(abs_disps, 90):.4f}")
        print(f"    max    = {max(abs_disps):.4f}")

        # How many would pass at various thresholds?
        thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]
        print()
        print(f"  Threshold sensitivity (how many Gate-6 rejects would pass):")
        for thresh in thresholds:
            would_pass = sum(1 for d in abs_disps if d >= thresh)
            pct = would_pass / total * 100
            marker = " ← current" if abs(thresh - 1.0) < 0.001 else ""
            colour = _green if pct > 0 else _dim
            print(f"    z_threshold = {thresh:.2f}  →  {colour(f'{would_pass:>4}')} pass  "
                  f"({pct:>5.1f}%){marker}")
        print()

    def _print_hourly_breakdown(self) -> None:
        """Per-hour gate rejection heatmap."""
        if not self.hourly_gates:
            return

        print(_bold("─" * 72))
        print(_bold(" HOURLY GATE REJECTION HEATMAP"))
        print(_bold("─" * 72))

        hours = sorted(self.hourly_gates.keys())
        # Compact table: hour | total | top-2 gates
        print(f"  {'Hour':>4}  {'Evals':>6}  {'Top rejection gate':<35}  {'Count':>5}")
        print(f"  {'----':>4}  {'-----':>6}  {'-' * 35}  {'-----':>5}")

        for h in hours:
            counts = self.hourly_gates[h]
            total = sum(counts.values())
            # Top gate (excluding all_passed)
            rejects = {k: v for k, v in counts.items() if k != "all_passed"}
            if rejects:
                top_gate = max(rejects, key=rejects.get)
                top_count = rejects[top_gate]
                label = GATE_LABELS.get(top_gate, top_gate)[:35]
            else:
                label = _green("(all passed)")
                top_count = counts.get("all_passed", 0)
            print(f"  {h:>4}  {total:>6}  {label:<35}  {top_count:>5}")

        print()

    def _print_per_market_summary(self) -> None:
        """Per-market gate rejection summary (top 10)."""
        if not self.per_market:
            return

        print(_bold("─" * 72))
        print(_bold(" PER-MARKET BREAKDOWN (top 10 by eval count)"))
        print(_bold("─" * 72))

        markets = sorted(
            self.per_market.items(),
            key=lambda kv: sum(kv[1].values()),
            reverse=True,
        )[:10]

        for market_id, counts in markets:
            total = sum(counts.values())
            passed = counts.get("all_passed", 0)
            top_reject = "—"
            rejects = {k: v for k, v in counts.items() if k != "all_passed"}
            if rejects:
                top_key = max(rejects, key=rejects.get)
                top_reject = f"{GATE_LABELS.get(top_key, top_key)} ({rejects[top_key]})"
            print(f"  {market_id:<18}  evals={total:<5}  fired={passed:<3}  "
                  f"top_block: {top_reject}")
        print()

    def _print_recommendations(self) -> None:
        """Data-driven parameter calibration advice."""
        print(_bold("=" * 72))
        print(_bold(" PARAMETER CALIBRATION RECOMMENDATIONS"))
        print(_bold("=" * 72))
        print()

        # ── Gate 1: Regime ────────────────────────────────────────
        g1 = self.gate_counts.get("regime", 0)
        g1_pct = (g1 / self.total_evals * 100) if self.total_evals else 0
        if g1_pct > 60:
            print(_yellow("  [Gate 1 — Regime]"))
            print(f"  {g1_pct:.0f}% of evaluations blocked by regime ≠ mean-revert.")
            print("  The overnight Asian/European session often lacks the")
            print("  autocorrelation and directional persistence needed to")
            print("  classify as mean-reverting.  Options:")
            print("    a) Lower regime_threshold from 0.40 → 0.30 conditionally")
            print("       during [18:00–06:00 UTC] (low-liquidity window).")
            print("    b) Add a 'dormant regime' fallback: if no regime can be")
            print("       classified after N bars, default to mean-revert (since")
            print("       absence of trend ≈ mean-reversion by definition).")
            print()

        # ── Gate 2: L2 ────────────────────────────────────────────
        g2 = self.gate_counts.get("l2_unreliable", 0)
        g2_pct = (g2 / self.total_evals * 100) if self.total_evals else 0
        if g2_pct > 10:
            print(_yellow("  [Gate 2 — L2 Reliability]"))
            print(f"  {g2_pct:.0f}% blocked by L2 unreliable.")
            print("  During low-liquidity sessions, WebSocket sequence gaps widen")
            print("  because Polygon.io thins out L2 updates.  The seq_gap_rate")
            print("  exceeds the 2% default threshold even though the book is")
            print("  not actually desynced — it's just quiet.")
            print("  Recommendation: raise l2_max_seq_gap_rate from 0.02 → 0.05")
            print("  for overnight hours, OR relax the L2 gate entirely for")
            print("  drift signals (which are less latency-sensitive than panic).")
            print()

        # ── Gate 4: EWMA vol ──────────────────────────────────────
        g4 = self.gate_counts.get("ewma_vol_bounds", 0)
        if g4 > 0 and self.ewma_vol_samples:
            zero_pct = (self.ewma_vol_zero_count / len(self.ewma_vol_samples) * 100)
            above_ceil = [v for v in self.ewma_vol_samples if v >= 0.015]

            print(_yellow("  [Gate 4 — EWMA Volatility Bounds]"))
            if self.ewma_vol_zero_count > 0:
                print(f"  {self.ewma_vol_zero_count} rejections from vol == 0.0 "
                      f"({zero_pct:.0f}% of gate-4 blocks).")
                print("  Zero-volume overnight bars collapse the EWMA to zero.")
                print("  Mathematical fix:")
                print("    ewma_vol = max(rolling_volatility_ewma, VOL_FLOOR)")
                print("    where VOL_FLOOR = 1e-5 (ensures displacement remains")
                print("    finite while correctly reflecting ultra-low vol).")
                print()
            if above_ceil:
                med = statistics.median(above_ceil)
                print(f"  {len(above_ceil)} rejections from vol > ceiling (0.015).")
                print(f"  Median rejected vol: {med:.6f}")
                if med < 0.020:
                    print("  The ceiling is marginally too tight for this session.")
                    print(f"  Recommendation: raise drift_vol_ceiling to "
                          f"{min(med * 1.3, 0.03):.4f}")
                print()

        # ── Gate 6: Displacement ──────────────────────────────────
        g6 = self.gate_counts.get("displacement_below_threshold", 0)
        if g6 > 0 and self.displacement_samples:
            abs_disps = [abs(d) for d in self.displacement_samples]
            p90 = _percentile(abs_disps, 90)
            median_d = statistics.median(abs_disps)

            print(_yellow("  [Gate 6 — Displacement Threshold]"))
            print(f"  {g6} rejections.  Median |displacement| = {median_d:.4f}, "
                  f"p90 = {p90:.4f}")

            if p90 < 1.0:
                # The displacement distribution is well below threshold
                # Recommend lowering
                suggested = max(0.50, round(p90 * 0.9, 2))
                print(f"  Even p90 displacement ({p90:.2f}) is below z_threshold (1.0).")
                print(f"  The overnight session simply doesn't produce enough")
                print(f"  drift to cross the bar.  Two approaches:")
                print()
                print(f"  A) Time-conditional threshold:")
                print(f"     if 18 <= hour_utc or hour_utc < 6:")
                print(f"         z_threshold = {suggested}")
                print(f"     This lowers the bar during quiet hours while keeping")
                print(f"     the daytime filter tight to avoid false positives.")
                print()
                print(f"  B) Rolling-vol-adaptive threshold:")
                print(f"     z_threshold = base_z * (σ_session / σ_24h)")
                print(f"     When overnight vol is 40% of daily average,")
                print(f"     z_threshold auto-scales to 0.40 × 1.0 = 0.40.")
                print(f"     This is mathematically cleaner but requires a")
                print(f"     reliable σ_24h estimate (use EMA with 24h half-life).")
                print()
                print(f"  Recommendation: start with (A) at z_threshold = {suggested},")
                print(f"  then graduate to (B) once you have ≥ 7 days of vol data.")
            elif median_d > 0.8:
                print(f"  Near-misses are close to threshold — small reduction")
                print(f"  to z_threshold = 0.85 would capture them without")
                print(f"  significant false-positive risk.")
            print()

        # ── Overall ───────────────────────────────────────────────
        if self.total_evals > 0 and len(self.fires) == 0:
            # Find the biggest bottleneck
            rejects = {k: v for k, v in self.gate_counts.items() if k != "all_passed"}
            if rejects:
                worst = max(rejects, key=rejects.get)
                worst_pct = rejects[worst] / self.total_evals * 100
                print(_bold("  SUMMARY"))
                print(f"  Primary bottleneck: {GATE_LABELS.get(worst, worst)} "
                      f"({worst_pct:.0f}% of all evals)")
                print(f"  Fix this gate first to unlock subsequent stages.")
                print()


def _percentile(data: list[float], p: int) -> float:
    """Simple percentile without numpy dependency."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


# ─── Main ────────────────────────────────────────────────────────────

def _open_files(paths: list[str]) -> Iterator[IO[str]]:
    """Open log files in order, handling globs and stdin."""
    if not paths or paths == ["-"]:
        yield sys.stdin
        return

    expanded: list[str] = []
    for p in paths:
        if "*" in p or "?" in p:
            expanded.extend(sorted(glob.glob(p)))
        else:
            expanded.append(p)

    if not expanded:
        # Auto-discover
        expanded = _discover_log_files()

    if not expanded:
        print("ERROR: No log files found. Pass a path or ensure logs/ exists.",
              file=sys.stderr)
        sys.exit(1)

    for fp in expanded:
        try:
            yield open(fp, "r", encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"WARNING: Cannot open {fp}: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Polymarket bot JSONL logs for MeanReversionDrift diagnostics.",
    )
    parser.add_argument(
        "files", nargs="*", default=[],
        help="Log file path(s) or glob patterns. Use '-' for stdin. "
             "If omitted, auto-discovers logs/ directory.",
    )
    parser.add_argument(
        "--after", default=None,
        help="Only include entries after this ISO-8601 timestamp.",
    )
    parser.add_argument(
        "--before", default=None,
        help="Only include entries before this ISO-8601 timestamp.",
    )
    parser.add_argument(
        "--recommend", action="store_true",
        help="Include parameter calibration recommendations.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw gate counts as JSON instead of the formatted report.",
    )
    args = parser.parse_args()

    after = _parse_bound(args.after)
    before = _parse_bound(args.before)

    diag = DriftDiagnostics()
    lines_read = 0
    files_read = 0

    for fh in _open_files(args.files or []):
        files_read += 1
        try:
            for entry in _iter_jsonl(fh):
                lines_read += 1

                # Time filter
                if after or before:
                    ts = _parse_ts(entry.get("timestamp"))
                    if ts:
                        if after and ts < after:
                            continue
                        if before and ts > before:
                            continue

                diag.ingest(entry)
        finally:
            if fh is not sys.stdin:
                fh.close()

    print(f"\n  Scanned {lines_read:,} log entries across {files_read} file(s).")

    if args.json:
        out = {
            "total_evals": diag.total_evals,
            "signals_fired": len(diag.fires),
            "gate_counts": dict(diag.gate_counts),
            "ewma_vol_zero_count": diag.ewma_vol_zero_count,
            "ewma_vol_samples": diag.ewma_vol_samples[:50],        # cap for readability
            "displacement_samples": diag.displacement_samples[:50],
        }
        print(json.dumps(out, indent=2))
    else:
        diag.print_report(recommend=args.recommend)


if __name__ == "__main__":
    main()
