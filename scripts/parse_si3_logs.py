#!/usr/bin/env python3
"""
SI-3 Cross-Market Shadow Signal Analyser — ``parse_si3_logs.py``
=================================================================

Parses structlog JSONL output and/or the TradeStore SQLite database to
extract all SI-3 (Cross-Market Stat-Arb) shadow signal events, then
simulates theoretical execution against historical 1-minute tick data
to evaluate expected value (EV) before graduating SI-3 from shadow mode.

Modelled on ``parse_drift_logs.py`` — same log discovery, time-window
filtering, and reporting conventions.

Usage
-----
    # Auto-discover logs and tick data:
    python scripts/parse_si3_logs.py

    # Parse a specific log file:
    python scripts/parse_si3_logs.py logs/bot.jsonl

    # Parse VPS data drops:
    python scripts/parse_si3_logs.py data/vps_march2026/logs/bot.jsonl.*

    # Read from stdin:
    python scripts/parse_si3_logs.py -

    # Time-window filter:
    python scripts/parse_si3_logs.py --after 2026-03-04T18:00:00Z --before 2026-03-05T06:00:00Z

    # Include graduation recommendation:
    python scripts/parse_si3_logs.py --recommend

    # Machine-readable JSON output:
    python scripts/parse_si3_logs.py --json

    # Specify tick data directory explicitly:
    python scripts/parse_si3_logs.py --tick-dir data/vps_march2026/ticks

    # Specify TradeStore DB path:
    python scripts/parse_si3_logs.py --db data/vps_march2026/db/trades.db
"""

from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import math
import os
import sqlite3
import statistics
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import IO, Iterator, NamedTuple

# ─── Ensure stdout can handle Unicode on Windows ─────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
        )


# ─── Constants ────────────────────────────────────────────────────────

# Default Polymarket dynamic fee parameters
F_MAX = 0.0156            # 1.56% peak fee at p=0.50
SLIPPAGE_TICKS = 1        # 1 tick (1¢) taker crossing slippage
TICK_SIZE = 0.01           # Polymarket tick = 1 cent

# Correlation bucket boundaries
CORR_BUCKETS = [
    (0.50, 0.65, "0.50 ≤ ρ < 0.65"),
    (0.65, 1.01, "ρ ≥ 0.65"),
]

# Z-score bucket boundaries
Z_BUCKETS = [
    (2.0, 2.5, "2.0 ≤ z < 2.5"),
    (2.5, 100.0, "z ≥ 2.5"),
]

# Assumed holding period for theoretical exit (minutes)
DEFAULT_HOLD_MINUTES = 5
# Maximum holding period (minutes) — exit at market regardless
MAX_HOLD_MINUTES = 30


class _RuntimeConfig:
    """Mutable runtime configuration — allows CLI overrides without globals."""
    f_max: float = F_MAX
    hold_minutes: float = DEFAULT_HOLD_MINUTES

# Module-level singleton; replaced in main() when CLI args are provided.
_cfg = _RuntimeConfig()


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


# ─── Fee curve ────────────────────────────────────────────────────────

def _fee_rate(price: float, f_max: float | None = None) -> float:
    """Polymarket dynamic fee: f_max · 4 · p · (1 - p)."""
    if f_max is None:
        f_max = _cfg.f_max
    if price <= 0.0 or price >= 1.0:
        return 0.0
    return f_max * 4.0 * price * (1.0 - price)


def _roundtrip_fee(entry_price: float, exit_price: float) -> float:
    """Total round-trip fee as a fraction."""
    return _fee_rate(entry_price) + _fee_rate(exit_price)


# ─── Data structures ─────────────────────────────────────────────────

class ShadowSignal(NamedTuple):
    """Parsed SI-3 shadow signal event."""
    timestamp: datetime
    lagging_market: str
    leading_market: str
    direction: str   # "YES" or "NO"
    z_score: float
    correlation: float
    confidence: float


class TheoreticalTrade(NamedTuple):
    """Result of simulating execution of a shadow signal."""
    signal: ShadowSignal
    entry_price: float
    exit_price: float
    gross_pnl_cents: float
    fee_cents: float
    slippage_cents: float
    net_pnl_cents: float
    hold_minutes: float
    tick_data_available: bool


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


def _discover_tick_dirs() -> list[Path]:
    """Find tick data directories."""
    base = Path(__file__).resolve().parent.parent
    tick_dirs: list[Path] = []

    # VPS tick data
    vps_ticks = base / "data" / "vps_march2026" / "ticks"
    if vps_ticks.is_dir():
        for d in sorted(vps_ticks.iterdir()):
            if d.is_dir():
                tick_dirs.append(d)

    # data/ticks/ (if exists)
    local_ticks = base / "data" / "ticks"
    if local_ticks.is_dir():
        for d in sorted(local_ticks.iterdir()):
            if d.is_dir():
                tick_dirs.append(d)

    return tick_dirs


def _discover_db_files() -> list[Path]:
    """Find TradeStore SQLite databases."""
    base = Path(__file__).resolve().parent.parent
    dbs: list[Path] = []

    for pattern in [
        base / "logs" / "trades.db",
        base / "data" / "vps_march2026" / "db" / "trades.db",
    ]:
        if pattern.exists():
            dbs.append(pattern)

    return dbs


# ─── JSONL streaming reader ──────────────────────────────────────────

def _iter_jsonl(stream: IO[str]) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL stream, skipping bad lines."""
    for raw in stream:
        raw = raw.strip()
        if not raw:
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            pass


# ─── Log file I/O ────────────────────────────────────────────────────

def _open_files(paths: list[str]) -> Iterator[IO[str]]:
    """Open log files in order, handling globs and stdin."""
    if paths == ["-"]:
        yield sys.stdin
        return

    expanded: list[str] = []
    for p in paths:
        if "*" in p or "?" in p:
            expanded.extend(sorted(glob.glob(p)))
        else:
            expanded.append(p)

    if not expanded:
        expanded = _discover_log_files()

    if not expanded:
        print("WARNING: No log files found. Pass a path or ensure logs/ exists.",
              file=sys.stderr)
        return

    for fp in expanded:
        try:
            yield open(fp, "r", encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"WARNING: Cannot open {fp}: {e}", file=sys.stderr)


# ─── Signal extraction from logs ─────────────────────────────────────

def _extract_signals_from_logs(
    file_paths: list[str],
    after: datetime | None,
    before: datetime | None,
) -> tuple[list[ShadowSignal], int, int]:
    """Parse JSONL logs for SI-3 shadow signal events.

    Returns (signals, lines_read, files_read).
    """
    signals: list[ShadowSignal] = []
    lines_read = 0
    files_read = 0

    for fh in _open_files(file_paths):
        files_read += 1
        try:
            for entry in _iter_jsonl(fh):
                lines_read += 1
                event = entry.get("event", "")

                # Match both shadow and live cross_market_signal events
                if "cross_market_signal" not in event:
                    continue

                ts = _parse_ts(entry.get("timestamp"))
                if not ts:
                    continue

                # Apply time filters
                if after and ts < after:
                    continue
                if before and ts > before:
                    continue

                sig = ShadowSignal(
                    timestamp=ts,
                    lagging_market=entry.get("lagging", ""),
                    leading_market=entry.get("leading", ""),
                    direction=entry.get("direction", "YES"),
                    z_score=float(entry.get("z", 0.0)),
                    correlation=float(entry.get("rho", 0.0)),
                    confidence=float(entry.get("confidence", 0.0)),
                )
                signals.append(sig)
        finally:
            if fh is not sys.stdin:
                fh.close()

    return signals, lines_read, files_read


# ─── Signal extraction from TradeStore DB ─────────────────────────────

def _extract_signals_from_db(
    db_path: Path,
    after: datetime | None,
    before: datetime | None,
) -> list[ShadowSignal]:
    """Query the TradeStore SQLite for SI-3 related trades.

    The SI-3 signals themselves aren't persisted in the trades table,
    but if the bot ever wrote them via a shadow_signals table or as
    special metadata in the trades table, we extract them here.
    """
    signals: list[ShadowSignal] = []
    if not db_path.exists():
        return signals

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if a shadow_signals table exists (future-proof)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('shadow_signals', 'si3_signals', 'cross_market_signals')"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            # Attempt generic extraction
            try:
                cursor.execute(f"SELECT * FROM {table}")
                cols = [desc[0] for desc in cursor.description]
                for row in cursor.fetchall():
                    rec = dict(zip(cols, row))
                    ts_raw = rec.get("timestamp") or rec.get("created_at")
                    if ts_raw:
                        if isinstance(ts_raw, (int, float)):
                            ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
                        else:
                            ts = _parse_ts(str(ts_raw))
                    else:
                        continue

                    if ts and after and ts < after:
                        continue
                    if ts and before and ts > before:
                        continue

                    if ts:
                        signals.append(ShadowSignal(
                            timestamp=ts,
                            lagging_market=str(rec.get("lagging_market", rec.get("lagging", ""))),
                            leading_market=str(rec.get("leading_market", rec.get("leading", ""))),
                            direction=str(rec.get("direction", "YES")),
                            z_score=float(rec.get("z_score", rec.get("z", 0.0))),
                            correlation=float(rec.get("correlation", rec.get("rho", 0.0))),
                            confidence=float(rec.get("confidence", 0.0)),
                        ))
            except sqlite3.OperationalError:
                pass

        conn.close()
    except sqlite3.DatabaseError as e:
        print(f"WARNING: Cannot read DB {db_path}: {e}", file=sys.stderr)

    return signals


# ─── Tick data loader ─────────────────────────────────────────────────

class TickDataIndex:
    """Loads and indexes tick data from JSONL files for price lookups.

    Builds a sparse time-series of mid-prices per asset_id for quick
    nearest-timestamp lookups.
    """

    def __init__(self, tick_dirs: list[Path]):
        # asset_id → sorted list of (unix_ts, mid_price)
        self._prices: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
        self._loaded_dirs: set[str] = set()
        self._tick_dirs = tick_dirs

    def _ensure_loaded(self, market_id: str) -> None:
        """Lazily load tick data for a market if not yet loaded."""
        if market_id in self._loaded_dirs:
            return
        self._loaded_dirs.add(market_id)

        for td in self._tick_dirs:
            # Try both hex-prefixed and plain market_id filenames
            candidates = [
                td / f"{market_id}.jsonl",
            ]
            # Also glob for partial match (condition_id might be truncated in logs)
            if len(market_id) >= 8:
                for f in td.glob(f"*{market_id}*"):
                    if f.suffix == ".jsonl":
                        candidates.append(f)

            for tick_file in candidates:
                if not tick_file.exists():
                    continue
                try:
                    self._load_tick_file(tick_file, market_id)
                except Exception as e:
                    print(f"WARNING: Error loading {tick_file}: {e}", file=sys.stderr)

        # Sort by timestamp for binary search
        if market_id in self._prices:
            self._prices[market_id].sort(key=lambda x: x[0])

    def _load_tick_file(self, path: Path, market_id: str) -> None:
        """Parse a tick JSONL file and extract mid-prices."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                local_ts = entry.get("local_ts")
                if not local_ts:
                    continue

                payload = entry.get("payload", {})
                price_changes = payload.get("price_changes", [])

                for pc in price_changes:
                    best_bid = pc.get("best_bid")
                    best_ask = pc.get("best_ask")
                    if best_bid and best_ask:
                        try:
                            bid = float(best_bid)
                            ask = float(best_ask)
                            mid = (bid + ask) / 2.0
                            asset_id = pc.get("asset_id", "")
                            self._prices[asset_id].append((float(local_ts), mid))
                            # Also index by the condition_id / market hex
                            self._prices[market_id].append((float(local_ts), mid))
                        except (ValueError, TypeError):
                            continue

    def get_price_at(self, market_id: str, ts: float) -> float | None:
        """Get the nearest mid-price for a market at the given unix timestamp.

        Returns None if no tick data is available.
        """
        self._ensure_loaded(market_id)
        series = self._prices.get(market_id, [])
        if not series:
            return None

        # Binary search for nearest timestamp
        lo, hi = 0, len(series) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if series[mid][0] < ts:
                lo = mid + 1
            else:
                hi = mid

        # Check closest between lo and lo-1
        best_idx = lo
        if lo > 0:
            if abs(series[lo - 1][0] - ts) < abs(series[lo][0] - ts):
                best_idx = lo - 1

        # Reject if more than 5 minutes away
        if abs(series[best_idx][0] - ts) > 300:
            return None

        return series[best_idx][1]

    def get_price_after(self, market_id: str, ts: float, offset_minutes: float) -> float | None:
        """Get the nearest mid-price at ts + offset_minutes.

        Searches for the closest price within a ±2 minute window around
        the target time.
        """
        target = ts + offset_minutes * 60
        return self.get_price_at(market_id, target)

    @property
    def available_markets(self) -> set[str]:
        """Return set of market IDs with loaded data."""
        return set(self._prices.keys())


# ─── Theoretical execution simulation ────────────────────────────────

def _simulate_trade(
    sig: ShadowSignal,
    tick_index: TickDataIndex,
) -> TheoreticalTrade:
    """Simulate theoretical execution of a shadow signal.

    Assumptions:
      - Entry at signal timestamp + 1-tick slippage (taker crossing).
      - Exit at earliest of: mean-reversion target (mid-price after
        DEFAULT_HOLD_MINUTES) or MAX_HOLD_MINUTES timeout.
      - Round-trip fees: f_max · 4 · p · (1-p) at entry and exit prices.
    """
    sig_unix = sig.timestamp.timestamp()
    hold_default = _cfg.hold_minutes

    # Attempt to find entry price from tick data
    entry_mid = tick_index.get_price_at(sig.lagging_market, sig_unix)

    if entry_mid is None:
        # No tick data — use a synthetic estimate based on correlation
        # and direction for reporting purposes
        return TheoreticalTrade(
            signal=sig,
            entry_price=0.0,
            exit_price=0.0,
            gross_pnl_cents=0.0,
            fee_cents=0.0,
            slippage_cents=0.0,
            net_pnl_cents=0.0,
            hold_minutes=0.0,
            tick_data_available=False,
        )

    # Apply 1-tick slippage for taker crossing
    if sig.direction == "YES":
        # Buying YES: cross the ask → pay 1 tick above mid
        entry_price = entry_mid + SLIPPAGE_TICKS * TICK_SIZE
    else:
        # Buying NO (selling YES): cross the bid → 1 tick below mid
        entry_price = entry_mid - SLIPPAGE_TICKS * TICK_SIZE

    entry_price = max(0.01, min(0.99, entry_price))

    # Find exit price — try multiple holding periods
    exit_price = None
    hold_minutes = hold_default

    for hold in [hold_default, 10, 15, 20, MAX_HOLD_MINUTES]:
        p = tick_index.get_price_after(sig.lagging_market, sig_unix, hold)
        if p is not None:
            exit_price = p
            hold_minutes = hold
            break

    if exit_price is None:
        # No exit data — mark as unavailable
        return TheoreticalTrade(
            signal=sig,
            entry_price=entry_price,
            exit_price=0.0,
            gross_pnl_cents=0.0,
            fee_cents=0.0,
            slippage_cents=0.0,
            net_pnl_cents=0.0,
            hold_minutes=0.0,
            tick_data_available=False,
        )

    # Calculate PnL
    if sig.direction == "YES":
        gross_pnl = exit_price - entry_price
    else:
        gross_pnl = entry_price - exit_price

    gross_pnl_cents = gross_pnl * 100.0

    # Fee drag: dynamic fee at entry and exit
    entry_fee = _fee_rate(entry_price)
    exit_fee = _fee_rate(exit_price)
    fee_cents = (entry_fee + exit_fee) * 100.0

    # Slippage already embedded in entry_price
    slippage_cents = SLIPPAGE_TICKS * TICK_SIZE * 100.0  # 1 cent

    net_pnl_cents = gross_pnl_cents - fee_cents

    return TheoreticalTrade(
        signal=sig,
        entry_price=round(entry_price, 4),
        exit_price=round(exit_price, 4),
        gross_pnl_cents=round(gross_pnl_cents, 2),
        fee_cents=round(fee_cents, 2),
        slippage_cents=round(slippage_cents, 2),
        net_pnl_cents=round(net_pnl_cents, 2),
        hold_minutes=hold_minutes,
        tick_data_available=True,
    )


# ─── Bucket helpers ──────────────────────────────────────────────────

def _bucket_by_correlation(trades: list[TheoreticalTrade]) -> dict[str, list[TheoreticalTrade]]:
    """Group trades into correlation buckets."""
    buckets: dict[str, list[TheoreticalTrade]] = {}
    for label_tuple in CORR_BUCKETS:
        buckets[label_tuple[2]] = []

    buckets["ρ < 0.50 (sub-threshold)"] = []

    for t in trades:
        rho = abs(t.signal.correlation)
        placed = False
        for lo, hi, label in CORR_BUCKETS:
            if lo <= rho < hi:
                buckets[label].append(t)
                placed = True
                break
        if not placed:
            buckets["ρ < 0.50 (sub-threshold)"].append(t)

    return {k: v for k, v in buckets.items() if v}


def _bucket_by_zscore(trades: list[TheoreticalTrade]) -> dict[str, list[TheoreticalTrade]]:
    """Group trades into z-score buckets."""
    buckets: dict[str, list[TheoreticalTrade]] = {}
    for label_tuple in Z_BUCKETS:
        buckets[label_tuple[2]] = []

    buckets["z < 2.0 (sub-threshold)"] = []

    for t in trades:
        z = abs(t.signal.z_score)
        placed = False
        for lo, hi, label in Z_BUCKETS:
            if lo <= z < hi:
                buckets[label].append(t)
                placed = True
                break
        if not placed:
            buckets["z < 2.0 (sub-threshold)"].append(t)

    return {k: v for k, v in buckets.items() if v}


# ─── Metrics ─────────────────────────────────────────────────────────

def _percentile(data: list[float], p: int) -> float:
    """Return the p-th percentile of data using the standard library."""
    return statistics.quantiles(data, n=100)[p - 1] if data else 0.0


def _compute_bucket_stats(trades: list[TheoreticalTrade]) -> dict:
    """Compute aggregate stats for a bucket of trades."""
    if not trades:
        return {"count": 0}

    pnls = [t.net_pnl_cents for t in trades if t.tick_data_available]
    gross_pnls = [t.gross_pnl_cents for t in trades if t.tick_data_available]
    holds = [t.hold_minutes for t in trades if t.tick_data_available and t.hold_minutes > 0]
    fees = [t.fee_cents for t in trades if t.tick_data_available]

    if not pnls:
        return {
            "count": len(trades),
            "with_tick_data": 0,
        }

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_pnl = total_pnl / len(pnls) if pnls else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    avg_hold = sum(holds) / len(holds) if holds else 0.0
    avg_fee = sum(fees) / len(fees) if fees else 0.0

    # Sharpe-like ratio: avg_pnl / stdev(pnl)
    if len(pnls) > 1:
        pnl_std = statistics.stdev(pnls)
        sharpe = avg_pnl / pnl_std if pnl_std > 0 else 0.0
    else:
        pnl_std = 0.0
        sharpe = 0.0

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = peak - cum
        max_dd = max(max_dd, dd)

    return {
        "count": len(trades),
        "with_tick_data": len(pnls),
        "win_rate": round(win_rate, 4),
        "avg_pnl_cents": round(avg_pnl, 2),
        "avg_win_cents": round(avg_win, 2),
        "avg_loss_cents": round(avg_loss, 2),
        "total_pnl_cents": round(total_pnl, 2),
        "avg_hold_minutes": round(avg_hold, 1),
        "avg_fee_cents": round(avg_fee, 2),
        "total_fee_cents": round(sum(fees), 2),
        "pnl_stdev": round(pnl_std, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_cents": round(max_dd, 2),
        "pnl_p10": round(_percentile(pnls, 10), 2),
        "pnl_p50": round(_percentile(pnls, 50), 2),
        "pnl_p90": round(_percentile(pnls, 90), 2),
    }


# ─── Recommendation engine ───────────────────────────────────────────

def _compute_recommendation(all_stats: dict, bucket_stats: dict) -> dict:
    """Determine whether SI-3 should graduate from shadow mode.

    Criteria for graduation:
      1. ≥ 20 signals with tick data for statistical significance.
      2. Win rate ≥ 52% (above break-even after fees).
      3. Positive average EV (net of fees + slippage).
      4. At least one correlation bucket with positive EV.
      5. Sharpe ratio > 0 (directionally correct on risk-adjusted basis).
    """
    result = {
        "graduate": False,
        "reasons": [],
        "config_change": None,
    }

    n = all_stats.get("with_tick_data", 0)
    if n < 20:
        result["reasons"].append(
            f"INSUFFICIENT DATA: Only {n} signals with tick data (need ≥ 20)."
        )
        return result

    wr = all_stats.get("win_rate", 0)
    if wr < 0.52:
        result["reasons"].append(
            f"WIN RATE TOO LOW: {wr:.1%} (need ≥ 52%)."
        )

    avg_ev = all_stats.get("avg_pnl_cents", 0)
    if avg_ev <= 0:
        result["reasons"].append(
            f"NEGATIVE EV: Average PnL {avg_ev:.2f}¢ (need > 0)."
        )

    sharpe = all_stats.get("sharpe_ratio", 0)
    if sharpe <= 0:
        result["reasons"].append(
            f"NEGATIVE SHARPE: {sharpe:.3f} (need > 0)."
        )

    # Check at least one bucket is profitable
    any_profitable_bucket = False
    best_bucket = None
    best_ev = -float("inf")
    for bucket_name, stats in bucket_stats.items():
        ev = stats.get("avg_pnl_cents", 0)
        if ev > 0 and stats.get("with_tick_data", 0) >= 5:
            any_profitable_bucket = True
        if ev > best_ev and stats.get("with_tick_data", 0) >= 3:
            best_ev = ev
            best_bucket = bucket_name

    if not any_profitable_bucket:
        result["reasons"].append(
            "NO PROFITABLE BUCKET: No correlation/z-score band with positive "
            "EV and ≥ 5 samples."
        )

    if not result["reasons"]:
        result["graduate"] = True
        result["reasons"].append("ALL CRITERIA MET — safe to graduate SI-3 from shadow mode.")
        result["config_change"] = {
            "file": "src/core/config.py",
            "field": "cross_mkt_shadow",
            "old_value": True,
            "new_value": False,
            "env_var": "CROSS_MKT_SHADOW=false",
        }
        if best_bucket:
            result["best_band"] = best_bucket

    return result


# ─── Report generation ───────────────────────────────────────────────

class SI3Analyser:
    """Aggregates SI-3 shadow signals and simulated trades for reporting."""

    def __init__(self):
        self.signals: list[ShadowSignal] = []
        self.trades: list[TheoreticalTrade] = []

    def ingest_signals(self, signals: list[ShadowSignal]) -> None:
        self.signals.extend(signals)

    def simulate_all(self, tick_index: TickDataIndex) -> None:
        """Run theoretical execution simulation for all signals."""
        for sig in self.signals:
            trade = _simulate_trade(sig, tick_index)
            self.trades.append(trade)

    def print_report(self, *, recommend: bool = False) -> None:
        """Print the full analysis report to stdout."""
        w = 72
        print()
        print(_bold("=" * w))
        print(_bold(" SI-3 CROSS-MARKET STAT-ARB — SHADOW SIGNAL ANALYSIS"))
        print(_bold("=" * w))

        # ── Overview ──
        total_signals = len(self.signals)
        with_data = sum(1 for t in self.trades if t.tick_data_available)
        without_data = total_signals - with_data

        if self.signals:
            first_ts = min(s.timestamp for s in self.signals)
            last_ts = max(s.timestamp for s in self.signals)
            duration = last_ts - first_ts
            hours = duration.total_seconds() / 3600
        else:
            first_ts = last_ts = None
            hours = 0

        print(f"  Total shadow signals : {_bold(str(total_signals))}")
        print(f"  With tick data       : {_green(str(with_data)) if with_data else _red('0')}")
        print(f"  Without tick data    : {_dim(str(without_data))}")
        if first_ts and last_ts:
            print(f"  Session window       : {first_ts.isoformat()} → {last_ts.isoformat()}")
            print(f"  Duration             : {hours:.1f} hours")
        print()

        if total_signals == 0:
            print(_yellow("  ⚠  No SI-3 shadow signal events found in logs."))
            print(_dim("     Ensure the bot is running with CROSS_MKT_ENABLED=true"))
            print(_dim("     and LOG_LEVEL=debug. Shadow signals are logged as:"))
            print(_dim("     event=\"[SHADOW] cross_market_signal\""))
            print()
            return

        # ── Signal distribution ──
        self._print_signal_distribution()

        # ── Overall performance ──
        all_stats = _compute_bucket_stats(self.trades)
        self._print_performance_summary(all_stats)

        # ── Correlation buckets ──
        corr_buckets = _bucket_by_correlation(self.trades)
        corr_bucket_stats = {}
        if corr_buckets:
            print(_bold("─" * w))
            print(_bold(" PERFORMANCE BY CORRELATION BUCKET"))
            print(_bold("─" * w))
            for label, trades in sorted(corr_buckets.items()):
                stats = _compute_bucket_stats(trades)
                corr_bucket_stats[label] = stats
                self._print_bucket(label, stats)

        # ── Z-score buckets ──
        z_buckets = _bucket_by_zscore(self.trades)
        z_bucket_stats = {}
        if z_buckets:
            print(_bold("─" * w))
            print(_bold(" PERFORMANCE BY Z-SCORE BUCKET"))
            print(_bold("─" * w))
            for label, trades in sorted(z_buckets.items()):
                stats = _compute_bucket_stats(trades)
                z_bucket_stats[label] = stats
                self._print_bucket(label, stats)

        # ── Per-pair breakdown ──
        self._print_pair_breakdown()

        # ── Fee impact analysis ──
        self._print_fee_analysis()

        # ── Recommendation ──
        if recommend:
            all_bucket_stats = {**corr_bucket_stats, **z_bucket_stats}
            rec = _compute_recommendation(all_stats, all_bucket_stats)
            self._print_recommendation(rec)

        print(_bold("=" * w))
        print()

    def _print_signal_distribution(self) -> None:
        """Print signal count distribution by ρ and z."""
        w = 72
        print(_bold("─" * w))
        print(_bold(" SIGNAL DISTRIBUTION"))
        print(_bold("─" * w))

        # Correlation distribution
        corr_counts: dict[str, int] = collections.Counter()
        for s in self.signals:
            rho = abs(s.correlation)
            for lo, hi, label in CORR_BUCKETS:
                if lo <= rho < hi:
                    corr_counts[label] += 1
                    break
            else:
                corr_counts["ρ < 0.50 (sub-threshold)"] += 1

        print("  By correlation:")
        for label, count in sorted(corr_counts.items()):
            pct = count / len(self.signals) * 100
            bar = "█" * int(pct / 2)
            print(f"    {label:<25}  {count:>4}  ({pct:>5.1f}%)  {_cyan(bar)}")

        # Z-score distribution
        z_counts: dict[str, int] = collections.Counter()
        for s in self.signals:
            z = abs(s.z_score)
            for lo, hi, label in Z_BUCKETS:
                if lo <= z < hi:
                    z_counts[label] += 1
                    break
            else:
                z_counts["z < 2.0 (sub-threshold)"] += 1

        print("  By z-score:")
        for label, count in sorted(z_counts.items()):
            pct = count / len(self.signals) * 100
            bar = "█" * int(pct / 2)
            print(f"    {label:<25}  {count:>4}  ({pct:>5.1f}%)  {_cyan(bar)}")

        # Direction distribution
        dir_counts = collections.Counter(s.direction for s in self.signals)
        print("  By direction:")
        for direction, count in dir_counts.most_common():
            pct = count / len(self.signals) * 100
            print(f"    {direction:<25}  {count:>4}  ({pct:>5.1f}%)")
        print()

    def _print_performance_summary(self, stats: dict) -> None:
        """Print overall theoretical performance."""
        w = 72
        print(_bold("─" * w))
        print(_bold(" OVERALL THEORETICAL PERFORMANCE"))
        print(_bold("─" * w))

        n = stats.get("with_tick_data", 0)
        if n == 0:
            print(_yellow("  ⚠  No signals with matching tick data for simulation."))
            print()
            return

        wr = stats["win_rate"]
        ev = stats["avg_pnl_cents"]
        total = stats["total_pnl_cents"]

        wr_colour = _green if wr >= 0.52 else (_yellow if wr >= 0.48 else _red)
        ev_colour = _green if ev > 0 else _red
        total_colour = _green if total > 0 else _red

        print(f"  Simulated trades     : {n}")
        print(f"  Win rate             : {wr_colour(f'{wr:.1%}')}")
        print(f"  Avg EV (net)         : {ev_colour(f'{ev:+.2f}¢')}")
        print(f"  Total PnL (net)      : {total_colour(f'{total:+.2f}¢')}")
        print(f"  Avg win              : {_green(f'{stats["avg_win_cents"]:+.2f}¢')}")
        print(f"  Avg loss             : {_red(f'{stats["avg_loss_cents"]:+.2f}¢')}")
        print(f"  Avg hold time        : {stats['avg_hold_minutes']:.1f} min")
        print(f"  Avg fee drag         : {stats['avg_fee_cents']:.2f}¢")
        print(f"  Sharpe ratio         : {stats['sharpe_ratio']:.3f}")
        print(f"  Max drawdown         : {stats['max_drawdown_cents']:.2f}¢")
        print(f"  PnL distribution     : p10={stats['pnl_p10']:.2f}¢  "
              f"p50={stats['pnl_p50']:.2f}¢  p90={stats['pnl_p90']:.2f}¢")
        print()

    def _print_bucket(self, label: str, stats: dict) -> None:
        """Print stats for one bucket."""
        n = stats.get("with_tick_data", 0)
        count = stats.get("count", 0)

        if n == 0:
            print(f"  {_dim(label)}")
            print(f"    Signals: {count}  (no tick data available)")
            print()
            return

        wr = stats["win_rate"]
        ev = stats["avg_pnl_cents"]
        total = stats["total_pnl_cents"]

        ev_colour = _green if ev > 0 else _red
        wr_colour = _green if wr >= 0.52 else _red

        marker = " ◀ BEST" if ev > 0 and n >= 5 else ""
        print(f"  {_bold(label)}{_green(marker) if marker else ''}")
        print(f"    Signals: {n}   WR: {wr_colour(f'{wr:.1%}')}   "
              f"Avg EV: {ev_colour(f'{ev:+.2f}¢')}   "
              f"Total: {ev_colour(f'{total:+.2f}¢')}   "
              f"Sharpe: {stats['sharpe_ratio']:.3f}")
        print()

    def _print_pair_breakdown(self) -> None:
        """Per-pair performance breakdown."""
        w = 72
        pairs: dict[tuple[str, str], list[TheoreticalTrade]] = collections.defaultdict(list)
        for t in self.trades:
            if t.tick_data_available:
                key = (t.signal.lagging_market, t.signal.leading_market)
                pairs[key].append(t)

        if not pairs:
            return

        print(_bold("─" * w))
        print(_bold(" PER-PAIR BREAKDOWN"))
        print(_bold("─" * w))

        sorted_pairs = sorted(
            pairs.items(),
            key=lambda kv: sum(t.net_pnl_cents for t in kv[1]),
            reverse=True,
        )

        print(f"  {'Lagger':<18} {'Leader':<18} {'N':>3}  {'WR':>5}  {'AvgEV':>7}  {'Total':>8}  {'ρ':>5}")
        print(f"  {'─' * 18} {'─' * 18} {'─' * 3}  {'─' * 5}  {'─' * 7}  {'─' * 8}  {'─' * 5}")

        for (lagger, leader), trades in sorted_pairs[:15]:
            n = len(trades)
            pnls = [t.net_pnl_cents for t in trades]
            wr = sum(1 for p in pnls if p > 0) / n if n else 0
            avg_ev = sum(pnls) / n if n else 0
            total = sum(pnls)
            avg_rho = sum(t.signal.correlation for t in trades) / n

            ev_colour = _green if avg_ev > 0 else _red
            print(f"  {lagger:<18} {leader:<18} {n:>3}  {wr:>4.0%}  "
                  f"{ev_colour(f'{avg_ev:>+6.2f}¢')}  {ev_colour(f'{total:>+7.2f}¢')}  "
                  f"{avg_rho:>5.3f}")
        print()

    def _print_fee_analysis(self) -> None:
        """Analyse fee impact on profitability."""
        w = 72
        valid = [t for t in self.trades if t.tick_data_available]
        if not valid:
            return

        print(_bold("─" * w))
        print(_bold(" FEE IMPACT ANALYSIS"))
        print(_bold("─" * w))

        total_gross = sum(t.gross_pnl_cents for t in valid)
        total_fees = sum(t.fee_cents for t in valid)
        total_net = sum(t.net_pnl_cents for t in valid)
        total_slippage = sum(t.slippage_cents for t in valid)

        pct_eaten = (total_fees / total_gross * 100) if total_gross > 0 else 0

        print(f"  Gross PnL        : {total_gross:+.2f}¢")
        print(f"  Total fees       : -{total_fees:.2f}¢ ({pct_eaten:.1f}% of gross)")
        print(f"  Total slippage   : -{total_slippage:.2f}¢ (embedded in entry)")
        print(f"  Net PnL          : {total_net:+.2f}¢")
        print()

        # Fee sensitivity: what f_max would break even?
        if total_gross > 0 and total_fees > 0:
            # Current f_max = 0.0156; if we scale fees proportionally
            break_even_ratio = total_gross / total_fees
            implied_f_max = _cfg.f_max * break_even_ratio
            print(f"  Break-even f_max : {implied_f_max:.4f} "
                  f"(current: {_cfg.f_max:.4f}, headroom: {(break_even_ratio - 1) * 100:.0f}%)")
        print()

    def _print_recommendation(self, rec: dict) -> None:
        """Print graduation recommendation."""
        w = 72
        print(_bold("=" * w))
        print(_bold(" GRADUATION RECOMMENDATION"))
        print(_bold("=" * w))
        print()

        if rec["graduate"]:
            print(_green("  ✓ RECOMMEND: Graduate SI-3 from shadow mode."))
            print()
            for reason in rec["reasons"]:
                print(f"    {_green(reason)}")
            print()
            cfg = rec.get("config_change", {})
            if cfg:
                print(f"  Config change required:")
                print(f"    File      : {cfg.get('file', 'src/core/config.py')}")
                print(f"    Field     : {cfg.get('field', 'cross_mkt_shadow')}")
                print(f"    Old value : {cfg.get('old_value', True)}")
                print(f"    New value : {cfg.get('new_value', False)}")
                print(f"    Env var   : {cfg.get('env_var', 'CROSS_MKT_SHADOW=false')}")
            if rec.get("best_band"):
                print(f"  Best band   : {rec['best_band']}")
        else:
            print(_red("  ✗ DO NOT GRADUATE — criteria not met."))
            print()
            for reason in rec["reasons"]:
                print(f"    {_red(reason)}")

        print()

    def to_json(self, *, recommend: bool = False) -> dict:
        """Return all results as a JSON-serialisable dict."""
        all_stats = _compute_bucket_stats(self.trades)

        corr_buckets = _bucket_by_correlation(self.trades)
        corr_bucket_stats = {}
        for label, trades in corr_buckets.items():
            corr_bucket_stats[label] = _compute_bucket_stats(trades)

        z_buckets = _bucket_by_zscore(self.trades)
        z_bucket_stats = {}
        for label, trades in z_buckets.items():
            z_bucket_stats[label] = _compute_bucket_stats(trades)

        result = {
            "total_signals": len(self.signals),
            "with_tick_data": sum(1 for t in self.trades if t.tick_data_available),
            "overall": all_stats,
            "by_correlation": corr_bucket_stats,
            "by_zscore": z_bucket_stats,
            "signals": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "lagging_market": s.lagging_market,
                    "leading_market": s.leading_market,
                    "direction": s.direction,
                    "z_score": s.z_score,
                    "correlation": s.correlation,
                    "confidence": s.confidence,
                }
                for s in self.signals[:200]  # cap for readability
            ],
            "trades": [
                {
                    "timestamp": t.signal.timestamp.isoformat(),
                    "lagging_market": t.signal.lagging_market,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "net_pnl_cents": t.net_pnl_cents,
                    "fee_cents": t.fee_cents,
                    "hold_minutes": t.hold_minutes,
                    "tick_data_available": t.tick_data_available,
                }
                for t in self.trades[:200]
            ],
        }

        if recommend:
            all_bucket_stats = {**corr_bucket_stats, **z_bucket_stats}
            result["recommendation"] = _compute_recommendation(all_stats, all_bucket_stats)

        return result


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse SI-3 Cross-Market shadow signals from Polymarket bot "
                    "JSONL logs and simulate theoretical execution performance.",
    )
    parser.add_argument(
        "files", nargs="*", default=[],
        help="Log file path(s) or glob patterns. Use '-' for stdin. "
             "If omitted, auto-discovers logs/ and data/vps_march2026/logs/.",
    )
    parser.add_argument(
        "--after", default=None,
        help="Only include signals after this ISO-8601 timestamp.",
    )
    parser.add_argument(
        "--before", default=None,
        help="Only include signals before this ISO-8601 timestamp.",
    )
    parser.add_argument(
        "--tick-dir", default=None,
        help="Path to tick data directory (contains date sub-dirs with per-asset JSONL).",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to TradeStore SQLite database for supplementary signal extraction.",
    )
    parser.add_argument(
        "--recommend", action="store_true",
        help="Include graduation recommendation (should cross_mkt_shadow be set to False?).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON instead of the formatted report.",
    )
    parser.add_argument(
        "--hold-minutes", type=float, default=None,
        help=f"Default holding period for exit simulation (default: {DEFAULT_HOLD_MINUTES} min).",
    )
    parser.add_argument(
        "--f-max", type=float, default=None,
        help=f"Peak fee rate for the dynamic fee curve (default: {F_MAX}).",
    )
    args = parser.parse_args()

    # Override module-level defaults from CLI args
    global _cfg
    if args.f_max is not None:
        _cfg.f_max = args.f_max
    if args.hold_minutes is not None:
        _cfg.hold_minutes = args.hold_minutes

    after = _parse_bound(args.after)
    before = _parse_bound(args.before)

    # ── Extract signals from logs ──
    signals, lines_read, files_read = _extract_signals_from_logs(
        args.files or [], after, before,
    )

    # ── Extract signals from DB (if available) ──
    db_paths: list[Path] = []
    if args.db:
        db_paths = [Path(args.db)]
    else:
        db_paths = _discover_db_files()

    for db_path in db_paths:
        db_signals = _extract_signals_from_db(db_path, after, before)
        if db_signals:
            print(f"  Found {len(db_signals)} signals in DB: {db_path}", file=sys.stderr)
            signals.extend(db_signals)

    # Deduplicate by (timestamp, lagging_market, leading_market)
    seen: set[tuple[str, str, str]] = set()
    unique_signals: list[ShadowSignal] = []
    for s in signals:
        key = (s.timestamp.isoformat(), s.lagging_market, s.leading_market)
        if key not in seen:
            seen.add(key)
            unique_signals.append(s)
    signals = unique_signals

    print(f"\n  Scanned {lines_read:,} log entries across {files_read} file(s).")
    if db_paths:
        print(f"  Checked {len(db_paths)} TradeStore DB(s).")
    print(f"  Found {len(signals)} unique SI-3 shadow signals.")

    # ── Load tick data ──
    tick_dirs: list[Path] = []
    if args.tick_dir:
        td = Path(args.tick_dir)
        if td.is_dir():
            # Check if it contains date sub-dirs or asset files directly
            sub_dirs = [d for d in td.iterdir() if d.is_dir()]
            if sub_dirs:
                tick_dirs = sub_dirs
            else:
                tick_dirs = [td]
        else:
            print(f"WARNING: Tick directory {td} does not exist.", file=sys.stderr)
    else:
        tick_dirs = _discover_tick_dirs()

    if tick_dirs:
        print(f"  Found {len(tick_dirs)} tick data directory/ies.")
    else:
        print(_dim("  No tick data directories found — simulation will be limited."))

    tick_index = TickDataIndex(tick_dirs)

    # ── Simulate trades ──
    analyser = SI3Analyser()
    analyser.ingest_signals(signals)
    analyser.simulate_all(tick_index)

    # ── Output ──
    if args.json:
        out = analyser.to_json(recommend=args.recommend)
        out["meta"] = {
            "lines_scanned": lines_read,
            "files_scanned": files_read,
            "db_paths": [str(p) for p in db_paths],
            "tick_dirs": [str(d) for d in tick_dirs],
            "f_max": _cfg.f_max,
            "slippage_ticks": SLIPPAGE_TICKS,
            "hold_minutes": _cfg.hold_minutes,
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        analyser.print_report(recommend=args.recommend)


if __name__ == "__main__":
    main()
