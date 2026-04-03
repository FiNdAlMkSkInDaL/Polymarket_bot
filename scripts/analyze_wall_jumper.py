from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from statistics import mean, median, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_universal_backtest import iter_replay_events, load_market_catalog, resolve_tick_root
from src.data.orderbook import OrderbookTracker
from src.trading.fees import get_fee_rate


HORIZONS = (5, 15, 60)
STRATEGY_NAME = "wall_jumper"
DIAGNOSTICS_PATTERN = re.compile(r"^strategy_diagnostics\s+(?P<payload>.+)$")
DEFAULT_MIN_WALL_SIZE = 10_000.0
DEFAULT_WALL_TO_OPPOSING_RATIO = 5.0
DEFAULT_DEPTH_LEVELS = 5
DEFAULT_TICK_SIZE = 0.01


@dataclass(slots=True)
class MakerFillRecord:
    fill_id: str
    market_id: str
    asset_id: str
    quote_side: str
    entry_price: float
    entry_size: float
    entry_time_ms: int
    entry_day: str
    fill_mid_yes: float | None = None
    spread_cents: float | None = None
    future_mid_yes: dict[int, float | None] = field(default_factory=dict)
    signal_metadata: dict[str, Any] = field(default_factory=dict)

    def edge_vs_yes_mid(self, yes_mid: float | None) -> float | None:
        if yes_mid is None:
            return None
        if self.quote_side == "ASK":
            return (self.entry_price - yes_mid) * 100.0
        return (yes_mid - self.entry_price) * 100.0


@dataclass(slots=True)
class WallObservationRecord:
    wall_id: str
    market_id: str
    asset_id: str
    wall_side: str
    wall_price: float
    wall_size_usd: float
    first_seen_at_ms: int
    price_level_vs_mid_ticks: float
    time_of_day_bucket: str
    outcome: str = "EXPIRED"
    outcome_time_ms: int | None = None

    def age_at_outcome_ms(self) -> int | None:
        if self.outcome_time_ms is None:
            return None
        return max(0, self.outcome_time_ms - self.first_seen_at_ms)


def _canonical_wall_price(price: float) -> str:
    normalized = f"{price:.4f}".rstrip("0").rstrip(".")
    return normalized or "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze WallJumper universal backtest fills and 60-second markouts.")
    parser.add_argument("--db", default="logs/universal_backtest_wall_jumper.db")
    parser.add_argument("--input-dir", default="logs/local_snapshot/l2_data")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--run-log", default=None, help="Optional stdout capture from run_universal_backtest containing strategy_diagnostics.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--wall-age-ms", type=int, default=None, help="Override the expected wall-age threshold for reporting/sweep runs.")
    parser.add_argument("--wall-csv", default=None, help="Optional CSV path for per-wall lifecycle records.")
    return parser.parse_args()


def _epoch_window(start_date: str, end_date: str) -> tuple[float, float]:
    lower = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    upper = (datetime.fromisoformat(end_date) + timedelta(days=1)).replace(tzinfo=timezone.utc)
    return lower.timestamp(), upper.timestamp()


def load_wall_jumper_fills(db_path: Path, *, start_date: str, end_date: str) -> list[MakerFillRecord]:
    start_ts, end_ts = _epoch_window(start_date, end_date)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
                        select s.id, s.market_id, s.asset_id, s.reference_price_band, s.entry_price, s.entry_size, s.entry_time, j.payload_json
                        from shadow_trades s
                        left join trade_persistence_journal j on j.journal_key = ('shadow_trades:' || s.id)
                        where s.exit_reason = 'MAKER_FILL' and s.reference_price_band like 'MAKER:%'
                            and s.state = 'CLOSED'
                            and s.entry_time >= ? and s.entry_time < ?
                        order by s.asset_id, s.entry_time, s.id
            """,
            (start_ts, end_ts),
        ).fetchall()
    finally:
        conn.close()

    fills: list[MakerFillRecord] = []
    for row in rows:
        payload = _decode_json_blob(row[7])
        extra_payload = dict(payload.get("extra_payload") or {})
        signal_metadata = dict(extra_payload.get("signal_metadata") or {})
        if str(signal_metadata.get("strategy") or "").strip() != STRATEGY_NAME:
            continue
        band = str(row[3])
        quote_side = band.split(":", 1)[1].upper() if ":" in band else "BID"
        entry_time = float(row[6])
        fills.append(
            MakerFillRecord(
                fill_id=str(row[0]),
                market_id=str(row[1]),
                asset_id=str(row[2]),
                quote_side=quote_side,
                entry_price=float(row[4]),
                entry_size=float(row[5]),
                entry_time_ms=int(round(entry_time * 1000)),
                entry_day=datetime.fromtimestamp(entry_time, tz=timezone.utc).date().isoformat(),
                future_mid_yes={horizon: None for horizon in HORIZONS},
                signal_metadata=signal_metadata,
            )
        )
    return fills


def reconstruct_markouts(
    fills: list[MakerFillRecord],
    *,
    input_dir: Path,
    start_date: str,
    end_date: str,
) -> None:
    fills_by_asset: dict[str, list[MakerFillRecord]] = defaultdict(list)
    for fill in fills:
        fills_by_asset[fill.asset_id].append(fill)

    fill_idx = {asset_id: 0 for asset_id in fills_by_asset}
    horizon_idx = {(asset_id, horizon): 0 for asset_id in fills_by_asset for horizon in HORIZONS}
    trackers: dict[str, OrderbookTracker] = {}

    for event in iter_replay_events(resolve_tick_root(input_dir), start_date=start_date, end_date=end_date):
        if event.event_type not in {"BOOK", "PRICE_CHANGE"}:
            continue
        asset_fills = fills_by_asset.get(event.asset_id)
        if not asset_fills:
            continue
        tracker = trackers.get(event.asset_id)
        if tracker is None:
            tracker = OrderbookTracker(event.asset_id)
            trackers[event.asset_id] = tracker
        if event.event_type == "BOOK":
            tracker.on_book_snapshot(event.payload)
        else:
            tracker.on_price_change(event.payload)

        snapshot = tracker.snapshot()
        if snapshot.best_bid <= 0 or snapshot.best_ask <= 0:
            continue
        yes_mid = snapshot.mid_price

        idx = fill_idx[event.asset_id]
        while idx < len(asset_fills) and asset_fills[idx].entry_time_ms <= event.timestamp_ms:
            fill = asset_fills[idx]
            if fill.fill_mid_yes is None:
                fill.fill_mid_yes = yes_mid
                fill.spread_cents = snapshot.spread * 100.0
            idx += 1
        fill_idx[event.asset_id] = idx

        for horizon in HORIZONS:
            key = (event.asset_id, horizon)
            idx = horizon_idx[key]
            while idx < len(asset_fills):
                fill = asset_fills[idx]
                target_time_ms = fill.entry_time_ms + horizon * 1000
                if target_time_ms > event.timestamp_ms:
                    break
                if fill.future_mid_yes[horizon] is None:
                    fill.future_mid_yes[horizon] = yes_mid
                idx += 1
            horizon_idx[key] = idx


def summarize(fills: list[MakerFillRecord], *, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    diagnostics = diagnostics or {}
    quote_counts = Counter(fill.quote_side for fill in fills)
    spread_capture = [edge for fill in fills if (edge := fill.edge_vs_yes_mid(fill.fill_mid_yes)) is not None]
    spreads = [fill.spread_cents for fill in fills if fill.spread_cents is not None]
    sizes = [fill.entry_size for fill in fills]

    unique_markets: set[str] = set()
    horizon_summary: dict[int, dict[str, float | int]] = {}

    for fill in fills:
        unique_markets.add(fill.market_id)

    for horizon in HORIZONS:
        net_edges: list[float] = []
        gross_markouts: list[float] = []
        adverse_selection: list[float] = []
        pnl_cents: list[float] = []
        wins = 0
        covered = 0
        for fill in fills:
            gross_markout = fill.edge_vs_yes_mid(fill.future_mid_yes[horizon])
            capture = fill.edge_vs_yes_mid(fill.fill_mid_yes)
            if gross_markout is None or capture is None:
                continue
            covered += 1
            exit_fee_cents = get_fee_rate(fill.future_mid_yes[horizon] or 0.0) * 100.0
            net_edge = gross_markout - exit_fee_cents
            net_edges.append(net_edge)
            gross_markouts.append(gross_markout)
            adverse_selection.append(capture - gross_markout)
            pnl_cents.append(net_edge * fill.entry_size)
            if net_edge > 0:
                wins += 1
        horizon_summary[horizon] = {
            "covered_fills": covered,
            "win_rate": (wins / covered) if covered else 0.0,
            "average_markout_cents": mean(gross_markouts) if gross_markouts else 0.0,
            "average_adverse_selection_cents": mean(adverse_selection) if adverse_selection else 0.0,
            "average_net_edge_cents": mean(net_edges) if net_edges else 0.0,
            "median_net_edge_cents": median(net_edges) if net_edges else 0.0,
            "total_pnl_usd": sum(pnl_cents) / 100.0,
        }

    total_shares = sum(sizes)
    jump_quotes_emitted = int(diagnostics.get("jump_quotes_emitted") or 0)
    cancel_all_triggered = int(diagnostics.get("cancel_all_triggered") or 0)
    return {
        "walls_identified": int(diagnostics.get("walls_identified") or 0),
        "walls_aged_past_threshold": int(diagnostics.get("walls_aged_past_threshold") or 0),
        "wall_age_ms_threshold": int(diagnostics.get("wall_age_ms_threshold") or 0),
        "min_distance_from_mid_ticks": diagnostics.get("min_distance_from_mid_ticks"),
        "min_structural_wall_size_usd": diagnostics.get("min_structural_wall_size_usd"),
        "total_fills": len(fills),
        "total_shares": total_shares,
        "average_fill_size": mean(sizes) if sizes else 0.0,
        "bid_fills": int(quote_counts.get("BID", 0)),
        "ask_fills": int(quote_counts.get("ASK", 0)),
        "unique_markets": len(unique_markets),
        "average_observed_full_spread_cents": mean(spreads) if spreads else 0.0,
        "average_spread_capture_cents": mean(spread_capture) if spread_capture else 0.0,
        "median_spread_capture_cents": median(spread_capture) if spread_capture else 0.0,
        "jump_quotes_emitted": jump_quotes_emitted,
        "cancel_all_triggered": cancel_all_triggered,
        "cancel_trigger_rate": (cancel_all_triggered / jump_quotes_emitted) if jump_quotes_emitted else 0.0,
        "horizons": horizon_summary,
    }


def summarize_daily(fills: list[MakerFillRecord], *, start_date: str, end_date: str) -> list[dict[str, Any]]:
    by_day: dict[str, dict[str, Any]] = {}
    current_day = datetime.fromisoformat(start_date).date()
    final_day = datetime.fromisoformat(end_date).date()
    while current_day <= final_day:
        day_label = current_day.isoformat()
        by_day[day_label] = {
            "date": day_label,
            "total_fills": 0,
            "covered_fills_60s": 0,
            "total_shares": 0.0,
            "deployed_notional_usd": 0.0,
            "average_net_edge_60s_cents": 0.0,
            "total_pnl_60s_usd": 0.0,
            "daily_return": 0.0,
        }
        current_day += timedelta(days=1)

    net_edges_by_day: dict[str, list[float]] = defaultdict(list)
    pnl_cents_by_day: dict[str, float] = defaultdict(float)

    for fill in fills:
        day_summary = by_day.setdefault(
            fill.entry_day,
            {
                "date": fill.entry_day,
                "total_fills": 0,
                "covered_fills_60s": 0,
                "total_shares": 0.0,
                "deployed_notional_usd": 0.0,
                "average_net_edge_60s_cents": 0.0,
                "total_pnl_60s_usd": 0.0,
                "daily_return": 0.0,
            },
        )
        day_summary["total_fills"] += 1
        day_summary["total_shares"] += fill.entry_size
        day_summary["deployed_notional_usd"] += fill.entry_price * fill.entry_size

        gross_markout = fill.edge_vs_yes_mid(fill.future_mid_yes[60])
        if gross_markout is None:
            continue
        exit_fee_cents = get_fee_rate(fill.future_mid_yes[60] or 0.0) * 100.0
        net_edge = gross_markout - exit_fee_cents
        net_edges_by_day[fill.entry_day].append(net_edge)
        pnl_cents_by_day[fill.entry_day] += net_edge * fill.entry_size
        day_summary["covered_fills_60s"] += 1

    ordered_days: list[dict[str, Any]] = []
    for day_label in sorted(by_day):
        day_summary = by_day[day_label]
        net_edges = net_edges_by_day.get(day_label, [])
        total_pnl_usd = pnl_cents_by_day.get(day_label, 0.0) / 100.0
        deployed_notional = float(day_summary["deployed_notional_usd"])
        day_summary["average_net_edge_60s_cents"] = mean(net_edges) if net_edges else 0.0
        day_summary["total_pnl_60s_usd"] = total_pnl_usd
        day_summary["daily_return"] = (total_pnl_usd / deployed_notional) if deployed_notional > 0 else 0.0
        ordered_days.append(day_summary)
    return ordered_days


def estimate_daily_sharpe(daily_rows: list[dict[str, Any]]) -> float | None:
    daily_returns = [float(row["daily_return"]) for row in daily_rows]
    if len(daily_returns) < 2:
        return None
    daily_vol = stdev(daily_returns)
    if daily_vol <= 0.0:
        return None
    return (mean(daily_returns) / daily_vol) * math.sqrt(365.0)


def reconstruct_wall_lifecycle(
    *,
    input_dir: Path,
    start_date: str,
    end_date: str,
    fills: list[MakerFillRecord],
    min_wall_size: float = DEFAULT_MIN_WALL_SIZE,
    wall_to_opposing_ratio: float = DEFAULT_WALL_TO_OPPOSING_RATIO,
    depth_levels: int = DEFAULT_DEPTH_LEVELS,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> list[WallObservationRecord]:
    market_catalog = load_market_catalog(Path("data/market_map.json"))
    fill_lookup: dict[tuple[str, str, str, int], list[MakerFillRecord]] = defaultdict(list)
    for fill in fills:
        wall_side = str(fill.signal_metadata.get("wall_side") or "").strip().upper()
        wall_price = str(fill.signal_metadata.get("wall_price") or "")
        if not wall_side or not wall_price:
            continue
        key = (fill.market_id, fill.asset_id, wall_side, _price_key(float(wall_price)))
        fill_lookup[key].append(fill)

    for matching_fills in fill_lookup.values():
        matching_fills.sort(key=lambda fill: (fill.entry_time_ms, fill.fill_id))

    active_walls: dict[tuple[str, str], WallObservationRecord] = {}
    completed_walls: list[WallObservationRecord] = []
    trackers: dict[str, OrderbookTracker] = {}
    last_timestamp_ms = 0

    for event in iter_replay_events(resolve_tick_root(input_dir), start_date=start_date, end_date=end_date):
        if event.event_type not in {"BOOK", "PRICE_CHANGE"}:
            continue
        last_timestamp_ms = event.timestamp_ms
        strategy_market_id = event.market_id if market_catalog.has_market(event.market_id) else event.asset_id
        tracker = trackers.get(event.asset_id)
        if tracker is None:
            tracker = OrderbookTracker(event.asset_id)
            trackers[event.asset_id] = tracker
        if event.event_type == "BOOK":
            tracker.on_book_snapshot(event.payload)
        else:
            tracker.on_price_change(event.payload)

        bids = tracker.levels("bid", n=depth_levels)
        asks = tracker.levels("ask", n=depth_levels)
        stream_key = (strategy_market_id, event.asset_id)
        active_wall = active_walls.get(stream_key)

        candidate = _select_wall_candidate(
            market_id=strategy_market_id,
            asset_id=event.asset_id,
            bids=bids,
            asks=asks,
            timestamp_ms=event.timestamp_ms,
            min_wall_size=min_wall_size,
            wall_to_opposing_ratio=wall_to_opposing_ratio,
            tick_size=tick_size,
        )

        if active_wall is not None:
            matching_fill = _consume_matching_fill(fill_lookup, active_wall)
            if matching_fill is not None and matching_fill.entry_time_ms <= event.timestamp_ms:
                active_wall.outcome = "FILLED"
                active_wall.outcome_time_ms = matching_fill.entry_time_ms
                completed_walls.append(active_wall)
                active_walls.pop(stream_key, None)
                active_wall = None

        if candidate is None:
            if active_wall is not None:
                _finalize_wall(active_wall, completed_walls, outcome="PULLED", outcome_time_ms=event.timestamp_ms)
                active_walls.pop(stream_key, None)
            continue

        if active_wall is None:
            active_walls[stream_key] = candidate
            continue

        if active_wall.wall_side != candidate.wall_side or _price_key(active_wall.wall_price) != _price_key(candidate.wall_price):
            current_size = _current_wall_size(active_wall, bids=bids, asks=asks)
            outcome = "PULLED" if current_size <= 0.0 else "EXPIRED"
            _finalize_wall(active_wall, completed_walls, outcome=outcome, outcome_time_ms=event.timestamp_ms)
            active_walls[stream_key] = candidate

    for active_wall in active_walls.values():
        _finalize_wall(active_wall, completed_walls, outcome="EXPIRED", outcome_time_ms=last_timestamp_ms)

    records_by_id = {record.wall_id: record for record in completed_walls}
    records_by_signature: dict[tuple[str, str, int], list[WallObservationRecord]] = defaultdict(list)
    for record in completed_walls:
        records_by_signature[(record.market_id, record.wall_side, _price_key(record.wall_price))].append(record)
    for candidates in records_by_signature.values():
        candidates.sort(key=lambda record: record.first_seen_at_ms)

    for fill in fills:
        wall_id = str(fill.signal_metadata.get("wall_id") or "").strip()
        if not wall_id:
            record = None
        else:
            record = records_by_id.get(wall_id)
        if record is None:
            wall_side = str(fill.signal_metadata.get("wall_side") or "").strip().upper()
            wall_price_raw = fill.signal_metadata.get("wall_price")
            if wall_side and wall_price_raw not in (None, ""):
                signature = (fill.market_id, wall_side, _price_key(float(wall_price_raw)))
                candidates = records_by_signature.get(signature, [])
                eligible = [candidate for candidate in candidates if candidate.first_seen_at_ms <= fill.entry_time_ms]
                if eligible:
                    record = max(eligible, key=lambda candidate: candidate.first_seen_at_ms)
        if record is None:
            continue
        if record.outcome != "FILLED" or (record.outcome_time_ms or 0) > fill.entry_time_ms:
            record.outcome = "FILLED"
            record.outcome_time_ms = fill.entry_time_ms
    return completed_walls


def write_wall_observation_csv(records: list[WallObservationRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "wall_id",
                "market_id",
                "asset_id",
                "wall_side",
                "wall_size_usd",
                "wall_age_at_pull_ms",
                "wall_age_at_fill_ms",
                "price_level_vs_mid_ticks",
                "time_of_day_bucket",
                "outcome",
            ],
        )
        writer.writeheader()
        for record in records:
            age_at_outcome_ms = record.age_at_outcome_ms()
            writer.writerow(
                {
                    "wall_id": record.wall_id,
                    "market_id": record.market_id,
                    "asset_id": record.asset_id,
                    "wall_side": record.wall_side,
                    "wall_size_usd": f"{record.wall_size_usd:.6f}",
                    "wall_age_at_pull_ms": age_at_outcome_ms if record.outcome == "PULLED" else "",
                    "wall_age_at_fill_ms": age_at_outcome_ms if record.outcome == "FILLED" else "",
                    "price_level_vs_mid_ticks": f"{record.price_level_vs_mid_ticks:.4f}",
                    "time_of_day_bucket": record.time_of_day_bucket,
                    "outcome": record.outcome,
                }
            )


def _select_wall_candidate(
    *,
    market_id: str,
    asset_id: str,
    bids: list[Any],
    asks: list[Any],
    timestamp_ms: int,
    min_wall_size: float,
    wall_to_opposing_ratio: float,
    tick_size: float,
) -> WallObservationRecord | None:
    if not bids or not asks:
        return None
    bid_candidate = _best_wall_candidate("BID", bids, asks, min_wall_size=min_wall_size, wall_to_opposing_ratio=wall_to_opposing_ratio)
    ask_candidate = _best_wall_candidate("ASK", asks, bids, min_wall_size=min_wall_size, wall_to_opposing_ratio=wall_to_opposing_ratio)
    candidates = [candidate for candidate in (bid_candidate, ask_candidate) if candidate is not None]
    if not candidates:
        return None
    wall_side, wall_price, wall_size, _ = max(candidates, key=lambda item: (item[3], item[2]))
    best_bid = float(bids[0].price)
    best_ask = float(asks[0].price)
    if best_bid <= 0.0 or best_ask <= 0.0:
        return None
    mid = (best_bid + best_ask) / 2.0
    signed_ticks = (float(wall_price) - mid) / tick_size if tick_size > 0 else 0.0
    time_bucket = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).strftime("%H:00-%H:59")
    wall_id = f"{market_id}:{wall_side}:{_canonical_wall_price(float(wall_price))}:{timestamp_ms}"
    return WallObservationRecord(
        wall_id=wall_id,
        market_id=market_id,
        asset_id=asset_id,
        wall_side=wall_side,
        wall_price=float(wall_price),
        wall_size_usd=float(wall_price) * float(wall_size),
        first_seen_at_ms=timestamp_ms,
        price_level_vs_mid_ticks=signed_ticks,
        time_of_day_bucket=time_bucket,
    )


def _best_wall_candidate(
    wall_side: str,
    levels: list[Any],
    opposing_levels: list[Any],
    *,
    min_wall_size: float,
    wall_to_opposing_ratio: float,
) -> tuple[str, float, float, float] | None:
    if not levels or not opposing_levels:
        return None
    opposing_average = sum(float(level.size) for level in opposing_levels) / float(len(opposing_levels))
    if opposing_average <= 0.0:
        return None
    best: tuple[str, float, float, float] | None = None
    for level in levels:
        size = float(level.size)
        if size < min_wall_size:
            continue
        ratio = size / opposing_average
        if ratio < wall_to_opposing_ratio:
            continue
        candidate = (wall_side, float(level.price), size, ratio)
        if best is None or (candidate[3], candidate[2]) > (best[3], best[2]):
            best = candidate
    return best


def _current_wall_size(record: WallObservationRecord, *, bids: list[Any], asks: list[Any]) -> float:
    levels = bids if record.wall_side == "BID" else asks
    target_price_key = _price_key(record.wall_price)
    for level in levels:
        if _price_key(float(level.price)) == target_price_key:
            return float(level.size)
    return 0.0


def _consume_matching_fill(
    fill_lookup: dict[tuple[str, str, str, int], list[MakerFillRecord]],
    record: WallObservationRecord,
) -> MakerFillRecord | None:
    key = (record.market_id, record.asset_id, record.wall_side, _price_key(record.wall_price))
    fills = fill_lookup.get(key)
    if not fills:
        return None
    while fills:
        fill = fills[0]
        if fill.entry_time_ms < record.first_seen_at_ms:
            fills.pop(0)
            continue
        return fills.pop(0)
    return None


def _finalize_wall(
    record: WallObservationRecord,
    completed_walls: list[WallObservationRecord],
    *,
    outcome: str,
    outcome_time_ms: int,
) -> None:
    record.outcome = outcome
    record.outcome_time_ms = outcome_time_ms
    completed_walls.append(record)


def _price_key(price: float) -> int:
    return int(round(price * 10_000))


def render_markdown(
    summary: dict[str, Any],
    *,
    date_label: str,
    db_path: Path,
    input_dir: Path,
    run_log: Path | None,
    daily_rows: list[dict[str, Any]],
    estimated_daily_sharpe: float | None,
) -> str:
    horizon_sixty = dict(summary["horizons"][60])
    net_edge = float(horizon_sixty["average_net_edge_cents"])
    verdict = "positive" if net_edge > 0 else "negative"
    structural_distance = summary.get("min_distance_from_mid_ticks")
    structural_size = summary.get("min_structural_wall_size_usd")
    total_pnl_60s = sum(float(row["total_pnl_60s_usd"]) for row in daily_rows)
    title = "WallJumper OOS Scorecard" if " to " in date_label else "WallJumper Toxic-Day Scorecard"
    lines = [
        f"# {title}",
        "",
        "## Run",
        "",
        f"- Date: `{date_label}`",
        "- Strategy: `src.signals.wall_jumper.WallJumper`",
        f"- Replay DB: `{db_path}`",
        f"- Replay source: `{input_dir}`",
    ]
    if run_log is not None:
        lines.append(f"- Backtest log: `{run_log}`")

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Fills were read from the universal backtest database and filtered to `signal_metadata.strategy = wall_jumper`.",
            "- Fill-time spread capture and 5s/15s/60s forward markouts were reconstructed from the same raw tick stream used by the replay engine.",
            "- Net edge per share was computed as signed fill-to-mid markout at each horizon, net of the simulated exit fee.",
            "- Emergency wall collapses were counted from the strategy diagnostics emitted by the replay run.",
            "- WallJumper v3 only jumps walls that are both structurally deep and structurally large; no time-of-day overlay is applied.",
            "",
            "## Counter Block",
            "",
            f"- Walls identified: `{summary['walls_identified']}`",
            f"- Walls aged past `{summary['wall_age_ms_threshold']}`ms: `{summary['walls_aged_past_threshold']}`",
            f"- Min distance from mid: `{structural_distance}` ticks" if structural_distance is not None else "- Min distance from mid: `n/a`",
            f"- Min wall size: `${structural_size}`" if structural_size is not None else "- Min wall size: `n/a`",
            f"- Jump quotes emitted: `{summary['jump_quotes_emitted']}`",
            f"- Emergency `CANCEL_ALL` triggers: `{summary['cancel_all_triggered']}`",
            f"- Maker fills: `{summary['total_fills']}`",
            f"- Filtered-set wall-pull rate: `{summary['cancel_trigger_rate']:.2%}`",
            f"- Unique markets touched: `{summary['unique_markets']}`",
            f"- Total filled shares: `{summary['total_shares']:.1f}`",
            f"- Average fill size: `{summary['average_fill_size']:.4f}`",
            f"- Bid fills: `{summary['bid_fills']}`",
            f"- Ask fills: `{summary['ask_fills']}`",
            f"- Average observed full spread at fill: `{summary['average_observed_full_spread_cents']:.5f}c/share`",
            f"- Average spread captured: `{summary['average_spread_capture_cents']:.5f}c/share`",
            f"- Median spread captured: `{summary['median_spread_capture_cents']:.5f}c/share`",
            "",
            "## Markout Table",
            "",
            "| Horizon | Covered Fills | Win Rate | Avg Markout (c/share) | Avg Adverse Selection (c/share) | Avg Net Edge (c/share) | Total PnL (USD) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon in HORIZONS:
        row = dict(summary["horizons"][horizon])
        lines.append(
            "| {h}s | {covered} | {win_rate:.2%} | {markout:.5f} | {adverse:.5f} | {edge:.5f} | {pnl:.2f} |".format(
                h=horizon,
                covered=int(row["covered_fills"]),
                win_rate=float(row["win_rate"]),
                markout=float(row["average_markout_cents"]),
                adverse=float(row["average_adverse_selection_cents"]),
                edge=float(row["average_net_edge_cents"]),
                pnl=float(row["total_pnl_usd"]),
            )
        )
    lines.extend(
        [
            "",
            "## Daily Breakdown",
            "",
            "| Day | Total Fills | Covered 60s Fills | Total Shares | Deployed Notional (USD) | 60s Avg Net Edge (c/share) | 60s PnL (USD) | Daily Return |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in daily_rows:
        lines.append(
            "| {date} | {fills} | {covered} | {shares:.1f} | {notional:.2f} | {edge:.5f} | {pnl:.2f} | {daily_return:.5%} |".format(
                date=row["date"],
                fills=int(row["total_fills"]),
                covered=int(row["covered_fills_60s"]),
                shares=float(row["total_shares"]),
                notional=float(row["deployed_notional_usd"]),
                edge=float(row["average_net_edge_60s_cents"]),
                pnl=float(row["total_pnl_60s_usd"]),
                daily_return=float(row["daily_return"]),
            )
        )
    lines.extend(
        [
            "",
            "## OOS Diagnostics",
            "",
            f"- 5-day average 60s net edge: `{float(horizon_sixty['average_net_edge_cents']):.5f}c/share`",
            f"- Aggregate filtered-set wall-pull rate: `{float(summary['cancel_trigger_rate']):.2%}`",
            f"- Aggregate 60s PnL across the sweep: `${total_pnl_60s:.2f}`",
            f"- Estimated OOS daily-return Sharpe: `{estimated_daily_sharpe:.3f}`" if estimated_daily_sharpe is not None else "- Estimated OOS daily-return Sharpe: `n/a`",
        ]
    )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "- The v3 structural gate was **{verdict}** over `{date}`: 60s net edge was `{edge:.5f}c/share` and total 60s simulated PnL was `${pnl:.2f}`."
            ).format(
                verdict=verdict,
                date=date_label,
                edge=float(horizon_sixty["average_net_edge_cents"]),
                pnl=total_pnl_60s,
            ),
            (
                "- Whale support still failed `{cancel_count}` times across `{jump_quotes}` filtered jump quotes, for a filtered-set wall-pull rate of `{cancel_rate:.2%}`."
            ).format(
                cancel_count=int(summary["cancel_all_triggered"]),
                jump_quotes=int(summary["jump_quotes_emitted"]),
                cancel_rate=float(summary["cancel_trigger_rate"]),
            ),
            (
                "- The realized 60s markout averaged `{markout:.5f}c/share` while immediate spread capture averaged `{capture:.5f}c/share`, which indicates the post-fill drift {drift_assessment}."
            ).format(
                markout=float(horizon_sixty["average_markout_cents"]),
                capture=float(summary["average_spread_capture_cents"]),
                drift_assessment="preserved the captured edge" if net_edge > 0 else "overwhelmed the captured edge",
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def parse_run_diagnostics(run_log: Path | None) -> dict[str, Any]:
    if run_log is None or not run_log.exists():
        return {}
    log_text = _read_text_with_fallbacks(run_log)
    for line in log_text.splitlines():
        match = DIAGNOSTICS_PATTERN.match(line.strip())
        if not match:
            continue
        diagnostics: dict[str, Any] = {}
        for token in match.group("payload").split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            diagnostics[key] = _parse_scalar(value)
        return diagnostics
    return {}


def _decode_json_blob(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_text_with_fallbacks(path: Path) -> str:
    raw_bytes = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    input_dir = Path(args.input_dir)
    run_log = Path(args.run_log) if args.run_log else None
    diagnostics = parse_run_diagnostics(run_log)
    if args.wall_age_ms is not None:
        diagnostics["wall_age_ms_threshold"] = args.wall_age_ms
    fills = load_wall_jumper_fills(db_path, start_date=args.start_date, end_date=args.end_date)
    reconstruct_markouts(fills, input_dir=input_dir, start_date=args.start_date, end_date=args.end_date)
    wall_records: list[WallObservationRecord] = []
    if args.wall_csv:
        wall_records = reconstruct_wall_lifecycle(
            input_dir=input_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            fills=fills,
        )
        write_wall_observation_csv(wall_records, Path(args.wall_csv))
    summary = summarize(fills, diagnostics=diagnostics)
    daily_rows = summarize_daily(fills, start_date=args.start_date, end_date=args.end_date)
    estimated_daily_sharpe = estimate_daily_sharpe(daily_rows)
    markdown = render_markdown(
        summary,
        date_label=args.start_date if args.start_date == args.end_date else f"{args.start_date} to {args.end_date}",
        db_path=db_path,
        input_dir=input_dir,
        run_log=run_log,
        daily_rows=daily_rows,
        estimated_daily_sharpe=estimated_daily_sharpe,
    )
    output_payload = dict(summary)
    output_payload["daily"] = daily_rows
    output_payload["estimated_daily_sharpe"] = estimated_daily_sharpe
    print(json.dumps(output_payload, indent=2, sort_keys=True))
    if args.wall_csv:
        print(f"wall_csv={args.wall_csv} wall_records={len(wall_records)}")
    print("\n---MARKDOWN---\n")
    print(markdown)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()