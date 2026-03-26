from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import statistics
from typing import Any

from src.backtest.data_loader import DataLoader
from src.backtest.wfo_optimizer import _build_data_loader, _load_market_configs
from src.data.orderbook import OrderbookTracker


@dataclass(frozen=True, slots=True)
class MarketPriceObservation:
    timestamp: float
    price: float


@dataclass(frozen=True, slots=True)
class MarketSeries:
    market_id: str
    observations: tuple[MarketPriceObservation, ...]
    days_observed: frozenset[str]
    event_count: int


def parse_iso_date(date_text: str) -> str:
    datetime.fromisoformat(date_text)
    return date_text


def parse_iso_datetime(timestamp_text: str) -> datetime:
    text = timestamp_text.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def date_range(start: datetime, end: datetime) -> list[str]:
    if end < start:
        raise ValueError("end must be >= start")
    cursor = start.date()
    end_date = end.date()
    values: list[str] = []
    while cursor <= end_date:
        values.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return values


def load_universe_market_configs(universe_path: str) -> list[dict[str, Any]]:
    configs = _load_market_configs(".", market_configs_path=universe_path)
    if not configs:
        raise ValueError(f"No valid market configs found in {universe_path!r}")
    return configs


def load_market_map_entries(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {str(candidate)!r}")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        market_id = str(item.get("market_id") or "").strip()
        yes_id = str(item.get("yes_asset_id") or item.get("yes_id") or "").strip()
        no_id = str(item.get("no_asset_id") or item.get("no_id") or "").strip()
        if not market_id or not yes_id or not no_id:
            continue
        entries.append(
            {
                "market_id": market_id or f"MARKET_{index}",
                "yes_asset_id": yes_id,
                "no_asset_id": no_id,
                "question": str(item.get("question") or market_id),
                "event_id": str(item.get("event_id") or item.get("group") or ""),
                "tags": item.get("tags") or item.get("theme") or "",
            }
        )
    return entries


def build_yes_price_series(
    archive_path: str,
    market_configs: list[dict[str, Any]],
    dates: list[str],
    *,
    max_events: int | None = None,
) -> dict[str, MarketSeries]:
    yes_asset_to_market = {
        str(config["yes_asset_id"]): str(config["market_id"])
        for config in market_configs
        if config.get("yes_asset_id") and config.get("market_id")
    }
    if not yes_asset_to_market:
        return {}

    loader = _build_data_loader(
        archive_path,
        dates,
        asset_ids=set(yes_asset_to_market),
    )
    if loader is None:
        return {}

    trackers = {asset_id: OrderbookTracker(asset_id) for asset_id in yes_asset_to_market}
    observations: dict[str, list[MarketPriceObservation]] = {market_id: [] for market_id in yes_asset_to_market.values()}
    days_observed: dict[str, set[str]] = {market_id: set() for market_id in yes_asset_to_market.values()}
    event_counts: dict[str, int] = {market_id: 0 for market_id in yes_asset_to_market.values()}

    processed = 0
    for event in loader:
        if max_events is not None and processed >= max_events:
            break
        processed += 1

        market_id = yes_asset_to_market.get(event.asset_id)
        if market_id is None:
            continue

        tracker = trackers[event.asset_id]
        if event.event_type == "l2_snapshot":
            tracker.on_book_snapshot(event.data)
        elif event.event_type == "l2_delta":
            tracker.on_price_change(event.data)
        else:
            continue

        snapshot = tracker.snapshot()
        if snapshot.mid_price <= 0.0:
            continue
        observations[market_id].append(MarketPriceObservation(timestamp=float(event.timestamp), price=float(snapshot.mid_price)))
        days_observed[market_id].add(datetime.fromtimestamp(event.timestamp, tz=timezone.utc).date().isoformat())
        event_counts[market_id] += 1

    return {
        market_id: MarketSeries(
            market_id=market_id,
            observations=tuple(series),
            days_observed=frozenset(days_observed[market_id]),
            event_count=event_counts[market_id],
        )
        for market_id, series in observations.items()
    }


def median_absolute_move_over_window(series_by_market: dict[str, MarketSeries], window_ms: int) -> float:
    if window_ms < 0:
        raise ValueError("window_ms must be >= 0")
    window_s = float(window_ms) / 1000.0
    moves: list[float] = []
    for series in series_by_market.values():
        if len(series.observations) < 2:
            continue
        timestamps = [point.timestamp for point in series.observations]
        prices = [point.price for point in series.observations]
        for index, timestamp in enumerate(timestamps[:-1]):
            target = timestamp + window_s
            future_index = bisect_left(timestamps, target, lo=index + 1)
            if future_index >= len(timestamps):
                continue
            moves.append(abs(prices[future_index] - prices[index]))
    if not moves:
        return 0.0
    return float(statistics.median(moves))


def events_per_day(series: MarketSeries) -> float:
    observed_days = max(1, len(series.days_observed))
    return float(series.event_count) / float(observed_days)


def compute_lagged_pair_metrics(
    leader: MarketSeries,
    lagger: MarketSeries,
    *,
    freshness_ms: int,
    response_window_ms: int,
) -> dict[str, float]:
    freshness_ms = max(0, int(freshness_ms))
    response_window_ms = max(0, int(response_window_ms))
    leader_obs = leader.observations
    lagger_obs = lagger.observations
    if len(leader_obs) < 3 or len(lagger_obs) < 3:
        return {
            "correlation": 0.0,
            "median_lagger_age_ms": math.inf,
            "freshness_coverage": 0.0,
            "leader_to_lagger_strength": 0.0,
            "lagger_to_leader_strength": 0.0,
            "aligned_samples": 0.0,
        }

    leader_times = [point.timestamp for point in leader_obs]
    leader_prices = [point.price for point in leader_obs]
    lagger_times = [point.timestamp for point in lagger_obs]
    lagger_prices = [point.price for point in lagger_obs]

    leader_returns: list[float] = []
    lagger_returns: list[float] = []
    lagger_ages_ms: list[float] = []
    prior_lagger_index: int | None = None

    for index in range(1, len(leader_times)):
        leader_time = leader_times[index]
        lagger_index = bisect_right(lagger_times, leader_time) - 1
        if lagger_index < 0:
            continue
        if prior_lagger_index is None:
            prior_lagger_index = lagger_index
            continue
        leader_returns.append(leader_prices[index] - leader_prices[index - 1])
        lagger_returns.append(lagger_prices[lagger_index] - lagger_prices[prior_lagger_index])
        lagger_ages_ms.append(max(0.0, (leader_time - lagger_times[lagger_index]) * 1000.0))
        prior_lagger_index = lagger_index

    correlation = max(0.0, pearson_correlation(leader_returns, lagger_returns))
    median_age = float(statistics.median(lagger_ages_ms)) if lagger_ages_ms else math.inf
    freshness_coverage = (
        float(sum(age <= float(freshness_ms) for age in lagger_ages_ms)) / float(len(lagger_ages_ms))
        if lagger_ages_ms
        else 0.0
    )

    leader_to_lagger_strength = response_correlation(leader_obs, lagger_obs, response_window_ms)
    lagger_to_leader_strength = response_correlation(lagger_obs, leader_obs, response_window_ms)
    return {
        "correlation": correlation,
        "median_lagger_age_ms": median_age,
        "freshness_coverage": freshness_coverage,
        "leader_to_lagger_strength": leader_to_lagger_strength,
        "lagger_to_leader_strength": lagger_to_leader_strength,
        "aligned_samples": float(len(lagger_ages_ms)),
    }


def response_correlation(
    source_observations: tuple[MarketPriceObservation, ...],
    target_observations: tuple[MarketPriceObservation, ...],
    window_ms: int,
) -> float:
    if len(source_observations) < 3 or len(target_observations) < 3:
        return 0.0
    window_s = float(window_ms) / 1000.0
    target_times = [point.timestamp for point in target_observations]
    target_prices = [point.price for point in target_observations]

    source_moves: list[float] = []
    target_moves: list[float] = []
    for index in range(1, len(source_observations)):
        timestamp = source_observations[index].timestamp
        source_move = source_observations[index].price - source_observations[index - 1].price
        base_index = bisect_right(target_times, timestamp) - 1
        future_index = bisect_left(target_times, timestamp + window_s, lo=max(base_index + 1, 0))
        if base_index < 0 or future_index >= len(target_times):
            continue
        source_moves.append(source_move)
        target_moves.append(target_prices[future_index] - target_prices[base_index])
    return max(0.0, pearson_correlation(source_moves, target_moves))


def pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((l - mean_left) * (r - mean_right) for l, r in zip(left, right, strict=False))
    variance_left = sum((l - mean_left) ** 2 for l in left)
    variance_right = sum((r - mean_right) ** 2 for r in right)
    denominator = math.sqrt(variance_left * variance_right)
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def percentile(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0.0, min(100.0, float(value))) / 100.0 * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction
