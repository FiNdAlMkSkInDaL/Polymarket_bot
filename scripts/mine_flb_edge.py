#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_universal_backtest import resolve_tick_root
from src.core.config import settings
from src.trading.fees import get_fee_rate


DEFAULT_DB = Path("logs/trades.db")
DEFAULT_INPUT_DIR = Path("logs/local_snapshot/l2_data")
DEFAULT_REPORT = Path("docs/flb_yield_analysis.md")
DEFAULT_JSON_OUTPUT = Path("data/flb_results_final.json")
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
SNAPSHOT_EVENT_TYPES = {"book", "snapshot", "book_snapshot", "l2_snapshot"}
DELTA_EVENT_TYPES = {"price_change", "delta", "l2_delta"}
DISPLAY_CATEGORIES = {"crypto", "geopolitics", "sports", "politics", "policy", "macro", "business", "technology", "culture"}
THEME_KEYWORDS = {
    "crypto": {
        "airdrop", "bitcoin", "btc", "consensys", "crypto", "defi", "eth", "ethereum", "kraken", "metamask", "microstrategy", "ostium", "pump", "sol", "solana", "theo", "token", "usdai",
    },
    "geopolitics": {
        "annex", "capture", "china", "clash", "cuba", "disarm", "hezbollah", "india", "iran", "israel", "nato", "nuke", "qassem", "russia", "territory", "ukraine", "withdraw",
    },
    "sports": {
        "champions", "conference", "esports", "fifa", "finals", "league", "lck", "lpl", "nba", "playoffs", "qualify", "season", "tournament", "win", "world",
    },
    "policy": {
        "cap", "deductions", "fed", "gambling", "interest", "rates", "repealed", "tax", "withdraw",
    },
    "macro": {
        "economy", "gdp", "inflation", "rates", "recession", "unemployment",
    },
    "politics": {
        "candidate", "democratic", "election", "electoral", "impeach", "nomination", "nominee", "party", "presidential", "president", "republican", "senate", "trump",
    },
    "business": {
        "bankruptcy", "cap", "company", "ipo", "market", "merger", "revenue", "stock", "valuation",
    },
    "technology": {
        "ai", "launch", "software", "tech", "token",
    },
    "culture": {
        "album", "film", "game", "gta", "movie", "released", "release", "tv",
    },
}


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.{digits}f}%"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_hours(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}h"


def _fmt_ts(timestamp: float | None) -> str:
    if timestamp is None:
        return "n/a"
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


@dataclass(slots=True)
class SqliteAudit:
    db_path: Path
    exists: bool
    tables: list[str]
    bbo_tables: list[str]


@dataclass(slots=True)
class TokenPathState:
    condition_id: str
    token_id: str
    min_ask: float
    max_ask: float
    min_duration_s: float
    max_gap_s: float
    last_ts: float | None = None
    last_ask: float | None = None
    in_band_started_at: float | None = None
    in_band_duration_s: float = 0.0
    longest_in_band_s: float = 0.0
    qualifying_entry_ts: float | None = None
    entry_ask: float | None = None
    post_entry_max_ask: float | None = None
    post_entry_min_ask: float | None = None
    post_entry_last_ask: float | None = None
    post_entry_last_ts: float | None = None
    first_above_10_ts: float | None = None
    first_above_25_ts: float | None = None
    first_above_50_ts: float | None = None
    first_above_95_ts: float | None = None
    quote_observation_count: int = 0

    def _is_in_band(self, ask: float | None) -> bool:
        return ask is not None and self.min_ask <= ask <= self.max_ask

    def _mark_post_entry(self, ask: float, timestamp: float) -> None:
        if self.post_entry_max_ask is None or ask > self.post_entry_max_ask:
            self.post_entry_max_ask = ask
        if self.post_entry_min_ask is None or ask < self.post_entry_min_ask:
            self.post_entry_min_ask = ask
        self.post_entry_last_ask = ask
        self.post_entry_last_ts = timestamp
        if ask >= 0.10 and self.first_above_10_ts is None:
            self.first_above_10_ts = timestamp
        if ask >= 0.25 and self.first_above_25_ts is None:
            self.first_above_25_ts = timestamp
        if ask >= 0.50 and self.first_above_50_ts is None:
            self.first_above_50_ts = timestamp
        if ask >= 0.95 and self.first_above_95_ts is None:
            self.first_above_95_ts = timestamp

    def advance_to(self, timestamp: float) -> None:
        if self.last_ts is None or self.last_ask is None:
            return
        gap_s = max(0.0, timestamp - self.last_ts)
        was_in_band = self._is_in_band(self.last_ask)
        if was_in_band and gap_s <= self.max_gap_s:
            prev_duration = self.in_band_duration_s
            self.in_band_duration_s += gap_s
            if self.qualifying_entry_ts is None and prev_duration < self.min_duration_s <= self.in_band_duration_s:
                offset_s = self.min_duration_s - prev_duration
                self.qualifying_entry_ts = self.last_ts + offset_s
                self.entry_ask = self.last_ask
                self._mark_post_entry(self.last_ask, self.qualifying_entry_ts)
        else:
            if was_in_band:
                self.longest_in_band_s = max(self.longest_in_band_s, self.in_band_duration_s)
            self.in_band_started_at = None
            self.in_band_duration_s = 0.0

    def observe(self, ask: float | None, timestamp: float) -> None:
        previous_in_band = self._is_in_band(self.last_ask)
        current_in_band = self._is_in_band(ask)

        if ask is not None:
            self.quote_observation_count += 1

        if current_in_band:
            if self.in_band_started_at is None:
                self.in_band_started_at = timestamp
                self.in_band_duration_s = 0.0
        elif previous_in_band or self.in_band_started_at is not None:
            self.longest_in_band_s = max(self.longest_in_band_s, self.in_band_duration_s)
            self.in_band_started_at = None
            self.in_band_duration_s = 0.0

        if ask is not None and self.qualifying_entry_ts is not None and timestamp >= self.qualifying_entry_ts:
            self._mark_post_entry(ask, timestamp)

        self.last_ts = timestamp
        self.last_ask = ask

    def finalize(self) -> None:
        if self._is_in_band(self.last_ask):
            self.longest_in_band_s = max(self.longest_in_band_s, self.in_band_duration_s)


@dataclass(slots=True)
class QualifiedTokenCandidate:
    condition_id: str
    token_id: str
    qualifying_entry_ts: float
    entry_ask: float
    longest_in_band_s: float
    post_entry_last_ts: float | None
    terminal_ask: float | None
    max_ask_after_entry: float | None
    min_ask_after_entry: float | None
    first_above_10_ts: float | None
    first_above_25_ts: float | None
    first_above_50_ts: float | None
    first_above_95_ts: float | None
    quote_observation_count: int

    @property
    def observed_post_entry_hours(self) -> float | None:
        if self.post_entry_last_ts is None:
            return None
        return max(0.0, self.post_entry_last_ts - self.qualifying_entry_ts) / 3600.0


@dataclass(slots=True)
class ResolvedConditionMetadata:
    condition_id: str
    question: str
    slug: str
    yes_token_id: str | None
    no_token_id: str | None
    event_title: str
    end_date: str | None
    closed: bool
    active: bool
    yes_outcome_price: float | None
    no_outcome_price: float | None
    market_category: str | None
    event_category: str | None

    @property
    def resolved_yes(self) -> bool:
        return bool(self.closed and self.yes_outcome_price is not None and self.yes_outcome_price >= 0.999)

    @property
    def resolved_no(self) -> bool:
        return bool(self.closed and self.yes_outcome_price is not None and self.yes_outcome_price <= 0.001)


@dataclass(slots=True)
class QualifiedYesLongshot:
    metadata: ResolvedConditionMetadata
    candidate: QualifiedTokenCandidate

    @property
    def bucket_label(self) -> str:
        cents = self.candidate.entry_ask * 100.0
        if cents < 2.0:
            return "1-2c"
        if cents < 3.0:
            return "2-3c"
        if cents < 4.0:
            return "3-4c"
        return "4-5c"

    @property
    def max_adverse_no_mark_to_market_cents(self) -> float:
        max_ask = self.candidate.max_ask_after_entry or self.candidate.entry_ask
        return max(0.0, (max_ask - 0.05) * 100.0)

    @property
    def theoretical_pnl_per_share(self) -> float:
        return -0.95 if self.metadata.resolved_yes else 0.05

    @property
    def theoretical_roc(self) -> float:
        return self.theoretical_pnl_per_share / 0.95

    @property
    def category(self) -> str:
        return infer_category(self.metadata)

    @property
    def fee_enabled(self) -> bool:
        configured = {
            part.strip().lower()
            for part in str(settings.strategy.fee_enabled_categories or "").split(",")
            if part.strip()
        }
        return self.category in configured

    @property
    def entry_fee_per_share(self) -> float:
        return get_fee_rate(0.95, fee_enabled=self.fee_enabled)

    @property
    def resolved_net_pnl_per_share(self) -> float:
        return self.theoretical_pnl_per_share - self.entry_fee_per_share

    @property
    def resolved_net_roc(self) -> float:
        return self.resolved_net_pnl_per_share / 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine sustained sub-5c longshot windows and estimate favorite-longshot short edge."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite DB path to audit for historical BBO coverage.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Replay root containing raw tick archives.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT, help="Markdown report output path.")
    parser.add_argument("--start-date", default=None, help="Inclusive start date YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Inclusive end date YYYY-MM-DD.")
    parser.add_argument("--min-yes-ask", type=float, default=0.01, help="Minimum YES ask to treat as longshot.")
    parser.add_argument("--max-yes-ask", type=float, default=0.05, help="Maximum YES ask to treat as longshot.")
    parser.add_argument("--min-sustain-hours", type=float, default=24.0, help="Required continuous time in band before a market qualifies.")
    parser.add_argument("--max-gap-hours", type=float, default=6.0, help="Maximum observation gap still treated as continuous state.")
    parser.add_argument("--gamma-timeout-seconds", type=float, default=20.0, help="HTTP timeout for Gamma metadata resolution.")
    parser.add_argument("--max-markets", type=int, default=None, help="Optional cap on grouped market files, for fast debugging.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Parallel worker count for market parsing.")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT, help="Machine-readable FLB results output path.")
    return parser.parse_args()


def _tokenise(text: str) -> set[str]:
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() else " ")
    return {part for part in "".join(cleaned).split() if part}


def infer_category(metadata: ResolvedConditionMetadata) -> str:
    for source in (metadata.event_category, metadata.market_category):
        text = str(source or "").strip().lower()
        if text in DISPLAY_CATEGORIES:
            return text
    tokens = _tokenise(" ".join(filter(None, [metadata.question, metadata.slug, metadata.event_title, metadata.market_category or "", metadata.event_category or ""])))
    best_category = "unknown"
    best_score = 0
    for category, keywords in THEME_KEYWORDS.items():
        score = len(tokens & keywords)
        if score > best_score:
            best_score = score
            best_category = category
    return best_category


def audit_sqlite(db_path: Path) -> SqliteAudit:
    if not db_path.exists():
        return SqliteAudit(db_path=db_path, exists=False, tables=[], bbo_tables=[])

    conn = sqlite3.connect(str(db_path))
    try:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()

    tables = [str(row[0]) for row in table_rows]
    bbo_tables = [
        table
        for table in tables
        if any(term in table.lower() for term in ("bbo", "quote", "book", "orderbook", "bid", "ask"))
    ]
    return SqliteAudit(db_path=db_path, exists=True, tables=tables, bbo_tables=bbo_tables)


def iter_date_dirs(tick_root: Path, start_date: str | None, end_date: str | None) -> list[Path]:
    lower = start_date or ""
    upper = end_date or "9999-12-31"
    return [
        path
        for path in sorted(candidate for candidate in tick_root.iterdir() if candidate.is_dir())
        if lower <= path.name <= upper
    ]


def _day_start_from_path(path: Path) -> float | None:
    try:
        day = datetime.fromisoformat(path.parent.name).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return day.timestamp()


def iter_market_records(path: Path) -> list[tuple[float, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("local_ts") is None:
                continue
            records.append(record)

    if not records:
        return []

    ts_vals = [float(record.get("local_ts", 0.0)) for record in records]
    ts_span = (max(ts_vals) - min(ts_vals)) if ts_vals else 0.0
    synth_day_start: float | None = None
    day_start = _day_start_from_path(path)
    min_real_ts = 1_577_836_800.0
    median_ts = sorted(ts_vals)[len(ts_vals) // 2] if ts_vals else 0.0
    ts_is_real = median_ts >= min_real_ts
    ts_far_from_date = day_start is not None and abs(median_ts - day_start) > 30 * 86400
    if ts_is_real and ts_far_from_date and day_start is not None:
        synth_day_start = day_start
    elif ts_span < 60.0 and ts_is_real and ts_far_from_date and day_start is not None:
        synth_day_start = day_start

    effective_records: list[tuple[float, dict[str, Any]]] = []
    prev_ts = 0.0
    total = len(records)
    for index, record in enumerate(records):
        timestamp = float(record.get("local_ts", 0.0))
        if synth_day_start is not None:
            timestamp = synth_day_start + index * (86400.0 / max(total, 1))
        if timestamp < prev_ts:
            timestamp = prev_ts
        prev_ts = timestamp
        effective_records.append((timestamp, record))
    return effective_records


def group_market_files(
    tick_root: Path,
    *,
    start_date: str | None,
    end_date: str | None,
    max_markets: int | None,
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for date_dir in iter_date_dirs(tick_root, start_date, end_date):
        for path in sorted(date_dir.glob("*.jsonl")):
            if path.is_file():
                grouped[path.stem].append(path)

    if max_markets is None or len(grouped) <= max_markets:
        return dict(grouped)

    selected_keys = sorted(grouped)[:max_markets]
    return {key: grouped[key] for key in selected_keys}


def mine_market_longshots(
    condition_id: str,
    paths: list[Path],
    *,
    min_ask: float,
    max_ask: float,
    min_duration_s: float,
    max_gap_s: float,
) -> list[QualifiedTokenCandidate]:
    states: dict[str, TokenPathState] = {}

    for path in paths:
        for timestamp, record in iter_market_records(path):
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            payload_type = str(payload.get("event_type") or "").lower()
            if payload_type not in DELTA_EVENT_TYPES and payload_type not in SNAPSHOT_EVENT_TYPES:
                continue

            changes = payload.get("price_changes") or payload.get("changes") or payload.get("data") or []
            if isinstance(changes, dict):
                changes = [changes]

            if changes:
                for change in changes:
                    if not isinstance(change, dict):
                        continue
                    token_id = str(change.get("asset_id") or payload.get("asset_id") or "").strip()
                    ask = _safe_float(change.get("best_ask"))
                    if not token_id or ask is None or ask <= 0.0:
                        continue
                    state = states.get(token_id)
                    if state is None:
                        state = TokenPathState(
                            condition_id=condition_id,
                            token_id=token_id,
                            min_ask=min_ask,
                            max_ask=max_ask,
                            min_duration_s=min_duration_s,
                            max_gap_s=max_gap_s,
                        )
                        states[token_id] = state
                    state.advance_to(timestamp)
                    state.observe(float(ask), timestamp)
                continue

            if payload_type in SNAPSHOT_EVENT_TYPES:
                token_id = str(payload.get("asset_id") or "").strip()
                asks = payload.get("asks") or []
                best_ask: float | None = None
                for level in asks:
                    if not isinstance(level, dict):
                        continue
                    price = _safe_float(level.get("price"))
                    if price is None or price <= 0.0:
                        continue
                    if best_ask is None or price < best_ask:
                        best_ask = price
                if not token_id or best_ask is None:
                    continue
                state = states.get(token_id)
                if state is None:
                    state = TokenPathState(
                        condition_id=condition_id,
                        token_id=token_id,
                        min_ask=min_ask,
                        max_ask=max_ask,
                        min_duration_s=min_duration_s,
                        max_gap_s=max_gap_s,
                    )
                    states[token_id] = state
                state.advance_to(timestamp)
                state.observe(float(best_ask), timestamp)

    candidates: list[QualifiedTokenCandidate] = []
    for state in states.values():
        state.finalize()
        if state.qualifying_entry_ts is None or state.entry_ask is None:
            continue
        candidates.append(
            QualifiedTokenCandidate(
                condition_id=state.condition_id,
                token_id=state.token_id,
                qualifying_entry_ts=state.qualifying_entry_ts,
                entry_ask=state.entry_ask,
                longest_in_band_s=state.longest_in_band_s,
                post_entry_last_ts=state.post_entry_last_ts,
                terminal_ask=state.post_entry_last_ask,
                max_ask_after_entry=state.post_entry_max_ask,
                min_ask_after_entry=state.post_entry_min_ask,
                first_above_10_ts=state.first_above_10_ts,
                first_above_25_ts=state.first_above_25_ts,
                first_above_50_ts=state.first_above_50_ts,
                first_above_95_ts=state.first_above_95_ts,
                quote_observation_count=state.quote_observation_count,
            )
        )
    return candidates


def mine_market_longshots_worker(payload: tuple[str, list[str], float, float, float, float]) -> list[QualifiedTokenCandidate]:
    condition_id, raw_paths, min_ask, max_ask, min_duration_s, max_gap_s = payload
    return mine_market_longshots(
        condition_id,
        [Path(item) for item in raw_paths],
        min_ask=min_ask,
        max_ask=max_ask,
        min_duration_s=min_duration_s,
        max_gap_s=max_gap_s,
    )


def resolve_condition_metadata(
    condition_ids: list[str],
    *,
    timeout_seconds: float,
) -> dict[str, ResolvedConditionMetadata]:
    session = requests.Session()
    resolved: dict[str, ResolvedConditionMetadata] = {}

    for index, condition_id in enumerate(sorted(set(condition_ids)), start=1):
        response = session.get(
            GAMMA_MARKETS_URL,
            params={"condition_ids": condition_id},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            continue
        row = payload[0]
        outcomes = [str(item) for item in _json_list(row.get("outcomes"))]
        token_ids = [str(item) for item in _json_list(row.get("clobTokenIds"))]
        outcome_prices = [_safe_float(item) for item in _json_list(row.get("outcomePrices"))]

        yes_token_id: str | None = None
        no_token_id: str | None = None
        yes_outcome_price: float | None = None
        no_outcome_price: float | None = None
        for idx, outcome in enumerate(outcomes):
            token_id = token_ids[idx] if idx < len(token_ids) else None
            price = outcome_prices[idx] if idx < len(outcome_prices) else None
            normalized = outcome.strip().lower()
            if normalized == "yes":
                yes_token_id = token_id
                yes_outcome_price = price
            elif normalized == "no":
                no_token_id = token_id
                no_outcome_price = price

        event_title = ""
        events = row.get("events")
        if isinstance(events, list) and events:
            first_event = events[0]
            if isinstance(first_event, dict):
                event_title = str(first_event.get("title") or "")

        resolved[condition_id] = ResolvedConditionMetadata(
            condition_id=condition_id,
            question=str(row.get("question") or condition_id),
            slug=str(row.get("slug") or ""),
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            event_title=event_title,
            end_date=str(row.get("endDate") or row.get("endDateIso") or "") or None,
            closed=bool(row.get("closed")),
            active=bool(row.get("active")),
            yes_outcome_price=yes_outcome_price,
            no_outcome_price=no_outcome_price,
            market_category=str(row.get("category") or "") or None,
            event_category=str(first_event.get("category") or "") if isinstance(events, list) and events and isinstance(events[0], dict) else None,
        )

        if index % 100 == 0:
            print(f"Resolved {index} condition ids...", flush=True)

    return resolved


def filter_yes_longshots(
    candidates: list[QualifiedTokenCandidate],
    metadata_by_condition: dict[str, ResolvedConditionMetadata],
) -> list[QualifiedYesLongshot]:
    filtered: list[QualifiedYesLongshot] = []
    for candidate in candidates:
        metadata = metadata_by_condition.get(candidate.condition_id)
        if metadata is None or metadata.yes_token_id is None:
            continue
        if candidate.token_id != metadata.yes_token_id:
            continue
        filtered.append(QualifiedYesLongshot(metadata=metadata, candidate=candidate))
    filtered.sort(key=lambda item: item.candidate.qualifying_entry_ts)
    return filtered


def build_bucket_rows(longshots: list[QualifiedYesLongshot]) -> list[dict[str, Any]]:
    buckets: dict[str, list[QualifiedYesLongshot]] = defaultdict(list)
    for item in longshots:
        buckets[item.bucket_label].append(item)

    rows: list[dict[str, Any]] = []
    for label in ("1-2c", "2-3c", "3-4c", "4-5c"):
        cohort = buckets.get(label, [])
        count = len(cohort)
        losses = sum(1 for item in cohort if item.metadata.resolved_yes)
        capital = count * 0.95
        pnl = sum(item.theoretical_pnl_per_share for item in cohort)
        rows.append(
            {
                "bucket": label,
                "count": count,
                "resolved_yes": losses,
                "theoretical_roc": (pnl / capital) if capital else None,
                "avg_observed_hours": _mean([
                    item.candidate.observed_post_entry_hours
                    for item in cohort
                    if item.candidate.observed_post_entry_hours is not None
                ]),
                "spike_10_rate": (
                    sum(1 for item in cohort if item.candidate.first_above_10_ts is not None) / count
                ) if count else None,
                "spike_25_rate": (
                    sum(1 for item in cohort if item.candidate.first_above_25_ts is not None) / count
                ) if count else None,
            }
        )
    return rows


def split_by_resolution(longshots: list[QualifiedYesLongshot]) -> tuple[list[QualifiedYesLongshot], list[QualifiedYesLongshot]]:
    resolved = [item for item in longshots if item.metadata.closed]
    active = [item for item in longshots if not item.metadata.closed]
    return resolved, active


def build_resolution_summary(longshots: list[QualifiedYesLongshot]) -> dict[str, Any]:
    count = len(longshots)
    gross_pnl = sum(item.theoretical_pnl_per_share for item in longshots)
    net_pnl = sum(item.resolved_net_pnl_per_share for item in longshots)
    capital = count * 0.95
    resolved_yes = sum(1 for item in longshots if item.metadata.resolved_yes)
    resolved_no = sum(1 for item in longshots if item.metadata.resolved_no)
    fee_drag = sum(item.entry_fee_per_share for item in longshots)
    category_counts: dict[str, int] = defaultdict(int)
    for item in longshots:
        category_counts[item.category] += 1
    return {
        "count": count,
        "resolved_yes": resolved_yes,
        "resolved_no": resolved_no,
        "gross_pnl_per_share_total": gross_pnl,
        "net_pnl_per_share_total": net_pnl,
        "gross_roc": (gross_pnl / capital) if capital else None,
        "net_roc": (net_pnl / capital) if capital else None,
        "entry_fee_drag_per_share_total": fee_drag,
        "avg_entry_fee_cents": ((fee_drag / count) * 100.0) if count else None,
        "category_counts": dict(sorted(category_counts.items())),
    }


def build_json_payload(
    *,
    sqlite_audit: SqliteAudit,
    tick_root: Path,
    start_date: str | None,
    end_date: str | None,
    market_group_count: int,
    all_candidates: list[QualifiedTokenCandidate],
    resolved_condition_count: int,
    yes_longshots: list[QualifiedYesLongshot],
) -> dict[str, Any]:
    resolved_bucket, active_bucket = split_by_resolution(yes_longshots)
    spike_markets = [item for item in yes_longshots if item.candidate.first_above_10_ts is not None]
    spike_markets.sort(key=lambda item: item.candidate.max_ask_after_entry or 0.0, reverse=True)
    return {
        "scope": {
            "sqlite_audit_target": str(sqlite_audit.db_path),
            "raw_tick_root": str(tick_root),
            "start_date": start_date,
            "end_date": end_date,
            "market_files_scanned": market_group_count,
            "token_candidates_pre_side_filter": len(all_candidates),
            "resolved_condition_count": resolved_condition_count,
        },
        "sqlite_audit": {
            "exists": sqlite_audit.exists,
            "tables": sqlite_audit.tables,
            "bbo_tables": sqlite_audit.bbo_tables,
        },
        "summary": {
            "qualified_yes_longshots": len(yes_longshots),
            "resolved_bucket": build_resolution_summary(resolved_bucket),
            "active_bucket": build_resolution_summary(active_bucket),
            "spike_above_10c_count": len(spike_markets),
        },
        "spike_markets_above_10c": [
            {
                "condition_id": item.metadata.condition_id,
                "question": item.metadata.question,
                "category": item.category,
                "entry_yes_ask": item.candidate.entry_ask,
                "max_yes_ask": item.candidate.max_ask_after_entry,
                "terminal_yes_ask": item.candidate.terminal_ask,
                "qualified_at": _fmt_ts(item.candidate.qualifying_entry_ts),
                "resolved_state": "YES" if item.metadata.resolved_yes else "NO" if item.metadata.resolved_no else "ACTIVE",
            }
            for item in spike_markets
        ],
        "resolved_markets": [
            {
                "condition_id": item.metadata.condition_id,
                "question": item.metadata.question,
                "category": item.category,
                "entry_yes_ask": item.candidate.entry_ask,
                "entry_no_price": 0.95,
                "entry_fee_per_share": item.entry_fee_per_share,
                "resolved_state": "YES" if item.metadata.resolved_yes else "NO",
                "gross_pnl_per_share": item.theoretical_pnl_per_share,
                "net_pnl_per_share": item.resolved_net_pnl_per_share,
                "gross_roc": item.theoretical_roc,
                "net_roc": item.resolved_net_roc,
            }
            for item in resolved_bucket
        ],
        "active_markets": [
            {
                "condition_id": item.metadata.condition_id,
                "question": item.metadata.question,
                "category": item.category,
                "entry_yes_ask": item.candidate.entry_ask,
                "terminal_yes_ask": item.candidate.terminal_ask,
                "max_yes_ask": item.candidate.max_ask_after_entry,
                "max_no_drawdown_cents": item.max_adverse_no_mark_to_market_cents,
            }
            for item in active_bucket
        ],
    }


def render_report(
    *,
    sqlite_audit: SqliteAudit,
    tick_root: Path,
    start_date: str | None,
    end_date: str | None,
    market_group_count: int,
    all_candidates: list[QualifiedTokenCandidate],
    resolved_condition_count: int,
    yes_longshots: list[QualifiedYesLongshot],
) -> str:
    resolved_bucket, active_bucket = split_by_resolution(yes_longshots)
    resolved_summary = build_resolution_summary(resolved_bucket)
    active_summary = build_resolution_summary(active_bucket)
    qualifying_hours = [item.candidate.longest_in_band_s / 3600.0 for item in yes_longshots]
    observed_hours = [
        item.candidate.observed_post_entry_hours
        for item in yes_longshots
        if item.candidate.observed_post_entry_hours is not None
    ]
    entry_asks = [item.candidate.entry_ask for item in yes_longshots]
    terminal_asks = [
        item.candidate.terminal_ask
        for item in yes_longshots
        if item.candidate.terminal_ask is not None
    ]
    max_asks = [
        item.candidate.max_ask_after_entry
        for item in yes_longshots
        if item.candidate.max_ask_after_entry is not None
    ]
    adverse_drawdowns = [item.max_adverse_no_mark_to_market_cents for item in yes_longshots]

    total_count = len(yes_longshots)
    resolved_yes_count = sum(1 for item in yes_longshots if item.metadata.resolved_yes)
    resolved_no_count = sum(1 for item in yes_longshots if item.metadata.resolved_no)
    assumed_no_count = total_count - resolved_yes_count
    gross_pnl_per_share = sum(item.theoretical_pnl_per_share for item in yes_longshots)
    deployed_capital = total_count * 0.95
    theoretical_roc = (gross_pnl_per_share / deployed_capital) if deployed_capital else None

    spike_10_count = sum(1 for item in yes_longshots if item.candidate.first_above_10_ts is not None)
    spike_25_count = sum(1 for item in yes_longshots if item.candidate.first_above_25_ts is not None)
    spike_50_count = sum(1 for item in yes_longshots if item.candidate.first_above_50_ts is not None)
    decay_to_one_cent_count = sum(
        1 for item in yes_longshots if item.candidate.terminal_ask is not None and item.candidate.terminal_ask <= 0.01
    )

    bucket_rows = build_bucket_rows(yes_longshots)

    worst_spikes = sorted(
        yes_longshots,
        key=lambda item: item.candidate.max_ask_after_entry or item.candidate.entry_ask,
        reverse=True,
    )[:10]
    spike_over_10 = [item for item in worst_spikes if (item.candidate.max_ask_after_entry or 0.0) >= 0.10]
    quiet_decays = sorted(
        [
            item for item in yes_longshots
            if item.candidate.terminal_ask is not None and item.candidate.terminal_ask <= 0.01
        ],
        key=lambda item: (
            item.candidate.max_ask_after_entry or item.candidate.entry_ask,
            item.candidate.terminal_ask or 0.0,
        ),
    )[:10]

    lines: list[str] = [
        "# Favorite-Longshot Bias Yield Analysis",
        "",
        "## Scope",
        "",
        f"- SQLite audit target: `{sqlite_audit.db_path}`",
        f"- Raw tick archive used for BBO reconstruction: `{tick_root}`",
        f"- Replay window: `{start_date or 'earliest available'}` to `{end_date or 'latest available'}`",
        f"- Market files scanned: `{market_group_count}`",
        f"- Token candidates that ever satisfied the sustained 1c-5c filter before side resolution: `{len(all_candidates)}`",
        f"- Condition ids resolved against Gamma for YES/NO token mapping: `{resolved_condition_count}`",
        "",
        "## Data Quality Notes",
        "",
    ]

    if not sqlite_audit.exists:
        lines.append(f"- The requested SQLite DB was not present at `{sqlite_audit.db_path}`.")
    else:
        lines.append(f"- SQLite tables present: `{', '.join(sqlite_audit.tables) if sqlite_audit.tables else 'none'}`")
        if sqlite_audit.bbo_tables:
            lines.append(f"- Quote-like tables detected: `{', '.join(sqlite_audit.bbo_tables)}`")
        else:
            lines.append("- No SQLite table storing historical bid/ask or orderbook snapshots was found; quote history was reconstructed from the raw tick archive instead.")
    lines.extend(
        [
            "- Qualification is strict: a token must stay inside the YES ask band continuously for at least 24h, with observation gaps larger than 6h treated as broken continuity.",
            "- Theoretical PnL follows the PM directive exactly: buy NO at 0.95 once the YES contract has already spent 24h in the 1c-5c band, then assume a +5c payoff unless the market is now explicitly resolved YES.",
            "",
            "## Headline Results",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Qualified YES longshots | {total_count} |",
            f"| Avg entry YES ask | {_fmt_num(_mean(entry_asks), 3)} |",
            f"| Median entry YES ask | {_fmt_num(_median(entry_asks), 3)} |",
            f"| Avg longest sub-5c window | {_fmt_hours(_mean(qualifying_hours))} |",
            f"| Median longest sub-5c window | {_fmt_hours(_median(qualifying_hours))} |",
            f"| Avg observed post-entry life | {_fmt_hours(_mean(observed_hours))} |",
            f"| Median observed post-entry life | {_fmt_hours(_median(observed_hours))} |",
            f"| Avg terminal YES ask | {_fmt_num(_mean(terminal_asks), 3)} |",
            f"| Median terminal YES ask | {_fmt_num(_median(terminal_asks), 3)} |",
            f"| Avg max YES ask after qualification | {_fmt_num(_mean(max_asks), 3)} |",
            f"| Avg max NO mark-to-market drawdown | {_fmt_num(_mean(adverse_drawdowns), 2)} cents |",
            f"| YES ask ever spiked above 10c | {spike_10_count} ({_fmt_pct(spike_10_count / total_count if total_count else None)}) |",
            f"| YES ask ever spiked above 25c | {spike_25_count} ({_fmt_pct(spike_25_count / total_count if total_count else None)}) |",
            f"| YES ask ever spiked above 50c | {spike_50_count} ({_fmt_pct(spike_50_count / total_count if total_count else None)}) |",
            f"| Terminal YES ask <= 1c | {decay_to_one_cent_count} ({_fmt_pct(decay_to_one_cent_count / total_count if total_count else None)}) |",
            f"| Markets currently resolved YES | {resolved_yes_count} |",
            f"| Markets currently resolved NO | {resolved_no_count} |",
            f"| Resolved bucket size | {len(resolved_bucket)} |",
            f"| Active bucket size | {len(active_bucket)} |",
            f"| Assumed NO wins under PM rule | {assumed_no_count} |",
            f"| Theoretical gross ROC at NO 0.95 | {_fmt_pct(theoretical_roc, 2)} |",
            "",
            "## Resolution Filter",
            "",
            "| Bucket | Count | YES Resolutions | NO Resolutions | Gross ROC | Net ROC After Fees | Avg Entry Fee |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| Resolved | {resolved_summary['count']} | {resolved_summary['resolved_yes']} | {resolved_summary['resolved_no']} | {_fmt_pct(resolved_summary['gross_roc'], 2)} | {_fmt_pct(resolved_summary['net_roc'], 2)} | {_fmt_num(resolved_summary['avg_entry_fee_cents'], 3)}c |",
            f"| Active | {active_summary['count']} | {active_summary['resolved_yes']} | {active_summary['resolved_no']} | {_fmt_pct(active_summary['gross_roc'], 2)} | {_fmt_pct(active_summary['net_roc'], 2)} | {_fmt_num(active_summary['avg_entry_fee_cents'], 3)}c |",
            "",
            "## Bucket Breakdown",
            "",
            "| Entry Band | Count | Resolved YES | Theoretical ROC | Avg Observed Life | Spike >10c | Spike >25c |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in bucket_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["bucket"],
                    str(row["count"]),
                    str(row["resolved_yes"]),
                    _fmt_pct(row["theoretical_roc"], 2),
                    _fmt_hours(row["avg_observed_hours"]),
                    _fmt_pct(row["spike_10_rate"]),
                    _fmt_pct(row["spike_25_rate"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )

    if theoretical_roc is None:
        lines.append("- No qualifying YES longshots were found, so the FLB short cannot be evaluated on the available archive.")
    else:
        if resolved_yes_count == 0:
            lines.append(
                f"- On this archive slice, no qualified YES longshot has yet gone on to resolve YES. Under the PM's assumption set, that leaves a mechanical gross ROC of {_fmt_pct(theoretical_roc, 2)} on capital deployed at 0.95 per contract."
            )
        else:
            lines.append(
                f"- The edge remains {'positive' if theoretical_roc > 0 else 'negative'} after counting the {resolved_yes_count} observed YES resolutions. Gross ROC is {_fmt_pct(theoretical_roc, 2)} at the mandated NO 0.95 entry assumption."
            )

        lines.append(
            f"- Ground-truth realized yield on already closed markets is {_fmt_pct(resolved_summary['net_roc'], 2)} net of modeled Polymarket entry fees, across {resolved_summary['count']} resolved contracts. That is the underwriter-grade ROC to trust rather than the full-sample 5.26% gross carry figure."
        )
        lines.append(
            f"- The bigger practical question is path risk, not terminal win rate: {spike_10_count} of {total_count} contracts ({_fmt_pct(spike_10_count / total_count if total_count else None)}) traded above 10c after already spending 24h in the longshot zone, and {spike_25_count} ({_fmt_pct(spike_25_count / total_count if total_count else None)}) traded above 25c."
        )
        lines.append(
            f"- Average observed max adverse mark-to-market for a standardized NO 0.95 entry was {_fmt_num(_mean(adverse_drawdowns), 2)} cents per share, with worst cases listed below. That makes this more of a slow-carry short-vol / structural-bias harvest than a low-latency hedge."
        )
        lines.append(
            f"- {decay_to_one_cent_count} contracts ({_fmt_pct(decay_to_one_cent_count / total_count if total_count else None)}) simply decayed back to <=1c by the end of observation, which is the behavioral pattern the FLB thesis needs."
        )

    lines.extend(
        [
            "",
            "## Worst Post-Qualification Spikes",
            "",
            "| Question | Category | Entry YES Ask | Max YES Ask | Terminal YES Ask | Max NO Drawdown | Resolved State | Qualified At |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in worst_spikes:
        resolved_state = "YES" if item.metadata.resolved_yes else "NO" if item.metadata.resolved_no else "open/assumed NO"
        lines.append(
            "| "
            + " | ".join(
                [
                    item.metadata.question.replace("|", "/"),
                    item.category,
                    _fmt_num(item.candidate.entry_ask, 3),
                    _fmt_num(item.candidate.max_ask_after_entry, 3),
                    _fmt_num(item.candidate.terminal_ask, 3),
                    _fmt_num(item.max_adverse_no_mark_to_market_cents, 2) + "c",
                    resolved_state,
                    _fmt_ts(item.candidate.qualifying_entry_ts),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Narrative Shock Sectors (>10c)",
            "",
            "| Question | Category | Max YES Ask | Resolved State |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for item in spike_over_10:
        resolved_state = "YES" if item.metadata.resolved_yes else "NO" if item.metadata.resolved_no else "ACTIVE"
        lines.append(
            "| "
            + " | ".join(
                [
                    item.metadata.question.replace("|", "/"),
                    item.category,
                    _fmt_num(item.candidate.max_ask_after_entry, 3),
                    resolved_state,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Quiet Decays",
            "",
            "| Question | Entry YES Ask | Max YES Ask | Terminal YES Ask | Longest Sub-5c Window | Resolved State |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in quiet_decays:
        resolved_state = "YES" if item.metadata.resolved_yes else "NO" if item.metadata.resolved_no else "open/assumed NO"
        lines.append(
            "| "
            + " | ".join(
                [
                    item.metadata.question.replace("|", "/"),
                    _fmt_num(item.candidate.entry_ask, 3),
                    _fmt_num(item.candidate.max_ask_after_entry, 3),
                    _fmt_num(item.candidate.terminal_ask, 3),
                    _fmt_hours(item.candidate.longest_in_band_s / 3600.0),
                    resolved_state,
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    sqlite_audit = audit_sqlite(args.db)
    tick_root = resolve_tick_root(args.input_dir)
    market_groups = group_market_files(
        tick_root,
        start_date=args.start_date,
        end_date=args.end_date,
        max_markets=args.max_markets,
    )
    if not market_groups:
        raise FileNotFoundError(f"No raw tick JSONL files found under {tick_root}")

    all_candidates: list[QualifiedTokenCandidate] = []
    min_duration_s = float(args.min_sustain_hours) * 3600.0
    max_gap_s = float(args.max_gap_hours) * 3600.0
    work_items = [
        (
            condition_id,
            [str(path) for path in paths],
            float(args.min_yes_ask),
            float(args.max_yes_ask),
            min_duration_s,
            max_gap_s,
        )
        for condition_id, paths in sorted(market_groups.items())
    ]
    worker_count = max(1, int(args.workers))

    if worker_count == 1:
        for index, work_item in enumerate(work_items, start=1):
            all_candidates.extend(mine_market_longshots_worker(work_item))
            if index % 250 == 0:
                print(f"Processed {index} / {len(market_groups)} market files...", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            for index, result in enumerate(executor.map(mine_market_longshots_worker, work_items, chunksize=16), start=1):
                all_candidates.extend(result)
                if index % 250 == 0:
                    print(f"Processed {index} / {len(market_groups)} market files...", flush=True)

    metadata_by_condition = resolve_condition_metadata(
        [candidate.condition_id for candidate in all_candidates],
        timeout_seconds=float(args.gamma_timeout_seconds),
    )
    yes_longshots = filter_yes_longshots(all_candidates, metadata_by_condition)

    report = render_report(
        sqlite_audit=sqlite_audit,
        tick_root=tick_root,
        start_date=args.start_date,
        end_date=args.end_date,
        market_group_count=len(market_groups),
        all_candidates=all_candidates,
        resolved_condition_count=len(metadata_by_condition),
        yes_longshots=yes_longshots,
    )
    json_payload = build_json_payload(
        sqlite_audit=sqlite_audit,
        tick_root=tick_root,
        start_date=args.start_date,
        end_date=args.end_date,
        market_group_count=len(market_groups),
        all_candidates=all_candidates,
        resolved_condition_count=len(metadata_by_condition),
        yes_longshots=yes_longshots,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    args.json_output.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print(f"Qualified YES longshots: {len(yes_longshots)}")
    print(f"Resolved condition ids: {len(metadata_by_condition)}")
    print(f"Report written to {args.output}")
    print(f"JSON written to {args.json_output}")


if __name__ == "__main__":
    main()