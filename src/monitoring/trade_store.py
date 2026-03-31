"""
Paper-trading statistics engine — persists trade records to SQLite and
computes aggregate metrics (win rate, expectancy, drawdown, etc.).

Also provides **state-persistence** tables for crash recovery: live
``orders`` and ``positions`` are periodically checkpointed so the bot
can reconcile on restart.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, Mapping

import aiosqlite

from src.core.config import settings
from src.core.logger import get_logger
from src.trading.position_manager import Position, PositionState

log = get_logger(__name__)


_SHADOW_PRICE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.1, "0.00-0.10"),
    (0.1, 0.2, "0.10-0.20"),
    (0.2, 0.4, "0.20-0.40"),
    (0.4, 0.6, "0.40-0.60"),
    (0.6, 0.8, "0.60-0.80"),
    (0.8, 0.95, "0.80-0.95"),
    (0.95, 1.01, "0.95-1.00"),
)

_SHADOW_CONFIDENCE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.2, "0.00-0.20"),
    (0.2, 0.4, "0.20-0.40"),
    (0.4, 0.6, "0.40-0.60"),
    (0.6, 0.8, "0.60-0.80"),
    (0.8, 1.01, "0.80-1.00"),
)

_SHADOW_HOLD_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 120.0, "<2m"),
    (120.0, 300.0, "2-5m"),
    (300.0, 900.0, "5-15m"),
    (900.0, 3600.0, "15-60m"),
    (3600.0, float("inf"), ">=60m"),
)

_SHADOW_TOXICITY_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (-1.0, 0.01, "0.00"),
    (0.01, 0.2, "0.01-0.20"),
    (0.2, 0.4, "0.20-0.40"),
    (0.4, 0.6, "0.40-0.60"),
    (0.6, 1.01, ">=0.60"),
)

_TRACKED_EXIT_REASONS: tuple[str, ...] = (
    "target",
    "stop_loss",
    "time_stop",
    "timeout",
    "preemptive_liquidity_drain",
    "vacuum_suppression",
)

_TRACKED_LOSS_BUCKETS: tuple[str, ...] = (
    "timeout",
    "preemptive_liquidity_drain",
    "time_stop",
    "stop_loss",
)

_REMEASUREMENT_SNAPSHOT_DIR = Path("artifacts") / "remeasurement_snapshots"


class TradePersistenceError(RuntimeError):
    """Raised when the store cannot durably persist a closed trade."""


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _normalize_json_safe_payload(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_json_safe_payload(item) for item in value]
    if isinstance(value, set):
        normalized_items = [_normalize_json_safe_payload(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), default=str),
        )
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _merge_shadow_journal_payload(
    payload: Mapping[str, Any],
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged_payload = _normalize_json_safe_payload(payload)
    if extra_payload is not None:
        merged_payload["extra_payload"] = _normalize_json_safe_payload(extra_payload)
    return merged_payload


def _shadow_fee_per_share_cents(row: dict[str, float | int | str]) -> float:
    entry_price = float(row.get("entry_price") or 0.0)
    exit_price = float(row.get("exit_price") or 0.0)
    entry_fee_rate = float(row.get("entry_fee_bps") or 0.0) / 10_000.0
    exit_fee_rate = float(row.get("exit_fee_bps") or 0.0) / 10_000.0
    return (entry_price * entry_fee_rate + exit_price * exit_fee_rate) * 100.0


def _annotate_shadow_row(row: dict[str, float | int | str]) -> dict[str, float | int | str]:
    annotated = dict(row)
    entry_price = float(annotated.get("entry_price") or 0.0)
    exit_price = float(annotated.get("exit_price") or 0.0)
    target_price = float(annotated.get("target_price") or 0.0)
    entry_size = float(annotated.get("entry_size") or 0.0)
    pnl_cents = float(annotated.get("pnl_cents") or 0.0)
    fee_per_share_cents = _shadow_fee_per_share_cents(annotated)
    expected_net_target_per_share_cents = 0.0
    if target_price > entry_price:
        expected_net_target_per_share_cents = (target_price - entry_price) * 100.0 - fee_per_share_cents
    ideal_target_pnl_cents = expected_net_target_per_share_cents * entry_size
    gross_move_cents = (exit_price - entry_price) * 100.0

    annotated["target_edge_cents"] = (target_price - entry_price) * 100.0
    annotated["gross_move_cents"] = gross_move_cents
    annotated["gross_move_cents_total"] = gross_move_cents * entry_size
    annotated["fee_per_share_cents"] = fee_per_share_cents
    annotated["total_fee_cents"] = fee_per_share_cents * entry_size
    annotated["expected_net_target_per_share_cents"] = expected_net_target_per_share_cents
    annotated["ideal_target_pnl_cents"] = ideal_target_pnl_cents
    annotated["realized_vs_ideal_target_cents"] = pnl_cents - ideal_target_pnl_cents
    return annotated


def _empty_exit_mix(total_trades: int = 0) -> dict[str, dict[str, float | int | None]]:
    empty_reason: dict[str, dict[str, float | int | None]] = {}
    for reason in _TRACKED_EXIT_REASONS:
        empty_reason[reason] = {
            "trades": 0,
            "rate": 0.0 if total_trades >= 0 else None,
            "pnl_cents": 0.0,
            "expectancy_cents": 0.0,
            "gross_move_cents_total": 0.0,
            "fee_burden_cents_total": 0.0,
            "realized_vs_ideal_gap_cents_total": 0.0,
            "avg_hold_seconds": 0.0,
        }
    return empty_reason


def _build_exit_mix(rows: list[dict[str, float | int | str]]) -> dict[str, dict[str, float | int | None]]:
    total = len(rows)
    if total == 0:
        return _empty_exit_mix()

    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        reason = str(row.get("exit_reason") or "unknown")
        grouped[reason].append(row)

    exit_mix = _empty_exit_mix(total)
    for reason in sorted(set(grouped) - set(_TRACKED_EXIT_REASONS)):
        exit_mix[reason] = {
            "trades": 0,
            "rate": 0.0,
            "pnl_cents": 0.0,
            "expectancy_cents": 0.0,
            "gross_move_cents_total": 0.0,
            "fee_burden_cents_total": 0.0,
            "realized_vs_ideal_gap_cents_total": 0.0,
            "avg_hold_seconds": 0.0,
        }

    for reason, reason_rows in grouped.items():
        total_pnl = sum(float(row.get("pnl_cents") or 0.0) for row in reason_rows)
        total_gross = sum(float(row.get("gross_move_cents_total") or 0.0) for row in reason_rows)
        total_fee = sum(float(row.get("total_fee_cents") or 0.0) for row in reason_rows)
        total_gap = sum(float(row.get("realized_vs_ideal_target_cents") or 0.0) for row in reason_rows)
        total_hold = sum(float(row.get("hold_seconds") or 0.0) for row in reason_rows)
        exit_mix[reason] = {
            "trades": len(reason_rows),
            "rate": round(len(reason_rows) / total, 4),
            "pnl_cents": round(total_pnl, 2),
            "expectancy_cents": round(total_pnl / len(reason_rows), 2),
            "gross_move_cents_total": round(total_gross, 2),
            "fee_burden_cents_total": round(total_fee, 2),
            "realized_vs_ideal_gap_cents_total": round(total_gap, 2),
            "avg_hold_seconds": round(total_hold / len(reason_rows), 2),
        }
    return exit_mix


def _build_loss_bucket_summary(
    exit_mix: dict[str, dict[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    return {reason: dict(exit_mix.get(reason) or {}) for reason in _TRACKED_LOSS_BUCKETS}


def _sqlite_backup(source_path: Path, snapshot_path: Path) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        snapshot_path.unlink()
    source_conn = sqlite3.connect(str(source_path), timeout=30.0)
    try:
        snapshot_conn = sqlite3.connect(str(snapshot_path), timeout=30.0)
        try:
            source_conn.backup(snapshot_conn)
        finally:
            snapshot_conn.close()
    finally:
        source_conn.close()


def _bucket_value(
    value: float,
    buckets: tuple[tuple[float, float, str], ...],
) -> str:
    for lower, upper, label in buckets:
        if lower <= value < upper:
            return label
    return buckets[-1][2]


def _build_shadow_summary(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    if not rows:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win_cents": 0.0,
            "avg_loss_cents": 0.0,
            "total_pnl_cents": 0.0,
            "max_drawdown_cents": 0.0,
            "target_exits": 0,
            "stop_exits": 0,
            "avg_hold_seconds": 0.0,
            "expectancy_cents": 0.0,
            "avg_entry_price": 0.0,
            "avg_confidence": 0.0,
            "avg_reference_price": 0.0,
            "avg_toxicity_index": 0.0,
            "positive_trade_count": 0,
            "avg_target_edge_cents": 0.0,
            "avg_fee_per_share_cents": 0.0,
            "avg_total_fee_cents": 0.0,
            "fee_burden_cents_total": 0.0,
            "avg_gross_move_cents": 0.0,
            "gross_move_cents_total": 0.0,
            "avg_expected_net_target_per_share_cents": 0.0,
            "avg_realized_vs_ideal_target_cents": 0.0,
            "realized_vs_ideal_gap_cents_total": 0.0,
            "timeout_exits": 0,
            "time_stop_exits": 0,
            "preemptive_liquidity_drain_exits": 0,
        }

    pnls = [float(row.get("pnl_cents") or 0.0) for row in rows]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl <= 0]
    holds = [float(row.get("hold_seconds") or 0.0) for row in rows if row.get("hold_seconds") is not None]
    total_pnl = sum(pnls)
    reference_prices = [
        float(row.get("reference_price") or 0.0)
        for row in rows
        if float(row.get("reference_price") or 0.0) > 0.0
    ]

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)

    total = len(rows)
    return {
        "total_trades": total,
        "win_rate": round(len(wins) / total, 4) if total else 0.0,
        "avg_win_cents": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss_cents": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "total_pnl_cents": round(total_pnl, 2),
        "max_drawdown_cents": round(max_drawdown, 2),
        "target_exits": sum(1 for row in rows if row.get("exit_reason") == "target"),
        "stop_exits": sum(1 for row in rows if row.get("exit_reason") == "stop_loss"),
        "avg_hold_seconds": round(sum(holds) / len(holds), 1) if holds else 0.0,
        "expectancy_cents": round(total_pnl / total, 2) if total else 0.0,
        "avg_entry_price": round(
            sum(float(row.get("entry_price") or 0.0) for row in rows) / total,
            4,
        ) if total else 0.0,
        "avg_confidence": round(
            sum(float(row.get("confidence") or 0.0) for row in rows) / total,
            4,
        ) if total else 0.0,
        "avg_reference_price": round(
            sum(reference_prices) / len(reference_prices),
            4,
        ) if reference_prices else 0.0,
        "avg_toxicity_index": round(
            sum(float(row.get("toxicity_index") or 0.0) for row in rows) / total,
            4,
        ) if total else 0.0,
        "positive_trade_count": sum(1 for pnl in pnls if pnl > 0),
        "avg_target_edge_cents": round(
            sum(float(row.get("target_edge_cents") or 0.0) for row in rows) / total,
            3,
        ) if total else 0.0,
        "avg_fee_per_share_cents": round(
            sum(float(row.get("fee_per_share_cents") or 0.0) for row in rows) / total,
            3,
        ) if total else 0.0,
        "avg_total_fee_cents": round(
            sum(float(row.get("total_fee_cents") or 0.0) for row in rows) / total,
            3,
        ) if total else 0.0,
        "fee_burden_cents_total": round(
            sum(float(row.get("total_fee_cents") or 0.0) for row in rows),
            2,
        ),
        "avg_gross_move_cents": round(
            sum(float(row.get("gross_move_cents") or 0.0) for row in rows) / total,
            3,
        ) if total else 0.0,
        "gross_move_cents_total": round(
            sum(float(row.get("gross_move_cents_total") or 0.0) for row in rows),
            2,
        ),
        "avg_expected_net_target_per_share_cents": round(
            sum(float(row.get("expected_net_target_per_share_cents") or 0.0) for row in rows) / total,
            3,
        ) if total else 0.0,
        "avg_realized_vs_ideal_target_cents": round(
            sum(float(row.get("realized_vs_ideal_target_cents") or 0.0) for row in rows) / total,
            3,
        ) if total else 0.0,
        "realized_vs_ideal_gap_cents_total": round(
            sum(float(row.get("realized_vs_ideal_target_cents") or 0.0) for row in rows),
            2,
        ),
        "timeout_exits": sum(1 for row in rows if row.get("exit_reason") == "timeout"),
        "time_stop_exits": sum(1 for row in rows if row.get("exit_reason") == "time_stop"),
        "preemptive_liquidity_drain_exits": sum(
            1 for row in rows if row.get("exit_reason") == "preemptive_liquidity_drain"
        ),
    }


def _group_shadow_rows(
    rows: list[dict[str, float | int | str]],
    *,
    key_fn,
) -> dict[str, dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)

    ranked_items = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            -sum(float(entry.get("pnl_cents") or 0.0) for entry in item[1]),
            item[0],
        ),
    )
    return {key: _build_shadow_summary(entries) for key, entries in ranked_items}


def _top_shadow_markets(
    rows: list[dict[str, float | int | str]],
    *,
    limit: int = 5,
) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("market_id") or "")].append(row)

    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            -sum(float(entry.get("pnl_cents") or 0.0) for entry in item[1]),
            item[0],
        ),
    )

    result: list[dict[str, float | int | str]] = []
    for market_id, entries in ranked[:limit]:
        summary = _build_shadow_summary(entries)
        result.append(
            {
                "market_id": market_id,
                "total_trades": int(summary["total_trades"]),
                "total_pnl_cents": float(summary["total_pnl_cents"]),
                "expectancy_cents": float(summary["expectancy_cents"]),
                "win_rate": float(summary["win_rate"]),
                "avg_entry_price": float(summary["avg_entry_price"]),
            }
        )
    return result


def _default_db_path() -> Path:
    """Derive the default DB path from the configured log directory."""
    return Path(settings.log_dir) / "trades.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              TEXT PRIMARY KEY,
    market_id       TEXT NOT NULL,
    state           TEXT NOT NULL,
    entry_price     REAL,
    entry_size      REAL,
    entry_time      REAL,
    target_price    REAL,
    exit_price      REAL,
    exit_time       REAL,
    exit_reason     TEXT,
    pnl_cents       REAL,
    hold_seconds    REAL,
    alpha           REAL,
    zscore          REAL,
    volume_ratio    REAL,
    whale           INTEGER,
    entry_fee_bps   INTEGER DEFAULT 0,
    exit_fee_bps    INTEGER DEFAULT 0,
    entry_toxicity_index REAL DEFAULT 0.0,
    exit_toxicity_index REAL DEFAULT 0.0,
    drawn_tp        REAL DEFAULT 0.0,
    drawn_stop      REAL DEFAULT 0.0,
    drawn_time      REAL DEFAULT 0.0,
    expected_net_target_per_share_cents REAL DEFAULT 0.0,
    expected_net_target_minus_one_tick_per_share_cents REAL DEFAULT 0.0,
    time_stop_delay_seconds REAL DEFAULT 0.0,
    time_stop_suppression_count INTEGER DEFAULT 0,
    exit_price_minus_drawn_stop_cents REAL DEFAULT 0.0,
    created_at      REAL,
    is_probe        INTEGER DEFAULT 0,
    signal_type     TEXT DEFAULT '',
    meta_weight     REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_trades_state ON trades(state);

-- Shadow Performance Tracker: counterfactual trades from shadow strategies
CREATE TABLE IF NOT EXISTS shadow_trades (
    id              TEXT PRIMARY KEY,
    signal_source   TEXT NOT NULL,
    market_id       TEXT NOT NULL,
    asset_id        TEXT DEFAULT '',
    state           TEXT NOT NULL,
    direction       TEXT DEFAULT 'NO',
    reference_price REAL DEFAULT 0.0,
    reference_price_band TEXT DEFAULT '',
    entry_price     REAL,
    entry_size      REAL,
    entry_time      REAL,
    target_price    REAL,
    stop_price      REAL,
    exit_price      REAL,
    exit_time       REAL,
    exit_reason     TEXT,
    pnl_cents       REAL,
    hold_seconds    REAL,
    entry_fee_bps   INTEGER DEFAULT 0,
    exit_fee_bps    INTEGER DEFAULT 0,
    zscore          REAL,
    confidence      REAL,
    toxicity_index  REAL DEFAULT 0.0,
    created_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_shadow_source ON shadow_trades(signal_source);
CREATE INDEX IF NOT EXISTS idx_shadow_state ON shadow_trades(state);

-- Persistence contract: journal every closed-trade intent before ledger write
CREATE TABLE IF NOT EXISTS trade_persistence_journal (
    journal_key      TEXT PRIMARY KEY,
    ledger_kind      TEXT NOT NULL,
    trade_id         TEXT NOT NULL,
    signal_source    TEXT DEFAULT '',
    market_id        TEXT NOT NULL,
    state            TEXT NOT NULL,
    entry_time       REAL,
    exit_time        REAL,
    exit_reason      TEXT DEFAULT '',
    payload_json     TEXT NOT NULL,
    journaled_at     REAL NOT NULL,
    ledger_recorded_at REAL DEFAULT 0.0,
    ledger_state     TEXT NOT NULL DEFAULT 'PENDING',
    last_error       TEXT DEFAULT '',
    last_error_at    REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_trade_persistence_journal_kind_source
    ON trade_persistence_journal(ledger_kind, signal_source);
CREATE INDEX IF NOT EXISTS idx_trade_persistence_journal_kind_state
    ON trade_persistence_journal(ledger_kind, ledger_state);

-- State-persistence: live orders snapshot
CREATE TABLE IF NOT EXISTS live_orders (
    order_id        TEXT PRIMARY KEY,
    market_id       TEXT NOT NULL,
    asset_id        TEXT NOT NULL,
    side            TEXT NOT NULL,
    price           REAL NOT NULL,
    size            REAL NOT NULL,
    status          TEXT NOT NULL,
    filled_size     REAL DEFAULT 0.0,
    filled_avg_price REAL DEFAULT 0.0,
    clob_order_id   TEXT DEFAULT '',
    post_only       INTEGER DEFAULT 0,
    created_at      REAL,
    updated_at      REAL
);

-- State-persistence: live positions snapshot
CREATE TABLE IF NOT EXISTS live_positions (
    id              TEXT PRIMARY KEY,
    market_id       TEXT NOT NULL,
    no_asset_id     TEXT NOT NULL,
    event_id        TEXT DEFAULT '',
    state           TEXT NOT NULL,
    entry_price     REAL,
    entry_size      REAL,
    entry_time      REAL,
    target_price    REAL,
    exit_reason     TEXT DEFAULT '',
    drawn_tp        REAL DEFAULT 0.0,
    drawn_stop      REAL DEFAULT 0.0,
    drawn_time      REAL DEFAULT 0.0,
    fee_enabled     INTEGER DEFAULT 1,
    sl_trigger_cents REAL DEFAULT 0.0,
    entry_fee_bps   INTEGER DEFAULT 0,
    exit_fee_bps    INTEGER DEFAULT 0,
    entry_order_id  TEXT DEFAULT '',
    exit_order_id   TEXT DEFAULT '',
    created_at      REAL
);
"""


class TradeStore:
    """Async SQLite-backed trade log with aggregate statistics."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

        # Cumulative counter of SQLite lock-contention retries
        # (application-level, supplements PRAGMA busy_timeout).
        # NOTE: currently only incremented by checkpoint methods;
        # PRAGMA busy_timeout handles most contention transparently.
        self.db_lock_retries: int = 0

    async def init(self) -> None:
        if self._db is not None:
            return  # already initialised — avoid leaking connections
        self._db = await aiosqlite.connect(str(self.db_path))

        # ── Concurrency hardening for async multi-task environment ────────
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")
        await self._db.execute("PRAGMA busy_timeout=5000;")         # retry internally for up to 5s on lock contention
        await self._db.execute("PRAGMA wal_autocheckpoint=1000;")   # checkpoint every 1000 pages
        await self._db.execute("PRAGMA journal_size_limit=67108864;")  # cap WAL at 64 MB

        await self._db.executescript(_SCHEMA)
        await self._db.commit()

        # ── Schema migration: add attribution columns if missing ─────────
        try:
            cursor = await self._db.execute("PRAGMA table_info(trades)")
            cols = {row[1] for row in await cursor.fetchall()}
            if "signal_type" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN signal_type TEXT DEFAULT ''"
                )
            if "meta_weight" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN meta_weight REAL DEFAULT 1.0"
                )
            if "entry_toxicity_index" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN entry_toxicity_index REAL DEFAULT 0.0"
                )
            if "exit_toxicity_index" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN exit_toxicity_index REAL DEFAULT 0.0"
                )
            if "drawn_tp" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN drawn_tp REAL DEFAULT 0.0"
                )
            if "drawn_stop" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN drawn_stop REAL DEFAULT 0.0"
                )
            if "drawn_time" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN drawn_time REAL DEFAULT 0.0"
                )
            if "expected_net_target_per_share_cents" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN expected_net_target_per_share_cents REAL DEFAULT 0.0"
                )
            if "expected_net_target_minus_one_tick_per_share_cents" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN expected_net_target_minus_one_tick_per_share_cents REAL DEFAULT 0.0"
                )
            if "time_stop_delay_seconds" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN time_stop_delay_seconds REAL DEFAULT 0.0"
                )
            if "time_stop_suppression_count" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN time_stop_suppression_count INTEGER DEFAULT 0"
                )
            if "exit_price_minus_drawn_stop_cents" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN exit_price_minus_drawn_stop_cents REAL DEFAULT 0.0"
                )
            if "entry_fee_bps" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN entry_fee_bps INTEGER DEFAULT 0"
                )
            if "exit_fee_bps" not in cols:
                await self._db.execute(
                    "ALTER TABLE trades ADD COLUMN exit_fee_bps INTEGER DEFAULT 0"
                )
            await self._db.commit()
        except Exception:
            log.warning("schema_migration_skipped", exc_info=True)

        try:
            cursor = await self._db.execute("PRAGMA table_info(live_positions)")
            cols = {row[1] for row in await cursor.fetchall()}
            if "drawn_tp" not in cols:
                await self._db.execute(
                    "ALTER TABLE live_positions ADD COLUMN drawn_tp REAL DEFAULT 0.0"
                )
            if "drawn_stop" not in cols:
                await self._db.execute(
                    "ALTER TABLE live_positions ADD COLUMN drawn_stop REAL DEFAULT 0.0"
                )
            if "drawn_time" not in cols:
                await self._db.execute(
                    "ALTER TABLE live_positions ADD COLUMN drawn_time REAL DEFAULT 0.0"
                )
            await self._db.commit()
        except Exception:
            log.warning("live_positions_migration_skipped", exc_info=True)

        # ── Schema migration: add shadow_trades if missing ────────────
        try:
            cursor = await self._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='shadow_trades'"
            )
            if not await cursor.fetchone():
                await self._db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS shadow_trades (
                        id              TEXT PRIMARY KEY,
                        signal_source   TEXT NOT NULL,
                        market_id       TEXT NOT NULL,
                        asset_id        TEXT DEFAULT '',
                        state           TEXT NOT NULL,
                        direction       TEXT DEFAULT 'NO',
                        reference_price REAL DEFAULT 0.0,
                        reference_price_band TEXT DEFAULT '',
                        entry_price     REAL,
                        entry_size      REAL,
                        entry_time      REAL,
                        target_price    REAL,
                        stop_price      REAL,
                        exit_price      REAL,
                        exit_time       REAL,
                        exit_reason     TEXT,
                        pnl_cents       REAL,
                        hold_seconds    REAL,
                        entry_fee_bps   INTEGER DEFAULT 0,
                        exit_fee_bps    INTEGER DEFAULT 0,
                        zscore          REAL,
                        confidence      REAL,
                        toxicity_index  REAL DEFAULT 0.0,
                        created_at      REAL
                    );
                    CREATE INDEX IF NOT EXISTS idx_shadow_source ON shadow_trades(signal_source);
                    CREATE INDEX IF NOT EXISTS idx_shadow_state ON shadow_trades(state);
                    """
                )
            cursor = await self._db.execute("PRAGMA table_info(shadow_trades)")
            cols = {row[1] for row in await cursor.fetchall()}
            if "asset_id" not in cols:
                await self._db.execute(
                    "ALTER TABLE shadow_trades ADD COLUMN asset_id TEXT DEFAULT ''"
                )
            if "reference_price" not in cols:
                await self._db.execute(
                    "ALTER TABLE shadow_trades ADD COLUMN reference_price REAL DEFAULT 0.0"
                )
            if "reference_price_band" not in cols:
                await self._db.execute(
                    "ALTER TABLE shadow_trades ADD COLUMN reference_price_band TEXT DEFAULT ''"
                )
            if "toxicity_index" not in cols:
                await self._db.execute(
                    "ALTER TABLE shadow_trades ADD COLUMN toxicity_index REAL DEFAULT 0.0"
                )
            await self._db.commit()
        except Exception:
            log.warning("shadow_trades_migration_skipped", exc_info=True)

        try:
            cursor = await self._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='trade_persistence_journal'"
            )
            if not await cursor.fetchone():
                await self._db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS trade_persistence_journal (
                        journal_key      TEXT PRIMARY KEY,
                        ledger_kind      TEXT NOT NULL,
                        trade_id         TEXT NOT NULL,
                        signal_source    TEXT DEFAULT '',
                        market_id        TEXT NOT NULL,
                        state            TEXT NOT NULL,
                        entry_time       REAL,
                        exit_time        REAL,
                        exit_reason      TEXT DEFAULT '',
                        payload_json     TEXT NOT NULL,
                        journaled_at     REAL NOT NULL,
                        ledger_recorded_at REAL DEFAULT 0.0,
                        ledger_state     TEXT NOT NULL DEFAULT 'PENDING',
                        last_error       TEXT DEFAULT '',
                        last_error_at    REAL DEFAULT 0.0
                    );
                    CREATE INDEX IF NOT EXISTS idx_trade_persistence_journal_kind_source
                        ON trade_persistence_journal(ledger_kind, signal_source);
                    CREATE INDEX IF NOT EXISTS idx_trade_persistence_journal_kind_state
                        ON trade_persistence_journal(ledger_kind, ledger_state);
                    """
                )
            cursor = await self._db.execute("PRAGMA table_info(trade_persistence_journal)")
            cols = {row[1] for row in await cursor.fetchall()}
            if "signal_source" not in cols:
                await self._db.execute(
                    "ALTER TABLE trade_persistence_journal ADD COLUMN signal_source TEXT DEFAULT ''"
                )
            if "entry_time" not in cols:
                await self._db.execute(
                    "ALTER TABLE trade_persistence_journal ADD COLUMN entry_time REAL"
                )
            if "exit_time" not in cols:
                await self._db.execute(
                    "ALTER TABLE trade_persistence_journal ADD COLUMN exit_time REAL"
                )
            if "exit_reason" not in cols:
                await self._db.execute(
                    "ALTER TABLE trade_persistence_journal ADD COLUMN exit_reason TEXT DEFAULT ''"
                )
            if "payload_json" not in cols:
                await self._db.execute(
                    "ALTER TABLE trade_persistence_journal ADD COLUMN payload_json TEXT DEFAULT '{}'"
                )
            if "journaled_at" not in cols:
                await self._db.execute(
                    "ALTER TABLE trade_persistence_journal ADD COLUMN journaled_at REAL DEFAULT 0.0"
                )
            if "ledger_recorded_at" not in cols:
                await self._db.execute(
                    "ALTER TABLE trade_persistence_journal ADD COLUMN ledger_recorded_at REAL DEFAULT 0.0"
                )
            if "ledger_state" not in cols:
                await self._db.execute(
                    "ALTER TABLE trade_persistence_journal ADD COLUMN ledger_state TEXT DEFAULT 'PENDING'"
                )
            if "last_error" not in cols:
                await self._db.execute(
                    "ALTER TABLE trade_persistence_journal ADD COLUMN last_error TEXT DEFAULT ''"
                )
            if "last_error_at" not in cols:
                await self._db.execute(
                    "ALTER TABLE trade_persistence_journal ADD COLUMN last_error_at REAL DEFAULT 0.0"
                )
            await self._db.commit()
        except Exception:
            log.warning("trade_persistence_journal_migration_skipped", exc_info=True)

        log.info("trade_store_initialised", path=str(self.db_path))

    async def _ensure_db(self) -> None:
        """Ensure the database connection is initialised."""
        if self._db is None:
            await self.init()

    async def close(self) -> None:
        if self._db:
            try:
                await self._db.close()
            except Exception:
                pass
            self._db = None

    def _journal_key(self, ledger_kind: str, trade_id: str) -> str:
        return f"{ledger_kind}:{trade_id}"

    async def _journal_trade_intent(self, journal_entry: dict[str, Any]) -> None:
        await self._db.execute(
            """
            INSERT INTO trade_persistence_journal
            (journal_key, ledger_kind, trade_id, signal_source, market_id, state,
             entry_time, exit_time, exit_reason, payload_json, journaled_at,
             ledger_recorded_at, ledger_state, last_error, last_error_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(journal_key) DO UPDATE SET
                ledger_kind = excluded.ledger_kind,
                trade_id = excluded.trade_id,
                signal_source = excluded.signal_source,
                market_id = excluded.market_id,
                state = excluded.state,
                entry_time = excluded.entry_time,
                exit_time = excluded.exit_time,
                exit_reason = excluded.exit_reason,
                payload_json = excluded.payload_json,
                journaled_at = excluded.journaled_at,
                ledger_state = CASE
                    WHEN trade_persistence_journal.ledger_recorded_at > 0 THEN 'RECORDED'
                    ELSE 'PENDING'
                END,
                last_error = '',
                last_error_at = 0.0
            """,
            (
                journal_entry["journal_key"],
                journal_entry["ledger_kind"],
                journal_entry["trade_id"],
                journal_entry["signal_source"],
                journal_entry["market_id"],
                journal_entry["state"],
                journal_entry["entry_time"],
                journal_entry["exit_time"],
                journal_entry["exit_reason"],
                journal_entry["payload_json"],
                journal_entry["journaled_at"],
                float(journal_entry.get("ledger_recorded_at") or 0.0),
                "PENDING",
                "",
                0.0,
            ),
        )
        await self._db.commit()

    async def _mark_trade_persistence_recorded(self, journal_key: str, *, recorded_at: float) -> None:
        cursor = await self._db.execute(
            """
            UPDATE trade_persistence_journal
            SET ledger_recorded_at = ?,
                ledger_state = 'RECORDED',
                last_error = '',
                last_error_at = 0.0
            WHERE journal_key = ?
            """,
            (recorded_at, journal_key),
        )
        if cursor.rowcount != 1:
            raise TradePersistenceError(f"Missing persistence journal entry for {journal_key}")

    async def _mark_trade_persistence_failed(
        self,
        journal_entry: dict[str, Any],
        *,
        error: str,
    ) -> None:
        try:
            await self._db.execute(
                """
                INSERT INTO trade_persistence_journal
                (journal_key, ledger_kind, trade_id, signal_source, market_id, state,
                 entry_time, exit_time, exit_reason, payload_json, journaled_at,
                 ledger_recorded_at, ledger_state, last_error, last_error_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(journal_key) DO UPDATE SET
                    ledger_kind = excluded.ledger_kind,
                    trade_id = excluded.trade_id,
                    signal_source = excluded.signal_source,
                    market_id = excluded.market_id,
                    state = excluded.state,
                    entry_time = excluded.entry_time,
                    exit_time = excluded.exit_time,
                    exit_reason = excluded.exit_reason,
                    payload_json = excluded.payload_json,
                    journaled_at = excluded.journaled_at,
                    ledger_state = 'FAILED',
                    last_error = excluded.last_error,
                    last_error_at = excluded.last_error_at
                """,
                (
                    journal_entry["journal_key"],
                    journal_entry["ledger_kind"],
                    journal_entry["trade_id"],
                    journal_entry["signal_source"],
                    journal_entry["market_id"],
                    journal_entry["state"],
                    journal_entry["entry_time"],
                    journal_entry["exit_time"],
                    journal_entry["exit_reason"],
                    journal_entry["payload_json"],
                    journal_entry["journaled_at"],
                    float(journal_entry.get("ledger_recorded_at") or 0.0),
                    "FAILED",
                    error[:1000],
                    time.time(),
                ),
            )
            await self._db.commit()
        except Exception:
            log.critical(
                "trade_persistence_failure_not_journalled",
                journal_key=journal_entry.get("journal_key"),
                market_id=journal_entry.get("market_id"),
                signal_source=journal_entry.get("signal_source"),
                db_path=str(self.db_path),
                exc_info=True,
            )

    # ── Record a closed position ────────────────────────────────────────────
    async def record(self, pos: Position) -> None:
        """Upsert a position record (with retry on transient lock)."""
        await self._ensure_db()

        hold = (pos.exit_time - pos.entry_time) if pos.exit_time else 0.0
        signal = pos.signal
        signal_zscore = pos.signal_zscore
        if signal_zscore is None and signal is not None:
            signal_zscore = getattr(signal, "zscore", None)
        signal_volume_ratio = pos.signal_volume_ratio
        if signal_volume_ratio is None and signal is not None:
            signal_volume_ratio = getattr(signal, "volume_ratio", None)
        signal_whale = int(pos.signal_whale_confluence)
        if signal_whale == 0 and signal is not None:
            signal_whale = int(getattr(signal, "whale_confluence", False))
        trade_state = pos.state.value if hasattr(pos.state, "value") else str(pos.state)
        signal_type = str(getattr(pos, "signal_type", "") or "")
        record_tuple = (
            pos.id,
            pos.market_id,
            trade_state,
            pos.entry_price,
            pos.entry_size,
            pos.entry_time,
            pos.target_price,
            pos.exit_price,
            pos.exit_time,
            pos.exit_reason,
            pos.pnl_cents,
            round(hold, 1),
            pos.tp_result.alpha if pos.tp_result else None,
            signal_zscore,
            signal_volume_ratio,
            signal_whale,
            int(getattr(pos, "entry_fee_bps", 0) or 0),
            int(getattr(pos, "exit_fee_bps", 0) or 0),
            float(getattr(pos, "entry_toxicity_index", 0.0) or 0.0),
            float(getattr(pos, "exit_toxicity_index", 0.0) or 0.0),
            float(getattr(pos, "drawn_tp", 0.0) or 0.0),
            float(getattr(pos, "drawn_stop", 0.0) or 0.0),
            float(getattr(pos, "drawn_time", 0.0) or 0.0),
            float(getattr(pos, "expected_net_target_per_share_cents", 0.0) or 0.0),
            float(getattr(pos, "expected_net_target_minus_one_tick_per_share_cents", 0.0) or 0.0),
            float(getattr(pos, "time_stop_delay_seconds", 0.0) or 0.0),
            int(getattr(pos, "time_stop_suppression_count", 0) or 0),
            float(getattr(pos, "exit_price_minus_drawn_stop_cents", 0.0) or 0.0),
            pos.created_at,
            int(getattr(pos, "is_probe", False)),
            signal_type,
            getattr(pos, "meta_weight", 1.0),
        )
        journal_entry = {
            "journal_key": self._journal_key("trades", pos.id),
            "ledger_kind": "trades",
            "trade_id": pos.id,
            "signal_source": signal_type,
            "market_id": pos.market_id,
            "state": trade_state,
            "entry_time": pos.entry_time,
            "exit_time": pos.exit_time,
            "exit_reason": str(pos.exit_reason or ""),
            "journaled_at": time.time(),
            "payload_json": _json_dumps(
                {
                    "id": pos.id,
                    "market_id": pos.market_id,
                    "state": trade_state,
                    "entry_price": pos.entry_price,
                    "entry_size": pos.entry_size,
                    "entry_time": pos.entry_time,
                    "target_price": pos.target_price,
                    "exit_price": pos.exit_price,
                    "exit_time": pos.exit_time,
                    "exit_reason": pos.exit_reason,
                    "pnl_cents": pos.pnl_cents,
                    "hold_seconds": round(hold, 1),
                    "alpha": pos.tp_result.alpha if pos.tp_result else None,
                    "zscore": signal_zscore,
                    "volume_ratio": signal_volume_ratio,
                    "whale": signal_whale,
                    "entry_fee_bps": int(getattr(pos, "entry_fee_bps", 0) or 0),
                    "exit_fee_bps": int(getattr(pos, "exit_fee_bps", 0) or 0),
                    "entry_toxicity_index": float(getattr(pos, "entry_toxicity_index", 0.0) or 0.0),
                    "exit_toxicity_index": float(getattr(pos, "exit_toxicity_index", 0.0) or 0.0),
                    "drawn_tp": float(getattr(pos, "drawn_tp", 0.0) or 0.0),
                    "drawn_stop": float(getattr(pos, "drawn_stop", 0.0) or 0.0),
                    "drawn_time": float(getattr(pos, "drawn_time", 0.0) or 0.0),
                    "expected_net_target_per_share_cents": float(
                        getattr(pos, "expected_net_target_per_share_cents", 0.0) or 0.0
                    ),
                    "expected_net_target_minus_one_tick_per_share_cents": float(
                        getattr(pos, "expected_net_target_minus_one_tick_per_share_cents", 0.0) or 0.0
                    ),
                    "time_stop_delay_seconds": float(getattr(pos, "time_stop_delay_seconds", 0.0) or 0.0),
                    "time_stop_suppression_count": int(getattr(pos, "time_stop_suppression_count", 0) or 0),
                    "exit_price_minus_drawn_stop_cents": float(
                        getattr(pos, "exit_price_minus_drawn_stop_cents", 0.0) or 0.0
                    ),
                    "created_at": pos.created_at,
                    "is_probe": int(getattr(pos, "is_probe", False)),
                    "signal_type": signal_type,
                    "meta_weight": getattr(pos, "meta_weight", 1.0),
                }
            ),
        }

        try:
            await self._journal_trade_intent(journal_entry)
            await self._db.execute("BEGIN IMMEDIATE")
            await self._db.execute(
                """
                INSERT OR REPLACE INTO trades
                (id, market_id, state, entry_price, entry_size, entry_time,
                 target_price, exit_price, exit_time, exit_reason, pnl_cents,
                 hold_seconds, alpha, zscore, volume_ratio, whale,
                 entry_fee_bps, exit_fee_bps, entry_toxicity_index, exit_toxicity_index,
                 drawn_tp, drawn_stop, drawn_time,
                 expected_net_target_per_share_cents, expected_net_target_minus_one_tick_per_share_cents,
                 time_stop_delay_seconds, time_stop_suppression_count, exit_price_minus_drawn_stop_cents,
                 created_at,
                 is_probe, signal_type, meta_weight)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                record_tuple,
            )
            await self._mark_trade_persistence_recorded(
                journal_entry["journal_key"],
                recorded_at=time.time(),
            )
            await self._db.commit()
        except Exception as exc:
            try:
                await self._db.rollback()
            except Exception:
                pass
            await self._mark_trade_persistence_failed(journal_entry, error=str(exc))
            log.critical(
                "trade_persistence_contract_failed",
                ledger_kind="trades",
                trade_id=pos.id,
                market_id=pos.market_id,
                signal_source=signal_type or "unknown",
                db_path=str(self.db_path),
                error=str(exc),
                exc_info=True,
            )
            raise TradePersistenceError(
                f"Failed to persist closed trade {pos.id} to {self.db_path}"
            ) from exc

    # ── Aggregate stats ─────────────────────────────────────────────────────
    async def get_stats(self, signal_type: str | None = None) -> dict:
        """Compute aggregate trading statistics.

        When ``signal_type`` is provided, only closed trades attributed to
        that strategy are included in the aggregate.
        """
        await self._ensure_db()

        if signal_type:
            cursor = await self._db.execute(
                "SELECT pnl_cents, exit_reason, hold_seconds FROM trades "
                "WHERE state = ? AND signal_type = ? ORDER BY exit_time ASC",
                (PositionState.CLOSED.value, signal_type),
            )
        else:
            cursor = await self._db.execute(
                "SELECT pnl_cents, exit_reason, hold_seconds FROM trades "
                "WHERE state = ? ORDER BY exit_time ASC",
                (PositionState.CLOSED.value,),
            )
        rows = await cursor.fetchall()

        if not rows:
            return {"total_trades": 0}

        pnls = [r[0] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        holds = [r[2] for r in rows if r[2]]

        target_exits = sum(1 for r in rows if r[1] == "target")
        timeout_exits = sum(1 for r in rows if r[1] == "timeout")

        total = len(pnls)
        win_rate = len(wins) / total if total else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        total_pnl = sum(pnls)

        # Max drawdown (cumulative)
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            dd = peak - cum
            max_dd = max(max_dd, dd)

        return {
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "avg_win_cents": round(avg_win, 2),
            "avg_loss_cents": round(avg_loss, 2),
            "total_pnl_cents": round(total_pnl, 2),
            "max_drawdown_cents": round(max_dd, 2),
            "target_exits": target_exits,
            "timeout_exits": timeout_exits,
            "avg_hold_seconds": round(sum(holds) / len(holds), 1) if holds else 0,
            "expectancy_cents": round(total_pnl / total, 2) if total else 0,
            "decayed_win_rate": round(self._compute_decayed_wr(pnls), 4),
        }

    async def get_ofi_toxicity_pnl_summary(self, buckets: int = 10) -> list[dict[str, float | int | str]]:
        """Aggregate OFI momentum trade quality by entry-toxicity bucket.

        Returns one row per fixed-width toxicity bucket across ``[0, 1]``.
        Each row includes trade count, win rate, average net PnL, and
        total taker-fee drag in cents. Agent 3 imports this helper for
        offline TCA analysis.
        """
        await self._ensure_db()

        bucket_count = max(1, int(buckets or 10))
        cursor = await self._db.execute(
            "SELECT entry_toxicity_index, pnl_cents, entry_price, exit_price, "
            "entry_size, entry_fee_bps, exit_fee_bps "
            "FROM trades WHERE state = ? AND signal_type = ? "
            "ORDER BY entry_toxicity_index ASC, exit_time ASC",
            (PositionState.CLOSED.value, "ofi_momentum"),
        )
        rows = await cursor.fetchall()

        if not rows:
            return []

        summaries: list[dict[str, float | int | str]] = []
        for bucket_idx in range(bucket_count):
            bucket_lo = bucket_idx / bucket_count
            bucket_hi = (bucket_idx + 1) / bucket_count
            bucket_trades: list[tuple] = []

            for row in rows:
                toxicity_index = max(0.0, min(1.0, float(row[0] or 0.0)))
                mapped_idx = min(bucket_count - 1, int(toxicity_index * bucket_count))
                if mapped_idx == bucket_idx:
                    bucket_trades.append(row)

            if not bucket_trades:
                continue

            total_trades = len(bucket_trades)
            wins = sum(1 for row in bucket_trades if float(row[1] or 0.0) > 0.0)
            total_pnl_cents = sum(float(row[1] or 0.0) for row in bucket_trades)
            total_taker_fee_drag_cents = 0.0
            for row in bucket_trades:
                entry_price = float(row[2] or 0.0)
                exit_price = float(row[3] or 0.0)
                size = float(row[4] or 0.0)
                entry_fee_bps = int(row[5] or 0)
                exit_fee_bps = int(row[6] or 0)
                total_taker_fee_drag_cents += (
                    entry_price * (entry_fee_bps / 10_000.0) * 100.0 * size
                )
                total_taker_fee_drag_cents += (
                    exit_price * (exit_fee_bps / 10_000.0) * 100.0 * size
                )

            summaries.append(
                {
                    "bucket_index": bucket_idx,
                    "toxicity_min": round(bucket_lo, 4),
                    "toxicity_max": round(bucket_hi, 4),
                    "bucket_label": f"[{bucket_lo:.1f}, {bucket_hi:.1f}{']' if bucket_idx == bucket_count - 1 else ')'}",
                    "trade_count": total_trades,
                    "win_rate": round(wins / total_trades, 4),
                    "avg_net_pnl_cents": round(total_pnl_cents / total_trades, 4),
                    "total_net_pnl_cents": round(total_pnl_cents, 4),
                    "total_taker_fee_drag_cents": round(total_taker_fee_drag_cents, 4),
                }
            )

        return summaries

    async def get_stochastic_execution_slippage(self) -> list[dict[str, float | int | str]]:
        """Summarise OFI exit slippage against private stochastic brackets.

        Groups closed OFI momentum trades into three buckets:
        ``target``, ``stop``, and ``time_stop``. For each bucket, reports
        how far the realised ``exit_price`` landed from both the private
        ``drawn_tp`` and ``drawn_stop`` levels, in cents.
        """
        await self._ensure_db()

        cursor = await self._db.execute(
            """
            SELECT
                CASE
                    WHEN exit_reason = 'target' THEN 'target'
                    WHEN exit_reason IN ('stop_loss', 'preemptive_liquidity_drain') THEN 'stop'
                    WHEN exit_reason = 'time_stop' THEN 'time_stop'
                    ELSE 'other'
                END AS exit_bucket,
                COUNT(*) AS trade_count,
                AVG(exit_price) AS avg_exit_price,
                AVG(drawn_tp) AS avg_drawn_tp,
                AVG(drawn_stop) AS avg_drawn_stop,
                AVG((exit_price - drawn_tp) * 100.0) AS avg_exit_minus_drawn_tp_cents,
                AVG((exit_price - drawn_stop) * 100.0) AS avg_exit_minus_drawn_stop_cents,
                AVG(
                    CASE
                        WHEN exit_reason = 'target' THEN (exit_price - drawn_tp) * 100.0
                        WHEN exit_reason IN ('stop_loss', 'preemptive_liquidity_drain') THEN (exit_price - drawn_stop) * 100.0
                        ELSE NULL
                    END
                ) AS avg_reference_slippage_cents,
                AVG(
                    CASE
                        WHEN exit_reason = 'target' THEN ABS((exit_price - drawn_tp) * 100.0)
                        WHEN exit_reason IN ('stop_loss', 'preemptive_liquidity_drain') THEN ABS((exit_price - drawn_stop) * 100.0)
                        ELSE NULL
                    END
                ) AS avg_abs_reference_slippage_cents,
                MIN(exit_price) AS min_exit_price,
                MAX(exit_price) AS max_exit_price
            FROM trades
            WHERE state = ?
              AND signal_type = ?
              AND COALESCE(exit_price, 0.0) > 0.0
              AND (
                    COALESCE(drawn_tp, 0.0) > 0.0
                    OR COALESCE(drawn_stop, 0.0) > 0.0
                  )
            GROUP BY exit_bucket
            HAVING exit_bucket IN ('target', 'stop', 'time_stop')
            ORDER BY CASE exit_bucket
                WHEN 'target' THEN 1
                WHEN 'stop' THEN 2
                WHEN 'time_stop' THEN 3
                ELSE 4
            END
            """,
            (PositionState.CLOSED.value, "ofi_momentum"),
        )
        rows = await cursor.fetchall()

        if not rows:
            return []

        summaries: list[dict[str, float | int | str]] = []
        for row in rows:
            summaries.append(
                {
                    "exit_bucket": row[0],
                    "trade_count": int(row[1] or 0),
                    "avg_exit_price": round(float(row[2] or 0.0), 6),
                    "avg_drawn_tp": round(float(row[3] or 0.0), 6),
                    "avg_drawn_stop": round(float(row[4] or 0.0), 6),
                    "avg_exit_minus_drawn_tp_cents": round(float(row[5] or 0.0), 4),
                    "avg_exit_minus_drawn_stop_cents": round(float(row[6] or 0.0), 4),
                    "avg_reference_slippage_cents": round(float(row[7] or 0.0), 4),
                    "avg_abs_reference_slippage_cents": round(float(row[8] or 0.0), 4),
                    "min_exit_price": round(float(row[9] or 0.0), 6),
                    "max_exit_price": round(float(row[10] or 0.0), 6),
                }
            )

        return summaries

    @staticmethod
    def _compute_decayed_wr(
        pnls: list[float], alpha: float = 0.10
    ) -> float:
        """Exponentially-decayed win rate.

        ŵ_t = α · 1[win_t] + (1-α) · ŵ_{t-1}

        Recent trades are weighted more heavily, so the sizer reacts
        to deteriorating or improving performance within ~10 trades
        instead of being anchored to the all-time aggregate.
        """
        if not pnls:
            return 0.0
        wr = 0.5  # seed
        for p in pnls:
            wr = alpha * (1.0 if p > 0 else 0.0) + (1.0 - alpha) * wr
        return wr

    async def passes_go_live_criteria(self) -> tuple[bool, dict]:
        """Check whether paper-trading results meet go-live thresholds.

        Criteria:
          - ≥ 20 trades
          - Win rate ≥ 55%
          - Positive expectancy
          - No single trade loss > 25% of initial simulated capital (assume $50)
        """
        stats = await self.get_stats()

        if stats["total_trades"] < 20:
            return False, stats

        if stats["win_rate"] < 0.55:
            return False, stats

        if stats.get("expectancy_cents", 0) <= 0:
            return False, stats

        # Check max single loss
        cursor = await self._db.execute(
            "SELECT MIN(pnl_cents) FROM trades WHERE state = ?",
            (PositionState.CLOSED.value,),
        )
        row = await cursor.fetchone()
        if row and row[0] is not None:
            max_loss_cents = abs(row[0])
            # 25% of $50 = $12.50 = 1250 cents
            if max_loss_cents > 1250:
                return False, stats

        return True, stats

    async def get_rolling_expectancy(self, window: int = 10) -> float:
        """Compute average PnL of the last *window* closed trades.

        Used by the adaptive cold-start Kelly sizer to detect negative
        expectancy and throttle sizing.

        Returns 0.0 if fewer than *window* trades are available.

        V4: Probe trades (is_probe=1) are excluded from the calculation
        to prevent micro-sized exploratory entries from polluting the
        expectancy estimator that governs full-size trades.
        """
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT pnl_cents FROM trades WHERE state = ? "
            "AND COALESCE(is_probe, 0) = 0 "
            "ORDER BY exit_time DESC LIMIT ?",
            (PositionState.CLOSED.value, window),
        )
        rows = await cursor.fetchall()
        if len(rows) < window:
            return 0.0
        return sum(r[0] for r in rows) / len(rows)

    async def get_strategy_expectancy(
        self, signal_type: str, window: int = 50,
    ) -> tuple[float, int]:
        """Return (avg_pnl_cents, n_trades) for the last *window* closed
        trades of a specific *signal_type*.

        Probe trades (``is_probe=1``) are excluded so the expectancy
        reflects full-sized execution quality only.

        Returns ``(0.0, 0)`` when fewer than 1 qualifying trade exists.
        """
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT pnl_cents FROM trades "
            "WHERE state = ? AND signal_type = ? "
            "AND COALESCE(is_probe, 0) = 0 "
            "ORDER BY exit_time DESC LIMIT ?",
            (PositionState.CLOSED.value, signal_type, window),
        )
        rows = await cursor.fetchall()
        n = len(rows)
        if n == 0:
            return 0.0, 0
        avg = sum(r[0] for r in rows) / n
        return avg, n

    # ═══════════════════════════════════════════════════════════════════════
    #  State Persistence — checkpoint / restore for crash recovery
    # ═══════════════════════════════════════════════════════════════════════

    async def checkpoint_orders(self, orders: list) -> None:
        """Snapshot all open orders to the ``live_orders`` table.

        Uses an IMMEDIATE transaction to ensure the DELETE + batch INSERT
        is atomic — preventing partial snapshots on crash.

        Parameters
        ----------
        orders:
            List of :class:`~src.trading.executor.Order` objects that are
            currently LIVE or PARTIALLY_FILLED.
        """
        await self._ensure_db()

        # Atomic truncate + rewrite inside a single transaction
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            await self._db.execute("DELETE FROM live_orders")
            for o in orders:
                await self._db.execute(
                    """
                    INSERT INTO live_orders
                    (order_id, market_id, asset_id, side, price, size, status,
                     filled_size, filled_avg_price, clob_order_id, post_only,
                     created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        o.order_id,
                        o.market_id,
                        o.asset_id,
                        o.side.value if hasattr(o.side, "value") else str(o.side),
                        o.price,
                        o.size,
                        o.status.value if hasattr(o.status, "value") else str(o.status),
                        o.filled_size,
                        o.filled_avg_price,
                        o.clob_order_id,
                        int(o.post_only),
                        o.created_at,
                        o.updated_at,
                    ),
                )
            await self._db.commit()
        except Exception:
            self.db_lock_retries += 1
            await self._db.rollback()
            raise
        log.debug("checkpoint_orders", count=len(orders))

    async def checkpoint_positions(self, positions: list) -> None:
        """Snapshot all open positions to the ``live_positions`` table.

        Uses an IMMEDIATE transaction to ensure atomicity.

        Parameters
        ----------
        positions:
            List of :class:`~src.trading.position_manager.Position` objects.
        """
        await self._ensure_db()

        await self._db.execute("BEGIN IMMEDIATE")
        try:
            await self._db.execute("DELETE FROM live_positions")
            for p in positions:
                await self._db.execute(
                    """
                    INSERT INTO live_positions
                    (id, market_id, no_asset_id, event_id, state, entry_price,
                     entry_size, entry_time, target_price, exit_reason,
                     drawn_tp, drawn_stop, drawn_time,
                     fee_enabled, sl_trigger_cents, entry_fee_bps, exit_fee_bps,
                     entry_order_id, exit_order_id, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        p.id,
                        p.market_id,
                        p.no_asset_id,
                        p.event_id,
                        p.state.value if hasattr(p.state, "value") else str(p.state),
                        p.entry_price,
                        p.entry_size,
                        p.entry_time,
                        p.target_price,
                        p.exit_reason,
                        float(getattr(p, "drawn_tp", 0.0) or 0.0),
                        float(getattr(p, "drawn_stop", 0.0) or 0.0),
                        float(getattr(p, "drawn_time", 0.0) or 0.0),
                        int(p.fee_enabled),
                        p.sl_trigger_cents,
                        p.entry_fee_bps,
                        p.exit_fee_bps,
                        p.entry_order.order_id if p.entry_order else "",
                        p.exit_order.order_id if p.exit_order else "",
                        p.created_at,
                    ),
                )
            await self._db.commit()
        except Exception:
            self.db_lock_retries += 1
            await self._db.rollback()
            raise
        log.debug("checkpoint_positions", count=len(positions))

    async def restore_orders(self) -> list[dict]:
        """Read persisted order snapshots from the DB."""
        await self._ensure_db()

        cursor = await self._db.execute(
            "SELECT order_id, market_id, asset_id, side, price, size, status, "
            "filled_size, filled_avg_price, clob_order_id, post_only, "
            "created_at, updated_at FROM live_orders"
        )
        rows = await cursor.fetchall()
        return [
            {
                "order_id": r[0],
                "market_id": r[1],
                "asset_id": r[2],
                "side": r[3],
                "price": r[4],
                "size": r[5],
                "status": r[6],
                "filled_size": r[7],
                "filled_avg_price": r[8],
                "clob_order_id": r[9],
                "post_only": bool(r[10]),
                "created_at": r[11],
                "updated_at": r[12],
            }
            for r in rows
        ]

    async def restore_positions(self) -> list[dict]:
        """Read persisted position snapshots from the DB."""
        await self._ensure_db()

        cursor = await self._db.execute(
            "SELECT id, market_id, no_asset_id, event_id, state, entry_price, "
            "entry_size, entry_time, target_price, exit_reason, drawn_tp, drawn_stop, drawn_time, "
            "fee_enabled, sl_trigger_cents, entry_fee_bps, exit_fee_bps, "
            "entry_order_id, exit_order_id, created_at FROM live_positions"
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": r[0],
                "market_id": r[1],
                "no_asset_id": r[2],
                "event_id": r[3],
                "state": r[4],
                "entry_price": r[5],
                "entry_size": r[6],
                "entry_time": r[7],
                "target_price": r[8],
                "exit_reason": r[9],
                "drawn_tp": r[10],
                "drawn_stop": r[11],
                "drawn_time": r[12],
                "fee_enabled": bool(r[13]),
                "sl_trigger_cents": r[14],
                "entry_fee_bps": r[15],
                "exit_fee_bps": r[16],
                "entry_order_id": r[17],
                "exit_order_id": r[18],
                "created_at": r[19],
            }
            for r in rows
        ]

    async def clear_live_state(self) -> None:
        """Wipe persisted live-state tables (call after clean shutdown)."""
        if not self._db:
            return
        await self._db.execute("BEGIN")
        try:
            await self._db.execute("DELETE FROM live_orders")
            await self._db.execute("DELETE FROM live_positions")
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

    # ═══════════════════════════════════════════════════════════════════════
    #  Shadow Performance Tracker — counterfactual trade persistence
    # ═══════════════════════════════════════════════════════════════════════

    async def record_shadow_trade(
        self,
        *,
        trade_id: str,
        signal_source: str,
        market_id: str,
        asset_id: str = "",
        direction: str = "NO",
        reference_price: float = 0.0,
        reference_price_band: str = "",
        entry_price: float,
        entry_size: float,
        entry_time: float,
        target_price: float,
        stop_price: float,
        exit_price: float,
        exit_time: float,
        exit_reason: str,
        pnl_cents: float,
        entry_fee_bps: int = 0,
        exit_fee_bps: int = 0,
        zscore: float = 0.0,
        confidence: float = 0.0,
        toxicity_index: float = 0.0,
        extra_payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a counterfactual shadow trade result."""
        await self._ensure_db()

        hold = (exit_time - entry_time) if exit_time and entry_time else 0.0
        created_at = time.time()
        record_tuple = (
            trade_id,
            signal_source,
            market_id,
            asset_id,
            PositionState.CLOSED.value,
            direction,
            reference_price,
            reference_price_band,
            entry_price,
            entry_size,
            entry_time,
            target_price,
            stop_price,
            exit_price,
            exit_time,
            exit_reason,
            pnl_cents,
            round(hold, 1),
            entry_fee_bps,
            exit_fee_bps,
            zscore,
            confidence,
            toxicity_index,
            created_at,
        )
        journal_entry = {
            "journal_key": self._journal_key("shadow_trades", trade_id),
            "ledger_kind": "shadow_trades",
            "trade_id": trade_id,
            "signal_source": signal_source,
            "market_id": market_id,
            "state": PositionState.CLOSED.value,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "exit_reason": str(exit_reason or ""),
            "journaled_at": created_at,
            "payload_json": _json_dumps(
                _merge_shadow_journal_payload(
                    {
                        "id": trade_id,
                        "signal_source": signal_source,
                        "market_id": market_id,
                        "asset_id": asset_id,
                        "state": PositionState.CLOSED.value,
                        "direction": direction,
                        "reference_price": reference_price,
                        "reference_price_band": reference_price_band,
                        "entry_price": entry_price,
                        "entry_size": entry_size,
                        "entry_time": entry_time,
                        "target_price": target_price,
                        "stop_price": stop_price,
                        "exit_price": exit_price,
                        "exit_time": exit_time,
                        "exit_reason": exit_reason,
                        "pnl_cents": pnl_cents,
                        "hold_seconds": round(hold, 1),
                        "entry_fee_bps": entry_fee_bps,
                        "exit_fee_bps": exit_fee_bps,
                        "zscore": zscore,
                        "confidence": confidence,
                        "toxicity_index": toxicity_index,
                        "created_at": created_at,
                    },
                    extra_payload=extra_payload,
                )
            ),
        }

        try:
            await self._journal_trade_intent(journal_entry)
            await self._db.execute("BEGIN IMMEDIATE")
            await self._db.execute(
                """
                INSERT OR REPLACE INTO shadow_trades
                (id, signal_source, market_id, asset_id, state, direction,
                 reference_price, reference_price_band,
                 entry_price, entry_size, entry_time,
                 target_price, stop_price, exit_price, exit_time,
                 exit_reason, pnl_cents, hold_seconds,
                 entry_fee_bps, exit_fee_bps, zscore, confidence, toxicity_index, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                record_tuple,
            )
            await self._mark_trade_persistence_recorded(
                journal_entry["journal_key"],
                recorded_at=time.time(),
            )
            await self._db.commit()
        except Exception as exc:
            try:
                await self._db.rollback()
            except Exception:
                pass
            await self._mark_trade_persistence_failed(journal_entry, error=str(exc))
            log.critical(
                "trade_persistence_contract_failed",
                ledger_kind="shadow_trades",
                trade_id=trade_id,
                market_id=market_id,
                signal_source=signal_source or "unknown",
                db_path=str(self.db_path),
                error=str(exc),
                exc_info=True,
            )
            raise TradePersistenceError(
                f"Failed to persist shadow trade {trade_id} to {self.db_path}"
            ) from exc

    async def _fetch_persistence_journal_rows(
        self,
        *,
        ledger_kind: str | None = None,
        signal_source: str | None = None,
    ) -> list[dict[str, Any]]:
        await self._ensure_db()

        sql = (
            "SELECT journal_key, ledger_kind, trade_id, signal_source, market_id, state, "
            "entry_time, exit_time, exit_reason, payload_json, journaled_at, "
            "ledger_recorded_at, ledger_state, last_error, last_error_at "
            "FROM trade_persistence_journal WHERE 1 = 1"
        )
        params: list[object] = []
        if ledger_kind is not None:
            sql += " AND ledger_kind = ?"
            params.append(ledger_kind)
        if signal_source is not None:
            sql += " AND signal_source = ?"
            params.append(signal_source)
        sql += " ORDER BY journaled_at ASC, trade_id ASC"

        cursor = await self._db.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            payload_json = str(row[9] or "{}")
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                payload = None
            result.append(
                {
                    "journal_key": row[0],
                    "ledger_kind": row[1],
                    "trade_id": row[2],
                    "signal_source": row[3],
                    "market_id": row[4],
                    "state": row[5],
                    "entry_time": row[6],
                    "exit_time": row[7],
                    "exit_reason": row[8],
                    "payload_json": payload_json,
                    "payload": payload,
                    "journaled_at": row[10],
                    "ledger_recorded_at": row[11],
                    "ledger_state": row[12],
                    "last_error": row[13],
                    "last_error_at": row[14],
                }
            )
        return result

    async def _fetch_ledger_trade_ids(
        self,
        *,
        ledger_kind: str,
        signal_source: str | None = None,
    ) -> set[str]:
        await self._ensure_db()

        if ledger_kind == "trades":
            sql = "SELECT id FROM trades WHERE state = ?"
            params: list[object] = [PositionState.CLOSED.value]
            if signal_source is not None:
                sql += " AND signal_type = ?"
                params.append(signal_source)
        elif ledger_kind == "shadow_trades":
            sql = "SELECT id FROM shadow_trades WHERE state = ?"
            params = [PositionState.CLOSED.value]
            if signal_source is not None:
                sql += " AND signal_source = ?"
                params.append(signal_source)
        else:
            raise ValueError(f"Unsupported ledger kind: {ledger_kind}")

        cursor = await self._db.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        return {str(row[0]) for row in rows}

    async def get_persistence_accounting_summary(
        self,
        *,
        ledger_kind: str,
        signal_source: str | None = None,
    ) -> dict[str, object]:
        await self._ensure_db()

        journal_rows = await self._fetch_persistence_journal_rows(
            ledger_kind=ledger_kind,
            signal_source=signal_source,
        )
        ledger_ids = await self._fetch_ledger_trade_ids(
            ledger_kind=ledger_kind,
            signal_source=signal_source,
        )
        journal_ids = {str(row["trade_id"]) for row in journal_rows}
        matched_ids = journal_ids & ledger_ids
        missing_ids = sorted(journal_ids - ledger_ids)
        legacy_ids = sorted(ledger_ids - journal_ids)
        failed_count = sum(1 for row in journal_rows if row.get("ledger_state") == "FAILED")
        pending_count = sum(1 for row in journal_rows if row.get("ledger_state") == "PENDING")
        recorded_count = sum(1 for row in journal_rows if row.get("ledger_state") == "RECORDED")
        journal_total = len(journal_rows)
        ledger_total = len(ledger_ids)
        completeness = round(len(matched_ids) / journal_total, 4) if journal_total else None
        coverage_ratio = round(journal_total / ledger_total, 4) if ledger_total else None
        accounting_complete: bool | None
        if journal_total == 0:
            accounting_complete = None
        else:
            accounting_complete = len(missing_ids) == 0 and failed_count == 0 and pending_count == 0

        return {
            "ledger_kind": ledger_kind,
            "signal_source": signal_source,
            "journal_rows": journal_total,
            "journal_recorded_rows": recorded_count,
            "journal_failed_rows": failed_count,
            "journal_pending_rows": pending_count,
            "matched_ledger_rows": len(matched_ids),
            "missing_ledger_rows": len(missing_ids),
            "legacy_ledger_rows_without_journal": len(legacy_ids),
            "ledger_rows": ledger_total,
            "accounting_completeness": completeness,
            "journal_coverage_ratio": coverage_ratio,
            "accounting_complete": accounting_complete,
            "sample_missing_ledger_ids": missing_ids[:20],
            "sample_legacy_ledger_ids": legacy_ids[:20],
            "last_journaled_at": journal_rows[-1]["journaled_at"] if journal_rows else None,
        }

    async def create_wal_safe_remeasurement_snapshot(
        self,
        *,
        label: str = "trade_store",
        output_dir: str | Path | None = None,
        include_journal_capture: bool = True,
    ) -> dict[str, object]:
        await self._ensure_db()

        snapshot_dir = Path(output_dir) if output_dir else _REMEASUREMENT_SNAPSHOT_DIR
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or "trade_store"
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        prefix = f"{safe_label}_{stamp}"
        snapshot_db_path = snapshot_dir / f"{prefix}.db"
        journal_capture_path = snapshot_dir / f"{prefix}.trade_persistence_journal.jsonl"
        manifest_path = snapshot_dir / f"{prefix}.manifest.json"

        checkpoint_cursor = await self._db.execute("PRAGMA wal_checkpoint(PASSIVE)")
        checkpoint_row = await checkpoint_cursor.fetchone()
        wal_checkpoint = None
        if checkpoint_row is not None:
            wal_checkpoint = {
                "busy": int(checkpoint_row[0] or 0),
                "log_frames": int(checkpoint_row[1] or 0),
                "checkpointed_frames": int(checkpoint_row[2] or 0),
            }

        await asyncio.to_thread(_sqlite_backup, self.db_path, snapshot_db_path)

        snapshot_store = TradeStore(snapshot_db_path)
        try:
            await snapshot_store.init()
            journal_rows = await snapshot_store._fetch_persistence_journal_rows()
            accounting = {
                "trades": await snapshot_store.get_persistence_accounting_summary(ledger_kind="trades"),
                "shadow_trades": await snapshot_store.get_persistence_accounting_summary(ledger_kind="shadow_trades"),
            }
        finally:
            await snapshot_store.close()

        if include_journal_capture:
            journal_lines = []
            for row in journal_rows:
                line = dict(row)
                payload = line.pop("payload", None)
                line["payload"] = payload
                journal_lines.append(json.dumps(line, sort_keys=True, default=str))
            journal_capture_path.write_text(
                ("\n".join(journal_lines) + "\n") if journal_lines else "",
                encoding="utf-8",
            )

        manifest: dict[str, object] = {
            "label": safe_label,
            "created_at": time.time(),
            "source_db_path": str(self.db_path),
            "snapshot_db_path": str(snapshot_db_path),
            "journal_capture_path": str(journal_capture_path) if include_journal_capture else None,
            "manifest_path": str(manifest_path),
            "wal_checkpoint": wal_checkpoint,
            "accounting": accounting,
            "journal_rows_exported": len(journal_rows),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    async def get_shadow_stats(self, signal_source: str) -> dict:
        """Compute aggregate stats for a specific shadow signal source.

        Returns a dict compatible with ``passes_go_live_criteria`` checks.
        """
        await self._ensure_db()

        rows = await self._fetch_shadow_rows(signal_source=signal_source)
        if not rows:
            return {"signal_source": signal_source, "total_trades": 0}
        return {
            "signal_source": signal_source,
            **_build_shadow_summary(rows),
        }

    async def _fetch_shadow_rows(
        self,
        *,
        signal_source: str | None = None,
    ) -> list[dict[str, float | int | str]]:
        await self._ensure_db()

        sql = (
            "SELECT signal_source, market_id, asset_id, direction, reference_price, reference_price_band, "
            "entry_price, entry_size, entry_time, "
            "target_price, stop_price, exit_price, exit_time, exit_reason, pnl_cents, hold_seconds, "
            "entry_fee_bps, exit_fee_bps, zscore, confidence, toxicity_index "
            "FROM shadow_trades WHERE state = ?"
        )
        params: list[object] = [PositionState.CLOSED.value]
        if signal_source is not None:
            sql += " AND signal_source = ?"
            params.append(signal_source)
        sql += " ORDER BY exit_time ASC"

        cursor = await self._db.execute(sql, tuple(params))
        fetched = await cursor.fetchall()
        columns = [
            "signal_source",
            "market_id",
            "asset_id",
            "direction",
            "reference_price",
            "reference_price_band",
            "entry_price",
            "entry_size",
            "entry_time",
            "target_price",
            "stop_price",
            "exit_price",
            "exit_time",
            "exit_reason",
            "pnl_cents",
            "hold_seconds",
            "entry_fee_bps",
            "exit_fee_bps",
            "zscore",
            "confidence",
            "toxicity_index",
        ]
        return [_annotate_shadow_row(dict(zip(columns, row, strict=False))) for row in fetched]

    async def get_shadow_cohort_report(
        self,
        signal_source: str | None = None,
        *,
        market_limit: int = 5,
    ) -> dict[str, object]:
        """Build a reusable cohort report for shadow strategies.

        The report is intentionally generic so multiple shadow candidates can
        share the same forward-study output shape.
        """
        rows = await self._fetch_shadow_rows(signal_source=signal_source)
        summary = _build_shadow_summary(rows)
        accounting = await self.get_persistence_accounting_summary(
            ledger_kind="shadow_trades",
            signal_source=signal_source,
        )
        summary.update(
            {
                "accounting_complete": accounting["accounting_complete"],
                "accounting_completeness": accounting["accounting_completeness"],
                "journal_rows": accounting["journal_rows"],
                "missing_ledger_rows": accounting["missing_ledger_rows"],
                "legacy_ledger_rows_without_journal": accounting["legacy_ledger_rows_without_journal"],
            }
        )
        exit_mix = _build_exit_mix(rows)

        report: dict[str, object] = {
            "summary": summary,
            "accounting": accounting,
            "exit_mix": exit_mix,
            "loss_buckets_to_beat": _build_loss_bucket_summary(exit_mix),
            "by_signal_source": _group_shadow_rows(
                rows,
                key_fn=lambda row: row.get("signal_source") or "",
            ),
            "by_direction": _group_shadow_rows(
                rows,
                key_fn=lambda row: row.get("direction") or "",
            ),
            "by_entry_price": _group_shadow_rows(
                rows,
                key_fn=lambda row: _bucket_value(
                    float(row.get("entry_price") or 0.0),
                    _SHADOW_PRICE_BUCKETS,
                ),
            ),
            "by_confidence": _group_shadow_rows(
                rows,
                key_fn=lambda row: _bucket_value(
                    float(row.get("confidence") or 0.0),
                    _SHADOW_CONFIDENCE_BUCKETS,
                ),
            ),
            "by_reference_price": _group_shadow_rows(
                rows,
                key_fn=lambda row: (
                    _bucket_value(
                        float(row.get("reference_price") or 0.0),
                        _SHADOW_PRICE_BUCKETS,
                    )
                    if float(row.get("reference_price") or 0.0) > 0.0
                    else "n/a"
                ),
            ),
            "by_reference_price_band": _group_shadow_rows(
                rows,
                key_fn=lambda row: row.get("reference_price_band") or "n/a",
            ),
            "by_toxicity": _group_shadow_rows(
                rows,
                key_fn=lambda row: _bucket_value(
                    float(row.get("toxicity_index") or 0.0),
                    _SHADOW_TOXICITY_BUCKETS,
                ),
            ),
            "by_hold": _group_shadow_rows(
                rows,
                key_fn=lambda row: _bucket_value(
                    float(row.get("hold_seconds") or 0.0),
                    _SHADOW_HOLD_BUCKETS,
                ),
            ),
            "by_exit_reason": _group_shadow_rows(
                rows,
                key_fn=lambda row: row.get("exit_reason") or "",
            ),
            "top_markets": _top_shadow_markets(rows, limit=market_limit),
        }
        if signal_source is not None:
            report["signal_source"] = signal_source
        return report

    async def get_shadow_source_overview(self) -> list[dict[str, object]]:
        """Return one compact row per shadow signal source.

        This is intended for periodic operator summaries and quick go-live
        triage without requiring a full cohort dump.
        """
        sources = await self.get_all_shadow_sources()
        overview: list[dict[str, object]] = []
        for source in sources:
            ready, stats = await self.passes_shadow_go_live(source)
            overview.append(
                {
                    "signal_source": source,
                    **stats,
                    "passes_go_live": ready,
                }
            )

        overview.sort(
            key=lambda row: (
                -int(row.get("total_trades", 0) or 0),
                -float(row.get("expectancy_cents", 0.0) or 0.0),
                str(row.get("signal_source") or ""),
            )
        )
        return overview

    async def get_all_shadow_sources(self) -> list[str]:
        """Return all distinct signal_source values in the shadow_trades table."""
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT DISTINCT signal_source FROM shadow_trades ORDER BY signal_source"
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def passes_shadow_go_live(self, signal_source: str) -> tuple[bool, dict]:
        """Check whether a shadow strategy meets go-live criteria.

        Criteria (same as live):
          - >= 20 trades
          - Win rate >= 55%
          - Positive expectancy
        """
        stats = await self.get_shadow_stats(signal_source)

        if stats["total_trades"] < 20:
            return False, stats

        if stats["win_rate"] < 0.55:
            return False, stats

        if stats.get("expectancy_cents", 0) <= 0:
            return False, stats

        return True, stats


async def get_ofi_toxicity_pnl_summary(
    buckets: int = 10,
    *,
    db_path: str | Path | None = None,
) -> list[dict[str, float | int | str]]:
    """Convenience wrapper for OFI toxicity aggregation from SQLite.

    This thin helper exists so offline analysis scripts can import a single
    public function from ``src.monitoring.trade_store`` without manually
    managing a ``TradeStore`` lifecycle.
    """
    store = TradeStore(db_path=db_path)
    try:
        return await store.get_ofi_toxicity_pnl_summary(buckets=buckets)
    finally:
        await store.close()


async def get_stochastic_execution_slippage(
    *,
    db_path: str | Path | None = None,
) -> list[dict[str, float | int | str]]:
    """Convenience wrapper for stochastic OFI execution slippage TCA."""
    store = TradeStore(db_path=db_path)
    try:
        return await store.get_stochastic_execution_slippage()
    finally:
        await store.close()


async def get_shadow_cohort_report(
    *,
    db_path: str | Path | None = None,
    signal_source: str | None = None,
    market_limit: int = 5,
) -> dict[str, object]:
    """Convenience wrapper for shadow forward-study cohort reporting."""
    store = TradeStore(db_path=db_path)
    try:
        return await store.get_shadow_cohort_report(
            signal_source=signal_source,
            market_limit=market_limit,
        )
    finally:
        await store.close()


async def get_persistence_accounting_summary(
    *,
    db_path: str | Path | None = None,
    ledger_kind: str,
    signal_source: str | None = None,
) -> dict[str, object]:
    """Convenience wrapper for journal-vs-ledger accounting checks."""
    store = TradeStore(db_path=db_path)
    try:
        return await store.get_persistence_accounting_summary(
            ledger_kind=ledger_kind,
            signal_source=signal_source,
        )
    finally:
        await store.close()


async def create_wal_safe_remeasurement_snapshot(
    *,
    db_path: str | Path | None = None,
    label: str = "trade_store",
    output_dir: str | Path | None = None,
    include_journal_capture: bool = True,
) -> dict[str, object]:
    """Create a point-in-time SQLite snapshot plus journal capture for analysis."""
    store = TradeStore(db_path=db_path)
    try:
        return await store.create_wal_safe_remeasurement_snapshot(
            label=label,
            output_dir=output_dir,
            include_journal_capture=include_journal_capture,
        )
    finally:
        await store.close()
