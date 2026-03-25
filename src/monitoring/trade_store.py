"""
Paper-trading statistics engine — persists trade records to SQLite and
computes aggregate metrics (win rate, expectancy, drawdown, etc.).

Also provides **state-persistence** tables for crash recovery: live
``orders`` and ``positions`` are periodically checkpointed so the bot
can reconcile on restart.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from functools import wraps
from pathlib import Path

import aiosqlite

from src.core.config import settings
from src.core.logger import get_logger
from src.trading.position_manager import Position, PositionState

log = get_logger(__name__)


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
    state           TEXT NOT NULL,
    direction       TEXT DEFAULT 'NO',
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
    created_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_shadow_source ON shadow_trades(signal_source);
CREATE INDEX IF NOT EXISTS idx_shadow_state ON shadow_trades(state);

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
                        state           TEXT NOT NULL,
                        direction       TEXT DEFAULT 'NO',
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
                        created_at      REAL
                    );
                    CREATE INDEX IF NOT EXISTS idx_shadow_source ON shadow_trades(signal_source);
                    CREATE INDEX IF NOT EXISTS idx_shadow_state ON shadow_trades(state);
                    """
                )
                await self._db.commit()
        except Exception:
            log.warning("shadow_trades_migration_skipped", exc_info=True)

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

        await self._db.execute(
            """
            INSERT OR REPLACE INTO trades
            (id, market_id, state, entry_price, entry_size, entry_time,
             target_price, exit_price, exit_time, exit_reason, pnl_cents,
             hold_seconds, alpha, zscore, volume_ratio, whale,
             entry_fee_bps, exit_fee_bps, entry_toxicity_index, exit_toxicity_index,
             drawn_tp, drawn_stop, drawn_time, created_at,
             is_probe, signal_type, meta_weight)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                pos.id,
                pos.market_id,
                pos.state.value,
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
                pos.created_at,
                int(getattr(pos, 'is_probe', False)),
                getattr(pos, 'signal_type', ''),
                getattr(pos, 'meta_weight', 1.0),
            ),
        )
        await self._db.commit()

    # ── Aggregate stats ─────────────────────────────────────────────────────
    async def get_stats(self) -> dict:
        """Compute aggregate trading statistics."""
        await self._ensure_db()

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
        direction: str = "NO",
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
    ) -> None:
        """Persist a counterfactual shadow trade result."""
        await self._ensure_db()

        hold = (exit_time - entry_time) if exit_time and entry_time else 0.0

        await self._db.execute(
            """
            INSERT OR REPLACE INTO shadow_trades
            (id, signal_source, market_id, state, direction,
             entry_price, entry_size, entry_time,
             target_price, stop_price, exit_price, exit_time,
             exit_reason, pnl_cents, hold_seconds,
             entry_fee_bps, exit_fee_bps, zscore, confidence, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade_id,
                signal_source,
                market_id,
                PositionState.CLOSED.value,
                direction,
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
                time.time(),
            ),
        )
        await self._db.commit()

    async def get_shadow_stats(self, signal_source: str) -> dict:
        """Compute aggregate stats for a specific shadow signal source.

        Returns a dict compatible with ``passes_go_live_criteria`` checks.
        """
        await self._ensure_db()

        cursor = await self._db.execute(
            "SELECT pnl_cents, exit_reason, hold_seconds "
            "FROM shadow_trades "
            "WHERE signal_source = ? AND state = ? "
            "ORDER BY exit_time ASC",
            (signal_source, PositionState.CLOSED.value),
        )
        rows = await cursor.fetchall()

        if not rows:
            return {"signal_source": signal_source, "total_trades": 0}

        pnls = [r[0] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        holds = [r[2] for r in rows if r[2]]

        target_exits = sum(1 for r in rows if r[1] == "target")
        stop_exits = sum(1 for r in rows if r[1] == "stop_loss")

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
            "signal_source": signal_source,
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "avg_win_cents": round(avg_win, 2),
            "avg_loss_cents": round(avg_loss, 2),
            "total_pnl_cents": round(total_pnl, 2),
            "max_drawdown_cents": round(max_dd, 2),
            "target_exits": target_exits,
            "stop_exits": stop_exits,
            "avg_hold_seconds": round(sum(holds) / len(holds), 1) if holds else 0,
            "expectancy_cents": round(total_pnl / total, 2) if total else 0,
        }

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
