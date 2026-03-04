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
    created_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_trades_state ON trades(state);

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

        await self._db.execute(
            """
            INSERT OR REPLACE INTO trades
            (id, market_id, state, entry_price, entry_size, entry_time,
             target_price, exit_price, exit_time, exit_reason, pnl_cents,
             hold_seconds, alpha, zscore, volume_ratio, whale, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                signal.zscore if signal else None,
                signal.volume_ratio if signal else None,
                int(signal.whale_confluence) if signal else 0,
                pos.created_at,
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
        """
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT pnl_cents FROM trades WHERE state = ? "
            "ORDER BY exit_time DESC LIMIT ?",
            (PositionState.CLOSED.value, window),
        )
        rows = await cursor.fetchall()
        if len(rows) < window:
            return 0.0
        return sum(r[0] for r in rows) / len(rows)

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
                     fee_enabled, sl_trigger_cents, entry_fee_bps, exit_fee_bps,
                     entry_order_id, exit_order_id, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            "entry_size, entry_time, target_price, exit_reason, "
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
                "fee_enabled": bool(r[10]),
                "sl_trigger_cents": r[11],
                "entry_fee_bps": r[12],
                "exit_fee_bps": r[13],
                "entry_order_id": r[14],
                "exit_order_id": r[15],
                "created_at": r[16],
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
