"""
Paper-trading statistics engine — persists trade records to SQLite and
computes aggregate metrics (win rate, expectancy, drawdown, etc.).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import aiosqlite

from src.core.logger import get_logger
from src.trading.position_manager import Position, PositionState

log = get_logger(__name__)

DB_PATH = Path("logs/trades.db")

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
"""


class TradeStore:
    """Async SQLite-backed trade log with aggregate statistics."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        log.info("trade_store_initialised", path=str(self.db_path))

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ── Record a closed position ────────────────────────────────────────────
    async def record(self, pos: Position) -> None:
        """Upsert a position record."""
        if not self._db:
            await self.init()

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
        if not self._db:
            await self.init()

        cursor = await self._db.execute(
            "SELECT pnl_cents, exit_reason, hold_seconds FROM trades WHERE state = ?",
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
        }

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
