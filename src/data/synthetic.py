"""
Synthetic Polymarket L2/Trade data generator.

Produces realistic-looking JSONL tick data that matches the
``MarketDataRecorder`` schema, allowing the DataPrepPipeline and
BacktestEngine to be tested without a live WebSocket connection.

Usage
─────
    from src.data.synthetic import SyntheticGenerator
    gen = SyntheticGenerator(seed=42)
    path = gen.generate(Path("data"), num_rows=100_000)

Output layout mirrors the live recorder::

    <output_dir>/raw_ticks/YYYY-MM-DD/<asset_id>.jsonl

Each line:
    {"local_ts": ..., "source": "l2"|"trade", "asset_id": "...", "payload": {...}}
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.logger import get_logger

log = get_logger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────
_DEFAULT_YES_ASSET = "0x" + "a1" * 16  # fake 32-byte hex
_DEFAULT_NO_ASSET = "0x" + "b2" * 16
_BOOK_DEPTH = 10  # levels per side in snapshots
_SNAPSHOT_INTERVAL_S = 60.0  # seconds between full snapshots


@dataclass
class _AssetState:
    """Mutable state for one simulated asset."""

    asset_id: str
    mid_price: float  # current mid (0, 1)
    seq: int = 0
    is_yes: bool = True


@dataclass
class _GeneratedRecord:
    """An intermediate record before writing to disk."""

    local_ts: float
    source: str  # "l2", "trade"
    asset_id: str
    payload: dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "local_ts": self.local_ts,
                "source": self.source,
                "asset_id": self.asset_id,
                "payload": self.payload,
            },
            separators=(",", ":"),
            default=str,
        )


class SyntheticGenerator:
    """Generate synthetic Polymarket L2 & trade JSONL data.

    Parameters
    ----------
    seed:
        Random seed for reproducibility (``None`` → non-deterministic).
    volatility:
        Annualised volatility for the GBM price walk (default 0.8).
    trade_probability:
        Probability that any given tick is a trade vs L2 delta
        (default 0.25 ≈ 1 trade per 3 deltas).
    snapshot_interval_s:
        Seconds between full L2 book snapshots (default 60).
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        volatility: float = 0.8,
        trade_probability: float = 0.25,
        snapshot_interval_s: float = _SNAPSHOT_INTERVAL_S,
    ) -> None:
        self._rng = random.Random(seed)
        self._volatility = volatility
        self._trade_prob = trade_probability
        self._snapshot_interval = snapshot_interval_s

    # ── Public API ─────────────────────────────────────────────────────

    def generate(
        self,
        output_dir: str | Path,
        *,
        num_rows: int = 100_000,
        duration_hours: float = 24.0,
        num_assets: int = 2,
        base_time: float | None = None,
    ) -> Path:
        """Generate synthetic JSONL files and write to disk.

        Parameters
        ----------
        output_dir:
            Root data directory (<output_dir>/raw_ticks/YYYY-MM-DD/...).
        num_rows:
            Total number of records to generate across all assets.
        duration_hours:
            Simulated time span in hours.
        num_assets:
            Number of distinct assets (default 2 = YES + NO tokens).
        base_time:
            Unix epoch start time.  ``None`` → midnight UTC today.

        Returns
        -------
        Path to the ``raw_ticks`` directory containing generated files.
        """
        output_dir = Path(output_dir)

        if base_time is None:
            now = datetime.now(tz=timezone.utc)
            base_time = datetime(
                now.year, now.month, now.day, tzinfo=timezone.utc
            ).timestamp()

        # Build asset states
        assets = self._init_assets(num_assets)

        # Calculate time step between events
        duration_s = duration_hours * 3600.0
        dt = duration_s / num_rows  # average interval

        # Generate all records
        records: list[_GeneratedRecord] = []
        current_time = base_time
        last_snapshot_time = base_time - self._snapshot_interval  # force initial snapshot

        for i in range(num_rows):
            # Pick a random asset for this tick
            asset = self._rng.choice(assets)

            # Advance time with jitter
            jitter = self._rng.uniform(0.5, 1.5) * dt
            current_time += jitter

            # Advance price via GBM step
            asset.mid_price = self._step_price(asset.mid_price, dt)

            # Decide event type
            time_since_snap = current_time - last_snapshot_time

            if time_since_snap >= self._snapshot_interval:
                # Snapshot time
                record = self._make_snapshot(asset, current_time)
                last_snapshot_time = current_time
            elif self._rng.random() < self._trade_prob:
                record = self._make_trade(asset, current_time)
            else:
                record = self._make_delta(asset, current_time)

            records.append(record)

        # Sort by local_ts (should already be mostly sorted)
        records.sort(key=lambda r: r.local_ts)

        # Write to disk grouped by (date, asset_id)
        written = self._write_records(records, output_dir)

        raw_ticks_dir = output_dir / "raw_ticks"

        log.info(
            "synthetic_generated",
            num_rows=len(records),
            num_assets=len(assets),
            duration_hours=duration_hours,
            files_written=written,
            output_dir=str(raw_ticks_dir),
        )

        return raw_ticks_dir

    # ── Internal helpers ───────────────────────────────────────────────

    def _init_assets(self, num_assets: int) -> list[_AssetState]:
        """Create initial asset states with random starting prices."""
        assets: list[_AssetState] = []

        if num_assets >= 2:
            # First two are YES/NO pair
            yes_price = self._rng.uniform(0.30, 0.70)
            assets.append(
                _AssetState(
                    asset_id=_DEFAULT_YES_ASSET,
                    mid_price=yes_price,
                    is_yes=True,
                )
            )
            assets.append(
                _AssetState(
                    asset_id=_DEFAULT_NO_ASSET,
                    mid_price=1.0 - yes_price,
                    is_yes=False,
                )
            )
            # Additional assets beyond the pair
            for idx in range(2, num_assets):
                aid = "0x" + f"{idx:032x}"
                assets.append(
                    _AssetState(
                        asset_id=aid,
                        mid_price=self._rng.uniform(0.20, 0.80),
                        is_yes=True,
                    )
                )
        elif num_assets == 1:
            assets.append(
                _AssetState(
                    asset_id=_DEFAULT_YES_ASSET,
                    mid_price=self._rng.uniform(0.30, 0.70),
                    is_yes=True,
                )
            )

        return assets

    def _step_price(self, price: float, dt: float) -> float:
        """Advance price one step via geometric Brownian motion,
        clamped to (0.02, 0.98) to keep it in the prediction-market range."""
        # GBM: dS = sigma * S * sqrt(dt) * Z
        annual_dt = dt / (365.25 * 86400)
        z = self._rng.gauss(0, 1)
        log_return = -0.5 * self._volatility**2 * annual_dt + self._volatility * math.sqrt(annual_dt) * z
        new_price = price * math.exp(log_return)
        return max(0.02, min(0.98, new_price))

    def _exchange_ts(self, local_ts: float) -> float:
        """Exchange timestamp is local_ts minus a small latency jitter."""
        latency_ms = self._rng.uniform(1.0, 50.0)
        return local_ts - latency_ms / 1000.0

    def _make_snapshot(self, asset: _AssetState, ts: float) -> _GeneratedRecord:
        """Generate a full L2 book snapshot."""
        asset.seq += 1
        exch_ts = self._exchange_ts(ts)
        spread = self._rng.uniform(0.005, 0.03)
        best_bid = max(0.01, asset.mid_price - spread / 2)
        best_ask = min(0.99, asset.mid_price + spread / 2)

        bids = []
        asks = []
        for i in range(_BOOK_DEPTH):
            bid_price = round(best_bid - i * 0.01, 4)
            ask_price = round(best_ask + i * 0.01, 4)
            if bid_price > 0:
                bids.append({
                    "price": str(round(bid_price, 2)),
                    "size": str(round(self._rng.uniform(50, 500), 1)),
                })
            if ask_price < 1.0:
                asks.append({
                    "price": str(round(ask_price, 2)),
                    "size": str(round(self._rng.uniform(50, 500), 1)),
                })

        payload = {
            "event_type": "book",
            "asset_id": asset.asset_id,
            "bids": bids,
            "asks": asks,
            "seq": asset.seq,
            "timestamp": exch_ts,
        }

        return _GeneratedRecord(
            local_ts=ts,
            source="l2",
            asset_id=asset.asset_id,
            payload=payload,
        )

    def _make_delta(self, asset: _AssetState, ts: float) -> _GeneratedRecord:
        """Generate an L2 book delta (price_change)."""
        asset.seq += 1
        exch_ts = self._exchange_ts(ts)

        num_changes = self._rng.randint(1, 3)
        changes = []
        for _ in range(num_changes):
            side = self._rng.choice(["BUY", "SELL"])
            if side == "BUY":
                price = round(
                    asset.mid_price - self._rng.uniform(0.001, 0.05), 2
                )
            else:
                price = round(
                    asset.mid_price + self._rng.uniform(0.001, 0.05), 2
                )
            price = max(0.01, min(0.99, price))
            # 10% chance of level removal (size=0)
            size = 0.0 if self._rng.random() < 0.10 else round(self._rng.uniform(10, 300), 1)
            changes.append({
                "side": side,
                "price": str(price),
                "size": str(size),
            })

        payload = {
            "event_type": "price_change",
            "asset_id": asset.asset_id,
            "seq": asset.seq,
            "timestamp": exch_ts,
            "changes": changes,
        }

        return _GeneratedRecord(
            local_ts=ts,
            source="l2",
            asset_id=asset.asset_id,
            payload=payload,
        )

    def _make_trade(self, asset: _AssetState, ts: float) -> _GeneratedRecord:
        """Generate a trade event."""
        asset.seq += 1
        exch_ts = self._exchange_ts(ts)

        side = self._rng.choice(["buy", "sell"])
        slippage = self._rng.uniform(-0.005, 0.005)
        price = max(0.01, min(0.99, round(asset.mid_price + slippage, 2)))
        size = round(self._rng.uniform(5, 200), 1)

        payload = {
            "price": str(price),
            "size": str(size),
            "asset_id": asset.asset_id,
            "side": side,
            "timestamp": exch_ts,
            "market": "MOCK_CONDITION_" + asset.asset_id[:10],
            "outcome": "Yes" if asset.is_yes else "No",
        }

        return _GeneratedRecord(
            local_ts=ts,
            source="trade",
            asset_id=asset.asset_id,
            payload=payload,
        )

    def _write_records(
        self, records: list[_GeneratedRecord], output_dir: Path
    ) -> int:
        """Write records to JSONL files grouped by (date, asset_id).

        Returns the number of files written.
        """
        # Group by (date_str, asset_id)
        groups: dict[tuple[str, str], list[str]] = {}

        for rec in records:
            date_str = datetime.fromtimestamp(
                rec.local_ts, tz=timezone.utc
            ).strftime("%Y-%m-%d")
            key = (date_str, rec.asset_id)
            groups.setdefault(key, []).append(rec.to_jsonl())

        files_written = 0
        for (date_str, asset_id), lines in groups.items():
            safe_name = asset_id.replace("/", "_").replace("\\", "_")
            dir_path = output_dir / "raw_ticks" / date_str
            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / f"{safe_name}.jsonl"

            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

            files_written += 1

        return files_written
