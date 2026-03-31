"""
Tests for the trade store (SQLite persistence and stats).
"""

import asyncio
from decimal import Decimal
import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.monitoring.trade_store import TradeStore
from src.trading.executor import OrderExecutor
from src.trading.position_manager import Position, PositionManager, PositionState, ShadowExecutionTracker
from src.trading.take_profit import TakeProfitResult
from src.signals.panic_detector import PanicSignal


def _make_position(
    pos_id: str,
    entry: float,
    exit_p: float,
    size: float = 10.0,
    reason: str = "target",
    whale: bool = False,
) -> Position:
    pnl = round((exit_p - entry) * size * 100, 2)
    now = time.time()
    return Position(
        id=pos_id,
        market_id="MKT_TEST",
        no_asset_id="NO_T",
        state=PositionState.CLOSED,
        entry_price=entry,
        entry_size=size,
        entry_time=now - 120,
        target_price=exit_p,
        exit_price=exit_p,
        exit_time=now,
        exit_reason=reason,
        pnl_cents=pnl,
        tp_result=TakeProfitResult(
            entry_price=entry, target_price=exit_p,
            alpha=0.5, spread_cents=abs(exit_p - entry) * 100, viable=True,
        ),
        signal=PanicSignal(
            market_id="MKT_TEST", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.70, yes_vwap=0.50, zscore=2.5, volume_ratio=4.0,
            no_best_ask=entry + 0.01, whale_confluence=whale,
        ),
    )


def _make_ofi_position(
    pos_id: str,
    entry: float,
    exit_p: float,
    *,
    toxicity_index: float,
    size: float = 10.0,
    entry_fee_bps: int = 0,
    exit_fee_bps: int = 0,
) -> Position:
    pos = _make_position(pos_id, entry, exit_p, size=size)
    pos.signal_type = "ofi_momentum"
    pos.entry_toxicity_index = toxicity_index
    pos.entry_fee_bps = entry_fee_bps
    pos.exit_fee_bps = exit_fee_bps
    pos.drawn_tp = exit_p
    pos.drawn_stop = round(entry * 0.985, 4)
    pos.drawn_time = 240.0
    pos.expected_net_target_per_share_cents = 1.8
    pos.expected_net_target_minus_one_tick_per_share_cents = 0.7
    pos.time_stop_delay_seconds = 0.0
    pos.time_stop_suppression_count = 0
    pos.exit_price_minus_drawn_stop_cents = round((exit_p - pos.drawn_stop) * 100.0, 4)
    return pos


class TestTradeStore:
    @pytest.fixture
    def store(self, tmp_path):
        return TradeStore(tmp_path / "test_trades.db")

    @pytest.mark.asyncio
    async def test_record_and_stats(self, store):
        await store.init()

        # Record 3 winning trades and 1 losing trade
        await store.record(_make_position("P1", 0.45, 0.55))  # +100¢
        await store.record(_make_position("P2", 0.40, 0.52))  # +120¢
        await store.record(_make_position("P3", 0.50, 0.58))  # +80¢
        await store.record(_make_position("P4", 0.48, 0.40, reason="timeout"))  # -80¢

        stats = await store.get_stats()
        assert stats["total_trades"] == 4
        assert stats["win_rate"] == 0.75
        assert stats["total_pnl_cents"] > 0
        assert stats["target_exits"] == 3
        assert stats["timeout_exits"] == 1

        await store.close()

    @pytest.mark.asyncio
    async def test_record_persists_fill_time_toxicity(self, store):
        await store.init()

        pos = _make_position("P_TOX", 0.45, 0.55)
        pos.entry_toxicity_index = 0.87
        pos.exit_toxicity_index = 0.24
        pos.drawn_tp = 0.56
        pos.drawn_stop = 0.441
        pos.drawn_time = 245.0
        pos.expected_net_target_per_share_cents = 2.4
        pos.expected_net_target_minus_one_tick_per_share_cents = 1.1
        pos.time_stop_delay_seconds = 19.0
        pos.time_stop_suppression_count = 3
        pos.exit_price_minus_drawn_stop_cents = 10.9
        pos.signal = None

        await store.record(pos)

        cursor = await store._db.execute(
            "SELECT entry_toxicity_index, exit_toxicity_index, drawn_tp, drawn_stop, drawn_time, "
            "expected_net_target_per_share_cents, expected_net_target_minus_one_tick_per_share_cents, "
            "time_stop_delay_seconds, time_stop_suppression_count, exit_price_minus_drawn_stop_cents "
            "FROM trades WHERE id = ?",
            ("P_TOX",),
        )
        row = await cursor.fetchone()

        assert row is not None
        assert row[0] == pytest.approx(0.87)
        assert row[1] == pytest.approx(0.24)
        assert row[2] == pytest.approx(0.56)
        assert row[3] == pytest.approx(0.441)
        assert row[4] == pytest.approx(245.0)
        assert row[5] == pytest.approx(2.4)
        assert row[6] == pytest.approx(1.1)
        assert row[7] == pytest.approx(19.0)
        assert row[8] == 3
        assert row[9] == pytest.approx(10.9)

        await store.close()

    @pytest.mark.asyncio
    async def test_go_live_criteria_not_met_few_trades(self, store):
        await store.init()
        await store.record(_make_position("P1", 0.45, 0.55))
        ready, stats = await store.passes_go_live_criteria()
        assert ready is False  # <20 trades
        await store.close()

    @pytest.mark.asyncio
    async def test_empty_stats(self, store):
        await store.init()
        stats = await store.get_stats()
        assert stats["total_trades"] == 0
        await store.close()

    @pytest.mark.asyncio
    async def test_shadow_trade_persists_reverse_ofi_metadata(self, store):
        await store.init()

        tracker = ShadowExecutionTracker(trade_store=store)
        pos = tracker.open_shadow_position(
            signal_source="OFI_REVERSE_SHADOW",
            market_id="MKT-SHADOW",
            asset_id="YES-SHADOW",
            direction="YES",
            best_ask=0.12,
            best_bid=0.10,
            entry_price=0.12,
            entry_size=125.0,
            zscore=0.91,
            confidence=0.91,
            reference_price=0.90,
            reference_price_band="0.80-0.95",
            toxicity_index=0.33,
        )

        assert pos is not None

        closed = await tracker.tick("YES-SHADOW", best_bid=0.17, best_ask=0.18)

        assert len(closed) == 1

        cursor = await store._db.execute(
            "SELECT signal_source, market_id, asset_id, direction, reference_price, "
            "reference_price_band, toxicity_index, confidence, zscore "
            "FROM shadow_trades WHERE id = ?",
            (pos.id,),
        )
        row = await cursor.fetchone()

        assert row is not None
        assert row[0] == "OFI_REVERSE_SHADOW"
        assert row[1] == "MKT-SHADOW"
        assert row[2] == "YES-SHADOW"
        assert row[3] == "YES"
        assert row[4] == pytest.approx(0.90, abs=1e-6)
        assert row[5] == "0.80-0.95"
        assert row[6] == pytest.approx(0.33, abs=1e-6)
        assert row[7] == pytest.approx(0.91, abs=1e-6)
        assert row[8] == pytest.approx(0.91, abs=1e-6)

        await store.close()

    @pytest.mark.asyncio
    async def test_get_stats_filters_by_signal_type(self, store):
        await store.init()

        panic_a = _make_position("PANIC-1", 0.45, 0.55)
        panic_a.signal_type = "panic"
        panic_b = _make_position("PANIC-2", 0.40, 0.50)
        panic_b.signal_type = "panic"
        ofi = _make_ofi_position(
            "OFI-1",
            0.50,
            0.44,
            toxicity_index=0.35,
        )

        await store.record(panic_a)
        await store.record(panic_b)
        await store.record(ofi)

        panic_stats = await store.get_stats(signal_type="panic")
        ofi_stats = await store.get_stats(signal_type="ofi_momentum")
        all_stats = await store.get_stats()

        assert panic_stats["total_trades"] == 2
        assert panic_stats["win_rate"] == 1.0
        assert panic_stats["total_pnl_cents"] == pytest.approx(200.0)
        assert ofi_stats["total_trades"] == 1
        assert ofi_stats["win_rate"] == 0.0
        assert ofi_stats["total_pnl_cents"] == pytest.approx(-60.0)
        assert all_stats["total_trades"] == 3

        await store.close()

    @pytest.mark.asyncio
    async def test_live_trade_persistence_journal_reconciles_to_ledger(self, store):
        await store.init()

        pos = _make_position("P-JOURNAL", 0.45, 0.55)
        pos.signal_type = "panic"
        await store.record(pos)

        accounting = await store.get_persistence_accounting_summary(
            ledger_kind="trades",
            signal_source="panic",
        )

        assert accounting["journal_rows"] == 1
        assert accounting["matched_ledger_rows"] == 1
        assert accounting["missing_ledger_rows"] == 0
        assert accounting["accounting_complete"] is True

        cursor = await store._db.execute(
            "SELECT ledger_state, signal_source FROM trade_persistence_journal WHERE journal_key = ?",
            ("trades:P-JOURNAL",),
        )
        row = await cursor.fetchone()
        assert row == ("RECORDED", "panic")

        await store.close()

    @pytest.mark.asyncio
    async def test_accounting_summary_detects_missing_live_ledger_row(self, store):
        await store.init()

        pos = _make_position("P-MISSING", 0.45, 0.55)
        pos.signal_type = "panic"
        await store.record(pos)
        await store._db.execute("DELETE FROM trades WHERE id = ?", ("P-MISSING",))
        await store._db.commit()

        accounting = await store.get_persistence_accounting_summary(
            ledger_kind="trades",
            signal_source="panic",
        )

        assert accounting["journal_rows"] == 1
        assert accounting["missing_ledger_rows"] == 1
        assert accounting["accounting_complete"] is False
        assert accounting["sample_missing_ledger_ids"] == ["P-MISSING"]

        await store.close()

    @pytest.mark.asyncio
    async def test_wal_safe_snapshot_exports_manifest_and_journal(self, store, tmp_path):
        await store.init()

        pos = _make_position("P-SNAPSHOT", 0.45, 0.55)
        pos.signal_type = "panic"
        await store.record(pos)
        await store.record_shadow_trade(
            trade_id="SH-SNAPSHOT",
            signal_source="OFI_REVERSE_SHADOW",
            market_id="MKT-SNAPSHOT",
            asset_id="YES-SNAPSHOT",
            direction="YES",
            reference_price=0.84,
            reference_price_band="0.80-0.95",
            entry_price=0.84,
            entry_size=10.0,
            entry_time=time.time() - 120,
            target_price=0.89,
            stop_price=0.80,
            exit_price=0.89,
            exit_time=time.time(),
            exit_reason="target",
            pnl_cents=50.0,
            confidence=0.72,
            zscore=2.1,
            toxicity_index=0.33,
        )

        manifest = await store.create_wal_safe_remeasurement_snapshot(
            label="unit_test_snapshot",
            output_dir=tmp_path / "snapshots",
        )

        snapshot_db_path = Path(str(manifest["snapshot_db_path"]))
        journal_capture_path = Path(str(manifest["journal_capture_path"]))
        manifest_path = Path(str(manifest["manifest_path"]))

        assert snapshot_db_path.exists()
        assert journal_capture_path.exists()
        assert manifest_path.exists()

        persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert persisted_manifest["accounting"]["trades"]["journal_rows"] == 1
        assert persisted_manifest["accounting"]["shadow_trades"]["journal_rows"] == 1

        snapshot_conn = sqlite3.connect(str(snapshot_db_path))
        try:
            trade_count = snapshot_conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            shadow_count = snapshot_conn.execute("SELECT COUNT(*) FROM shadow_trades").fetchone()[0]
        finally:
            snapshot_conn.close()

        assert trade_count == 1
        assert shadow_count == 1

        journal_lines = [line for line in journal_capture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(journal_lines) == 2

        await store.close()


class TestPositionManagerStatsCache:
    @pytest.fixture
    def store(self, tmp_path):
        return TradeStore(tmp_path / "test_trades.db")

    @pytest.mark.asyncio
    async def test_cached_stats_are_scoped_per_signal_type(self):
        executor = OrderExecutor(paper_mode=True)
        trade_store = AsyncMock()
        trade_store.get_stats.side_effect = [
            {"total_trades": 7, "win_rate": 0.6},
            {"total_trades": 2, "win_rate": 0.5},
        ]
        pm = PositionManager(executor, trade_store=trade_store, max_open_positions=10)

        panic_stats = await pm._get_cached_stats("panic")
        rpe_stats = await pm._get_cached_stats("rpe")

        assert panic_stats["total_trades"] == 7
        assert rpe_stats["total_trades"] == 2
        assert trade_store.get_stats.await_args_list[0].kwargs == {"signal_type": "panic"}
        assert trade_store.get_stats.await_args_list[1].kwargs == {"signal_type": "rpe"}

    @pytest.mark.asyncio
    async def test_get_ofi_toxicity_pnl_summary_buckets_pnl_and_fee_drag(self, store):
        await store.init()

        low_bucket = _make_ofi_position(
            "OFI-LOW",
            0.50,
            0.55,
            toxicity_index=0.12,
            entry_fee_bps=100,
            exit_fee_bps=0,
        )
        mid_bucket_win = _make_ofi_position(
            "OFI-MID-WIN",
            0.45,
            0.50,
            toxicity_index=0.44,
            entry_fee_bps=100,
            exit_fee_bps=50,
        )
        mid_bucket_loss = _make_ofi_position(
            "OFI-MID-LOSS",
            0.48,
            0.44,
            toxicity_index=0.47,
            entry_fee_bps=0,
            exit_fee_bps=100,
        )
        ignored = _make_position("PANIC-IGNORE", 0.40, 0.42)
        ignored.signal_type = "panic"
        ignored.entry_toxicity_index = 0.92
        ignored.entry_fee_bps = 100
        ignored.exit_fee_bps = 100

        await store.record(low_bucket)
        await store.record(mid_bucket_win)
        await store.record(mid_bucket_loss)
        await store.record(ignored)

        summary = await store.get_ofi_toxicity_pnl_summary(buckets=5)

        assert len(summary) == 2

        low = summary[0]
        assert low["bucket_index"] == 0
        assert low["trade_count"] == 1
        assert low["win_rate"] == pytest.approx(1.0)
        assert low["avg_net_pnl_cents"] == pytest.approx(50.0)
        assert low["total_taker_fee_drag_cents"] == pytest.approx(5.0)

        mid = summary[1]
        assert mid["bucket_index"] == 2
        assert mid["trade_count"] == 2
        assert mid["win_rate"] == pytest.approx(0.5)
        assert mid["avg_net_pnl_cents"] == pytest.approx(5.0)
        assert mid["total_net_pnl_cents"] == pytest.approx(10.0)
        assert mid["total_taker_fee_drag_cents"] == pytest.approx(11.4)

        await store.close()

    @pytest.mark.asyncio
    async def test_get_ofi_toxicity_pnl_summary_empty_when_no_ofi_trades(self, store):
        await store.init()
        await store.record(_make_position("PANIC-1", 0.45, 0.55))

        summary = await store.get_ofi_toxicity_pnl_summary()

        assert summary == []
        await store.close()

    @pytest.mark.asyncio
    async def test_get_stochastic_execution_slippage_groups_ofi_exits(self, store):
        await store.init()

        target_trade = _make_ofi_position(
            "OFI-TARGET",
            0.50,
            0.56,
            toxicity_index=0.20,
        )
        target_trade.drawn_tp = 0.55
        target_trade.drawn_stop = 0.49
        target_trade.exit_reason = "target"

        stop_trade = _make_ofi_position(
            "OFI-STOP",
            0.50,
            0.48,
            toxicity_index=0.35,
        )
        stop_trade.drawn_tp = 0.54
        stop_trade.drawn_stop = 0.49
        stop_trade.exit_reason = "stop_loss"

        time_stop_trade = _make_ofi_position(
            "OFI-TIMESTOP",
            0.50,
            0.53,
            toxicity_index=0.52,
        )
        time_stop_trade.drawn_tp = 0.57
        time_stop_trade.drawn_stop = 0.48
        time_stop_trade.exit_reason = "time_stop"

        await store.record(target_trade)
        await store.record(stop_trade)
        await store.record(time_stop_trade)

        summary = await store.get_stochastic_execution_slippage()

        assert [row["exit_bucket"] for row in summary] == ["target", "stop", "time_stop"]

        target = summary[0]
        assert target["trade_count"] == 1
        assert target["avg_exit_minus_drawn_tp_cents"] == pytest.approx(1.0)
        assert target["avg_exit_minus_drawn_stop_cents"] == pytest.approx(7.0)
        assert target["avg_reference_slippage_cents"] == pytest.approx(1.0)
        assert target["avg_abs_reference_slippage_cents"] == pytest.approx(1.0)

        stop = summary[1]
        assert stop["trade_count"] == 1
        assert stop["avg_exit_minus_drawn_tp_cents"] == pytest.approx(-6.0)
        assert stop["avg_exit_minus_drawn_stop_cents"] == pytest.approx(-1.0)
        assert stop["avg_reference_slippage_cents"] == pytest.approx(-1.0)
        assert stop["avg_abs_reference_slippage_cents"] == pytest.approx(1.0)

        time_stop = summary[2]
        assert time_stop["trade_count"] == 1
        assert time_stop["avg_exit_minus_drawn_tp_cents"] == pytest.approx(-4.0)
        assert time_stop["avg_exit_minus_drawn_stop_cents"] == pytest.approx(5.0)
        assert time_stop["avg_reference_slippage_cents"] == pytest.approx(0.0)
        assert time_stop["avg_abs_reference_slippage_cents"] == pytest.approx(0.0)

        await store.close()

    @pytest.mark.asyncio
    async def test_get_shadow_cohort_report_groups_shadow_candidates(self, store):
        await store.init()

        base_time = time.time()
        await store.record_shadow_trade(
            trade_id="SH-1",
            signal_source="OFI_REVERSE_SHADOW",
            market_id="MKT-1",
            asset_id="YES-1",
            direction="YES",
            reference_price=0.84,
            reference_price_band="0.80-0.95",
            entry_price=0.84,
            entry_size=10.0,
            entry_time=base_time - 120,
            target_price=0.89,
            stop_price=0.80,
            exit_price=0.89,
            exit_time=base_time,
            exit_reason="target",
            pnl_cents=50.0,
            confidence=0.72,
            zscore=2.1,
            toxicity_index=0.33,
        )
        await store.record_shadow_trade(
            trade_id="SH-2",
            signal_source="OFI_REVERSE_SHADOW",
            market_id="MKT-2",
            asset_id="YES-2",
            direction="YES",
            reference_price=0.92,
            reference_price_band="0.80-0.95",
            entry_price=0.96,
            entry_size=10.0,
            entry_time=base_time - 420,
            target_price=0.98,
            stop_price=0.92,
            exit_price=0.92,
            exit_time=base_time - 60,
            exit_reason="stop_loss",
            pnl_cents=-20.0,
            confidence=0.61,
            zscore=1.8,
            toxicity_index=0.41,
        )
        await store.record_shadow_trade(
            trade_id="SH-3",
            signal_source="SI-3_CrossMarket",
            market_id="MKT-1",
            asset_id="NO-1",
            direction="NO",
            entry_price=0.35,
            entry_size=10.0,
            entry_time=base_time - 1000,
            target_price=0.40,
            stop_price=0.31,
            exit_price=0.40,
            exit_time=base_time - 100,
            exit_reason="target",
            pnl_cents=30.0,
            confidence=0.45,
            zscore=2.7,
        )

        report = await store.get_shadow_cohort_report()

        assert report["summary"]["total_trades"] == 3
        assert report["summary"]["expectancy_cents"] == pytest.approx(20.0)
        assert report["summary"]["avg_reference_price"] == pytest.approx(0.88)
        assert report["summary"]["avg_toxicity_index"] == pytest.approx(0.2467)
        assert report["summary"]["gross_move_cents_total"] == pytest.approx(60.0)
        assert report["summary"]["fee_burden_cents_total"] == pytest.approx(0.0)
        assert report["summary"]["accounting_complete"] is True
        assert report["accounting"]["journal_rows"] == 3
        assert report["exit_mix"]["target"]["trades"] == 2
        assert report["loss_buckets_to_beat"]["stop_loss"]["trades"] == 1
        assert set(report["by_signal_source"].keys()) == {"OFI_REVERSE_SHADOW", "SI-3_CrossMarket"}
        assert report["by_signal_source"]["OFI_REVERSE_SHADOW"]["total_trades"] == 2
        assert report["by_direction"]["YES"]["total_trades"] == 2
        assert report["by_direction"]["NO"]["total_trades"] == 1
        assert report["by_entry_price"]["0.80-0.95"]["total_trades"] == 1
        assert report["by_entry_price"]["0.95-1.00"]["total_trades"] == 1
        assert report["by_entry_price"]["0.20-0.40"]["total_trades"] == 1
        assert report["by_confidence"]["0.60-0.80"]["total_trades"] == 2
        assert report["by_confidence"]["0.40-0.60"]["total_trades"] == 1
        assert report["by_reference_price"]["0.80-0.95"]["total_trades"] == 2
        assert report["by_reference_price_band"]["0.80-0.95"]["total_trades"] == 2
        assert report["by_reference_price_band"]["n/a"]["total_trades"] == 1
        assert report["by_toxicity"]["0.20-0.40"]["total_trades"] == 1
        assert report["by_toxicity"]["0.40-0.60"]["total_trades"] == 1
        assert report["by_toxicity"]["0.00"]["total_trades"] == 1
        assert report["top_markets"][0]["market_id"] == "MKT-1"
        assert report["top_markets"][0]["total_trades"] == 2

        filtered = await store.get_shadow_cohort_report(signal_source="OFI_REVERSE_SHADOW")
        assert filtered["signal_source"] == "OFI_REVERSE_SHADOW"
        assert filtered["summary"]["total_trades"] == 2
        assert set(filtered["by_signal_source"].keys()) == {"OFI_REVERSE_SHADOW"}

        await store.close()

    @pytest.mark.asyncio
    async def test_get_shadow_source_overview_marks_go_live_candidates(self, store):
        await store.init()

        base_time = time.time()
        for idx in range(20):
            await store.record_shadow_trade(
                trade_id=f"READY-{idx}",
                signal_source="READY_SHADOW",
                market_id=f"READY-MKT-{idx % 3}",
                direction="YES",
                entry_price=0.70,
                entry_size=10.0,
                entry_time=base_time - (idx + 2) * 90,
                target_price=0.75,
                stop_price=0.66,
                exit_price=0.75,
                exit_time=base_time - idx * 30,
                exit_reason="target",
                pnl_cents=15.0,
                confidence=0.7,
                zscore=1.9,
            )

        for idx in range(3):
            await store.record_shadow_trade(
                trade_id=f"EARLY-{idx}",
                signal_source="EARLY_SHADOW",
                market_id="EARLY-MKT",
                direction="NO",
                entry_price=0.30,
                entry_size=10.0,
                entry_time=base_time - (idx + 2) * 60,
                target_price=0.35,
                stop_price=0.27,
                exit_price=0.27,
                exit_time=base_time - idx * 20,
                exit_reason="stop_loss",
                pnl_cents=-8.0,
                confidence=0.3,
                zscore=1.2,
            )

        overview = await store.get_shadow_source_overview()

        assert [row["signal_source"] for row in overview] == ["READY_SHADOW", "EARLY_SHADOW"]
        assert overview[0]["passes_go_live"] is True
        assert overview[0]["total_trades"] == 20
        assert overview[0]["expectancy_cents"] == pytest.approx(15.0)
        assert overview[1]["passes_go_live"] is False
        assert overview[1]["total_trades"] == 3

        await store.close()

    @pytest.mark.asyncio
    async def test_shadow_trade_extra_payload_is_normalized_in_journal(self, store):
        await store.init()

        await store.record_shadow_trade(
            trade_id="SH-EXTRA",
            signal_source="REWARD_SHADOW",
            market_id="MKT-EXTRA",
            direction="YES",
            entry_price=0.42,
            entry_size=15.0,
            entry_time=100.0,
            target_price=0.45,
            stop_price=0.39,
            exit_price=0.44,
            exit_time=190.0,
            exit_reason="target",
            pnl_cents=30.0,
            extra_payload={
                "reward_daily_usd": Decimal("12.50"),
                "quote_tags": ("maker", "reward"),
                "checkpoints": [Decimal("1.0"), {"markout": Decimal("-0.5")}],
                "legs": {"yes", "no"},
            },
        )

        cursor = await store._db.execute(
            "SELECT payload_json FROM trade_persistence_journal WHERE journal_key = ?",
            ("shadow_trades:SH-EXTRA",),
        )
        row = await cursor.fetchone()

        assert row is not None
        payload = json.loads(row[0])
        assert payload["extra_payload"]["reward_daily_usd"] == pytest.approx(12.5)
        assert payload["extra_payload"]["quote_tags"] == ["maker", "reward"]
        assert payload["extra_payload"]["checkpoints"] == [1.0, {"markout": -0.5}]
        assert payload["extra_payload"]["legs"] == ["no", "yes"]

        await store.close()

    @pytest.mark.asyncio
    async def test_shadow_trade_extra_payload_does_not_change_shadow_schema(self, store):
        await store.init()

        cursor = await store._db.execute("PRAGMA table_info(shadow_trades)")
        before_columns = [row[1] for row in await cursor.fetchall()]

        await store.record_shadow_trade(
            trade_id="SH-NO-SCHEMA",
            signal_source="REWARD_SHADOW",
            market_id="MKT-NO-SCHEMA",
            direction="YES",
            entry_price=0.40,
            entry_size=10.0,
            entry_time=10.0,
            target_price=0.43,
            stop_price=0.37,
            exit_price=0.42,
            exit_time=70.0,
            exit_reason="target",
            pnl_cents=20.0,
            extra_payload={"quote_id": "Q-123", "fill_occurred": True},
        )

        cursor = await store._db.execute("PRAGMA table_info(shadow_trades)")
        after_columns = [row[1] for row in await cursor.fetchall()]

        assert after_columns == before_columns
        assert "extra_payload" not in after_columns

        cursor = await store._db.execute("SELECT COUNT(*) FROM shadow_trades WHERE id = ?", ("SH-NO-SCHEMA",))
        assert (await cursor.fetchone())[0] == 1

        await store.close()

    @pytest.mark.asyncio
    async def test_shadow_trade_without_extra_payload_keeps_backwards_compatibility(self, store):
        await store.init()

        await store.record_shadow_trade(
            trade_id="SH-BACKCOMPAT",
            signal_source="PLAIN_SHADOW",
            market_id="MKT-BACKCOMPAT",
            direction="NO",
            entry_price=0.61,
            entry_size=9.0,
            entry_time=50.0,
            target_price=0.56,
            stop_price=0.64,
            exit_price=0.55,
            exit_time=90.0,
            exit_reason="target",
            pnl_cents=54.0,
        )

        cursor = await store._db.execute(
            "SELECT payload_json FROM trade_persistence_journal WHERE journal_key = ?",
            ("shadow_trades:SH-BACKCOMPAT",),
        )
        payload = json.loads((await cursor.fetchone())[0])

        assert "extra_payload" not in payload
        assert payload["signal_source"] == "PLAIN_SHADOW"

        await store.close()


# ── State-persistence tests ─────────────────────────────────────────────────

from src.trading.executor import Order, OrderSide, OrderStatus


class TestStatePersistence:
    @pytest.fixture
    def store(self, tmp_path):
        return TradeStore(tmp_path / "persist_test.db")

    @pytest.mark.asyncio
    async def test_checkpoint_and_restore_orders(self, store):
        await store.init()

        orders = [
            Order(
                order_id="LIVE-1",
                market_id="MKT_A",
                asset_id="NO_A",
                side=OrderSide.BUY,
                price=0.45,
                size=10.0,
                status=OrderStatus.LIVE,
                clob_order_id="CLOB-1",
            ),
            Order(
                order_id="LIVE-2",
                market_id="MKT_B",
                asset_id="NO_B",
                side=OrderSide.SELL,
                price=0.60,
                size=5.0,
                status=OrderStatus.PARTIALLY_FILLED,
                filled_size=2.0,
                filled_avg_price=0.59,
                clob_order_id="CLOB-2",
            ),
        ]

        await store.checkpoint_orders(orders)
        restored = await store.restore_orders()

        assert len(restored) == 2
        by_id = {r["order_id"]: r for r in restored}

        o1 = by_id["LIVE-1"]
        assert o1["market_id"] == "MKT_A"
        assert o1["side"] == "BUY"
        assert o1["price"] == 0.45
        assert o1["clob_order_id"] == "CLOB-1"
        assert o1["status"] == "LIVE"

        o2 = by_id["LIVE-2"]
        assert o2["filled_size"] == 2.0
        assert o2["filled_avg_price"] == 0.59

        await store.close()

    @pytest.mark.asyncio
    async def test_checkpoint_and_restore_positions(self, store):
        await store.init()

        pos = _make_position("POS-1", 0.45, 0.55)
        pos.state = PositionState.EXIT_PENDING  # simulate open position

        await store.checkpoint_positions([pos])
        restored = await store.restore_positions()

        assert len(restored) == 1
        p = restored[0]
        assert p["id"] == "POS-1"
        assert p["market_id"] == "MKT_TEST"
        assert p["state"] == "EXIT_PENDING"
        assert p["entry_price"] == 0.45
        assert p["target_price"] == 0.55
        assert p["drawn_tp"] == 0.0
        assert p["drawn_stop"] == 0.0
        assert p["drawn_time"] == 0.0

        await store.close()

    @pytest.mark.asyncio
    async def test_checkpoint_replaces_previous_data(self, store):
        await store.init()

        orders_v1 = [
            Order(
                order_id="OLD-1", market_id="M", asset_id="A",
                side=OrderSide.BUY, price=0.40, size=5.0,
                status=OrderStatus.LIVE, clob_order_id="C-1",
            ),
        ]
        await store.checkpoint_orders(orders_v1)

        orders_v2 = [
            Order(
                order_id="NEW-1", market_id="M2", asset_id="A2",
                side=OrderSide.SELL, price=0.60, size=3.0,
                status=OrderStatus.LIVE, clob_order_id="C-2",
            ),
        ]
        await store.checkpoint_orders(orders_v2)

        restored = await store.restore_orders()
        assert len(restored) == 1
        assert restored[0]["order_id"] == "NEW-1"

        await store.close()

    @pytest.mark.asyncio
    async def test_clear_live_state(self, store):
        await store.init()

        orders = [
            Order(
                order_id="X-1", market_id="M", asset_id="A",
                side=OrderSide.BUY, price=0.45, size=10.0,
                status=OrderStatus.LIVE, clob_order_id="C-X",
            ),
        ]
        pos = _make_position("POS-X", 0.45, 0.55)
        pos.state = PositionState.ENTRY_PENDING

        await store.checkpoint_orders(orders)
        await store.checkpoint_positions([pos])

        await store.clear_live_state()

        assert len(await store.restore_orders()) == 0
        assert len(await store.restore_positions()) == 0

        await store.close()

    @pytest.mark.asyncio
    async def test_restore_empty_tables(self, store):
        await store.init()
        assert await store.restore_orders() == []
        assert await store.restore_positions() == []
        await store.close()


class TestTradeStorePragmas:
    """Verify WAL-mode concurrency hardening pragmas are applied."""

    @pytest.fixture
    def store(self, tmp_path):
        return TradeStore(tmp_path / "pragma_test.db")

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, store):
        await store.init()
        async with store._db.execute("PRAGMA journal_mode") as cur:
            row = await cur.fetchone()
            assert row[0].lower() == "wal"
        await store.close()

    @pytest.mark.asyncio
    async def test_busy_timeout_set(self, store):
        await store.init()
        async with store._db.execute("PRAGMA busy_timeout") as cur:
            row = await cur.fetchone()
            assert row[0] == 5000
        await store.close()