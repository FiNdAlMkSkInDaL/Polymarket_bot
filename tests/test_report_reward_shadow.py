from __future__ import annotations

from pathlib import Path

import pytest

from scripts.report_reward_shadow import build_reward_shadow_report, load_reward_shadow_rows, render_markdown
from src.monitoring.trade_store import TradeStore


@pytest.mark.asyncio
async def test_reward_shadow_report_groups_reward_buckets(tmp_path: Path) -> None:
    store = TradeStore(tmp_path / "reward_shadow.db")
    await store.init()
    try:
        await store.record_shadow_trade(
            trade_id="RW-1",
            signal_source="REWARD_SHADOW",
            market_id="MKT-1",
            direction="YES",
            entry_price=0.45,
            entry_size=20.0,
            entry_time=100.0,
            target_price=0.48,
            stop_price=0.42,
            exit_price=0.47,
            exit_time=160.0,
            exit_reason="target",
            pnl_cents=40.0,
            extra_payload={
                "reward_daily_usd": 100.0,
                "reward_to_competition": 0.4,
                "fill_occurred": True,
                "estimated_reward_capture_usd": 1.2,
                "estimated_net_edge_usd": 0.8,
                "emergency_flatten": False,
            },
        )
        await store.record_shadow_trade(
            trade_id="RW-2",
            signal_source="REWARD_SHADOW",
            market_id="MKT-2",
            direction="NO",
            entry_price=0.60,
            entry_size=10.0,
            entry_time=200.0,
            target_price=0.56,
            stop_price=0.63,
            exit_price=0.63,
            exit_time=260.0,
            exit_reason="stop_loss",
            pnl_cents=-30.0,
            extra_payload={
                "reward_daily_usd": 100.0,
                "reward_to_competition": 1.4,
                "fill_occurred": False,
                "estimated_reward_capture_usd": 0.4,
                "estimated_net_edge_usd": 0.35,
                "emergency_flatten": True,
            },
        )
        await store.record_shadow_trade(
            trade_id="RW-3",
            signal_source="REWARD_PARTIAL",
            market_id="MKT-3",
            direction="YES",
            entry_price=0.30,
            entry_size=10.0,
            entry_time=300.0,
            target_price=0.33,
            stop_price=0.27,
            exit_price=0.32,
            exit_time=360.0,
            exit_reason="timeout",
            pnl_cents=20.0,
            extra_payload={
                "reward_daily_usd": 80.0,
                "quote_id": "Q-PARTIAL",
            },
        )
        await store.record_shadow_trade(
            trade_id="RW-4",
            signal_source="PLAIN_SHADOW",
            market_id="MKT-4",
            direction="YES",
            entry_price=0.30,
            entry_size=10.0,
            entry_time=300.0,
            target_price=0.33,
            stop_price=0.27,
            exit_price=0.33,
            exit_time=360.0,
            exit_reason="target",
            pnl_cents=30.0,
        )
    finally:
        await store.close()

    rows = await load_reward_shadow_rows(db_path=tmp_path / "reward_shadow.db")
    report = build_reward_shadow_report(rows)
    markdown = render_markdown(report)

    assert report["summary"]["total_trades"] == 3
    assert report["by_market"]["MKT-1"]["total_trades"] == 1
    assert report["by_signal_source"]["REWARD_SHADOW"]["total_trades"] == 2
    assert report["by_signal_source"]["REWARD_PARTIAL"]["total_trades"] == 1
    assert report["by_reward_to_competition_bucket"]["0.25-0.50"]["total_trades"] == 1
    assert report["by_reward_to_competition_bucket"]["1.00-2.00"]["total_trades"] == 1
    assert report["by_reward_to_competition_bucket"]["missing"]["total_trades"] == 1
    assert report["by_fill_occurred"]["true"]["total_trades"] == 1
    assert report["by_fill_occurred"]["false"]["total_trades"] == 1
    assert report["by_fill_occurred"]["missing"]["total_trades"] == 1
    assert report["by_emergency_flatten"]["false"]["total_trades"] == 1
    assert report["by_emergency_flatten"]["true"]["total_trades"] == 1
    assert report["by_emergency_flatten"]["missing"]["total_trades"] == 1
    assert "Reward Shadow Report" in markdown