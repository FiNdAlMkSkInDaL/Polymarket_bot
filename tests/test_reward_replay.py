from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

import pytest

from src.backtest.reward_replay import ReplayConfig, RewardReplayEngine, iter_replay_events, load_reward_markets
from src.data.market_discovery import MarketInfo
from src.rewards.reward_selector import RewardSelector, RewardSelectorConfig


def _market(*, end_date: datetime) -> MarketInfo:
    return MarketInfo(
        condition_id="mkt-a",
        question="Will BTC stay in range next week?",
        yes_token_id="yes-a",
        no_token_id="no-a",
        daily_volume_usd=50_000.0,
        end_date=end_date,
        active=True,
        event_id="evt-a",
        liquidity_usd=20_000.0,
        accepting_orders=True,
        tags="crypto range",
        neg_risk=False,
        reward_program_active=True,
        reward_daily_rate_usd=40.0,
        reward_min_size=5.0,
        reward_max_spread_cents=3.0,
        reward_competition_score=4.0,
    )


def test_reward_selector_static_candidates_uses_replay_timestamp() -> None:
    selector = RewardSelector()
    replay_now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    market = _market(end_date=replay_now + timedelta(days=10))

    candidates = selector.static_candidates([market], int(replay_now.timestamp() * 1000))

    assert [entry.condition_id for entry in candidates] == ["mkt-a"]


def test_reward_selector_override_relaxes_static_candidate_gate() -> None:
    selector = RewardSelector(RewardSelectorConfig(min_reward_usd=0.0, min_days_to_resolution=0, max_days_to_resolution=365))
    replay_now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    market = _market(end_date=replay_now + timedelta(days=1))
    market.reward_daily_rate_usd = 1.0

    candidates = selector.static_candidates([market], int(replay_now.timestamp() * 1000))

    assert [entry.condition_id for entry in candidates] == ["mkt-a"]


def test_iter_replay_events_dedupes_mirrored_sources(tmp_path: Path) -> None:
    day_dir = tmp_path / "2026-03-01"
    day_dir.mkdir(parents=True)
    market_path = day_dir / "mkt-a.jsonl"
    asset_path = day_dir / "yes-a.jsonl"
    market_rows = [
        {
            "source": "l2",
            "asset_id": "mkt-a",
            "payload": {
                "market": "mkt-a",
                "timestamp": "1772323200100",
                "event_type": "price_change",
                "price_changes": [
                    {"asset_id": "yes-a", "price": "0.48", "size": "25", "side": "BUY", "hash": "dup-1"}
                ],
            },
        },
        {
            "source": "trade",
            "asset_id": "mkt-a",
            "payload": {
                "market": "mkt-a",
                "timestamp": "1772323200100",
                "event_type": "price_change",
                "price_changes": [
                    {"asset_id": "yes-a", "price": "0.48", "size": "25", "side": "BUY", "hash": "dup-1"}
                ],
            },
        },
    ]
    asset_rows = [
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323200200",
                "event_type": "last_trade_price",
                "price": "0.47",
                "size": "5",
                "side": "SELL",
                "transaction_hash": "tx-1",
            },
        }
    ]
    market_path.write_text("\n".join(json.dumps(row) for row in market_rows), encoding="utf-8")
    asset_path.write_text("\n".join(json.dumps(row) for row in asset_rows), encoding="utf-8")

    events = list(iter_replay_events(tmp_path, tracked_ids=frozenset({"mkt-a", "yes-a"})))

    assert [event.event_type for event in events] == ["PRICE_CHANGE", "TRADE", "TRADE"]
    assert events[0].asset_id == "yes-a"
    assert events[1].trade_price == Decimal("0.48")


@pytest.mark.asyncio
async def test_reward_replay_engine_respects_activation_latency_and_persists_shadow_rows(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw_ticks"
    day_dir = input_dir / "2026-03-01"
    day_dir.mkdir(parents=True)
    reward_universe_path = tmp_path / "reward_universe.json"
    market_map_path = tmp_path / "market_map.json"
    db_path = tmp_path / "backtest.db"

    reward_universe_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": "mkt-a",
                    "question": "Will BTC stay in range by March 20?",
                    "volume_24h": 50_000.0,
                    "liquidity": 20_000.0,
                    "daily_reward_usd": 40.0,
                    "reward_max_spread_cents": 3.0,
                    "reward_min_size": 5.0,
                    "competition_usd": 4.0,
                    "token_audit": [
                        {"token_id": "yes-a", "outcome": "Yes"},
                        {"token_id": "no-a", "outcome": "No"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    market_map_path.write_text(
        json.dumps([{"market_id": "mkt-a", "yes_id": "yes-a", "no_id": "no-a"}]),
        encoding="utf-8",
    )

    book_rows = [
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323200000",
                "event_type": "book",
                "hash": "book-1",
                "bids": [{"price": "0.47", "size": "100"}],
                "asks": [{"price": "0.49", "size": "100"}],
            },
        },
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323200200",
                "event_type": "last_trade_price",
                "price": "0.48",
                "size": "5",
                "side": "SELL",
                "transaction_hash": "too-early",
            },
        },
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323200600",
                "event_type": "last_trade_price",
                "price": "0.48",
                "size": "5",
                "side": "SELL",
                "transaction_hash": "fill-1",
            },
        },
    ]
    (day_dir / "yes-a.jsonl").write_text("\n".join(json.dumps(row) for row in book_rows), encoding="utf-8")

    markets = load_reward_markets(
        reward_universe_path,
        replay_anchor_ms=1772323200000,
        market_map_path=market_map_path,
    )
    config = ReplayConfig(
        input_dir=input_dir,
        db_path=db_path,
        reward_universe_path=reward_universe_path,
        market_map_path=market_map_path,
        activation_latency_ms=500,
    )

    summary = await RewardReplayEngine(config, markets).run()

    assert summary.trade_events == 2
    assert summary.matched_fills == 1
    assert summary.persisted_shadow_rows >= 1
    assert db_path.exists() is True


@pytest.mark.asyncio
async def test_reward_replay_price_change_keeps_book_fresh_for_matching(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw_ticks"
    day_dir = input_dir / "2026-03-01"
    day_dir.mkdir(parents=True)
    reward_universe_path = tmp_path / "reward_universe.json"
    market_map_path = tmp_path / "market_map.json"
    db_path = tmp_path / "backtest.db"

    reward_universe_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": "mkt-a",
                    "question": "Will BTC stay in range by March 20?",
                    "volume_24h": 50_000.0,
                    "liquidity": 20_000.0,
                    "daily_reward_usd": 40.0,
                    "reward_max_spread_cents": 10.0,
                    "reward_min_size": 5.0,
                    "competition_usd": 4.0,
                    "token_audit": [
                        {"token_id": "yes-a", "outcome": "Yes"},
                        {"token_id": "no-a", "outcome": "No"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    market_map_path.write_text(
        json.dumps([{"market_id": "mkt-a", "yes_id": "yes-a", "no_id": "no-a"}]),
        encoding="utf-8",
    )

    market_rows = [
        {
            "source": "trade",
            "asset_id": "mkt-a",
            "payload": {
                "market": "mkt-a",
                "timestamp": "1772323200100",
                "event_type": "price_change",
                "price_changes": [
                    {"asset_id": "yes-a", "price": "0.48", "size": "25", "side": "BUY", "hash": "px-1"}
                ],
            },
        }
    ]
    asset_rows = [
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323200000",
                "event_type": "book",
                "hash": "book-1",
                "bids": [{"price": "0.47", "size": "100"}],
                "asks": [{"price": "0.49", "size": "100"}],
            },
        },
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323200600",
                "event_type": "last_trade_price",
                "price": "0.48",
                "size": "5",
                "side": "SELL",
                "transaction_hash": "fill-1",
            },
        },
    ]
    (day_dir / "mkt-a.jsonl").write_text("\n".join(json.dumps(row) for row in market_rows), encoding="utf-8")
    (day_dir / "yes-a.jsonl").write_text("\n".join(json.dumps(row) for row in asset_rows), encoding="utf-8")

    markets = load_reward_markets(
        reward_universe_path,
        replay_anchor_ms=1772323200000,
        market_map_path=market_map_path,
    )
    config = ReplayConfig(
        input_dir=input_dir,
        db_path=db_path,
        reward_universe_path=reward_universe_path,
        market_map_path=market_map_path,
        activation_latency_ms=500,
        selector_config=RewardSelectorConfig(max_spread_cents=10.0),
        reward_cancel_on_stale_ms=300_000,
        reward_replace_only_if_price_moves_ticks=10_000,
        reward_refresh_interval_ms=3_600_000,
    )

    summary = await RewardReplayEngine(config, markets).run()

    assert summary.trade_events == 2
    assert summary.matched_fills == 1
    assert summary.persisted_shadow_rows >= 1


@pytest.mark.asyncio
async def test_reward_replay_releases_inventory_after_flatten_delay(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw_ticks"
    day_dir = input_dir / "2026-03-01"
    day_dir.mkdir(parents=True)
    reward_universe_path = tmp_path / "reward_universe.json"
    market_map_path = tmp_path / "market_map.json"
    db_path = tmp_path / "backtest.db"

    reward_universe_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": "mkt-a",
                    "question": "Will BTC stay in range by March 20?",
                    "volume_24h": 50_000.0,
                    "liquidity": 20_000.0,
                    "daily_reward_usd": 40.0,
                    "reward_max_spread_cents": 10.0,
                    "reward_min_size": 5.0,
                    "competition_usd": 4.0,
                    "token_audit": [
                        {"token_id": "yes-a", "outcome": "Yes"},
                        {"token_id": "no-a", "outcome": "No"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    market_map_path.write_text(
        json.dumps([{"market_id": "mkt-a", "yes_id": "yes-a", "no_id": "no-a"}]),
        encoding="utf-8",
    )

    asset_rows = [
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323200000",
                "event_type": "book",
                "hash": "book-1",
                "bids": [{"price": "0.47", "size": "100"}],
                "asks": [{"price": "0.49", "size": "100"}],
            },
        },
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323200600",
                "event_type": "last_trade_price",
                "price": "0.48",
                "size": "5",
                "side": "SELL",
                "transaction_hash": "fill-1",
            },
        },
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323232000",
                "event_type": "book",
                "hash": "book-2",
                "bids": [{"price": "0.47", "size": "100"}],
                "asks": [{"price": "0.49", "size": "100"}],
            },
        },
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323232600",
                "event_type": "last_trade_price",
                "price": "0.48",
                "size": "5",
                "side": "SELL",
                "transaction_hash": "fill-2",
            },
        },
    ]
    (day_dir / "yes-a.jsonl").write_text("\n".join(json.dumps(row) for row in asset_rows), encoding="utf-8")

    markets = load_reward_markets(
        reward_universe_path,
        replay_anchor_ms=1772323200000,
        market_map_path=market_map_path,
    )
    config = ReplayConfig(
        input_dir=input_dir,
        db_path=db_path,
        reward_universe_path=reward_universe_path,
        market_map_path=market_map_path,
        activation_latency_ms=500,
        reward_cancel_on_stale_ms=300_000,
        reward_replace_only_if_price_moves_ticks=10_000,
        reward_refresh_interval_ms=3_600_000,
    )

    summary = await RewardReplayEngine(config, markets).run()

    assert summary.trade_events == 2
    assert summary.matched_fills == 2


@pytest.mark.asyncio
async def test_reward_replay_populates_forward_markouts_and_net_edge(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw_ticks"
    day_dir = input_dir / "2026-03-01"
    day_dir.mkdir(parents=True)
    reward_universe_path = tmp_path / "reward_universe.json"
    market_map_path = tmp_path / "market_map.json"
    db_path = tmp_path / "backtest.db"

    reward_universe_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": "mkt-a",
                    "question": "Will BTC stay in range by March 20?",
                    "volume_24h": 50_000.0,
                    "liquidity": 20_000.0,
                    "daily_reward_usd": 40.0,
                    "reward_max_spread_cents": 10.0,
                    "reward_min_size": 5.0,
                    "competition_usd": 4.0,
                    "token_audit": [
                        {"token_id": "yes-a", "outcome": "Yes"},
                        {"token_id": "no-a", "outcome": "No"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    market_map_path.write_text(
        json.dumps([{"market_id": "mkt-a", "yes_id": "yes-a", "no_id": "no-a"}]),
        encoding="utf-8",
    )

    asset_rows = [
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323200000",
                "event_type": "book",
                "hash": "book-1",
                "bids": [{"price": "0.47", "size": "100"}],
                "asks": [{"price": "0.49", "size": "100"}],
            },
        },
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323200600",
                "event_type": "last_trade_price",
                "price": "0.48",
                "size": "5",
                "side": "SELL",
                "transaction_hash": "fill-1",
            },
        },
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323205800",
                "event_type": "book",
                "hash": "book-2",
                "bids": [{"price": "0.45", "size": "100"}],
                "asks": [{"price": "0.47", "size": "100"}],
            },
        },
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323215800",
                "event_type": "book",
                "hash": "book-3",
                "bids": [{"price": "0.44", "size": "100"}],
                "asks": [{"price": "0.46", "size": "100"}],
            },
        },
        {
            "source": "l2",
            "asset_id": "yes-a",
            "payload": {
                "market": "mkt-a",
                "asset_id": "yes-a",
                "timestamp": "1772323265800",
                "event_type": "book",
                "hash": "book-4",
                "bids": [{"price": "0.42", "size": "100"}],
                "asks": [{"price": "0.44", "size": "100"}],
            },
        },
    ]
    (day_dir / "yes-a.jsonl").write_text("\n".join(json.dumps(row) for row in asset_rows), encoding="utf-8")

    markets = load_reward_markets(
        reward_universe_path,
        replay_anchor_ms=1772323200000,
        market_map_path=market_map_path,
    )
    config = ReplayConfig(
        input_dir=input_dir,
        db_path=db_path,
        reward_universe_path=reward_universe_path,
        market_map_path=market_map_path,
        activation_latency_ms=500,
        reward_cancel_on_stale_ms=300_000,
        reward_replace_only_if_price_moves_ticks=10_000,
        reward_refresh_interval_ms=3_600_000,
    )

    summary = await RewardReplayEngine(config, markets).run()

    assert summary.matched_fills == 1
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "select json_extract(payload_json, '$.extra_payload.markout_5s_cents'), "
        "json_extract(payload_json, '$.extra_payload.markout_15s_cents'), "
        "json_extract(payload_json, '$.extra_payload.markout_60s_cents'), "
        "json_extract(payload_json, '$.extra_payload.estimated_reward_capture_usd'), "
        "json_extract(payload_json, '$.extra_payload.estimated_net_edge_usd') "
        "from trade_persistence_journal where ledger_kind='shadow_trades' and signal_source='REWARD_FILLED'"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == pytest.approx(-2.0)
    assert row[1] == pytest.approx(-3.0)
    assert row[2] == pytest.approx(-5.0)
    assert row[3] is not None
    assert row[4] is not None
    assert row[4] < row[3]