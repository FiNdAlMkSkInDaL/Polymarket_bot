from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import sqlite3

import pytest

from src.backtest.multi_market_streamer import iter_multiplexed_market_records
from src.execution.priority_context import PriorityOrderContext
from src.signals.lead_lag_maker import LeadLagMaker
from src.signals.base_strategy import BaseStrategy
from scripts.run_universal_backtest import MarketCatalog, MarketTokens, UniversalReplayEngine, load_strategy


class RecordAndFireStrategy(BaseStrategy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bbo_updates: list[tuple[str, list[dict], list[dict]]] = []
        self.trades: list[dict] = []
        self.sent = False

    def on_bbo_update(self, market_id: str, top_bids, top_asks) -> None:
        self.bbo_updates.append((market_id, list(top_bids), list(top_asks)))
        if self.sent or not top_bids:
            return
        context = PriorityOrderContext(
            market_id=market_id,
            side="YES",
            signal_source="MANUAL",
            conviction_scalar=Decimal("1"),
            target_price=Decimal("0.45"),
            anchor_volume=Decimal("5"),
            max_capital=Decimal("2.25"),
        )
        self.submit_order(context)
        self.sent = True

    def on_trade(self, market_id: str, trade_data: dict[str, object]) -> None:
        self.trades.append({"market_id": market_id, **trade_data})

    def on_tick(self) -> None:
        return None


class PassiveBidStrategy(BaseStrategy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sent = False

    def on_bbo_update(self, market_id: str, top_bids, top_asks) -> None:
        if self.sent:
            return
        context = PriorityOrderContext(
            market_id=market_id,
            side="YES",
            signal_source="MANUAL",
            conviction_scalar=Decimal("1"),
            target_price=Decimal("0.44"),
            anchor_volume=Decimal("5"),
            max_capital=Decimal("2.20"),
            signal_metadata={"post_only": True, "liquidity_intent": "maker"},
        )
        self.submit_order(context)
        self.sent = True

    def on_trade(self, market_id: str, trade_data: dict[str, object]) -> None:
        del market_id, trade_data

    def on_tick(self) -> None:
        return None


class PassiveAskStrategy(BaseStrategy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sent = False

    def on_bbo_update(self, market_id: str, top_bids, top_asks) -> None:
        if self.sent:
            return
        context = PriorityOrderContext(
            market_id=market_id,
            side="YES",
            signal_source="MANUAL",
            conviction_scalar=Decimal("1"),
            target_price=Decimal("0.45"),
            anchor_volume=Decimal("5"),
            max_capital=Decimal("2.25"),
            signal_metadata={"post_only": True, "liquidity_intent": "maker", "quote_side": "ASK", "quote_id": "ask-1"},
        )
        self.submit_order(context)
        self.sent = True

    def on_trade(self, market_id: str, trade_data: dict[str, object]) -> None:
        del market_id, trade_data

    def on_tick(self) -> None:
        return None


class ToxicCancelStrategy(BaseStrategy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stage = 0

    def on_bbo_update(self, market_id: str, top_bids, top_asks) -> None:
        if self.stage == 0:
            self.submit_order(
                PriorityOrderContext(
                    market_id=market_id,
                    side="YES",
                    signal_source="MANUAL",
                    conviction_scalar=Decimal("1"),
                    target_price=Decimal("0.44"),
                    anchor_volume=Decimal("5"),
                    max_capital=Decimal("2.20"),
                    signal_metadata={
                        "strategy": "tox-cancel",
                        "post_only": True,
                        "liquidity_intent": "maker",
                        "quote_id": "tox-bid",
                    },
                )
            )
            self.stage = 1
            return
        if self.stage == 1:
            self.submit_order(
                PriorityOrderContext(
                    market_id=market_id,
                    side="YES",
                    signal_source="MANUAL",
                    conviction_scalar=Decimal("1"),
                    target_price=Decimal("0.01"),
                    anchor_volume=Decimal("1"),
                    max_capital=Decimal("0.01"),
                    signal_metadata={"strategy": "tox-cancel", "action": "CANCEL_ALL"},
                )
            )
            self.stage = 2

    def on_trade(self, market_id: str, trade_data: dict[str, object]) -> None:
        del market_id, trade_data

    def on_tick(self) -> None:
        return None


class MarketSequenceStrategy(BaseStrategy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.market_sequence: list[str] = []

    def on_bbo_update(self, market_id: str, top_bids, top_asks) -> None:
        del top_bids, top_asks
        self.market_sequence.append(market_id)

    def on_trade(self, market_id: str, trade_data: dict[str, object]) -> None:
        del market_id, trade_data

    def on_tick(self) -> None:
        return None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_multi_market_streamer_yields_time_ordered_market_records(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    _write_jsonl(
        primary,
        [
            {
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1000",
                    "event_type": "book",
                }
            },
            {
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "3000",
                    "event_type": "book",
                }
            },
        ],
    )
    _write_jsonl(
        secondary,
        [
            {
                "payload": {
                    "market": "mkt-2",
                    "asset_id": "yes-2",
                    "timestamp": "2000",
                    "event_type": "book",
                }
            },
            {
                "payload": {
                    "market": "mkt-2",
                    "asset_id": "yes-2",
                    "timestamp": "3000",
                    "event_type": "book",
                }
            },
        ],
    )

    records = list(iter_multiplexed_market_records(primary, secondary))

    assert [market_id for market_id, _ in records] == ["mkt-1", "mkt-2", "mkt-1", "mkt-2"]
    assert [str(record["payload"]["timestamp"]) for _, record in records] == ["1000", "2000", "3000", "3000"]


def test_load_strategy_binds_dispatcher_and_catalog() -> None:
    catalog = MarketCatalog([MarketTokens(market_id="mkt-1", yes_asset_id="yes-1", no_asset_id="no-1")])
    strategy = load_strategy(
        "tests.test_universal_backtest.RecordAndFireStrategy",
        dispatcher=object(),
        market_catalog=catalog,
        strategy_config={},
        clock=lambda: 123,
    )

    assert isinstance(strategy, RecordAndFireStrategy)
    assert strategy.market_catalog is catalog
    assert strategy.current_timestamp_ms == 123


def test_load_strategy_expands_strategy_config_kwargs() -> None:
    catalog = MarketCatalog([MarketTokens(market_id="primary", yes_asset_id="yes-primary", no_asset_id="no-primary")])
    strategy = load_strategy(
        "src.signals.lead_lag_maker.LeadLagMaker",
        dispatcher=object(),
        market_catalog=catalog,
        strategy_config={
            "primary_market_id": "primary",
            "secondary_market_id": "secondary",
            "cooldown_ms": 2_500,
        },
        clock=lambda: 456,
    )

    assert isinstance(strategy, LeadLagMaker)
    assert strategy.primary_market_id == "primary"
    assert strategy.secondary_market_id == "secondary"
    assert strategy.cooldown_ms == 2_500


@pytest.mark.asyncio
async def test_universal_replay_engine_executes_taker_and_persists_shadow_trade(tmp_path: Path) -> None:
    raw_root = tmp_path / "logs" / "local_snapshot" / "l2_data" / "data" / "raw_ticks" / "2026-03-01"
    raw_root.mkdir(parents=True)
    market_map = tmp_path / "market_map.json"
    db_path = tmp_path / "universal_backtest.db"
    market_map.write_text(
        json.dumps([{"market_id": "mkt-1", "yes_id": "yes-1", "no_id": "no-1"}]),
        encoding="utf-8",
    )
    _write_jsonl(
        raw_root / "yes-1.jsonl",
        [
            {
                "local_ts": 1772323200.0,
                "source": "l2",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323200000",
                    "event_type": "book",
                    "bids": [{"price": "0.44", "size": "20"}],
                    "asks": [{"price": "0.45", "size": "10"}],
                },
            }
        ],
    )

    engine = UniversalReplayEngine(
        input_dir=tmp_path / "logs" / "local_snapshot" / "l2_data",
        db_path=db_path,
        strategy_path="tests.test_universal_backtest.RecordAndFireStrategy",
        market_catalog=MarketCatalog([MarketTokens(market_id="mkt-1", yes_asset_id="yes-1", no_asset_id="no-1")]),
    )

    summary = await engine.run()

    assert summary.dispatches == 1
    assert summary.taker_fills == 1
    assert summary.persisted_shadow_rows == 1
    assert db_path.exists() is True


@pytest.mark.asyncio
async def test_universal_replay_engine_matches_passive_order_on_trade(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_ticks" / "2026-03-01"
    raw_root.mkdir(parents=True)
    db_path = tmp_path / "universal_backtest.db"
    _write_jsonl(
        raw_root / "yes-1.jsonl",
        [
            {
                "local_ts": 1772323200.0,
                "source": "l2",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323200000",
                    "event_type": "book",
                    "bids": [{"price": "0.43", "size": "20"}],
                    "asks": [{"price": "0.45", "size": "20"}],
                },
            },
            {
                "local_ts": 1772323201.0,
                "source": "trade",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323201000",
                    "event_type": "last_trade_price",
                    "price": "0.44",
                    "size": "5",
                    "side": "SELL",
                    "transaction_hash": "tx-1",
                },
            },
        ],
    )

    engine = UniversalReplayEngine(
        input_dir=tmp_path / "raw_ticks",
        db_path=db_path,
        strategy_path="tests.test_universal_backtest.PassiveBidStrategy",
        market_catalog=MarketCatalog([MarketTokens(market_id="mkt-1", yes_asset_id="yes-1", no_asset_id="no-1")]),
    )

    summary = await engine.run()

    assert summary.dispatches == 1
    assert summary.maker_fills == 1
    assert summary.persisted_shadow_rows == 1
    assert summary.open_orders == 0


@pytest.mark.asyncio
async def test_universal_replay_engine_matches_passive_ask_on_buy_trade(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_ticks" / "2026-03-01"
    raw_root.mkdir(parents=True)
    db_path = tmp_path / "universal_backtest.db"
    _write_jsonl(
        raw_root / "yes-1.jsonl",
        [
            {
                "local_ts": 1772323200.0,
                "source": "l2",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323200000",
                    "event_type": "book",
                    "bids": [{"price": "0.44", "size": "20"}],
                    "asks": [{"price": "0.45", "size": "20"}],
                },
            },
            {
                "local_ts": 1772323201.0,
                "source": "trade",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323201000",
                    "event_type": "last_trade_price",
                    "price": "0.45",
                    "size": "5",
                    "side": "BUY",
                    "transaction_hash": "tx-ask-1",
                },
            },
        ],
    )

    engine = UniversalReplayEngine(
        input_dir=tmp_path / "raw_ticks",
        db_path=db_path,
        strategy_path="tests.test_universal_backtest.PassiveAskStrategy",
        market_catalog=MarketCatalog([MarketTokens(market_id="mkt-1", yes_asset_id="yes-1", no_asset_id="no-1")]),
    )

    summary = await engine.run()

    assert summary.dispatches == 1
    assert summary.maker_fills == 1
    assert summary.maker_ask_fills == 1
    assert summary.persisted_shadow_rows == 1


@pytest.mark.asyncio
async def test_universal_replay_engine_cancel_all_removes_working_quotes(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_ticks" / "2026-03-01"
    raw_root.mkdir(parents=True)
    db_path = tmp_path / "universal_backtest.db"
    _write_jsonl(
        raw_root / "yes-1.jsonl",
        [
            {
                "local_ts": 1772323200.0,
                "source": "l2",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323200000",
                    "event_type": "book",
                    "bids": [{"price": "0.44", "size": "20"}],
                    "asks": [{"price": "0.45", "size": "20"}],
                },
            },
            {
                "local_ts": 1772323201.0,
                "source": "l2",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323201000",
                    "event_type": "book",
                    "bids": [{"price": "0.43", "size": "20"}],
                    "asks": [{"price": "0.46", "size": "20"}],
                },
            },
        ],
    )

    engine = UniversalReplayEngine(
        input_dir=tmp_path / "raw_ticks",
        db_path=db_path,
        strategy_path="tests.test_universal_backtest.ToxicCancelStrategy",
        market_catalog=MarketCatalog([MarketTokens(market_id="mkt-1", yes_asset_id="yes-1", no_asset_id="no-1")]),
    )

    summary = await engine.run()

    assert summary.dispatches == 2
    assert summary.cancel_requests == 1
    assert summary.cancelled_orders == 1
    assert summary.open_orders == 0


@pytest.mark.asyncio
async def test_universal_replay_engine_falls_back_to_asset_level_market_id(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_ticks" / "2026-03-01"
    raw_root.mkdir(parents=True)
    db_path = tmp_path / "universal_backtest.db"
    _write_jsonl(
        raw_root / "market-hex.jsonl",
        [
            {
                "local_ts": 1772323200.0,
                "source": "l2",
                "asset_id": "market-hex",
                "payload": {
                    "market": "market-hex",
                    "timestamp": "1772323200000",
                    "event_type": "price_change",
                    "price_changes": [
                        {
                            "asset_id": "market-hex",
                            "price": "0.45",
                            "size": "10",
                            "side": "SELL",
                        },
                        {
                            "asset_id": "market-hex",
                            "price": "0.44",
                            "size": "15",
                            "side": "BUY",
                        },
                    ],
                },
            }
        ],
    )

    engine = UniversalReplayEngine(
        input_dir=tmp_path / "raw_ticks",
        db_path=db_path,
        strategy_path="tests.test_universal_backtest.RecordAndFireStrategy",
        market_catalog=MarketCatalog(),
    )

    summary = await engine.run()

    assert summary.dispatches == 1
    assert summary.rejections == 0
    assert summary.taker_fills == 1
    assert summary.persisted_shadow_rows == 1


@pytest.mark.asyncio
async def test_universal_replay_engine_multiplexes_secondary_market_updates(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    secondary = tmp_path / "secondary.jsonl"
    db_path = tmp_path / "universal_backtest.db"
    _write_jsonl(
        primary,
        [
            {
                "local_ts": 1772323200.0,
                "source": "l2",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323200000",
                    "event_type": "book",
                    "bids": [{"price": "0.44", "size": "20"}],
                    "asks": [{"price": "0.45", "size": "10"}],
                },
            },
            {
                "local_ts": 1772323202.0,
                "source": "l2",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323202000",
                    "event_type": "book",
                    "bids": [{"price": "0.43", "size": "20"}],
                    "asks": [{"price": "0.46", "size": "10"}],
                },
            },
        ],
    )
    _write_jsonl(
        secondary,
        [
            {
                "local_ts": 1772323201.0,
                "source": "l2",
                "asset_id": "yes-2",
                "payload": {
                    "market": "mkt-2",
                    "asset_id": "yes-2",
                    "timestamp": "1772323201000",
                    "event_type": "book",
                    "bids": [{"price": "0.54", "size": "20"}],
                    "asks": [{"price": "0.55", "size": "10"}],
                },
            }
        ],
    )

    engine = UniversalReplayEngine(
        input_dir=primary,
        secondary_data_path=secondary,
        db_path=db_path,
        strategy_path="tests.test_universal_backtest.MarketSequenceStrategy",
        market_catalog=MarketCatalog(
            [
                MarketTokens(market_id="mkt-1", yes_asset_id="yes-1", no_asset_id="no-1"),
                MarketTokens(market_id="mkt-2", yes_asset_id="yes-2", no_asset_id="no-2"),
            ]
        ),
    )

    summary = await engine.run()

    assert summary.book_events == 3
    assert isinstance(engine._strategy, MarketSequenceStrategy)
    assert engine._strategy.market_sequence == ["mkt-1", "mkt-2", "mkt-1"]


@pytest.mark.asyncio
async def test_universal_replay_engine_accepts_single_raw_file_input(tmp_path: Path) -> None:
    raw_file = tmp_path / "single_market.jsonl"
    db_path = tmp_path / "single_file_backtest.db"
    _write_jsonl(
        raw_file,
        [
            {
                "local_ts": 1772323200.0,
                "source": "l2",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323200000",
                    "event_type": "book",
                    "bids": [{"price": "0.43", "size": "20"}],
                    "asks": [{"price": "0.45", "size": "20"}],
                },
            },
            {
                "local_ts": 1772323201.0,
                "source": "trade",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323201000",
                    "event_type": "last_trade_price",
                    "price": "0.44",
                    "size": "5",
                    "side": "SELL",
                    "transaction_hash": "tx-single-1",
                },
            },
        ],
    )

    engine = UniversalReplayEngine(
        input_dir=raw_file,
        db_path=db_path,
        strategy_path="tests.test_universal_backtest.PassiveBidStrategy",
        market_catalog=MarketCatalog([MarketTokens(market_id="mkt-1", yes_asset_id="yes-1", no_asset_id="no-1")]),
    )

    summary = await engine.run()

    assert summary.total_events == 2
    assert summary.maker_fills == 1
    assert summary.persisted_shadow_rows == 1


@pytest.mark.asyncio
async def test_universal_replay_engine_appends_unique_trade_ids_with_order_prefix(tmp_path: Path) -> None:
    raw_file = tmp_path / "single_market.jsonl"
    db_path = tmp_path / "batch_backtest.db"
    _write_jsonl(
        raw_file,
        [
            {
                "local_ts": 1772323200.0,
                "source": "l2",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323200000",
                    "event_type": "book",
                    "bids": [{"price": "0.43", "size": "20"}],
                    "asks": [{"price": "0.45", "size": "20"}],
                },
            },
            {
                "local_ts": 1772323201.0,
                "source": "trade",
                "asset_id": "yes-1",
                "payload": {
                    "market": "mkt-1",
                    "asset_id": "yes-1",
                    "timestamp": "1772323201000",
                    "event_type": "last_trade_price",
                    "price": "0.44",
                    "size": "5",
                    "side": "SELL",
                    "transaction_hash": "tx-prefix-1",
                },
            },
        ],
    )

    catalog = MarketCatalog([MarketTokens(market_id="mkt-1", yes_asset_id="yes-1", no_asset_id="no-1")])
    first_engine = UniversalReplayEngine(
        input_dir=raw_file,
        db_path=db_path,
        strategy_path="tests.test_universal_backtest.PassiveBidStrategy",
        market_catalog=catalog,
        order_id_prefix="batch-a",
    )
    second_engine = UniversalReplayEngine(
        input_dir=raw_file,
        db_path=db_path,
        strategy_path="tests.test_universal_backtest.PassiveBidStrategy",
        market_catalog=catalog,
        order_id_prefix="batch-b",
    )

    await first_engine.run()
    await second_engine.run()

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "select id from shadow_trades order by id"
        ).fetchall()
    finally:
        conn.close()

    assert [row[0] for row in rows] == [
        "batch-a-1-fill-1",
        "batch-b-1-fill-1",
    ]