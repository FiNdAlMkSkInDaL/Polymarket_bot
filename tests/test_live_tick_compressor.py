from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts.live_tick_compressor import (
    AssetSubscription,
    CompressorHeartbeatLoop,
    L2RecordingPool,
    LowDiskSpaceError,
    ParquetTickWriter,
    UniverseRefreshLoop,
    build_tick_row,
)


def _subscription(asset_id: str, *, market_id: str | None = None, outcome: str = "YES") -> AssetSubscription:
    return AssetSubscription(
        asset_id=asset_id,
        market_id=market_id or f"condition-{asset_id}",
        outcome=outcome,
        question=f"Question for {asset_id}",
        daily_volume_usd=1000.0,
    )


class _FakeSocket:
    def __init__(
        self,
        *,
        socket_id: int,
        asset_ids: list[str],
        writer: object,
        subscriptions_by_asset: dict[str, AssetSubscription],
        ws_url: str,
        silence_timeout_seconds: float,
    ) -> None:
        self.socket_id = socket_id
        self.asset_ids = list(asset_ids)
        self.history: list[tuple[str, list[str]]] = []
        self.started = False
        self.stopped = False
        self.reconnect_count = 0

    async def start(self) -> None:
        self.started = True
        while not self.stopped:
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        self.stopped = True

    async def add_assets(self, new_ids: list[str]) -> None:
        added = [asset_id for asset_id in new_ids if asset_id not in self.asset_ids]
        if not added:
            return
        self.asset_ids.extend(added)
        self.history.append(("add", list(added)))

    async def remove_assets(self, ids_to_remove: list[str]) -> None:
        removing = [asset_id for asset_id in ids_to_remove if asset_id in self.asset_ids]
        if not removing:
            return
        for asset_id in removing:
            self.asset_ids.remove(asset_id)
        self.history.append(("remove", list(removing)))


def test_build_tick_row_classifies_snapshot_and_uses_subscription_metadata() -> None:
    subscriptions = {
        "yes-1": AssetSubscription(
            asset_id="yes-1",
            market_id="condition-1",
            outcome="YES",
            question="Will BTC close above 100k?",
            daily_volume_usd=12345.0,
        )
    }

    row = build_tick_row(
        {
            "event_type": "book",
            "asset_id": "yes-1",
            "timestamp": "1712145600123",
            "bids": [{"price": "0.42", "size": "10"}],
            "asks": [{"price": "0.44", "size": "12"}],
        },
        received_at=1712145600.456,
        subscriptions_by_asset=subscriptions,
    )

    assert row is not None
    assert row["msg_type"] == "snapshot"
    assert row["asset_id"] == "yes-1"
    assert row["market_id"] == "condition-1"
    assert row["outcome"] == "YES"
    assert row["exchange_ts"] == pytest.approx(1712145600.123)
    assert row["sequence_id"] is None
    assert '"event_type":"book"' in row["payload"]


def test_writer_flush_groups_rows_into_hourly_chunks(tmp_path: Path) -> None:
    writer = ParquetTickWriter(
        output_dir=tmp_path,
        flush_rows=10,
        flush_seconds=60.0,
        queue_size=100,
        rotation="hourly",
        compression="zstd",
        min_free_gb=0.0,
    )

    writer._flush_rows_sync(
        [
            {
                "local_ts": 1712145600.0,
                "exchange_ts": 1712145600.0,
                "msg_type": "snapshot",
                "asset_id": "yes-1",
                "market_id": "condition-1",
                "outcome": "YES",
                "price": None,
                "size": None,
                "sequence_id": 10,
                "side": None,
                "payload": '{"event_type":"book"}',
            },
            {
                "local_ts": 1712149200.0,
                "exchange_ts": 1712149200.0,
                "msg_type": "delta",
                "asset_id": "yes-2",
                "market_id": "condition-2",
                "outcome": "NO",
                "price": 0.45,
                "size": 7.0,
                "sequence_id": 11,
                "side": "SELL",
                "payload": '{"event_type":"price_change"}',
            },
        ]
    )

    files = sorted(tmp_path.rglob("*.parquet"))
    assert [path.name for path in files] == [
        "ticks_2024-04-03_12_000001.parquet",
        "ticks_2024-04-03_13_000001.parquet",
    ]

    first_table = pq.read_table(files[0])
    second_table = pq.read_table(files[1])
    assert first_table.num_rows == 1
    assert second_table.num_rows == 1
    assert first_table.column("msg_type").to_pylist() == ["snapshot"]
    assert second_table.column("side").to_pylist() == ["SELL"]


def test_writer_raises_when_disk_guard_is_breached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = ParquetTickWriter(
        output_dir=tmp_path,
        flush_rows=10,
        flush_seconds=60.0,
        queue_size=100,
        rotation="daily",
        compression="zstd",
        min_free_gb=5.0,
    )

    class _Usage:
        total = 10 * 1024**3
        used = 9 * 1024**3
        free = 1 * 1024**3

    monkeypatch.setattr("scripts.live_tick_compressor.shutil.disk_usage", lambda _path: _Usage())

    with pytest.raises(LowDiskSpaceError, match="Free space below threshold"):
        writer._flush_rows_sync(
            [
                {
                    "local_ts": 1712145600.0,
                    "exchange_ts": 1712145600.0,
                    "msg_type": "snapshot",
                    "asset_id": "yes-1",
                    "market_id": "condition-1",
                    "outcome": "YES",
                    "price": None,
                    "size": None,
                    "sequence_id": 10,
                    "side": None,
                    "payload": '{"event_type":"book"}',
                }
            ]
        )


@pytest.mark.asyncio
async def test_pool_apply_universe_adds_before_removing_on_existing_socket() -> None:
    created_sockets: list[_FakeSocket] = []

    def socket_factory(**kwargs: object) -> _FakeSocket:
        socket = _FakeSocket(**kwargs)
        created_sockets.append(socket)
        return socket

    subscriptions = {
        "a": _subscription("a"),
        "b": _subscription("b"),
    }
    pool = L2RecordingPool(
        asset_ids=["a", "b"],
        writer=object(),
        subscriptions_by_asset=dict(subscriptions),
        max_assets_per_socket=4,
        ws_url="wss://example.test/ws",
        silence_timeout_seconds=30.0,
        connect_stagger_seconds=0.0,
        socket_factory=socket_factory,
    )

    await pool.start()
    additions, removals, updated = await pool.apply_universe(
        {
            "b": _subscription("b"),
            "c": _subscription("c"),
        }
    )

    assert additions == ["c"]
    assert removals == ["a"]
    assert updated == []
    assert pool.asset_ids == ["b", "c"]
    assert pool.socket_count == 1
    assert created_sockets[0].history == [("add", ["c"]), ("remove", ["a"])]

    await pool.stop()


@pytest.mark.asyncio
async def test_pool_apply_universe_creates_and_retires_sockets() -> None:
    created_sockets: list[_FakeSocket] = []

    def socket_factory(**kwargs: object) -> _FakeSocket:
        socket = _FakeSocket(**kwargs)
        created_sockets.append(socket)
        return socket

    pool = L2RecordingPool(
        asset_ids=["a"],
        writer=object(),
        subscriptions_by_asset={"a": _subscription("a")},
        max_assets_per_socket=1,
        ws_url="wss://example.test/ws",
        silence_timeout_seconds=30.0,
        connect_stagger_seconds=0.0,
        socket_factory=socket_factory,
    )

    await pool.start()
    additions, removals, updated = await pool.apply_universe(
        {
            "a": _subscription("a"),
            "b": _subscription("b"),
        }
    )
    await asyncio.sleep(0)

    assert additions == ["b"]
    assert removals == []
    assert updated == []
    assert len(created_sockets) == 2
    assert created_sockets[1].started is True
    assert pool.socket_count == 2

    additions, removals, updated = await pool.apply_universe({"b": _subscription("b")})

    assert additions == []
    assert removals == ["a"]
    assert updated == []
    assert created_sockets[0].stopped is True
    assert pool.asset_ids == ["b"]
    assert pool.socket_count == 1

    await pool.stop()


@pytest.mark.asyncio
async def test_universe_refresh_loop_applies_periodic_updates() -> None:
    applied_universes: list[list[str]] = []
    applied_event = asyncio.Event()

    class _FakePool:
        asset_count = 2
        socket_count = 1

        async def apply_universe(self, subscriptions_by_asset: dict[str, AssetSubscription]) -> tuple[list[str], list[str], list[str]]:
            applied_universes.append(sorted(subscriptions_by_asset))
            applied_event.set()
            return ["b"], [], []

    async def resolve_subscriptions(
        args: argparse.Namespace,
        *,
        reason: str,
    ) -> dict[str, AssetSubscription]:
        assert reason == "periodic_refresh"
        return {
            "a": _subscription("a"),
            "b": _subscription("b"),
        }

    refresh_loop = UniverseRefreshLoop(
        args=argparse.Namespace(),
        pool=_FakePool(),
        refresh_seconds=0.01,
        resolve_subscriptions=resolve_subscriptions,
    )

    task = asyncio.create_task(refresh_loop.run())
    await asyncio.wait_for(applied_event.wait(), timeout=1.0)
    refresh_loop.stop()
    await asyncio.wait_for(task, timeout=1.0)

    assert applied_universes == [["a", "b"]]


@pytest.mark.asyncio
async def test_heartbeat_loop_logs_pool_and_refresh_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    logged_events: list[tuple[str, dict[str, object]]] = []
    heartbeat_seen = asyncio.Event()

    def fake_info(event: str, **kwargs: object) -> None:
        logged_events.append((event, kwargs))
        if event == "tick_compressor_heartbeat":
            heartbeat_seen.set()

    monkeypatch.setattr("scripts.live_tick_compressor.log.info", fake_info)

    class _FakePool:
        asset_count = 12
        socket_count = 3
        reconnect_count = 7

    refresh_loop = UniverseRefreshLoop(
        args=argparse.Namespace(),
        pool=_FakePool(),
        refresh_seconds=60.0,
    )
    refresh_loop.note_refresh(reason="periodic_refresh")

    heartbeat_loop = CompressorHeartbeatLoop(
        pool=_FakePool(),
        refresh_loop=refresh_loop,
        interval_seconds=0.01,
    )

    task = asyncio.create_task(heartbeat_loop.run())
    await asyncio.wait_for(heartbeat_seen.wait(), timeout=1.0)
    heartbeat_loop.stop()
    await asyncio.wait_for(task, timeout=1.0)

    heartbeat_entries = [payload for event, payload in logged_events if event == "tick_compressor_heartbeat"]
    assert len(heartbeat_entries) == 1
    assert heartbeat_entries[0]["asset_count"] == 12
    assert heartbeat_entries[0]["socket_count"] == 3
    assert heartbeat_entries[0]["reconnect_count"] == 7
    assert heartbeat_entries[0]["refresh_count"] == 1
    assert heartbeat_entries[0]["last_refresh_reason"] == "periodic_refresh"
    assert heartbeat_entries[0]["last_refresh_age_s"] is not None
    assert float(heartbeat_entries[0]["last_refresh_age_s"]) >= 0.0