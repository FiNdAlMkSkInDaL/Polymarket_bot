from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import polars as pl

from scripts.enrich_lake_metadata import GammaMarketRow
from scripts.sync_lake_from_vps import (
    ENRICHED_MANIFEST_NAME,
    MANIFEST_NAME,
    RemoteFileSnapshot,
    SYNC_STATE_NAME,
    SyncTool,
    _build_rsync_command,
    _build_scp_command,
    _plan_delta_scp_transfers,
    _prune_local_partitions,
    _refresh_lake_metadata,
)


def _strict_row(
    *,
    timestamp: datetime,
    event_id: str,
    market_id: str,
    token_id: str,
    best_bid: float,
    best_ask: float,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "market_id": market_id,
        "event_id": event_id,
        "token_id": token_id,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_depth": 100.0,
        "ask_depth": 125.0,
    }


def test_build_scp_command_copies_remote_root_contents_into_custom_local_root(tmp_path: Path) -> None:
    tool = SyncTool(name="scp", command=["scp"])
    local_root = tmp_path / "artifacts" / "l2_parquet_lake_rolling" / "l2_book"

    command = _build_scp_command(
        tool,
        remote="botuser@135.181.85.32",
        remote_root="/home/botuser/polymarket-bot/data/l2_book_live",
        local_root=local_root,
        subpath=None,
        dry_run=False,
        delete=False,
    )

    assert command == [
        "scp",
        "-r",
        "botuser@135.181.85.32:/home/botuser/polymarket-bot/data/l2_book_live/.",
        str(local_root),
    ]


def test_build_rsync_command_uses_msys_local_path_for_msys_rsync(tmp_path: Path) -> None:
    tool = SyncTool(name="msys-rsync", command=["C:/msys64/usr/bin/rsync.exe"], transfer_family="rsync", path_style="msys")
    local_root = tmp_path / "artifacts" / "l2_parquet_lake_rolling" / "l2_book"

    command = _build_rsync_command(
        tool,
        remote="botuser@135.181.85.32",
        remote_root="/home/botuser/polymarket-bot/data/l2_book_live",
        local_root=local_root,
        subpath=None,
        dry_run=False,
        delete=False,
    )

    assert command[:6] == [
        "C:/msys64/usr/bin/rsync.exe",
        "-avz",
        "--partial",
        "--progress",
        "--update",
        "--rsync-path=/usr/bin/rsync",
    ]
    assert command[-2] == "botuser@135.181.85.32:/home/botuser/polymarket-bot/data/l2_book_live/"
    assert command[-1].startswith("/")
    assert ":" not in command[-1]


def test_plan_delta_scp_transfers_groups_changed_parquet_by_parent_directory(tmp_path: Path) -> None:
    local_root = tmp_path / "l2_book"
    local_hour_dir = local_root / "date=2026-04-04" / "hour=09"
    local_hour_dir.mkdir(parents=True)
    local_file = local_hour_dir / "l2_book_2026-04-04_09_000011.parquet"
    local_file.write_bytes(b"old")

    snapshots = [
        RemoteFileSnapshot(
            relative_path=PurePosixPath("date=2026-04-04/hour=09/l2_book_2026-04-04_09_000011.parquet"),
            size=10,
            mtime_ns=local_file.stat().st_mtime_ns + 1,
        ),
        RemoteFileSnapshot(
            relative_path=PurePosixPath("_state/writer_handoff.json"),
            size=50,
            mtime_ns=1,
        ),
    ]

    changed_dirs, changed_files = _plan_delta_scp_transfers(snapshots, local_root=local_root)

    assert changed_dirs == [PurePosixPath("date=2026-04-04/hour=09")]
    assert changed_files == [PurePosixPath("_state/writer_handoff.json")]


def test_prune_local_partitions_removes_days_before_floor(tmp_path: Path) -> None:
    local_root = tmp_path / "l2_book"
    (local_root / "date=2026-04-03" / "hour=08").mkdir(parents=True)
    (local_root / "date=2026-04-04" / "hour=09").mkdir(parents=True)

    removed = _prune_local_partitions(local_root, "2026-04-04")

    assert removed == ["2026-04-03"]
    assert not (local_root / "date=2026-04-03").exists()
    assert (local_root / "date=2026-04-04").exists()


def test_refresh_lake_metadata_writes_manifest_enriched_manifest_and_sync_state(tmp_path: Path) -> None:
    local_root = tmp_path / "l2_book"
    hour_dir = local_root / "date=2026-04-04" / "hour=09"
    hour_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            _strict_row(
                timestamp=datetime(2026, 4, 4, 9, 49, tzinfo=UTC),
                event_id="evt-live",
                market_id="mkt-live",
                token_id="YES",
                best_bid=0.43,
                best_ask=0.44,
            ),
            _strict_row(
                timestamp=datetime(2026, 4, 4, 9, 49, tzinfo=UTC),
                event_id="evt-live",
                market_id="mkt-live",
                token_id="NO",
                best_bid=0.56,
                best_ask=0.57,
            ),
        ]
    ).write_parquet(hour_dir / "l2_book_2026-04-04_09_000011.parquet")

    handoff_dir = local_root / "_state"
    handoff_dir.mkdir(parents=True)
    (handoff_dir / "writer_handoff.json").write_text(
        json.dumps(
            {
                "schema": "tick_writer_handoff_v1",
                "status": "running",
                "session_id": "session-1",
                "updated_at": "2026-04-04T09:49:02+00:00",
                "last_written_files": ["date=2026-04-04/hour=09/l2_book_2026-04-04_09_000011.parquet"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_fetcher(market_ids: list[str]) -> dict[str, GammaMarketRow]:
        assert sorted(market_ids) == ["mkt-live"]
        return {
            "mkt-live": GammaMarketRow(
                market_id="mkt-live",
                event_id="evt-live",
                question="Live rolling market",
                gamma_closed=False,
                gamma_market_status="open",
                resolution_timestamp=None,
                final_resolution_value=None,
            )
        }

    result = _refresh_lake_metadata(
        local_root=local_root,
        remote="botuser@135.181.85.32",
        remote_root="/home/botuser/polymarket-bot/data/l2_book_live",
        transfer_tool="scp",
        transfer_command=[
            "scp",
            "-r",
            "botuser@135.181.85.32:/home/botuser/polymarket-bot/data/l2_book_live/.",
            str(local_root),
        ],
        min_date="2026-04-04",
        started_at="2026-04-04T10:00:00+00:00",
        duration_seconds=7.5,
        interval_seconds=3600.0,
        subpath=None,
        gamma_batch_size=5,
        gamma_timeout_seconds=5.0,
        fetcher=fake_fetcher,
    )

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    enriched_manifest = json.loads((tmp_path / ENRICHED_MANIFEST_NAME).read_text(encoding="utf-8"))
    sync_state = json.loads((tmp_path / SYNC_STATE_NAME).read_text(encoding="utf-8"))

    assert result["manifest_path"] == tmp_path / MANIFEST_NAME
    assert manifest["days"] == ["2026-04-04"]
    assert manifest["stats"]["parquet_file_count"] == 1
    assert manifest["stats"]["scan_column_count"] == 10
    assert manifest["stats"]["base_column_count"] == 8
    assert manifest["stats"]["market_count"] == 1
    assert manifest["stats"]["latest_parquet_file"] == "l2_book/date=2026-04-04/hour=09/l2_book_2026-04-04_09_000011.parquet"
    assert manifest["current_run"]["handoff_state"]["session_id"] == "session-1"
    assert manifest["current_run"]["handoff_state"]["path"] == "l2_book/_state/writer_handoff.json"

    assert enriched_manifest["market_count"] == 1
    assert enriched_manifest["open_market_count"] == 1
    assert enriched_manifest["markets"][0]["market_id"] == "mkt-live"

    assert sync_state["schema"] == "rolling_lake_sync_state_v1"
    assert sync_state["last_started_at"] == "2026-04-04T10:00:00+00:00"
    assert sync_state["last_successful_sync_at"]
    assert sync_state["last_duration_seconds"] == 7.5
    assert sync_state["latest_parquet_file"] == "l2_book/date=2026-04-04/hour=09/l2_book_2026-04-04_09_000011.parquet"
    assert sync_state["handoff_state"]["session_id"] == "session-1"
