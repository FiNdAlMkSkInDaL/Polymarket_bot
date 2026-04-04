from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_lake_builder


def _write_metadata(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "market_id": "0xabc",
                    "event_id": "evt-1",
                    "yes_id": "111",
                    "no_id": "222",
                }
            ]
        ),
        encoding="utf-8",
    )


def _write_metadata_without_event_id(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "market_id": "0xabc",
                    "yes_id": "111",
                    "no_id": "222",
                }
            ]
        ),
        encoding="utf-8",
    )


def _touch_unit_files(raw_root: Path) -> None:
    day_dir = raw_root / "2026-03-20"
    day_dir.mkdir(parents=True)
    for stem in ("0xabc", "111", "222"):
        (day_dir / f"{stem}.jsonl").write_text("{}\n", encoding="utf-8")


def test_run_lake_builder_skips_completed_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "out"
    metadata_path = tmp_path / "metadata.json"
    _touch_unit_files(raw_root)
    _write_metadata(metadata_path)

    call_count = 0

    def fake_process_market_day(**kwargs) -> None:
        nonlocal call_count
        call_count += 1
        output_root_local: Path = kwargs["output_root"]
        metadata = kwargs["metadata"]
        target_path = output_root_local / "l2_book" / "date=2026-03-20" / "hour=00" / f"part-{metadata.market_id}-fake.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("parquet", encoding="utf-8")
        on_file_written = kwargs.get("on_file_written")
        if on_file_written is not None:
            on_file_written(target_path)
        stats = kwargs["stats"]
        stats.output_rows += 2
        stats.markets_completed += 1

    monkeypatch.setattr(run_lake_builder, "process_market_day", fake_process_market_day)

    args = [
        "--raw-root",
        str(raw_root),
        "--metadata",
        str(metadata_path),
        "--output-root",
        str(output_root),
    ]
    first_manifest = run_lake_builder.run_builder(run_lake_builder.parse_args(args))
    second_manifest = run_lake_builder.run_builder(run_lake_builder.parse_args(args))

    assert call_count == 1
    assert first_manifest["stats"]["output_rows"] == 2
    assert second_manifest["current_run"]["skipped_completed_units"] == 1


def test_run_lake_builder_fails_fast_when_metadata_rows_are_unusable(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "out"
    metadata_path = tmp_path / "bad_metadata.json"
    _touch_unit_files(raw_root)
    _write_metadata_without_event_id(metadata_path)

    with pytest.raises(ValueError, match="No market metadata rows were loaded"):
        run_lake_builder.run_builder(
            run_lake_builder.parse_args(
                [
                    "--raw-root",
                    str(raw_root),
                    "--metadata",
                    str(metadata_path),
                    "--output-root",
                    str(output_root),
                ]
            )
        )


def test_run_lake_builder_resumes_partial_units_and_cleans_stale_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "out"
    metadata_path = tmp_path / "metadata.json"
    _touch_unit_files(raw_root)
    _write_metadata(metadata_path)

    unit = run_lake_builder.WorkUnit(
        day="2026-03-20",
        day_dir=raw_root / "2026-03-20",
        metadata=run_lake_builder.MarketMetadata(
            market_id="0xabc",
            event_id="evt-1",
            yes_asset_id="111",
            no_asset_id="222",
        ),
    )
    state_root = output_root / run_lake_builder.STATE_DIR_NAME
    stale_path = output_root / "l2_book" / "date=2026-03-20" / "hour=00" / "part-0xabc-stale.parquet"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("stale", encoding="utf-8")
    run_lake_builder._write_json(
        run_lake_builder._in_progress_marker_path(state_root, unit),
        {
            "job_id": unit.job_id,
            "status": "in_progress",
            "day": unit.day,
            "market_id": unit.metadata.market_id,
            "event_id": unit.metadata.event_id,
            "started_at": "2026-04-04T00:00:00+00:00",
            "updated_at": "2026-04-04T00:00:00+00:00",
            "source_files": run_lake_builder._source_signature(unit),
            "written_files": [stale_path.relative_to(output_root).as_posix()],
        },
    )

    def fake_process_market_day(**kwargs) -> None:
        output_root_local: Path = kwargs["output_root"]
        metadata = kwargs["metadata"]
        target_path = output_root_local / "validation" / "date=2026-03-20" / "hour=00" / f"rejects-{metadata.market_id}-new.parquet"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("fresh", encoding="utf-8")
        on_file_written = kwargs.get("on_file_written")
        if on_file_written is not None:
            on_file_written(target_path)
        stats = kwargs["stats"]
        stats.rejected_rows += 1
        stats.markets_completed += 1

    monkeypatch.setattr(run_lake_builder, "process_market_day", fake_process_market_day)

    manifest = run_lake_builder.run_builder(
        run_lake_builder.parse_args(
            [
                "--raw-root",
                str(raw_root),
                "--metadata",
                str(metadata_path),
                "--output-root",
                str(output_root),
            ]
        )
    )

    assert not stale_path.exists()
    assert manifest["current_run"]["resumed_partial_units"] == 1
    assert manifest["stats"]["rejected_rows"] == 1


def test_run_lake_builder_aggregates_malformed_raw_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "out"
    metadata_path = tmp_path / "metadata.json"
    _touch_unit_files(raw_root)
    _write_metadata(metadata_path)

    def fake_process_market_day(**kwargs) -> None:
        stats = kwargs["stats"]
        stats.raw_records_read["snapshot:yes"] += 4
        stats.raw_records_parsed["snapshot:yes"] += 3
        stats.raw_records_malformed["snapshot:yes"] += 1
        stats.raw_batches_salvaged["snapshot:yes"] += 1
        stats.markets_completed += 1

    monkeypatch.setattr(run_lake_builder, "process_market_day", fake_process_market_day)

    manifest = run_lake_builder.run_builder(
        run_lake_builder.parse_args(
            [
                "--raw-root",
                str(raw_root),
                "--metadata",
                str(metadata_path),
                "--output-root",
                str(output_root),
            ]
        )
    )

    assert manifest["stats"]["raw_records_malformed"] == {"snapshot:yes": 1}
    assert manifest["stats"]["raw_batches_salvaged"] == {"snapshot:yes": 1}


def test_source_signature_matches_windows_drive_case_only_difference() -> None:
    current = {
        "market_delta": {
            "path": "c:/tmp/raw/0xabc.jsonl",
            "exists": True,
            "size_bytes": 123,
            "mtime_ns": 456,
        }
    }
    stored = {
        "market_delta": {
            "path": "C:\\tmp\\raw\\0xabc.jsonl",
            "exists": True,
            "size_bytes": 123,
            "mtime_ns": 456,
        }
    }

    assert run_lake_builder._source_signature_matches(current, stored) is True


def test_source_signature_matches_rejects_real_size_drift() -> None:
    current = {
        "market_delta": {
            "path": "C:\\tmp\\raw\\0xabc.jsonl",
            "exists": True,
            "size_bytes": 124,
            "mtime_ns": 456,
        }
    }
    stored = {
        "market_delta": {
            "path": "c:/tmp/raw/0xabc.jsonl",
            "exists": True,
            "size_bytes": 123,
            "mtime_ns": 456,
        }
    }

    assert run_lake_builder._source_signature_matches(current, stored) is False