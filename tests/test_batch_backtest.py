from __future__ import annotations

import json
from pathlib import Path

from scripts.run_batch_backtest import resolve_batch_input_dir, select_files_from_list
from scripts.scan_extreme_obi import compute_top3_abs_obi_from_book, scan_directory


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_compute_top3_abs_obi_from_book_sorts_before_taking_top_levels() -> None:
    abs_obi = compute_top3_abs_obi_from_book(
        {
            "bids": [
                {"price": "0.10", "size": "1"},
                {"price": "0.95", "size": "100"},
                {"price": "0.94", "size": "90"},
                {"price": "0.93", "size": "80"},
            ],
            "asks": [
                {"price": "0.11", "size": "1"},
                {"price": "0.12", "size": "1"},
                {"price": "0.13", "size": "1"},
                {"price": "0.90", "size": "999"},
            ],
        }
    )

    assert abs_obi is not None
    assert abs_obi > 0.95


def test_scan_directory_returns_only_files_exceeding_threshold(tmp_path: Path) -> None:
    quiet_file = tmp_path / "quiet.jsonl"
    crash_file = tmp_path / "crash.jsonl"
    _write_jsonl(
        quiet_file,
        [
            {
                "payload": {
                    "event_type": "book",
                    "bids": [{"price": "0.40", "size": "10"}, {"price": "0.39", "size": "10"}, {"price": "0.38", "size": "10"}],
                    "asks": [{"price": "0.41", "size": "10"}, {"price": "0.42", "size": "10"}, {"price": "0.43", "size": "10"}],
                }
            }
        ],
    )
    _write_jsonl(
        crash_file,
        [
            {
                "payload": {
                    "event_type": "book",
                    "bids": [{"price": "0.40", "size": "100"}, {"price": "0.39", "size": "90"}, {"price": "0.38", "size": "80"}],
                    "asks": [{"price": "0.41", "size": "1"}, {"price": "0.42", "size": "1"}, {"price": "0.43", "size": "1"}],
                }
            }
        ],
    )

    candidates = scan_directory(tmp_path, threshold=0.95)

    assert candidates == [(crash_file.resolve(), candidates[0][1])]
    assert candidates[0][1] > 0.95


def test_select_files_from_list_resolves_relative_paths_from_list_directory(tmp_path: Path) -> None:
    replay_dir = tmp_path / "raw_ticks"
    replay_dir.mkdir()
    first_file = replay_dir / "first.jsonl"
    second_file = replay_dir / "second.jsonl"
    first_file.write_text("", encoding="utf-8")
    second_file.write_text("", encoding="utf-8")
    file_list = tmp_path / "candidates.txt"
    file_list.write_text("# comment\nraw_ticks/second.jsonl\nraw_ticks/first.jsonl\n", encoding="utf-8")

    selected = select_files_from_list(file_list)

    assert selected == [first_file.resolve(), second_file.resolve()]


def test_resolve_batch_input_dir_uses_common_parent_for_selected_files(tmp_path: Path) -> None:
    replay_dir = tmp_path / "raw_ticks"
    replay_dir.mkdir()
    first_file = replay_dir / "first.jsonl"
    second_file = replay_dir / "second.jsonl"
    first_file.write_text("", encoding="utf-8")
    second_file.write_text("", encoding="utf-8")

    resolved = resolve_batch_input_dir(input_dir_arg=None, selected_files=[first_file, second_file])

    assert resolved == replay_dir