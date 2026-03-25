from __future__ import annotations

from scripts.l2_region_matrix import build_matrix_document, rank_summaries
from scripts.l2_region_probe import extract_event_payloads, parse_timestamp_ms, summarize


def test_parse_timestamp_ms_accepts_seconds_and_milliseconds() -> None:
    assert parse_timestamp_ms("1711300000") == 1711300000000.0
    assert parse_timestamp_ms("1711300000123") == 1711300000123.0
    assert parse_timestamp_ms(None) is None


def test_extract_event_payloads_handles_list_and_dict() -> None:
    assert extract_event_payloads({"event_type": "book"}) == [{"event_type": "book"}]
    assert extract_event_payloads([{"event_type": "book"}, "x", 1]) == [{"event_type": "book"}]
    assert extract_event_payloads("bad") == []


def test_summarize_reports_population_stddev() -> None:
    stats = summarize([10.0, 20.0, 30.0])
    assert stats.count == 3
    assert stats.mean == 20.0
    assert round(stats.stdev or 0.0, 3) == 8.165
    assert stats.p95 == 29.0


def test_region_matrix_prefers_stable_region() -> None:
    summaries = [
        {
            "label": "tokyo",
            "disconnect_count": 2,
            "silence_gap_count": 5,
            "total_frames": 100,
            "total_events": 120,
            "exchange_lag_ms": {"stdev": 12.0, "p95": 90.0, "mean": 55.0},
            "frame_gap_ms": {"stdev": 40.0},
        },
        {
            "label": "virginia",
            "disconnect_count": 0,
            "silence_gap_count": 1,
            "total_frames": 140,
            "total_events": 200,
            "exchange_lag_ms": {"stdev": 6.0, "p95": 45.0, "mean": 28.0},
            "frame_gap_ms": {"stdev": 20.0},
        },
    ]
    ranked = rank_summaries(summaries)
    assert ranked[0].label == "virginia"

    matrix = build_matrix_document(summaries)
    assert matrix["recommended_region"]["label"] == "virginia"


def test_region_matrix_penalizes_insufficient_samples() -> None:
    summaries = [
        {
            "label": "quiet",
            "disconnect_count": 0,
            "silence_gap_count": 0,
            "total_frames": 1,
            "total_events": 1,
            "exchange_lag_ms": {"stdev": None, "p95": None, "mean": None},
            "frame_gap_ms": {"stdev": None},
        },
        {
            "label": "active",
            "disconnect_count": 0,
            "silence_gap_count": 2,
            "total_frames": 50,
            "total_events": 60,
            "exchange_lag_ms": {"stdev": 10.0, "p95": 55.0, "mean": 25.0},
            "frame_gap_ms": {"stdev": 120.0},
        },
    ]
    matrix = build_matrix_document(summaries)
    assert matrix["recommended_region"]["label"] == "active"