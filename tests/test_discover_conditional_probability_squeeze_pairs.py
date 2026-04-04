from __future__ import annotations

import json
import sys
import importlib
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

discover_pairs = importlib.import_module("discover_conditional_probability_squeeze_pairs")


def _write_lake_fixture(input_root: Path) -> None:
    output_dir = input_root / "l2_book" / "date=2026-04-01" / "hour=00"
    output_dir.mkdir(parents=True, exist_ok=True)
    common_rows = [
        {
            "timestamp": 1_700_000_000_000,
            "event_id": "event-1",
            "token_id": "YES",
            "best_bid": 0.45,
            "best_ask": 0.46,
            "bid_depth": 200.0,
            "ask_depth": 200.0,
        },
        {
            "timestamp": 1_700_000_000_000,
            "event_id": "event-1",
            "token_id": "NO",
            "best_bid": 0.54,
            "best_ask": 0.55,
            "bid_depth": 200.0,
            "ask_depth": 200.0,
        },
    ]
    for market_id in ("0xaaa", "0xbbb"):
        rows = [{**row, "market_id": market_id} for row in common_rows]
        pl.DataFrame(rows).write_parquet(output_dir / f"part-{market_id}-fixture.parquet")


def _write_metadata_fixture(metadata_path: Path) -> None:
    payload = {
        "markets_by_token": {
            "token-a-yes": {
                "conditionId": "0xaaa",
                "question": "Will Alpha win?",
                "slug": "alpha-win",
                "negRisk": True,
                "clobTokenIds": ["token-a-yes", "token-a-no"],
                "outcomes": ["YES", "NO"],
                "events": [
                    {
                        "id": "event-1",
                        "slug": "winner-market",
                        "title": "Winner Market",
                        "negRisk": True,
                    }
                ],
            },
            "token-b-yes": {
                "conditionId": "0xbbb",
                "question": "Will Beta win?",
                "slug": "beta-win",
                "negRisk": True,
                "clobTokenIds": ["token-b-yes", "token-b-no"],
                "outcomes": ["YES", "NO"],
                "events": [
                    {
                        "id": "event-1",
                        "slug": "winner-market",
                        "title": "Winner Market",
                        "negRisk": True,
                    }
                ],
            },
        }
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_seed_config(config_path: Path) -> None:
    payload = {
        "pairs": [
            {
                "pair_id": "manual-seed-pair",
                "parent_market_id": "0xparent",
                "child_market_id": "0xchild",
                "notes": "Preserve existing manually curated pair.",
            }
        ]
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_discovery_merges_seed_pairs_with_mutual_exclusion_pairs(tmp_path: Path) -> None:
    lake_root = tmp_path / "lake"
    metadata_path = tmp_path / "metadata.json"
    seed_config_path = tmp_path / "seed_pairs.json"

    _write_lake_fixture(lake_root)
    _write_metadata_fixture(metadata_path)
    _write_seed_config(seed_config_path)

    pairs, summary = discover_pairs.build_squeeze_pairs_config(
        lake_root,
        metadata_path,
        seed_config_path,
    )

    assert summary["available_market_count"] == 2
    assert summary["mutually_exclusive_event_count"] == 1
    assert summary["mutually_exclusive_pair_count"] == 2
    assert summary["total_pair_count"] == 3

    pair_ids = {entry["pair_id"] for entry in pairs}
    assert "manual-seed-pair" in pair_ids

    discovered = [entry for entry in pairs if entry.get("relationship_type") == "mutually_exclusive"]
    assert len(discovered) == 2
    assert {entry["parent_token_id"] for entry in discovered} == {"NO"}
    assert {entry["child_token_id"] for entry in discovered} == {"YES"}
