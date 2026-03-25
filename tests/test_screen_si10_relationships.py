from __future__ import annotations

from scripts.screen_si10_relationships import (
    GammaMarketRecord,
    _is_joint_market_text,
    _payload,
    build_relationship_candidates,
)


def _market(
    condition_id: str,
    question: str,
    *,
    event_id: str,
    event_title: str,
    tags: tuple[str, ...],
    category: str = "crypto",
    end_date: str = "2026-04-01T00:00:00Z",
) -> GammaMarketRecord:
    return GammaMarketRecord(
        market_id=condition_id.replace("0x", "m-"),
        condition_id=condition_id,
        question=question,
        event_id=event_id,
        event_title=event_title,
        tags=tags,
        category=category,
        end_date=end_date,
        yes_token_id=f"YES-{condition_id[-4:]}",
        no_token_id=f"NO-{condition_id[-4:]}",
        volume_24h=5000.0,
        liquidity=3000.0,
        outcomes=("Yes", "No"),
    )


def test_joint_market_detection_uses_connectors_and_hints() -> None:
    assert _is_joint_market_text("Will BTC hit 100k and ETH hit 5k?", "") is True
    assert _is_joint_market_text("BTC Trifecta special", "") is True
    assert _is_joint_market_text("Will Harvey Weinstein be sentenced to between 20 and 30 years in prison?", "") is False
    assert _is_joint_market_text("Will BTC hit 100k in April?", "") is False


def test_build_relationship_candidates_finds_base_pair_for_joint_market() -> None:
    markets = [
        _market(
            "0xaaa1",
            "Will BTC hit 100k in April?",
            event_id="E1",
            event_title="BTC targets",
            tags=("crypto", "btc", "april"),
        ),
        _market(
            "0xbbb2",
            "Will ETH hit 5k in April?",
            event_id="E2",
            event_title="ETH targets",
            tags=("crypto", "eth", "april"),
        ),
        _market(
            "0xccc3",
            "Will BTC hit 100k in April and ETH hit 5k in April?",
            event_id="E3",
            event_title="Crypto combo",
            tags=("crypto", "btc", "eth", "april"),
        ),
    ]

    candidates = build_relationship_candidates(
        markets,
        min_score=1.0,
        min_volume=0.0,
        min_liquidity=0.0,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.base_a_condition_id == "0xaaa1"
    assert candidate.base_b_condition_id == "0xbbb2"
    assert candidate.joint_condition_id == "0xccc3"
    assert "joint_text_split_match" in candidate.heuristic_reasons
    assert "shared_tags" in candidate.heuristic_reasons
    assert "crypto" in candidate.shared_tags


def test_payload_is_human_readable_and_preserves_condition_ids() -> None:
    markets = [
        _market(
            "0xaaa1",
            "Will BTC hit 100k?",
            event_id="E1",
            event_title="BTC",
            tags=("crypto", "btc"),
        ),
        _market(
            "0xbbb2",
            "Will ETH hit 5k?",
            event_id="E2",
            event_title="ETH",
            tags=("crypto", "eth"),
        ),
        _market(
            "0xccc3",
            "Will BTC hit 100k and ETH hit 5k?",
            event_id="E3",
            event_title="Combo",
            tags=("crypto", "btc", "eth"),
        ),
    ]
    candidates = build_relationship_candidates(markets, min_score=1.0, min_volume=0.0, min_liquidity=0.0)

    payload = _payload(candidates, top=10)

    assert len(payload) == 1
    row = payload[0]
    assert row["base_a_condition_id"].startswith("0x")
    assert row["base_b_condition_id"].startswith("0x")
    assert row["joint_condition_id"].startswith("0x")
    assert row["joint_question"] == "Will BTC hit 100k and ETH hit 5k?"