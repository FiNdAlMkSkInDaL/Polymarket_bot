from __future__ import annotations

from scripts.live_flb_scanner import LiveFlbTarget, infer_category, rank_targets, select_reference_yes_price


def test_select_reference_prefers_best_ask_below_threshold() -> None:
    reference, midpoint, source = select_reference_yes_price(
        best_bid=0.031,
        best_ask=0.041,
        gamma_yes_price=0.039,
        max_yes_price=0.05,
        min_yes_price=0.001,
    )

    assert reference == 0.041
    assert midpoint == 0.036
    assert source == "best_ask"


def test_select_reference_falls_back_to_midpoint_when_ask_not_sub5c() -> None:
    reference, midpoint, source = select_reference_yes_price(
        best_bid=0.041,
        best_ask=0.057,
        gamma_yes_price=0.048,
        max_yes_price=0.05,
        min_yes_price=0.001,
    )

    assert reference == 0.049
    assert midpoint == 0.049
    assert source == "midpoint"


def test_select_reference_uses_gamma_midpoint_fallback_when_book_is_one_sided() -> None:
    reference, midpoint, source = select_reference_yes_price(
        best_bid=0.0,
        best_ask=0.0,
        gamma_yes_price=0.032,
        max_yes_price=0.05,
        min_yes_price=0.001,
    )

    assert reference == 0.032
    assert midpoint == 0.032
    assert source == "gamma_outcome_price"


def test_infer_category_matches_flb_taxonomy() -> None:
    category = infer_category(
        question="Will GTA VI release before June 2026?",
        market_slug="gta-vi-release-before-june-2026",
        event_title="Entertainment release schedule",
    )

    assert category == "culture"


def test_rank_targets_uses_volume_then_liquidity_then_caps() -> None:
    targets = [
        LiveFlbTarget(
            condition_id=f"cid-{idx}",
            market_id=f"mid-{idx}",
            event_id=f"evt-{idx}",
            question=f"Question {idx}",
            category="unknown",
            market_slug=f"slug-{idx}",
            event_title=f"Event {idx}",
            yes_token_id=f"yes-{idx}",
            no_token_id=f"no-{idx}",
            entry_yes_ask=0.04,
            terminal_yes_ask=0.04,
            max_yes_ask=0.04,
            max_no_drawdown_cents=None,
            best_yes_bid=0.03,
            best_yes_ask=0.04,
            yes_midpoint=0.035,
            gamma_yes_price=0.04,
            price_source="best_ask",
            market_volume_24h=volume,
            liquidity_clob_usd=liquidity,
            discovered_at="2026-04-03T00:00:00Z",
        )
        for idx, volume, liquidity in [
            (1, 10.0, 50.0),
            (2, 100.0, 10.0),
            (3, 100.0, 25.0),
        ]
    ]

    ranked = rank_targets(targets, max_targets=2)

    assert [row.condition_id for row in ranked] == ["cid-3", "cid-2"]