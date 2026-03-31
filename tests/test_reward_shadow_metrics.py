from __future__ import annotations

from decimal import Decimal

import pytest

from src.rewards.reward_shadow_metrics import (
    build_shadow_extra_payload,
    compute_markout_bundle,
    compute_quote_residency_seconds,
    estimate_net_edge_usd,
    estimate_reward_capture_usd,
)


def test_compute_quote_residency_seconds_clamps_negative() -> None:
    assert compute_quote_residency_seconds(120.0, 100.0) == 0.0
    assert compute_quote_residency_seconds(100.0, 115.5) == pytest.approx(15.5)


def test_compute_markout_bundle_respects_direction() -> None:
    yes_bundle = compute_markout_bundle(
        reference_price=0.50,
        direction="YES",
        mark_price_5s=0.53,
        mark_price_15s=0.49,
        mark_price_60s=0.55,
    )
    no_bundle = compute_markout_bundle(
        reference_price=0.50,
        direction="NO",
        mark_price_5s=0.53,
        mark_price_15s=0.49,
        mark_price_60s=0.55,
    )

    assert yes_bundle == {
        "markout_5s_cents": pytest.approx(3.0),
        "markout_15s_cents": pytest.approx(-1.0),
        "markout_60s_cents": pytest.approx(5.0),
    }
    assert no_bundle == {
        "markout_5s_cents": pytest.approx(-3.0),
        "markout_15s_cents": pytest.approx(1.0),
        "markout_60s_cents": pytest.approx(-5.0),
    }


def test_estimate_reward_capture_usd_scales_with_residency_and_queue_share() -> None:
    reward_capture = estimate_reward_capture_usd(
        reward_daily_usd=120.0,
        queue_residency_seconds=3600.0,
        quote_size_usd=25.0,
        queue_depth_ahead_usd=75.0,
    )

    assert reward_capture == pytest.approx(1.25)


def test_estimate_net_edge_usd_combines_reward_and_markout() -> None:
    net_edge = estimate_net_edge_usd(
        estimated_reward_capture_usd=1.25,
        fill_occurred=True,
        quote_size_usd=25.0,
        markout_cents=-2.0,
        fees_paid_usd=0.10,
    )

    assert net_edge == pytest.approx(0.65)


def test_build_shadow_extra_payload_derives_missing_fields() -> None:
    payload = build_shadow_extra_payload(
        reward_daily_usd=Decimal("240"),
        reward_min_size=Decimal("50"),
        reward_max_spread_cents=Decimal("1.5"),
        competition_usd=Decimal("120"),
        queue_depth_ahead_usd=Decimal("75"),
        quoted_at=100.0,
        terminal_at=3700.0,
        fill_occurred=True,
        fill_latency_ms=Decimal("180"),
        reference_price=Decimal("0.45"),
        direction="YES",
        mark_price_5s=Decimal("0.46"),
        mark_price_15s=Decimal("0.47"),
        mark_price_60s=Decimal("0.44"),
        quote_size_usd=Decimal("25"),
        fees_paid_usd=Decimal("0.05"),
        quote_id="Q-1",
        quote_reason="top_of_book",
    )

    assert payload["reward_to_competition"] == pytest.approx(2.0)
    assert payload["queue_residency_seconds"] == pytest.approx(3600.0)
    assert payload["markout_5s_cents"] == pytest.approx(1.0)
    assert payload["markout_15s_cents"] == pytest.approx(2.0)
    assert payload["markout_60s_cents"] == pytest.approx(-1.0)
    assert payload["estimated_reward_capture_usd"] == pytest.approx(2.5)
    assert payload["estimated_net_edge_usd"] == pytest.approx(2.20)
    assert payload["quote_id"] == "Q-1"