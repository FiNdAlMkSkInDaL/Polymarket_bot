from __future__ import annotations

from decimal import Decimal
from typing import Any


def _as_float(value: float | int | Decimal | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _direction_sign(direction: str) -> float:
    normalized = str(direction or "").strip().upper()
    if normalized in {"YES", "BUY", "LONG"}:
        return 1.0
    if normalized in {"NO", "SELL", "SHORT"}:
        return -1.0
    raise ValueError(f"Unsupported direction: {direction}")


def compute_quote_residency_seconds(
    quoted_at: float | int | Decimal | None,
    terminal_at: float | int | Decimal | None,
) -> float:
    start = _as_float(quoted_at)
    end = _as_float(terminal_at)
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def compute_markout_bundle(
    *,
    reference_price: float | int | Decimal,
    direction: str,
    mark_price_5s: float | int | Decimal | None = None,
    mark_price_15s: float | int | Decimal | None = None,
    mark_price_60s: float | int | Decimal | None = None,
) -> dict[str, float | None]:
    reference = _as_float(reference_price)
    if reference is None:
        raise ValueError("reference_price is required")
    sign = _direction_sign(direction)

    def _markout(mark_price: float | int | Decimal | None) -> float | None:
        future_price = _as_float(mark_price)
        if future_price is None:
            return None
        return sign * (future_price - reference) * 100.0

    return {
        "markout_5s_cents": _markout(mark_price_5s),
        "markout_15s_cents": _markout(mark_price_15s),
        "markout_60s_cents": _markout(mark_price_60s),
    }


def estimate_reward_capture_usd(
    *,
    reward_daily_usd: float | int | Decimal,
    queue_residency_seconds: float | int | Decimal,
    quote_size_usd: float | int | Decimal,
    queue_depth_ahead_usd: float | int | Decimal = 0.0,
    day_seconds: float = 86400.0,
) -> float:
    reward = max(0.0, _as_float(reward_daily_usd) or 0.0)
    residency_seconds = max(0.0, _as_float(queue_residency_seconds) or 0.0)
    quote_size = max(0.0, _as_float(quote_size_usd) or 0.0)
    queue_depth = max(0.0, _as_float(queue_depth_ahead_usd) or 0.0)
    if reward <= 0.0 or residency_seconds <= 0.0 or quote_size <= 0.0 or day_seconds <= 0.0:
        return 0.0
    residency_share = min(1.0, residency_seconds / day_seconds)
    queue_share = quote_size / max(quote_size + queue_depth, quote_size)
    return reward * residency_share * queue_share


def estimate_net_edge_usd(
    *,
    estimated_reward_capture_usd: float | int | Decimal,
    fill_occurred: bool,
    quote_size_usd: float | int | Decimal,
    markout_cents: float | int | Decimal | None = None,
    fees_paid_usd: float | int | Decimal = 0.0,
) -> float:
    reward_capture = _as_float(estimated_reward_capture_usd) or 0.0
    fees_paid = max(0.0, _as_float(fees_paid_usd) or 0.0)
    realized_markout_usd = 0.0
    if fill_occurred:
        size_usd = max(0.0, _as_float(quote_size_usd) or 0.0)
        realized_markout_usd = size_usd * ((_as_float(markout_cents) or 0.0) / 100.0)
    return reward_capture + realized_markout_usd - fees_paid


def build_shadow_extra_payload(
    *,
    reward_daily_usd: float | int | Decimal | None = None,
    reward_min_size: float | int | Decimal | None = None,
    reward_max_spread_cents: float | int | Decimal | None = None,
    competition_usd: float | int | Decimal | None = None,
    reward_to_competition: float | int | Decimal | None = None,
    queue_depth_ahead_usd: float | int | Decimal | None = None,
    quoted_at: float | int | Decimal | None = None,
    terminal_at: float | int | Decimal | None = None,
    queue_residency_seconds: float | int | Decimal | None = None,
    fill_occurred: bool | None = None,
    fill_latency_ms: float | int | Decimal | None = None,
    reference_price: float | int | Decimal | None = None,
    direction: str = "YES",
    mark_price_5s: float | int | Decimal | None = None,
    mark_price_15s: float | int | Decimal | None = None,
    mark_price_60s: float | int | Decimal | None = None,
    markout_5s_cents: float | int | Decimal | None = None,
    markout_15s_cents: float | int | Decimal | None = None,
    markout_60s_cents: float | int | Decimal | None = None,
    estimated_reward_capture_usd: float | int | Decimal | None = None,
    estimated_net_edge_usd: float | int | Decimal | None = None,
    quote_size_usd: float | int | Decimal | None = None,
    fees_paid_usd: float | int | Decimal | None = None,
    quote_id: str | None = None,
    quote_reason: str | None = None,
    emergency_flatten: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    normalized_reward_daily = _as_float(reward_daily_usd)
    normalized_competition = _as_float(competition_usd)
    normalized_queue_depth = _as_float(queue_depth_ahead_usd)
    normalized_quote_size = _as_float(quote_size_usd)
    normalized_residency = _as_float(queue_residency_seconds)
    if normalized_residency is None:
        normalized_residency = compute_quote_residency_seconds(quoted_at, terminal_at)

    if reward_to_competition is None and normalized_reward_daily is not None and normalized_competition not in (None, 0.0):
        reward_to_competition = normalized_reward_daily / normalized_competition

    computed_markouts = None
    if reference_price is not None and any(price is not None for price in (mark_price_5s, mark_price_15s, mark_price_60s)):
        computed_markouts = compute_markout_bundle(
            reference_price=reference_price,
            direction=direction,
            mark_price_5s=mark_price_5s,
            mark_price_15s=mark_price_15s,
            mark_price_60s=mark_price_60s,
        )

    resolved_markout_5s = _as_float(markout_5s_cents)
    resolved_markout_15s = _as_float(markout_15s_cents)
    resolved_markout_60s = _as_float(markout_60s_cents)
    if computed_markouts is not None:
        if resolved_markout_5s is None:
            resolved_markout_5s = computed_markouts["markout_5s_cents"]
        if resolved_markout_15s is None:
            resolved_markout_15s = computed_markouts["markout_15s_cents"]
        if resolved_markout_60s is None:
            resolved_markout_60s = computed_markouts["markout_60s_cents"]

    resolved_reward_capture = _as_float(estimated_reward_capture_usd)
    if (
        resolved_reward_capture is None
        and normalized_reward_daily is not None
        and normalized_residency is not None
        and normalized_quote_size is not None
    ):
        resolved_reward_capture = estimate_reward_capture_usd(
            reward_daily_usd=normalized_reward_daily,
            queue_residency_seconds=normalized_residency,
            quote_size_usd=normalized_quote_size,
            queue_depth_ahead_usd=normalized_queue_depth or 0.0,
        )

    resolved_net_edge = _as_float(estimated_net_edge_usd)
    if resolved_net_edge is None and resolved_reward_capture is not None and fill_occurred is not None and normalized_quote_size is not None:
        fallback_markout = resolved_markout_60s
        if fallback_markout is None:
            fallback_markout = resolved_markout_15s
        if fallback_markout is None:
            fallback_markout = resolved_markout_5s
        resolved_net_edge = estimate_net_edge_usd(
            estimated_reward_capture_usd=resolved_reward_capture,
            fill_occurred=fill_occurred,
            quote_size_usd=normalized_quote_size,
            markout_cents=fallback_markout,
            fees_paid_usd=fees_paid_usd or 0.0,
        )

    candidate_values = {
        "reward_daily_usd": normalized_reward_daily,
        "reward_min_size": _as_float(reward_min_size),
        "reward_max_spread_cents": _as_float(reward_max_spread_cents),
        "competition_usd": normalized_competition,
        "reward_to_competition": _as_float(reward_to_competition),
        "queue_depth_ahead_usd": normalized_queue_depth,
        "queue_residency_seconds": normalized_residency,
        "fill_occurred": fill_occurred,
        "fill_latency_ms": _as_float(fill_latency_ms),
        "markout_5s_cents": resolved_markout_5s,
        "markout_15s_cents": resolved_markout_15s,
        "markout_60s_cents": resolved_markout_60s,
        "estimated_reward_capture_usd": resolved_reward_capture,
        "estimated_net_edge_usd": resolved_net_edge,
        "quote_id": quote_id,
        "quote_reason": quote_reason,
        "emergency_flatten": emergency_flatten,
    }
    for key, value in candidate_values.items():
        if value is not None:
            payload[key] = value
    return payload