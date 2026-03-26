"""Dry-run JSON serializer for MEV execution batches."""

from __future__ import annotations

import json
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from typing import Any

from src.execution.mev_router import MevExecutionBatch, MevOrderPayload, format_native_price
from src.execution.priority_context import PriorityOrderContext


_PRICE_METADATA_KEYS = {
    "capital_limit",
    "anchor_volume",
    "base_size",
    "effective_size",
    "target_price",
    "optimized_price",
    "priority_epsilon",
    "trigger_price",
    "stop_price",
    "limit_price",
    "exit_anchor_price",
}


def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in _PRICE_METADATA_KEYS:
            serialized[key] = format_native_price(value)
            continue
        serialized[key] = value
    return serialized


def _serialize_priority_context(context: PriorityOrderContext) -> dict[str, Any]:
    payload = {
        "market_id": context.market_id,
        "side": context.side,
        "signal_source": context.signal_source,
        "conviction_scalar": format_native_price(context.conviction_scalar),
    }
    if context.leg_role is not None:
        payload["leg_role"] = context.leg_role
    return payload


def _serialize_order_payload(order_payload: MevOrderPayload) -> dict[str, Any]:
    payload = asdict(order_payload)
    payload["price"] = format_native_price(order_payload.price)
    payload["metadata"] = _serialize_metadata(order_payload.metadata)
    if order_payload.context is None:
        payload.pop("context", None)
    else:
        payload["context"] = _serialize_priority_context(order_payload.context)
    return payload


def serialize_mev_execution_batch(batch: MevExecutionBatch) -> str:
    payload: dict[str, Any] = {
        "route_id": batch.route_id,
        "playbook": batch.playbook,
        "payload_count": len(batch.payloads),
        "payloads": [_serialize_order_payload(order_payload) for order_payload in batch.payloads],
        "responses": [dict(response) for response in batch.responses],
    }
    return json.dumps(payload, indent=2, sort_keys=False)


def deserialize_conviction_scalar(raw: str) -> Decimal:
    """
    conviction_scalar is serialized as a fixed 6-decimal string.
    This is the sole authorized parse path for downstream consumers.
    Do not cast directly from JSON numeric - precision is not guaranteed.
    """

    try:
        value = Decimal(raw)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"conviction_scalar is not a valid decimal: {raw!r}") from exc
    if not (Decimal("0") <= value <= Decimal("1")):
        raise ValueError(f"conviction_scalar out of range: {raw}")
    return value


def deserialize_envelope(json_str: str) -> dict[str, Any]:
    payload = json.loads(json_str)
    for order_payload in payload.get("payloads", []):
        context = order_payload.get("context")
        if isinstance(context, dict) and "conviction_scalar" in context:
            context["conviction_scalar"] = deserialize_conviction_scalar(context["conviction_scalar"])
    return payload