"""Dry-run JSON serializer for MEV execution batches."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from src.execution.mev_router import MevExecutionBatch, MevOrderPayload, format_native_price


_PRICE_METADATA_KEYS = {
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


def _serialize_order_payload(order_payload: MevOrderPayload) -> dict[str, Any]:
    payload = asdict(order_payload)
    payload["price"] = format_native_price(order_payload.price)
    payload["metadata"] = _serialize_metadata(order_payload.metadata)
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