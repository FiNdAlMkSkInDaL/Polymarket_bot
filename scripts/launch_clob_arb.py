#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Literal

from py_clob_client.client import ClobClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.core.config import EXCHANGE_MIN_SHARES, EXCHANGE_MIN_USD, settings
from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.clob_signer import ClobSigner
from src.execution.clob_transport import AiohttpClobTransport
from src.execution.nonce_manager import ClobNonceManager
from src.execution.polymarket_clob_translator import ClobOrderIntent, ClobPayloadBuilder, ClobReceiptParser, ClobTimeInForce
from src.execution.venue_adapter_interface import VenueOrderResponse, VenueOrderStatus


DEFAULT_INPUT = PROJECT_ROOT / "config" / "live_executable_strips.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "logs" / "clob_arb_receipt.json"
DEFAULT_MARKDOWN_OUTPUT = PROJECT_ROOT / "docs" / "clob_arb_receipt.md"
CHAIN_ID = 137
SIZE_QUANT = Decimal("0.000001")
USD_CENT = Decimal("0.01")
SAFE_NOTIONAL_QUANT = Decimal("1")
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
DEFAULT_CLOB_URL = "https://clob.polymarket.com"
PRICE_MIN = Decimal("0.001")
PRICE_MAX = Decimal("0.999")
FLATTEN_ESCALATION_PCT = Decimal("0.05")
FLATTEN_ESCALATION_ABS = Decimal("0.05")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _session_id(prefix: str = "clob-arb") -> str:
    return f"{prefix}-{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"


def _safe_decimal(value: Any) -> Decimal:
    return Decimal(str(value or "0")).normalize()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _quantize_shares(value: Decimal) -> Decimal:
    return value.quantize(SIZE_QUANT, rounding=ROUND_DOWN)


def _round_down_whole_dollars(value: Decimal) -> Decimal:
    if value <= Decimal("0"):
        return Decimal("0")
    return value.quantize(SAFE_NOTIONAL_QUANT, rounding=ROUND_DOWN)


@dataclass(frozen=True, slots=True)
class StripLeg:
    condition_id: str
    market_id: str
    question: str
    outcome_label: str
    yes_token_id: str
    no_token_id: str
    best_bid: Decimal
    best_bid_size_shares: Decimal
    best_bid_notional_usd: Decimal
    best_ask: Decimal
    best_ask_size_shares: Decimal
    best_ask_notional_usd: Decimal


@dataclass(frozen=True, slots=True)
class StripTarget:
    event_id: str
    event_title: str
    event_slug: str
    recommended_action: Literal["BUY_YES_STRIP", "SELL_NO_STRIP"]
    execution_price_sum: Decimal
    min_leg_depth_usd_observed: Decimal
    strip_max_size_shares_at_bbo: Decimal
    legs: list[StripLeg]


@dataclass(frozen=True, slots=True)
class OrderPlan:
    event_id: str
    event_title: str
    action: Literal["BUY", "SELL"]
    side: Literal["YES", "NO"]
    strip_notional_cap_usd: Decimal
    strip_shares: Decimal
    total_notional_usd: Decimal
    total_execution_price: Decimal
    time_in_force: ClobTimeInForce
    legs: list[StripLeg]


@dataclass(frozen=True, slots=True)
class PreparedOrder:
    event_id: str
    event_title: str
    condition_id: str
    market_id: str
    client_order_id: str
    action: Literal["BUY", "SELL"]
    side: Literal["YES", "NO"]
    price: Decimal
    size: Decimal
    notional_usd: Decimal
    token_id: str
    source_client_order_id: str | None
    payload: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class LegExecutionResult:
    event_id: str
    event_title: str
    client_order_id: str
    condition_id: str
    action: str
    side: str
    price: str
    size: str
    notional_usd: str
    submit_status: str
    rejection_reason: str | None
    fill_status: str | None
    filled_size: str | None
    remaining_size: str | None
    average_fill_price: str | None


@dataclass(frozen=True, slots=True)
class FlattenExecutionResult:
    event_id: str
    event_title: str
    source_client_order_id: str
    flatten_client_order_id: str | None
    condition_id: str
    action: str
    side: str
    price: str | None
    size: str
    submit_status: str
    rejection_reason: str | None
    fill_status: str | None
    filled_size: str | None
    remaining_size: str | None
    average_fill_price: str | None
    remaining_inventory_after_stage: str | None
    flatten_stage: int
    flatten_price_mode: str
    flatten_status: str


@dataclass(frozen=True, slots=True)
class FlattenExposure:
    source_client_order_id: str
    event_id: str
    event_title: str
    condition_id: str
    market_id: str
    action: Literal["BUY", "SELL"]
    side: Literal["YES", "NO"]
    token_id: str
    remaining_size: Decimal


class PaperClobTransport:
    def __init__(self, *, now_ms: callable) -> None:
        self._now_ms = now_ms
        self.intercepted_orders: list[dict[str, Any]] = []
        self._orders_by_client_id: dict[str, dict[str, Any]] = {}

    async def close(self) -> None:
        return None

    async def post_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        client_order_id = str(payload.get("clientOrderId") or payload.get("order", {}).get("client_order_id") or "").strip()
        venue_order_id = f"paper-{client_order_id}"
        timestamp_ms = int(self._now_ms())
        envelope = {
            "clientOrderId": client_order_id,
            "orderID": venue_order_id,
            "status": "LIVE",
            "timestamp": timestamp_ms,
            "latencyMs": 0,
            "paperIntercepted": True,
            "payload": dict(payload),
        }
        self.intercepted_orders.append(envelope)
        self._orders_by_client_id[client_order_id] = envelope
        return envelope

    async def get_order(self, order_id: str) -> dict[str, Any]:
        existing = self._orders_by_client_id.get(str(order_id).strip())
        if existing is None:
            return {
                "clientOrderId": str(order_id).strip(),
                "status": "UNKNOWN",
                "filled_size": "0",
                "remaining_size": "0",
            }
        order_payload = existing.get("payload", {}).get("order", {})
        return {
            "clientOrderId": existing["clientOrderId"],
            "orderID": existing["orderID"],
            "status": "OPEN",
            "filled_size": "0",
            "remaining_size": str(order_payload.get("size") or "0"),
        }

    async def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        return Decimal("1000000")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch live or PAPER grouped CLOB arbitrage strips.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to config/live_executable_strips.json")
    parser.add_argument("--env", choices=("PAPER", "LIVE"), default="PAPER", help="Execution mode.")
    parser.add_argument("--time-in-force", choices=("FOK", "IOC"), default="FOK", help="Immediate execution policy per leg.")
    parser.add_argument("--max-strips", type=int, default=0, help="Optional cap on number of strips to launch. 0 means all.")
    parser.add_argument("--event-id", action="append", default=[], help="Optional event_id filter; can be provided multiple times.")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT, help="Optional machine-readable launch summary path.")
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT, help="Optional human-readable launch summary path.")
    parser.add_argument("--allow-empty", action="store_true", help="Write empty launch summaries instead of failing when no strips remain.")
    parser.add_argument("--log-dir", default="logs", help="Reserved for future structured logging compatibility.")
    return parser.parse_args()


def build_json_summary(*, env: str, input_path: Path, plans: list[OrderPlan], results: list[dict[str, Any]], paper_intercepted_total: int) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in results:
        status = _clean_text(row.get("status")) or "UNKNOWN"
        status_counts[status] = status_counts.get(status, 0) + 1
    total_notional = sum((plan.total_notional_usd for plan in plans), start=Decimal("0")).quantize(USD_CENT, rounding=ROUND_DOWN)
    return {
        "generated_at": _utc_now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "strategy": "CLOB_GROUP_ARB",
        "env": env,
        "input_path": str(input_path),
        "targets_loaded": len(plans),
        "paper_intercepted_payloads": paper_intercepted_total,
        "flatten_events": sum(1 for row in results if row.get("flatten_triggered")),
        "flatten_stage2_events": sum(1 for row in results if row.get("flatten_stage2_triggered")),
        "flatten_failures": sum(1 for row in results if row.get("flatten_failed")),
        "status_counts": dict(sorted(status_counts.items())),
        "total_planned_notional_usd": format(total_notional, "f"),
        "results": results,
    }


def build_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# CLOB Arb Receipt",
        "",
        f"- Generated at: `{summary.get('generated_at', '')}`",
        f"- Env: `{summary.get('env', '')}`",
        f"- Targets loaded: `{summary.get('targets_loaded', 0)}`",
        f"- Total planned notional USD: `${summary.get('total_planned_notional_usd', '0')}`",
        f"- Paper intercepted payloads: `{summary.get('paper_intercepted_payloads', 0)}`",
        f"- Flatten events: `{summary.get('flatten_events', 0)}`",
        f"- Stage 2 escalations: `{summary.get('flatten_stage2_events', 0)}`",
        f"- Flatten failures: `{summary.get('flatten_failures', 0)}`",
        "",
        "## Strip Results",
        "",
    ]
    for row in summary.get("results", []):
        event_title = _clean_text(row.get("event_title")) or _clean_text(row.get("event_id"))
        lines.extend(
            [
                f"### {event_title}",
                "",
                f"- Status: `{row.get('status', 'UNKNOWN')}`",
                f"- Recommended action: `{row.get('recommended_action', 'UNKNOWN')}`",
                f"- Strip notional cap USD: `${row.get('strip_notional_cap_usd', '0')}`",
                f"- Strip shares: `{row.get('strip_shares', '0')}`",
                f"- Total notional USD: `${row.get('total_notional_usd', '0')}`",
                f"- Time in force: `{row.get('time_in_force', 'UNKNOWN')}`",
                f"- Flatten triggered: `{bool(row.get('flatten_triggered'))}`",
                f"- Stage 2 triggered: `{bool(row.get('flatten_stage2_triggered'))}`",
                f"- Flatten failed: `{bool(row.get('flatten_failed'))}`",
                f"- Flatten unresolved inventory: `{row.get('flatten_unresolved_inventory', '0')}`",
                "",
                "| Leg | Submit | Fill | Filled Size | Remaining | Reason |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for leg in row.get("legs", []):
            lines.append(
                f"| {leg.get('client_order_id', '')} | {leg.get('submit_status', '')} | {leg.get('fill_status', '')} | {leg.get('filled_size', '')} | {leg.get('remaining_size', '')} | {str(leg.get('rejection_reason') or '').replace('|', '/')} |"
            )
        flatten_rows = row.get("flatten_legs") or []
        if flatten_rows:
            lines.extend(
                [
                    "",
                    "| Flatten Leg | Stage | Mode | Status | Fill | Remaining Inventory | Reason |",
                    "| --- | ---: | --- | --- | --- | ---: | --- |",
                ]
            )
            for leg in flatten_rows:
                lines.append(
                    f"| {leg.get('flatten_client_order_id') or leg.get('source_client_order_id', '')} | {leg.get('flatten_stage', '')} | {leg.get('flatten_price_mode', '')} | {leg.get('flatten_status', '')} | {leg.get('fill_status', '')} | {leg.get('remaining_inventory_after_stage', '')} | {str(leg.get('rejection_reason') or '').replace('|', '/')} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_summaries(*, json_output: Path, markdown_output: Path, summary: dict[str, Any]) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(build_markdown_summary(summary), encoding="utf-8")


def _filled_size_decimal(result: LegExecutionResult) -> Decimal:
    return _safe_decimal(result.filled_size)


def _has_any_fill(result: LegExecutionResult) -> bool:
    return _filled_size_decimal(result) > Decimal("0") or result.fill_status in {"FILLED", "PARTIAL"}


def _is_full_fill(result: LegExecutionResult) -> bool:
    return result.fill_status == "FILLED" and _filled_size_decimal(result) > Decimal("0")


def _inverse_action(action: str) -> Literal["BUY", "SELL"]:
    return "SELL" if action.upper() == "BUY" else "BUY"


def _best_level(levels: list[Any], *, side: Literal["bid", "ask"]) -> Decimal:
    if not levels:
        return Decimal("0")
    level = levels[-1]
    value = getattr(level, "price", "0")
    return _safe_decimal(value)


def _lookup_book_prices(clob_client: ClobClient, token_id: str) -> tuple[Decimal, Decimal]:
    book = clob_client.get_order_book(token_id)
    bids = getattr(book, "bids", []) or []
    asks = getattr(book, "asks", []) or []
    return _best_level(bids, side="bid"), _best_level(asks, side="ask")


def _flatten_stage1_price(clob_client: ClobClient, token_id: str, *, flatten_action: Literal["BUY", "SELL"]) -> Decimal:
    best_bid, best_ask = _lookup_book_prices(clob_client, token_id)
    return best_bid if flatten_action == "SELL" else best_ask


def _panic_liquidation_price(stage1_price: Decimal, *, flatten_action: Literal["BUY", "SELL"]) -> Decimal:
    aggression = max(FLATTEN_ESCALATION_ABS, (stage1_price * FLATTEN_ESCALATION_PCT).quantize(USD_CENT, rounding=ROUND_DOWN))
    if flatten_action == "SELL":
        return max(PRICE_MIN, stage1_price - aggression)
    return min(PRICE_MAX, stage1_price + aggression)


def _flatten_exposures(
    leg_results: list[LegExecutionResult],
    *,
    prepared_by_client_id: dict[str, PreparedOrder],
) -> list[FlattenExposure]:
    exposures: list[FlattenExposure] = []
    for result in leg_results:
        filled_size = _filled_size_decimal(result)
        if filled_size <= Decimal("0"):
            continue
        source = prepared_by_client_id[result.client_order_id]
        exposures.append(
            FlattenExposure(
                source_client_order_id=source.client_order_id,
                event_id=source.event_id,
                event_title=source.event_title,
                condition_id=source.condition_id,
                market_id=source.market_id,
                action=_inverse_action(source.action),
                side=source.side,
                token_id=source.token_id,
                remaining_size=_quantize_shares(filled_size),
            )
        )
    return exposures


def _prepare_flatten_orders(
    exposures: list[FlattenExposure],
    *,
    clob_client: ClobClient,
    owner_id: str,
    signer: ClobSigner,
    payload_builder: ClobPayloadBuilder,
    nonce_manager: ClobNonceManager,
    client_ids: ClientOrderIdGenerator,
    flatten_stage: int,
) -> tuple[list[PreparedOrder], list[FlattenExecutionResult]]:
    prepared_flatten_orders: list[PreparedOrder] = []
    skipped_flatten: list[FlattenExecutionResult] = []
    usable_exposures = [exposure for exposure in exposures if exposure.remaining_size > Decimal("0")]
    if not usable_exposures:
        return prepared_flatten_orders, skipped_flatten

    nonces = nonce_manager.reserve_nonces(len(usable_exposures))
    timestamp_ms = _now_ms()
    nonce_index = 0
    for index, exposure in enumerate(usable_exposures):
        if flatten_stage == 1:
            flatten_price = _flatten_stage1_price(clob_client, exposure.token_id, flatten_action=exposure.action)
            price_mode = "BBO_IOC"
        else:
            stage1_price = _flatten_stage1_price(clob_client, exposure.token_id, flatten_action=exposure.action)
            flatten_price = _panic_liquidation_price(stage1_price, flatten_action=exposure.action)
            price_mode = "PANIC_IOC"
        flatten_size = _quantize_shares(exposure.remaining_size)
        if flatten_price <= Decimal("0") or flatten_size <= Decimal("0"):
            skipped_flatten.append(
                FlattenExecutionResult(
                    event_id=exposure.event_id,
                    event_title=exposure.event_title,
                    source_client_order_id=exposure.source_client_order_id,
                    flatten_client_order_id=None,
                    condition_id=exposure.condition_id,
                    action=exposure.action,
                    side=exposure.side,
                    price=None,
                    size=format(flatten_size, "f"),
                    submit_status="SKIPPED",
                    rejection_reason="NO_LIVE_FLATTEN_PRICE",
                    fill_status=None,
                    filled_size=None,
                    remaining_size=None,
                    average_fill_price=None,
                    remaining_inventory_after_stage=format(flatten_size, "f"),
                    flatten_stage=flatten_stage,
                    flatten_price_mode=price_mode,
                    flatten_status=f"STAGE{flatten_stage}_FAILED_TO_PREPARE",
                )
            )
            continue

        flatten_client_order_id = client_ids.generate(exposure.condition_id, exposure.side, timestamp_ms + index + (flatten_stage * 10_000))
        intent = ClobOrderIntent(
            condition_id=exposure.condition_id,
            token_id=exposure.token_id,
            outcome=exposure.side,
            action=exposure.action,
            price=flatten_price,
            size=flatten_size,
            time_in_force=ClobTimeInForce.IOC,
            client_order_id=flatten_client_order_id,
            post_only=False,
            fee_rate_bps=0,
            nonce=nonces[nonce_index],
            expiration=0,
            taker=ZERO_ADDRESS,
        )
        nonce_index += 1
        signed_order = signer.sign_create_order_payload(payload_builder.build_create_order_payload(intent))
        payload = payload_builder.build_post_order_payload(
            signed_order=signed_order,
            owner_id=owner_id,
            time_in_force=ClobTimeInForce.IOC,
            post_only=False,
        )
        payload["clientOrderId"] = flatten_client_order_id
        prepared_flatten_orders.append(
            PreparedOrder(
                event_id=exposure.event_id,
                event_title=exposure.event_title,
                condition_id=exposure.condition_id,
                market_id=exposure.market_id,
                client_order_id=flatten_client_order_id,
                action=exposure.action,
                side=exposure.side,
                price=flatten_price,
                size=flatten_size,
                notional_usd=(flatten_price * flatten_size).quantize(USD_CENT, rounding=ROUND_DOWN),
                token_id=exposure.token_id,
                source_client_order_id=exposure.source_client_order_id,
                payload=payload,
            )
        )

    return prepared_flatten_orders, skipped_flatten


async def _execute_flatten_stage(
    *,
    transport: Any,
    receipt_parser: ClobReceiptParser,
    prepared_orders: list[PreparedOrder],
    flatten_stage: int,
    price_mode: str,
) -> tuple[list[FlattenExecutionResult], list[FlattenExposure]]:
    if not prepared_orders:
        return [], []

    raw_submit_responses = await _post_batch(transport, prepared_orders)
    raw_status_responses = await _fetch_status_batch(transport, prepared_orders)
    flatten_results: list[FlattenExecutionResult] = []
    unresolved: list[FlattenExposure] = []
    for order, raw_submit, raw_status in zip(prepared_orders, raw_submit_responses, raw_status_responses, strict=True):
        if isinstance(raw_submit, Exception):
            flatten_results.append(
                FlattenExecutionResult(
                    event_id=order.event_id,
                    event_title=order.event_title,
                    source_client_order_id=order.source_client_order_id or "",
                    flatten_client_order_id=order.client_order_id,
                    condition_id=order.condition_id,
                    action=order.action,
                    side=order.side,
                    price=format(order.price, "f"),
                    size=format(order.size, "f"),
                    submit_status="ERROR",
                    rejection_reason=str(raw_submit),
                    fill_status=None,
                    filled_size=None,
                    remaining_size=None,
                    average_fill_price=None,
                    remaining_inventory_after_stage=format(order.size, "f"),
                    flatten_stage=flatten_stage,
                    flatten_price_mode=price_mode,
                    flatten_status=f"STAGE{flatten_stage}_FAILED",
                )
            )
            unresolved.append(
                FlattenExposure(
                    source_client_order_id=order.source_client_order_id or "",
                    event_id=order.event_id,
                    event_title=order.event_title,
                    condition_id=order.condition_id,
                    market_id=order.market_id,
                    action=order.action,
                    side=order.side,
                    token_id=order.token_id,
                    remaining_size=order.size,
                )
            )
            continue

        submit_result = receipt_parser.parse_submit_response(raw_submit, expected_client_order_id=order.client_order_id)
        parsed_status = None
        if not isinstance(raw_status, Exception):
            parsed_status = receipt_parser.parse_order_status(raw_status, expected_client_order_id=order.client_order_id)

        stage_filled = _safe_decimal(parsed_status.filled_size if parsed_status is not None else "0")
        remaining_inventory = _quantize_shares(max(order.size - stage_filled, Decimal("0")))
        if remaining_inventory > Decimal("0"):
            unresolved.append(
                FlattenExposure(
                    source_client_order_id=order.source_client_order_id or "",
                    event_id=order.event_id,
                    event_title=order.event_title,
                    condition_id=order.condition_id,
                    market_id=order.market_id,
                    action=order.action,
                    side=order.side,
                    token_id=order.token_id,
                    remaining_size=remaining_inventory,
                )
            )

        if remaining_inventory <= Decimal("0"):
            stage_status = f"STAGE{flatten_stage}_SUCCESS"
        elif flatten_stage == 1:
            stage_status = "STAGE1_ESCALATE_TO_STAGE2"
        else:
            stage_status = "STAGE2_FAILED"

        flatten_results.append(
            FlattenExecutionResult(
                event_id=order.event_id,
                event_title=order.event_title,
                source_client_order_id=order.source_client_order_id or "",
                flatten_client_order_id=order.client_order_id,
                condition_id=order.condition_id,
                action=order.action,
                side=order.side,
                price=format(order.price, "f"),
                size=format(order.size, "f"),
                submit_status=submit_result.status,
                rejection_reason=submit_result.rejection_reason,
                fill_status=parsed_status.fill_status if parsed_status is not None else None,
                filled_size=format(parsed_status.filled_size, "f") if parsed_status is not None else None,
                remaining_size=format(parsed_status.remaining_size, "f") if parsed_status is not None else None,
                average_fill_price=(format(parsed_status.average_fill_price, "f") if parsed_status and parsed_status.average_fill_price is not None else None),
                remaining_inventory_after_stage=format(remaining_inventory, "f"),
                flatten_stage=flatten_stage,
                flatten_price_mode=price_mode,
                flatten_status=stage_status,
            )
        )

    return flatten_results, unresolved


async def _execute_flatten_workflow(
    *,
    transport: Any,
    clob_client: ClobClient,
    owner_id: str,
    signer: ClobSigner,
    payload_builder: ClobPayloadBuilder,
    receipt_parser: ClobReceiptParser,
    nonce_manager: ClobNonceManager,
    client_ids: ClientOrderIdGenerator,
    prepared_by_client_id: dict[str, PreparedOrder],
    leg_results: list[LegExecutionResult],
) -> list[FlattenExecutionResult]:
    exposures = _flatten_exposures(leg_results, prepared_by_client_id=prepared_by_client_id)
    prepared_flatten_orders, skipped_flatten = _prepare_flatten_orders(
        exposures,
        clob_client=clob_client,
        owner_id=owner_id,
        signer=signer,
        payload_builder=payload_builder,
        nonce_manager=nonce_manager,
        client_ids=client_ids,
        flatten_stage=1,
    )
    flatten_results: list[FlattenExecutionResult] = list(skipped_flatten)
    if not prepared_flatten_orders:
        return flatten_results

    stage1_results, unresolved = await _execute_flatten_stage(
        transport=transport,
        receipt_parser=receipt_parser,
        prepared_orders=prepared_flatten_orders,
        flatten_stage=1,
        price_mode="BBO_IOC",
    )
    flatten_results.extend(stage1_results)
    if not unresolved:
        return flatten_results

    stage2_orders, skipped_stage2 = _prepare_flatten_orders(
        unresolved,
        clob_client=clob_client,
        owner_id=owner_id,
        signer=signer,
        payload_builder=payload_builder,
        nonce_manager=nonce_manager,
        client_ids=client_ids,
        flatten_stage=2,
    )
    flatten_results.extend(skipped_stage2)
    if not stage2_orders:
        return flatten_results

    stage2_results, _ = await _execute_flatten_stage(
        transport=transport,
        receipt_parser=receipt_parser,
        prepared_orders=stage2_orders,
        flatten_stage=2,
        price_mode="PANIC_IOC",
    )
    flatten_results.extend(stage2_results)
    return flatten_results


def _validate_live_credentials() -> None:
    missing: list[str] = []
    for attr, env_name in (
        ("polymarket_api_key", "POLYMARKET_API_KEY"),
        ("eoa_private_key", "EOA_PRIVATE_KEY"),
    ):
        if not getattr(settings, attr, ""):
            missing.append(env_name)
    if missing:
        raise RuntimeError("Missing required credentials: " + ", ".join(missing))


def load_targets(path: Path) -> list[StripTarget]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = payload.get("targets") if isinstance(payload, dict) else None
    if not isinstance(targets, list):
        raise ValueError(f"Expected targets list in {path}")

    resolved: list[StripTarget] = []
    for item in targets:
        if not isinstance(item, dict):
            continue
        action = _clean_text(item.get("recommended_action"))
        if action not in {"BUY_YES_STRIP", "SELL_NO_STRIP"}:
            continue
        raw_legs = item.get("legs")
        if not isinstance(raw_legs, list) or not raw_legs:
            continue
        legs = [
            StripLeg(
                condition_id=_clean_text(leg.get("condition_id")),
                market_id=_clean_text(leg.get("market_id")),
                question=_clean_text(leg.get("question")),
                outcome_label=_clean_text(leg.get("outcome_label")),
                yes_token_id=_clean_text(leg.get("yes_token_id")),
                no_token_id=_clean_text(leg.get("no_token_id")),
                best_bid=_safe_decimal(leg.get("best_bid")),
                best_bid_size_shares=_safe_decimal(leg.get("best_bid_size_shares")),
                best_bid_notional_usd=_safe_decimal(leg.get("best_bid_notional_usd")),
                best_ask=_safe_decimal(leg.get("best_ask")),
                best_ask_size_shares=_safe_decimal(leg.get("best_ask_size_shares")),
                best_ask_notional_usd=_safe_decimal(leg.get("best_ask_notional_usd")),
            )
            for leg in raw_legs
            if isinstance(leg, dict)
        ]
        if not legs:
            continue
        resolved.append(
            StripTarget(
                event_id=_clean_text(item.get("event_id")),
                event_title=_clean_text(item.get("event_title")),
                event_slug=_clean_text(item.get("event_slug")),
                recommended_action=action,
                execution_price_sum=_safe_decimal(item.get("execution_price_sum")),
                min_leg_depth_usd_observed=_safe_decimal(item.get("min_leg_depth_usd_observed")),
                strip_max_size_shares_at_bbo=_safe_decimal(item.get("strip_max_size_shares_at_bbo")),
                legs=legs,
            )
        )
    if not resolved:
        raise ValueError(f"No executable strips found in {path}")
    return resolved


def filter_targets(targets: list[StripTarget], args: argparse.Namespace) -> list[StripTarget]:
    filtered = targets
    if args.event_id:
        allowed = {value.strip() for value in args.event_id if value.strip()}
        filtered = [target for target in filtered if target.event_id in allowed]
    if args.max_strips > 0:
        filtered = filtered[: args.max_strips]
    if not filtered:
        raise ValueError("No strips remain after applying filters")
    return filtered


def build_order_plan(target: StripTarget, *, time_in_force: ClobTimeInForce) -> OrderPlan:
    safe_notional_cap = _round_down_whole_dollars(target.min_leg_depth_usd_observed)
    if safe_notional_cap < Decimal(str(EXCHANGE_MIN_USD)):
        raise RuntimeError(
            f"{target.event_title}: dynamic strip cap ${safe_notional_cap} is below exchange minimum ${EXCHANGE_MIN_USD}"
        )
    if target.execution_price_sum <= Decimal("0"):
        raise RuntimeError(f"{target.event_title}: execution_price_sum must be positive")

    shares_from_cap = safe_notional_cap / target.execution_price_sum
    strip_shares = _quantize_shares(min(shares_from_cap, target.strip_max_size_shares_at_bbo))
    if strip_shares < Decimal(str(EXCHANGE_MIN_SHARES)):
        raise RuntimeError(
            f"{target.event_title}: computed strip size {strip_shares} is below exchange minimum shares {EXCHANGE_MIN_SHARES}"
        )

    total_notional = (strip_shares * target.execution_price_sum).quantize(USD_CENT, rounding=ROUND_DOWN)
    if total_notional < Decimal(str(EXCHANGE_MIN_USD)):
        raise RuntimeError(
            f"{target.event_title}: computed strip notional ${total_notional} is below exchange minimum ${EXCHANGE_MIN_USD}"
        )

    if target.recommended_action == "BUY_YES_STRIP":
        action: Literal["BUY", "SELL"] = "BUY"
        side: Literal["YES", "NO"] = "YES"
    else:
        action = "SELL"
        side = "NO"

    return OrderPlan(
        event_id=target.event_id,
        event_title=target.event_title,
        action=action,
        side=side,
        strip_notional_cap_usd=safe_notional_cap,
        strip_shares=strip_shares,
        total_notional_usd=total_notional,
        total_execution_price=target.execution_price_sum,
        time_in_force=time_in_force,
        legs=target.legs,
    )


def prepare_orders(
    plan: OrderPlan,
    *,
    owner_id: str,
    signer: ClobSigner,
    payload_builder: ClobPayloadBuilder,
    nonce_manager: ClobNonceManager,
    client_ids: ClientOrderIdGenerator,
) -> list[PreparedOrder]:
    nonces = nonce_manager.reserve_nonces(len(plan.legs))
    timestamp_ms = _now_ms()
    prepared: list[PreparedOrder] = []

    for index, leg in enumerate(plan.legs):
        if plan.side == "YES":
            token_id = leg.yes_token_id
            price = leg.best_ask
        else:
            token_id = leg.no_token_id
            price = leg.best_bid

        notional = (price * plan.strip_shares).quantize(USD_CENT, rounding=ROUND_DOWN)
        client_order_id = client_ids.generate(leg.condition_id, plan.side, timestamp_ms + index)
        intent = ClobOrderIntent(
            condition_id=leg.condition_id,
            token_id=token_id,
            outcome=plan.side,
            action=plan.action,
            price=price,
            size=plan.strip_shares,
            time_in_force=plan.time_in_force,
            client_order_id=client_order_id,
            post_only=False,
            fee_rate_bps=0,
            nonce=nonces[index],
            expiration=0,
            taker=ZERO_ADDRESS,
        )
        create_order_payload = payload_builder.build_create_order_payload(intent)
        signed_order = signer.sign_create_order_payload(create_order_payload)
        post_payload = payload_builder.build_post_order_payload(
            signed_order=signed_order,
            owner_id=owner_id,
            time_in_force=plan.time_in_force,
            post_only=False,
        )
        post_payload["clientOrderId"] = client_order_id
        prepared.append(
            PreparedOrder(
                event_id=plan.event_id,
                event_title=plan.event_title,
                condition_id=leg.condition_id,
                market_id=leg.market_id,
                client_order_id=client_order_id,
                action=plan.action,
                side=plan.side,
                price=price,
                size=plan.strip_shares,
                notional_usd=notional,
                token_id=token_id,
                source_client_order_id=None,
                payload=post_payload,
            )
        )
    return prepared


async def _post_batch(transport: Any, prepared_orders: list[PreparedOrder]) -> list[Any]:
    coros = [transport.post_order(order.payload or {}) for order in prepared_orders]
    return await asyncio.gather(*coros, return_exceptions=True)


async def _fetch_status_batch(transport: Any, prepared_orders: list[PreparedOrder]) -> list[Any]:
    coros = [transport.get_order(order.client_order_id) for order in prepared_orders]
    return await asyncio.gather(*coros, return_exceptions=True)


async def execute_strip(
    plan: OrderPlan,
    *,
    env: str,
    transport: Any,
    signer: ClobSigner,
    payload_builder: ClobPayloadBuilder,
    receipt_parser: ClobReceiptParser,
    nonce_manager: ClobNonceManager,
    client_ids: ClientOrderIdGenerator,
    owner_id: str,
) -> tuple[list[LegExecutionResult], list[PreparedOrder], int]:
    intercepted_before = len(getattr(transport, "intercepted_orders", [])) if env == "PAPER" else 0
    prepared_orders = prepare_orders(
        plan,
        owner_id=owner_id,
        signer=signer,
        payload_builder=payload_builder,
        nonce_manager=nonce_manager,
        client_ids=client_ids,
    )
    raw_submit_responses = await _post_batch(transport, prepared_orders)

    parsed_submit: list[VenueOrderResponse | Exception] = []
    for order, raw_response in zip(prepared_orders, raw_submit_responses, strict=True):
        if isinstance(raw_response, Exception):
            parsed_submit.append(raw_response)
            continue
        parsed_submit.append(
            receipt_parser.parse_submit_response(
                raw_response,
                expected_client_order_id=order.client_order_id,
            )
        )

    raw_status_responses = await _fetch_status_batch(transport, prepared_orders)
    results: list[LegExecutionResult] = []
    for order, submit_result, status_raw in zip(prepared_orders, parsed_submit, raw_status_responses, strict=True):
        if isinstance(submit_result, Exception):
            results.append(
                LegExecutionResult(
                    event_id=order.event_id,
                    event_title=order.event_title,
                    client_order_id=order.client_order_id,
                    condition_id=order.condition_id,
                    action=order.action,
                    side=order.side,
                    price=format(order.price, "f"),
                    size=format(order.size, "f"),
                    notional_usd=format(order.notional_usd, "f"),
                    submit_status="ERROR",
                    rejection_reason=str(submit_result),
                    fill_status=None,
                    filled_size=None,
                    remaining_size=None,
                    average_fill_price=None,
                )
            )
            continue

        parsed_status: VenueOrderStatus | None = None
        if not isinstance(status_raw, Exception):
            parsed_status = receipt_parser.parse_order_status(
                status_raw,
                expected_client_order_id=order.client_order_id,
            )

        results.append(
            LegExecutionResult(
                event_id=order.event_id,
                event_title=order.event_title,
                client_order_id=order.client_order_id,
                condition_id=order.condition_id,
                action=order.action,
                side=order.side,
                price=format(order.price, "f"),
                size=format(order.size, "f"),
                notional_usd=format(order.notional_usd, "f"),
                submit_status=submit_result.status,
                rejection_reason=submit_result.rejection_reason,
                fill_status=parsed_status.fill_status if parsed_status is not None else None,
                filled_size=format(parsed_status.filled_size, "f") if parsed_status is not None else None,
                remaining_size=format(parsed_status.remaining_size, "f") if parsed_status is not None else None,
                average_fill_price=(format(parsed_status.average_fill_price, "f") if parsed_status and parsed_status.average_fill_price is not None else None),
            )
        )

    paper_intercepted_after = len(getattr(transport, "intercepted_orders", [])) if env == "PAPER" else 0
    paper_intercepted = paper_intercepted_after - intercepted_before
    return results, prepared_orders, paper_intercepted


async def _run_async(
    plans: list[OrderPlan],
    *,
    env: str,
    owner_id: str,
    signer: ClobSigner,
    payload_builder: ClobPayloadBuilder,
    receipt_parser: ClobReceiptParser,
    nonce_manager: ClobNonceManager,
    client_ids: ClientOrderIdGenerator,
) -> tuple[list[dict[str, Any]], int]:
    if env == "PAPER":
        transport = PaperClobTransport(now_ms=_now_ms)
    else:
        transport = AiohttpClobTransport(base_url=settings.clob_http_url, now_ms=_now_ms)
    clob_client = ClobClient(DEFAULT_CLOB_URL)

    all_results: list[dict[str, Any]] = []
    paper_intercepted_total = 0
    try:
        if env == "LIVE":
            wallet_balance = await transport.get_wallet_balance("USDC")
        else:
            wallet_balance = Decimal("1000000")

        for plan in plans:
            if plan.action == "BUY" and wallet_balance < plan.total_notional_usd:
                all_results.append(
                    {
                        "event_id": plan.event_id,
                        "event_title": plan.event_title,
                        "recommended_action": f"{plan.action}_{plan.side}",
                        "strip_notional_cap_usd": format(plan.strip_notional_cap_usd, "f"),
                        "strip_shares": format(plan.strip_shares, "f"),
                        "total_notional_usd": format(plan.total_notional_usd, "f"),
                        "time_in_force": plan.time_in_force.value,
                        "status": "SKIPPED",
                        "reason": "INSUFFICIENT_USDC",
                        "legs": [],
                    }
                )
                continue

            leg_results, prepared_orders, paper_intercepted = await execute_strip(
                plan,
                env=env,
                transport=transport,
                signer=signer,
                payload_builder=payload_builder,
                receipt_parser=receipt_parser,
                nonce_manager=nonce_manager,
                client_ids=client_ids,
                owner_id=owner_id,
            )
            paper_intercepted_total += paper_intercepted
            if plan.action == "BUY":
                wallet_balance -= plan.total_notional_usd

            prepared_by_client_id = {order.client_order_id: order for order in prepared_orders}
            filled_leg_results = [result for result in leg_results if _has_any_fill(result)]
            fully_filled = len(leg_results) == len(plan.legs) and all(_is_full_fill(result) for result in leg_results)
            flatten_results: list[FlattenExecutionResult] = []
            flatten_triggered = False
            flatten_stage2_triggered = False
            flatten_failed = False
            flatten_unresolved_inventory = Decimal("0")
            if fully_filled:
                strip_status = "FULLY_FILLED"
            elif filled_leg_results:
                flatten_triggered = True
                flatten_results = await _execute_flatten_workflow(
                    transport=transport,
                    clob_client=clob_client,
                    owner_id=owner_id,
                    signer=signer,
                    payload_builder=payload_builder,
                    receipt_parser=receipt_parser,
                    nonce_manager=nonce_manager,
                    client_ids=client_ids,
                    prepared_by_client_id=prepared_by_client_id,
                    leg_results=leg_results,
                )
                flatten_stage2_triggered = any(result.flatten_stage == 2 for result in flatten_results)
                latest_flatten_by_source: dict[str, FlattenExecutionResult] = {}
                for result in flatten_results:
                    source_key = result.source_client_order_id or result.flatten_client_order_id or ""
                    existing = latest_flatten_by_source.get(source_key)
                    if existing is None or result.flatten_stage >= existing.flatten_stage:
                        latest_flatten_by_source[source_key] = result
                flatten_unresolved_inventory = sum(
                    (_safe_decimal(result.remaining_inventory_after_stage) for result in latest_flatten_by_source.values()),
                    start=Decimal("0"),
                )
                if flatten_unresolved_inventory <= Decimal("0"):
                    if flatten_stage2_triggered:
                        strip_status = "PARTIAL_FILL_FLATTEN_STAGE2_SUCCESS"
                    else:
                        strip_status = "PARTIAL_FILL_FLATTEN_STAGE1_SUCCESS"
                else:
                    flatten_failed = True
                    strip_status = "PARTIAL_FILL_FLATTEN_FAILED"
            else:
                leg_submit_statuses = {result.submit_status for result in leg_results}
                leg_fill_statuses = {result.fill_status for result in leg_results if result.fill_status}
                if leg_submit_statuses == {"REJECTED"} or leg_fill_statuses == {"CANCELLED"}:
                    strip_status = "NO_FILL"
                else:
                    strip_status = "SUBMITTED"

            all_results.append(
                {
                    "event_id": plan.event_id,
                    "event_title": plan.event_title,
                    "recommended_action": f"{plan.action}_{plan.side}",
                    "strip_notional_cap_usd": format(plan.strip_notional_cap_usd, "f"),
                    "strip_shares": format(plan.strip_shares, "f"),
                    "total_notional_usd": format(plan.total_notional_usd, "f"),
                    "time_in_force": plan.time_in_force.value,
                    "filled_legs": len(filled_leg_results),
                    "attempted_legs": len(plan.legs),
                    "flatten_triggered": flatten_triggered,
                    "flatten_stage2_triggered": flatten_stage2_triggered,
                    "flatten_failed": flatten_failed,
                    "flatten_unresolved_inventory": format(flatten_unresolved_inventory, "f"),
                    "status": strip_status,
                    "legs": [asdict(result) for result in leg_results],
                    "flatten_legs": [asdict(result) for result in flatten_results],
                }
            )
    finally:
        await transport.close()

    return all_results, paper_intercepted_total


def main() -> None:
    args = parse_args()
    loaded_targets = load_targets(args.input)
    try:
        targets = filter_targets(loaded_targets, args)
    except ValueError:
        if not args.allow_empty:
            raise
        summary = build_json_summary(
            env=args.env,
            input_path=args.input,
            plans=[],
            results=[],
            paper_intercepted_total=0,
        )
        write_summaries(
            json_output=args.json_output,
            markdown_output=args.markdown_output,
            summary=summary,
        )
        print(f"Launch mode: {args.env}")
        print("Targets loaded: 0")
        print("Strips attempted: 0")
        print("Paper intercepted payloads: 0")
        print(f"Launch summary written to {args.json_output}")
        print(f"Markdown receipt written to {args.markdown_output}")
        print(json.dumps(summary, indent=2))
        return
    time_in_force = ClobTimeInForce(args.time_in_force)
    plans = [build_order_plan(target, time_in_force=time_in_force) for target in targets]

    if args.env == "LIVE":
        _validate_live_credentials()
    owner_id = _clean_text(settings.polymarket_api_key)
    private_key = _clean_text(settings.eoa_private_key)
    if not owner_id or not private_key:
        raise RuntimeError("POLYMARKET_API_KEY and EOA_PRIVATE_KEY are required for launch_clob_arb")

    from py_clob_client.config import get_contract_config

    contract_config = get_contract_config(CHAIN_ID)
    signer = ClobSigner(
        private_key=private_key,
        chain_id=CHAIN_ID,
        exchange_address=contract_config.exchange,
    )
    payload_builder = ClobPayloadBuilder()
    receipt_parser = ClobReceiptParser()
    nonce_manager = ClobNonceManager()
    client_ids = ClientOrderIdGenerator("MANUAL", _session_id())

    results, paper_intercepted_total = asyncio.run(
        _run_async(
            plans,
            env=args.env,
            owner_id=owner_id,
            signer=signer,
            payload_builder=payload_builder,
            receipt_parser=receipt_parser,
            nonce_manager=nonce_manager,
            client_ids=client_ids,
        )
    )

    print(f"Launch mode: {args.env}")
    print(f"Targets loaded: {len(targets)}")
    print(f"Strips attempted: {len(plans)}")
    print(f"Paper intercepted payloads: {paper_intercepted_total}")
    summary = build_json_summary(
        env=args.env,
        input_path=args.input,
        plans=plans,
        results=results,
        paper_intercepted_total=paper_intercepted_total,
    )
    write_summaries(
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        summary=summary,
    )
    print(f"Launch summary written to {args.json_output}")
    print(f"Markdown receipt written to {args.markdown_output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()