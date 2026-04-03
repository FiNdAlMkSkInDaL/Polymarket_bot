#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Callable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.clob_signer import ClobSigner
from src.execution.clob_transport import AiohttpClobTransport
from src.execution.live_execution_boundary import _AsyncTransportLoopRunner
from src.execution.nonce_manager import ClobNonceManager
from src.execution.polymarket_clob_adapter import PolymarketClobAdapter
from src.execution.polymarket_clob_translator import ClobPayloadBuilder, ClobReceiptParser
from src.execution.venue_adapter_interface import VenueOrderStatus


DEFAULT_INPUT = PROJECT_ROOT / "data" / "flb_results_final.json"
DEFAULT_STATE = PROJECT_ROOT / "data" / "underwriter_state.json"
DEFAULT_PAPER_STATE = PROJECT_ROOT / "data" / "underwriter_state_paper.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "docs" / "underwriter_launch_report.md"
DEFAULT_PAPER_MARKDOWN = PROJECT_ROOT / "docs" / "underwriter_launch_report_paper.md"
DEFAULT_JSON = PROJECT_ROOT / "data" / "underwriter_launch_summary.json"
DEFAULT_PAPER_JSON = PROJECT_ROOT / "data" / "underwriter_launch_summary_paper.json"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CHAIN_ID = 137
ENTRY_PRICE = Decimal("0.95")
MAX_NOTIONAL_PER_CONDITION = Decimal("50")
ORDER_SIZE_QUANT = Decimal("0.000001")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        result = Decimal(str(value))
    except Exception:
        return None
    if not result.is_finite():
        return None
    return result


def _normalize_condition_id(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError("condition_id is required")
    if not normalized.startswith("0x"):
        normalized = "0x" + normalized
    body = normalized[2:]
    if len(body) != 64:
        raise ValueError(f"condition_id must be 64 hex chars: {value!r}")
    int(body, 16)
    return normalized


def _session_id(prefix: str = "underwriter") -> str:
    return f"{prefix}-{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"


def _quantize_size(value: Decimal) -> Decimal:
    return value.quantize(ORDER_SIZE_QUANT, rounding=ROUND_DOWN)


@dataclass(frozen=True, slots=True)
class ActiveTarget:
    condition_id: str
    question: str
    category: str
    entry_yes_ask: Decimal
    terminal_yes_ask: Decimal | None
    max_yes_ask: Decimal | None
    max_no_drawdown_cents: Decimal | None
    yes_token_id: str | None = None
    no_token_id: str | None = None

    @property
    def target_notional_usd(self) -> Decimal:
        return MAX_NOTIONAL_PER_CONDITION

    @property
    def order_price(self) -> Decimal:
        return ENTRY_PRICE

    @property
    def order_size(self) -> Decimal:
        return _quantize_size(self.target_notional_usd / self.order_price)


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    condition_id: str
    question: str
    category: str
    no_token_id: str
    yes_token_id: str


@dataclass(slots=True)
class StateEntry:
    condition_id: str
    client_order_id: str
    venue_order_id: str | None
    last_status: str
    last_seen_at: str
    question: str
    category: str
    order_price: str
    order_size: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "client_order_id": self.client_order_id,
            "venue_order_id": self.venue_order_id,
            "last_status": self.last_status,
            "last_seen_at": self.last_seen_at,
            "question": self.question,
            "category": self.category,
            "order_price": self.order_price,
            "order_size": self.order_size,
        }


class _InlineTransportRunner:
    def run(self, coro: Any) -> Any:
        return asyncio.run(coro)

    def shutdown(self) -> None:
        return None


class PaperClobTransport:
    def __init__(self, *, now_ms: Callable[[], int]) -> None:
        self._now_ms = now_ms
        self.intercepted_orders: list[dict[str, Any]] = []
        self._orders_by_client_id: dict[str, dict[str, Any]] = {}

    async def close(self) -> None:
        return None

    async def post_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        client_order_id = str(payload.get("order", {}).get("client_order_id") or "").strip()
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

    async def cancel_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        client_order_id = str(payload.get("clientOrderId") or "").strip()
        existing = self._orders_by_client_id.get(client_order_id)
        if existing is None:
            return {
                "clientOrderId": client_order_id,
                "status": "REJECTED",
                "reason": "ORDER_NOT_FOUND",
                "timestamp": int(self._now_ms()),
            }
        existing["status"] = "CANCELLED"
        return {
            "clientOrderId": client_order_id,
            "orderID": existing["orderID"],
            "status": "CANCELLED",
            "timestamp": int(self._now_ms()),
        }

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

    async def get_expected_nonce(self, payload: dict[str, Any]) -> int:
        return 0

    async def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        return Decimal("1000000")


class PaperClobPayloadBuilder(ClobPayloadBuilder):
    def build_post_order_payload(
        self,
        *,
        signed_order: dict[str, Any],
        owner_id: str,
        time_in_force: Any,
        post_only: bool,
    ) -> dict[str, Any]:
        payload = super().build_post_order_payload(
            signed_order=signed_order,
            owner_id=owner_id,
            time_in_force=time_in_force,
            post_only=post_only,
        )
        client_order_id = str(signed_order.get("client_order_id") or "").strip()
        if client_order_id:
            payload["clientOrderId"] = client_order_id
        return payload


class _ClientTokenResolver:
    def __init__(self, owner_id: str, targets_by_condition: dict[str, ResolvedTarget]) -> None:
        self.owner_id = owner_id
        self._targets_by_condition = targets_by_condition

    def resolve_market_token(self, payload: dict[str, str]) -> dict[str, str]:
        condition_id = _normalize_condition_id(str(payload.get("conditionId") or ""))
        outcome = str(payload.get("outcome") or "").strip().upper()
        target = self._targets_by_condition.get(condition_id)
        if target is None:
            raise ValueError(f"Unknown condition_id: {condition_id}")
        if outcome == "NO":
            return {"token_id": target.no_token_id}
        if outcome == "YES":
            return {"token_id": target.yes_token_id}
        raise ValueError(f"Unsupported outcome: {outcome!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the FLB underwriter resting-NO strategy.")
    parser.add_argument(
        "--env",
        choices=("PAPER", "LIVE"),
        default="LIVE",
        help="Execution mode. PAPER builds and signs live-equivalent payloads then intercepts them locally; LIVE submits to the CLOB.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to data/flb_results_final.json")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE, help="State file used to avoid duplicate live orders.")
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN, help="Human-readable launch summary path.")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON, help="Machine-readable launch summary path.")
    parser.add_argument("--gamma-timeout-seconds", type=float, default=20.0, help="HTTP timeout for Gamma token resolution.")
    parser.add_argument("--dry-run", action="store_true", help="Deprecated alias for resolving and sizing orders without building submission payloads.")
    return parser.parse_args()


def validate_live_credentials() -> None:
    missing = settings.validate_credentials()
    if missing:
        raise RuntimeError("Missing required live credentials: " + ", ".join(missing))


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    state_path = args.state_path
    markdown_output = args.markdown_output
    json_output = args.json_output
    if args.env == "PAPER":
        if state_path == DEFAULT_STATE:
            state_path = DEFAULT_PAPER_STATE
        if markdown_output == DEFAULT_MARKDOWN:
            markdown_output = DEFAULT_PAPER_MARKDOWN
        if json_output == DEFAULT_JSON:
            json_output = DEFAULT_PAPER_JSON
    return state_path, markdown_output, json_output


def load_active_targets(path: Path) -> list[ActiveTarget]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    active_rows = raw.get("active_markets") if isinstance(raw, dict) else None
    if not isinstance(active_rows, list):
        raise ValueError(f"Expected active_markets list in {path}")
    targets: list[ActiveTarget] = []
    for row in active_rows:
        if not isinstance(row, dict):
            continue
        condition_id = _normalize_condition_id(str(row.get("condition_id") or ""))
        entry_yes_ask = _safe_decimal(row.get("entry_yes_ask"))
        if entry_yes_ask is None:
            continue
        targets.append(
            ActiveTarget(
                condition_id=condition_id,
                question=str(row.get("question") or condition_id),
                category=str(row.get("category") or "unknown"),
                entry_yes_ask=entry_yes_ask,
                terminal_yes_ask=_safe_decimal(row.get("terminal_yes_ask")),
                max_yes_ask=_safe_decimal(row.get("max_yes_ask")),
                max_no_drawdown_cents=_safe_decimal(row.get("max_no_drawdown_cents")),
                yes_token_id=str(row.get("yes_token_id") or "").strip() or None,
                no_token_id=str(row.get("no_token_id") or "").strip() or None,
            )
        )
    if not targets:
        raise ValueError(f"No active underwriter targets found in {path}")
    return targets


def resolve_targets_via_gamma(targets: list[ActiveTarget], *, timeout_seconds: float) -> dict[str, ResolvedTarget]:
    session = requests.Session()
    resolved: dict[str, ResolvedTarget] = {}
    for index, target in enumerate(targets, start=1):
        if target.yes_token_id and target.no_token_id:
            resolved[target.condition_id] = ResolvedTarget(
                condition_id=target.condition_id,
                question=target.question,
                category=target.category,
                no_token_id=target.no_token_id,
                yes_token_id=target.yes_token_id,
            )
            if index % 250 == 0:
                print(f"Resolved {index} / {len(targets)} underwriter targets...", flush=True)
            continue

        response = session.get(
            GAMMA_MARKETS_URL,
            params={"condition_ids": target.condition_id},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            raise RuntimeError(f"Gamma returned no metadata for {target.condition_id}")
        row = payload[0]
        outcomes = row.get("outcomes")
        token_ids = row.get("clobTokenIds")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        if not isinstance(outcomes, list) or not isinstance(token_ids, list):
            raise RuntimeError(f"Gamma metadata malformed for {target.condition_id}")
        yes_token_id = ""
        no_token_id = ""
        for item_index, outcome in enumerate(outcomes):
            label = str(outcome or "").strip().upper()
            token_id = str(token_ids[item_index] if item_index < len(token_ids) else "").strip()
            if label == "YES":
                yes_token_id = token_id
            elif label == "NO":
                no_token_id = token_id
        if not yes_token_id or not no_token_id:
            raise RuntimeError(f"Could not resolve YES/NO tokens for {target.condition_id}")
        resolved[target.condition_id] = ResolvedTarget(
            condition_id=target.condition_id,
            question=target.question,
            category=target.category,
            no_token_id=no_token_id,
            yes_token_id=yes_token_id,
        )
        if index % 25 == 0:
            print(f"Resolved {index} / {len(targets)} underwriter targets...", flush=True)
    return resolved


def load_state(path: Path) -> dict[str, StateEntry]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    entries: dict[str, StateEntry] = {}
    for condition_id, item in raw.items():
        if not isinstance(item, dict):
            continue
        try:
            normalized = _normalize_condition_id(condition_id)
        except ValueError:
            continue
        entries[normalized] = StateEntry(
            condition_id=normalized,
            client_order_id=str(item.get("client_order_id") or ""),
            venue_order_id=str(item.get("venue_order_id") or "") or None,
            last_status=str(item.get("last_status") or "UNKNOWN"),
            last_seen_at=str(item.get("last_seen_at") or ""),
            question=str(item.get("question") or normalized),
            category=str(item.get("category") or "unknown"),
            order_price=str(item.get("order_price") or ENTRY_PRICE),
            order_size=str(item.get("order_size") or "0"),
        )
    return entries


def save_state(path: Path, entries: dict[str, StateEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {condition_id: entry.to_dict() for condition_id, entry in sorted(entries.items())}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_adapter(
    targets_by_condition: dict[str, ResolvedTarget],
    *,
    env: str,
) -> tuple[PolymarketClobAdapter, Any, Any]:
    from py_clob_client.config import get_contract_config

    owner_id = str(settings.polymarket_api_key or "").strip()
    if not owner_id:
        raise RuntimeError("POLYMARKET_API_KEY is required")
    contract_config = get_contract_config(CHAIN_ID)
    if env == "PAPER":
        runner = _InlineTransportRunner()
        transport = PaperClobTransport(now_ms=_now_ms)
        transport_runner = runner.run
        payload_builder: ClobPayloadBuilder = PaperClobPayloadBuilder()
    else:
        runner = _AsyncTransportLoopRunner(thread_name="underwriter-clob-loop")
        transport = AiohttpClobTransport(
            base_url=settings.clob_http_url,
            now_ms=_now_ms,
        )
        transport_runner = runner.run
        payload_builder = ClobPayloadBuilder()
    adapter = PolymarketClobAdapter(
        client=_ClientTokenResolver(owner_id, targets_by_condition),
        transport=transport,
        payload_builder=payload_builder,
        receipt_parser=ClobReceiptParser(),
        nonce_manager=ClobNonceManager(),
        signer=ClobSigner(
            private_key=settings.eoa_private_key,
            chain_id=CHAIN_ID,
            exchange_address=contract_config.exchange,
        ),
        transport_runner=transport_runner,
    )
    return adapter, transport, runner


def is_live_resting_status(status: VenueOrderStatus) -> bool:
    return status.fill_status in {"OPEN", "PARTIAL", "UNKNOWN"}


def render_markdown(
    *,
    submitted: list[dict[str, str]],
    skipped_existing: list[dict[str, str]],
    rejected: list[dict[str, str]],
    dry_run_rows: list[dict[str, str]],
) -> str:
    lines = [
        "# Underwriter Launch Report",
        "",
        f"- Generated at: `{_utc_now().strftime('%Y-%m-%d %H:%M:%SZ')}`",
        f"- Resting order price: `{ENTRY_PRICE}` on NO",
        f"- Max notional per condition: `${MAX_NOTIONAL_PER_CONDITION}`",
        "- Execution posture: passive maker only, no chasing",
        "",
        "## Summary",
        "",
        f"- Submitted: `{len(submitted)}`",
        f"- Already live / skipped: `{len(skipped_existing)}`",
        f"- Rejected: `{len(rejected)}`",
        f"- Dry-run rows: `{len(dry_run_rows)}`",
        "",
    ]
    if dry_run_rows:
        lines.extend([
            "## Dry Run",
            "",
            "| Condition | Category | Price | Size | Notional USD |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for row in dry_run_rows:
            lines.append(f"| {row['question'].replace('|', '/')} | {row['category']} | {row['price']} | {row['size']} | {row['notional']} |")
        lines.append("")
    if submitted:
        lines.extend([
            "## Submitted Orders",
            "",
            "| Condition | Category | Client Order ID | Price | Size | Venue Status |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ])
        for row in submitted:
            lines.append(f"| {row['question'].replace('|', '/')} | {row['category']} | {row['client_order_id']} | {row['price']} | {row['size']} | {row['status']} |")
        lines.append("")
    if skipped_existing:
        lines.extend([
            "## Existing Live Orders",
            "",
            "| Condition | Category | Client Order ID | Fill Status |",
            "| --- | --- | --- | --- |",
        ])
        for row in skipped_existing:
            lines.append(f"| {row['question'].replace('|', '/')} | {row['category']} | {row['client_order_id']} | {row['status']} |")
        lines.append("")
    if rejected:
        lines.extend([
            "## Rejections",
            "",
            "| Condition | Category | Client Order ID | Reason |",
            "| --- | --- | --- | --- |",
        ])
        for row in rejected:
            lines.append(f"| {row['question'].replace('|', '/')} | {row['category']} | {row['client_order_id']} | {row['reason'].replace('|', '/')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def build_json_summary(
    *,
    env: str,
    input_path: Path,
    state_path: Path,
    markdown_output: Path,
    submitted: list[dict[str, str]],
    skipped_existing: list[dict[str, str]],
    rejected: list[dict[str, str]],
    dry_run_rows: list[dict[str, str]],
    paper_intercepted: int,
    targets: list[ActiveTarget],
) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    for target in targets:
        category_counts[target.category] = category_counts.get(target.category, 0) + 1
    total_notional = (Decimal(len(submitted)) * MAX_NOTIONAL_PER_CONDITION).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    return {
        "generated_at": _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": env,
        "strategy": "SHIELD",
        "input_path": str(input_path),
        "state_path": None if env == "PAPER" else str(state_path),
        "markdown_output": str(markdown_output),
        "active_targets_loaded": len(targets),
        "submitted_orders": len(submitted),
        "skipped_existing": len(skipped_existing),
        "rejected_orders": len(rejected),
        "dry_run_rows": len(dry_run_rows),
        "paper_intercepted_payloads": paper_intercepted,
        "submitted_notional_usd": format(total_notional, "f"),
        "category_counts": dict(sorted(category_counts.items())),
        "submitted": submitted,
        "skipped": skipped_existing,
        "rejected": rejected,
        "dry_run": dry_run_rows,
    }


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.env = "PAPER"
    validate_live_credentials()
    state_path, markdown_output, json_output = resolve_output_paths(args)

    targets = load_active_targets(args.input)
    resolved_targets = resolve_targets_via_gamma(targets, timeout_seconds=float(args.gamma_timeout_seconds))
    state = {} if args.env == "PAPER" else load_state(state_path)
    submitted: list[dict[str, str]] = []
    skipped_existing: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    dry_run_rows: list[dict[str, str]] = []
    paper_intercepted = 0

    adapter: PolymarketClobAdapter | None = None
    transport: Any | None = None
    runner: Any | None = None
    if not args.dry_run:
        adapter, transport, runner = build_adapter(resolved_targets, env=args.env)

    session_id = _session_id()
    client_ids = ClientOrderIdGenerator("MANUAL", session_id)

    try:
        for target in targets:
            resolved_target = resolved_targets[target.condition_id]
            notional = target.target_notional_usd
            if notional > MAX_NOTIONAL_PER_CONDITION:
                raise RuntimeError(f"Notional cap violation for {target.condition_id}: {notional}")
            order_size = target.order_size
            order_price = target.order_price
            existing = state.get(target.condition_id)
            if existing and adapter is not None:
                status = adapter.get_order_status(existing.client_order_id)
                if is_live_resting_status(status):
                    existing.last_status = status.fill_status
                    existing.last_seen_at = _utc_now().strftime("%Y-%m-%d %H:%M:%SZ")
                    skipped_existing.append(
                        {
                            "question": target.question,
                            "category": target.category,
                            "client_order_id": existing.client_order_id,
                            "status": status.fill_status,
                        }
                    )
                    continue

            if args.dry_run:
                dry_run_rows.append(
                    {
                        "question": target.question,
                        "category": target.category,
                        "price": str(order_price),
                        "size": str(order_size),
                        "notional": format((order_price * order_size).quantize(Decimal("0.01"), rounding=ROUND_DOWN), "f"),
                    }
                )
                continue

            assert adapter is not None
            client_order_id = client_ids.generate(target.condition_id, "NO", _now_ms())
            response = adapter.submit_order(
                market_id=resolved_target.condition_id,
                side="NO",
                price=order_price,
                size=order_size,
                order_type="LIMIT",
                client_order_id=client_order_id,
                time_in_force="GTC",
                post_only=True,
            )
            if response.status == "REJECTED":
                rejected.append(
                    {
                        "question": target.question,
                        "category": target.category,
                        "client_order_id": client_order_id,
                        "reason": str(response.rejection_reason or "UNKNOWN"),
                    }
                )
                continue

            state[target.condition_id] = StateEntry(
                condition_id=target.condition_id,
                client_order_id=client_order_id,
                venue_order_id=response.venue_order_id,
                last_status=response.status,
                last_seen_at=_utc_now().strftime("%Y-%m-%d %H:%M:%SZ"),
                question=target.question,
                category=target.category,
                order_price=str(order_price),
                order_size=str(order_size),
            )
            submitted.append(
                {
                    "question": target.question,
                    "category": target.category,
                    "client_order_id": client_order_id,
                    "price": str(order_price),
                    "size": str(order_size),
                    "status": response.status,
                }
            )
        if args.env == "PAPER" and isinstance(transport, PaperClobTransport):
            paper_intercepted = len(transport.intercepted_orders)
    finally:
        if transport is not None and runner is not None:
            runner.run(transport.close())
            runner.shutdown()

    if args.env != "PAPER":
        save_state(state_path, state)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(
        render_markdown(
            submitted=submitted,
            skipped_existing=skipped_existing,
            rejected=rejected,
            dry_run_rows=dry_run_rows,
        ),
        encoding="utf-8",
    )
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(
            build_json_summary(
                env=args.env,
                input_path=args.input,
                state_path=state_path,
                markdown_output=markdown_output,
                submitted=submitted,
                skipped_existing=skipped_existing,
                rejected=rejected,
                dry_run_rows=dry_run_rows,
                paper_intercepted=paper_intercepted,
                targets=targets,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.env == "PAPER" and not args.dry_run:
        print(f"Paper intercepted payloads: {paper_intercepted}")
    print(f"Active targets loaded: {len(targets)}")
    print(f"Submitted orders: {len(submitted)}")
    print(f"Skipped existing: {len(skipped_existing)}")
    print(f"Rejected orders: {len(rejected)}")
    if args.env != "PAPER":
        print(f"State written to {state_path}")
    else:
        print("State written to <skipped in PAPER mode>")
    print(f"Launch report written to {markdown_output}")
    print(f"Launch summary written to {json_output}")


if __name__ == "__main__":
    main()