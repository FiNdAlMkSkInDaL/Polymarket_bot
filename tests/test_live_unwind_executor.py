from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.ctf_execution_manifest import CtfExecutionManifest, build_ctf_execution_manifest
from src.execution.ctf_unwind_manifest import CtfUnwindLeg, CtfUnwindManifest
from src.execution.live_unwind_executor import LiveUnwindExecutor
from src.execution.si9_execution_manifest import Si9ExecutionManifest, Si9LegManifest
from src.execution.si9_unwind_manifest import Si9UnwindManifest, Si9UnwindLeg
from src.execution.unwind_executor_interface import UnwindExecutor
from src.execution.venue_adapter_interface import VenueAdapter, VenueOrderResponse, VenueOrderStatus


def _submit_response(
    *,
    status: str = "ACCEPTED",
    venue_order_id: str | None = "VENUE-1",
    rejection_reason: str | None = None,
    venue_timestamp_ms: int = 1000,
    latency_ms: int = 4,
) -> VenueOrderResponse:
    return VenueOrderResponse(
        client_order_id="template",
        venue_order_id=venue_order_id,
        status=status,  # type: ignore[arg-type]
        rejection_reason=rejection_reason,
        venue_timestamp_ms=venue_timestamp_ms,
        latency_ms=latency_ms,
    )


def _status(
    fill_status: str,
    *,
    filled_size: Decimal,
    remaining_size: Decimal,
    average_fill_price: Decimal | None,
    venue_order_id: str | None = "VENUE-1",
) -> VenueOrderStatus:
    return VenueOrderStatus(
        client_order_id="template",
        venue_order_id=venue_order_id,
        fill_status=fill_status,  # type: ignore[arg-type]
        filled_size=filled_size,
        remaining_size=remaining_size,
        average_fill_price=average_fill_price,
    )


class _ScriptedVenueAdapter(VenueAdapter):
    def __init__(
        self,
        *,
        submit_responses: list[VenueOrderResponse] | None = None,
        status_responses: list[VenueOrderStatus] | None = None,
    ) -> None:
        self._submit_responses = list(submit_responses or [])
        self._status_responses = list(status_responses or [])
        self.submit_calls: list[dict[str, object]] = []
        self.status_calls: list[str] = []
        self.cancel_calls: list[dict[str, object]] = []

    def submit_order(
        self,
        market_id: str,
        side: str,
        price: Decimal,
        size: Decimal,
        order_type: str,
        client_order_id: str,
    ) -> VenueOrderResponse:
        self.submit_calls.append(
            {
                "market_id": market_id,
                "side": side,
                "price": price,
                "size": size,
                "order_type": order_type,
                "client_order_id": client_order_id,
            }
        )
        index = len(self.submit_calls) - 1
        template = self._submit_responses[index] if index < len(self._submit_responses) else _submit_response()
        return replace(template, client_order_id=client_order_id)

    def cancel_order(self, client_order_id: str, market_id: str):
        self.cancel_calls.append({"client_order_id": client_order_id, "market_id": market_id})
        raise NotImplementedError("LiveUnwindExecutor does not cancel unwind orders")

    def get_order_status(self, client_order_id: str) -> VenueOrderStatus:
        self.status_calls.append(client_order_id)
        index = len(self.status_calls) - 1
        template = self._status_responses[index] if index < len(self._status_responses) else _status(
            "OPEN",
            filled_size=Decimal("0"),
            remaining_size=Decimal("1.000000"),
            average_fill_price=None,
        )
        return replace(template, client_order_id=client_order_id)

    def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        _ = asset_symbol
        return Decimal("100.000000")


def _si9_execution_manifest() -> Si9ExecutionManifest:
    return Si9ExecutionManifest(
        cluster_id="cluster-1",
        legs=(
            Si9LegManifest("MKT_A", "YES", Decimal("0.31"), Decimal("2"), True, 0),
            Si9LegManifest("MKT_B", "YES", Decimal("0.29"), Decimal("2"), False, 1),
        ),
        net_edge=Decimal("0.020000"),
        required_share_counts=Decimal("2.000000"),
        bottleneck_market_id="MKT_A",
        manifest_timestamp_ms=100,
        max_leg_fill_wait_ms=200,
        cancel_on_stale_ms=300,
    )


def _si9_unwind_manifest(
    *,
    recommended_action: str = "MARKET_SELL",
    hanging_legs: tuple[Si9UnwindLeg, ...] | None = None,
) -> Si9UnwindManifest:
    legs = hanging_legs or (
        Si9UnwindLeg(
            market_id="MKT_A",
            side="YES",
            filled_size=Decimal("2.000000"),
            filled_price=Decimal("0.330000"),
            current_best_bid=Decimal("0.320000"),
            estimated_unwind_cost=Decimal("0.020000"),
            leg_index=0,
        ),
        Si9UnwindLeg(
            market_id="MKT_B",
            side="YES",
            filled_size=Decimal("2.000000"),
            filled_price=Decimal("0.310000"),
            current_best_bid=Decimal("0.300000"),
            estimated_unwind_cost=Decimal("0.020000"),
            leg_index=1,
        ),
    )
    return Si9UnwindManifest(
        cluster_id="cluster-1",
        hanging_legs=legs,
        unwind_reason="SECOND_LEG_REJECTED",
        original_manifest=_si9_execution_manifest(),
        unwind_timestamp_ms=400,
        total_estimated_unwind_cost=Decimal("0.040000"),
        recommended_action=recommended_action,  # type: ignore[arg-type]
    )


def _ctf_execution_manifest() -> CtfExecutionManifest:
    return build_ctf_execution_manifest(
        market_id="MKT_CTF",
        yes_price=Decimal("0.380000"),
        no_price=Decimal("0.400000"),
        net_edge=Decimal("0.185000"),
        gas_estimate=Decimal("0.010000"),
        default_anchor_volume=Decimal("10.000000"),
        max_capital_per_signal=Decimal("25.000000"),
        max_size_per_leg=Decimal("8.000000"),
        taker_fee_yes=Decimal("0.010000"),
        taker_fee_no=Decimal("0.010000"),
        manifest_timestamp_ms=1000,
        cancel_on_stale_ms=100,
    )


def _ctf_unwind_manifest(*, recommended_action: str = "MARKET_SELL") -> CtfUnwindManifest:
    return CtfUnwindManifest(
        cluster_id="MKT_CTF",
        hanging_legs=(
            CtfUnwindLeg(
                market_id="MKT_CTF",
                side="NO",
                filled_size=Decimal("3.000000"),
                filled_price=Decimal("0.390000"),
                current_best_bid=Decimal("0.385000"),
                estimated_unwind_cost=Decimal("0.015000"),
                leg_index=0,
            ),
        ),
        unwind_reason="SECOND_LEG_REJECTED",
        original_manifest=_ctf_execution_manifest(),
        unwind_timestamp_ms=500,
        total_estimated_unwind_cost=Decimal("0.015000"),
        recommended_action=recommended_action,  # type: ignore[arg-type]
    )


def _make_executor(
    *,
    signal_source: str = "SI9",
    submit_responses: list[VenueOrderResponse] | None = None,
    status_responses: list[VenueOrderStatus] | None = None,
) -> tuple[LiveUnwindExecutor, _ScriptedVenueAdapter]:
    adapter = _ScriptedVenueAdapter(
        submit_responses=submit_responses,
        status_responses=status_responses,
    )
    return LiveUnwindExecutor(adapter, ClientOrderIdGenerator(signal_source, "abc12345-session")), adapter


def test_live_unwind_executor_satisfies_interface() -> None:
    executor, _ = _make_executor()

    assert isinstance(executor, UnwindExecutor)


def test_market_sell_si9_manifest_submits_market_orders_for_all_hanging_legs() -> None:
    executor, adapter = _make_executor(
        status_responses=[
            _status("FILLED", filled_size=Decimal("2.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.320000")),
            _status("FILLED", filled_size=Decimal("2.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.300000")),
        ]
    )

    receipt = executor.execute_unwind(_si9_unwind_manifest(), 700)

    assert receipt.action_taken == "MARKET_SELL"
    assert len(adapter.submit_calls) == 2
    assert all(call["order_type"] == "MARKET" for call in adapter.submit_calls)
    assert receipt.legs_acted_on == ("MKT_A", "MKT_B")


def test_passive_unwind_uses_limit_orders() -> None:
    executor, adapter = _make_executor(
        status_responses=[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
        ]
    )

    receipt = executor.execute_unwind(_si9_unwind_manifest(recommended_action="PASSIVE_UNWIND"), 700)

    assert receipt.action_taken == "PASSIVE_UNWIND"
    assert all(call["order_type"] == "LIMIT" for call in adapter.submit_calls)


def test_hold_for_recovery_skips_order_submission() -> None:
    executor, adapter = _make_executor()

    receipt = executor.execute_unwind(_si9_unwind_manifest(recommended_action="HOLD_FOR_RECOVERY"), 700)

    assert receipt.action_taken == "SKIPPED"
    assert adapter.submit_calls == []
    assert executor.active_unwind_count == 0


def test_ctf_unwind_manifest_is_parsed_with_ctf_signal_source_context() -> None:
    executor, _ = _make_executor(
        signal_source="CTF",
        status_responses=[
            _status("FILLED", filled_size=Decimal("3.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.385000")),
        ],
    )

    receipt = executor.execute_unwind(_ctf_unwind_manifest(), 800)

    assert receipt.per_leg_receipts[0].context.signal_source == "CTF"
    assert receipt.per_leg_receipts[0].context.side == "NO"


def test_si9_unwind_manifest_is_parsed_with_si9_signal_source_context() -> None:
    executor, _ = _make_executor(
        status_responses=[
            _status("FILLED", filled_size=Decimal("2.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.320000")),
            _status("FILLED", filled_size=Decimal("2.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.300000")),
        ]
    )

    receipt = executor.execute_unwind(_si9_unwind_manifest(), 700)

    assert receipt.per_leg_receipts[0].context.signal_source == "SI9"
    assert receipt.per_leg_receipts[1].context.signal_source == "SI9"


def test_client_order_ids_are_deterministic_for_si9_unwind() -> None:
    executor, adapter = _make_executor(
        status_responses=[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
        ]
    )

    executor.execute_unwind(_si9_unwind_manifest(), 700)

    assert [call["client_order_id"] for call in adapter.submit_calls] == [
        "SI9-abc12345-MKT_A-Y-700",
        "SI9-abc12345-MKT_B-Y-701",
    ]


def test_client_order_id_is_deterministic_for_ctf_unwind() -> None:
    executor, adapter = _make_executor(
        signal_source="CTF",
        status_responses=[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("3.000000"), average_fill_price=None),
        ],
    )

    executor.execute_unwind(_ctf_unwind_manifest(), 800)

    assert [call["client_order_id"] for call in adapter.submit_calls] == ["CTF-abc12345-MKT_CTF-N-800"]


def test_duplicate_execution_guard_reuses_existing_receipt_without_new_orders() -> None:
    executor, adapter = _make_executor(
        status_responses=[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
        ]
    )
    manifest = _si9_unwind_manifest()

    first = executor.execute_unwind(manifest, 700)
    second = executor.execute_unwind(manifest, 710)

    assert second is first
    assert len(adapter.submit_calls) == 2
    assert executor.active_unwind_count == 1


def test_clear_unwind_releases_state_lock_for_reexecution() -> None:
    executor, adapter = _make_executor(
        status_responses=[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
        ]
    )
    manifest = _si9_unwind_manifest()

    executor.execute_unwind(manifest, 700)
    executor.clear_unwind(manifest.cluster_id)
    executor.execute_unwind(manifest, 710)

    assert len(adapter.submit_calls) == 4
    assert executor.active_unwind_count == 1


def test_submit_rejection_is_mapped_without_crashing_the_loop() -> None:
    executor, _ = _make_executor(
        submit_responses=[
            _submit_response(status="REJECTED", venue_order_id=None, rejection_reason="INSUFFICIENT_BALANCE"),
            _submit_response(status="ACCEPTED", venue_order_id="VENUE-2", venue_timestamp_ms=1005),
        ],
        status_responses=[
            _status("UNKNOWN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None, venue_order_id=None),
            _status("FILLED", filled_size=Decimal("2.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.300000"), venue_order_id="VENUE-2"),
        ],
    )

    receipt = executor.execute_unwind(_si9_unwind_manifest(), 700)

    assert len(receipt.per_leg_receipts) == 2
    assert receipt.per_leg_receipts[0].executed is False
    assert receipt.per_leg_receipts[0].guard_reason == "INSUFFICIENT_BALANCE"
    assert receipt.per_leg_receipts[1].executed is True


def test_full_fill_status_maps_to_full_dispatch_receipt() -> None:
    executor, _ = _make_executor(
        status_responses=[
            _status("FILLED", filled_size=Decimal("2.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.320000")),
            _status("FILLED", filled_size=Decimal("2.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.300000")),
        ]
    )

    receipt = executor.execute_unwind(_si9_unwind_manifest(), 700)

    assert receipt.per_leg_receipts[0].fill_status == "FULL"
    assert receipt.per_leg_receipts[1].fill_status == "FULL"


def test_partial_fill_status_maps_to_partial_dispatch_receipt() -> None:
    executor, _ = _make_executor(
        status_responses=[
            _status("PARTIAL", filled_size=Decimal("1.000000"), remaining_size=Decimal("1.000000"), average_fill_price=Decimal("0.320000")),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
        ]
    )

    receipt = executor.execute_unwind(_si9_unwind_manifest(), 700)

    assert receipt.per_leg_receipts[0].fill_status == "PARTIAL"
    assert receipt.per_leg_receipts[0].partial_fill_size == Decimal("1.000000")


def test_open_status_maps_to_executed_none_receipt() -> None:
    executor, _ = _make_executor(
        status_responses=[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
        ]
    )

    receipt = executor.execute_unwind(_si9_unwind_manifest(), 700)

    assert receipt.per_leg_receipts[0].executed is True
    assert receipt.per_leg_receipts[0].fill_status == "NONE"


def test_execution_timestamp_uses_max_venue_timestamp_when_available() -> None:
    executor, _ = _make_executor(
        submit_responses=[
            _submit_response(venue_timestamp_ms=1002),
            _submit_response(venue_timestamp_ms=1010),
        ],
        status_responses=[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
        ]
    )

    receipt = executor.execute_unwind(_si9_unwind_manifest(), 700)

    assert receipt.execution_timestamp_ms == 1010


def test_leg_ordering_follows_leg_index_not_input_order() -> None:
    hanging_legs = (
        Si9UnwindLeg(
            market_id="MKT_B",
            side="YES",
            filled_size=Decimal("2.000000"),
            filled_price=Decimal("0.310000"),
            current_best_bid=Decimal("0.300000"),
            estimated_unwind_cost=Decimal("0.020000"),
            leg_index=1,
        ),
        Si9UnwindLeg(
            market_id="MKT_A",
            side="YES",
            filled_size=Decimal("2.000000"),
            filled_price=Decimal("0.330000"),
            current_best_bid=Decimal("0.320000"),
            estimated_unwind_cost=Decimal("0.020000"),
            leg_index=0,
        ),
    )
    executor, adapter = _make_executor(
        status_responses=[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("2.000000"), average_fill_price=None),
        ]
    )

    executor.execute_unwind(_si9_unwind_manifest(hanging_legs=hanging_legs), 700)

    assert [call["market_id"] for call in adapter.submit_calls] == ["MKT_A", "MKT_B"]


def test_estimated_cost_round_trips_from_manifest() -> None:
    executor, _ = _make_executor(
        status_responses=[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("3.000000"), average_fill_price=None),
        ],
        signal_source="CTF",
    )

    receipt = executor.execute_unwind(_ctf_unwind_manifest(), 800)

    assert receipt.estimated_cost == Decimal("0.015000")


def test_rejected_ctf_unwind_leg_preserves_ctf_side_and_remaining_size() -> None:
    executor, _ = _make_executor(
        signal_source="CTF",
        submit_responses=[_submit_response(status="REJECTED", venue_order_id=None, rejection_reason="NO_BALANCE")],
        status_responses=[_status("UNKNOWN", filled_size=Decimal("0"), remaining_size=Decimal("3.000000"), average_fill_price=None, venue_order_id=None)],
    )

    receipt = executor.execute_unwind(_ctf_unwind_manifest(), 800)

    assert receipt.per_leg_receipts[0].context.side == "NO"
    assert receipt.per_leg_receipts[0].remaining_size == Decimal("3.000000")
    assert receipt.per_leg_receipts[0].guard_reason == "NO_BALANCE"