from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from src.execution.ctf_execution_manifest import CtfExecutionManifest, build_ctf_execution_manifest
from src.execution.ctf_live_executor import CtfLiveExecutor
from src.execution.unwind_executor_interface import UnwindExecutionReceipt, UnwindExecutor
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


def _manifest(**overrides) -> CtfExecutionManifest:
    values = {
        "market_id": "MKT_CTF",
        "yes_price": Decimal("0.380000"),
        "no_price": Decimal("0.400000"),
        "net_edge": Decimal("0.185000"),
        "gas_estimate": Decimal("0.010000"),
        "default_anchor_volume": Decimal("10.000000"),
        "max_capital_per_signal": Decimal("25.000000"),
        "max_size_per_leg": Decimal("8.000000"),
        "taker_fee_yes": Decimal("0.010000"),
        "taker_fee_no": Decimal("0.010000"),
        "manifest_timestamp_ms": 1000,
        "cancel_on_stale_ms": 100,
    }
    values.update(overrides)
    return build_ctf_execution_manifest(**values)


def _submit_response(
    *,
    status: str = "ACCEPTED",
    rejection_reason: str | None = None,
    venue_order_id: str | None = "VENUE-1",
    venue_timestamp_ms: int = 1001,
    latency_ms: int = 7,
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


def _cancel_response(
    *,
    cancelled: bool = True,
    rejection_reason: str | None = None,
    venue_timestamp_ms: int = 1002,
) -> VenueCancelResponse:
    return VenueCancelResponse(
        client_order_id="template",
        cancelled=cancelled,
        rejection_reason=rejection_reason,
        venue_timestamp_ms=venue_timestamp_ms,
    )


class _RecordingUnwindExecutor(UnwindExecutor):
    def __init__(self) -> None:
        self.manifests: list[object] = []

    def execute_unwind(self, manifest, current_timestamp_ms: int) -> UnwindExecutionReceipt:  # type: ignore[override]
        self.manifests.append(manifest)
        return UnwindExecutionReceipt(
            manifest=manifest,
            action_taken="MARKET_SELL",
            legs_acted_on=tuple(leg.market_id for leg in manifest.hanging_legs),
            estimated_cost=manifest.total_estimated_unwind_cost,
            execution_timestamp_ms=int(current_timestamp_ms),
            notes="ctf unwind invoked",
        )


class _ScriptedVenueAdapter(VenueAdapter):
    def __init__(
        self,
        *,
        submit_responses: list[VenueOrderResponse] | None = None,
        status_sequences: list[list[VenueOrderStatus]] | None = None,
        cancel_responses: list[VenueCancelResponse] | None = None,
    ) -> None:
        self._submit_responses = list(submit_responses or [])
        self._status_sequences = [list(sequence) for sequence in (status_sequences or [])]
        self._cancel_responses = list(cancel_responses or [])
        self._order_index_by_client_id: dict[str, int] = {}
        self._status_reads: dict[str, int] = {}
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
        order_index = len(self.submit_calls) - 1
        self._order_index_by_client_id[client_order_id] = order_index
        template = self._submit_responses[order_index] if order_index < len(self._submit_responses) else _submit_response()
        return replace(template, client_order_id=client_order_id)

    def cancel_order(self, client_order_id: str, market_id: str) -> VenueCancelResponse:
        self.cancel_calls.append({"client_order_id": client_order_id, "market_id": market_id})
        order_index = self._order_index_by_client_id[client_order_id]
        template = self._cancel_responses[order_index] if order_index < len(self._cancel_responses) else _cancel_response()
        return replace(template, client_order_id=client_order_id)

    def get_order_status(self, client_order_id: str) -> VenueOrderStatus:
        self.status_calls.append(client_order_id)
        order_index = self._order_index_by_client_id[client_order_id]
        sequence = self._status_sequences[order_index]
        read_index = self._status_reads.get(client_order_id, 0)
        self._status_reads[client_order_id] = read_index + 1
        template = sequence[min(read_index, len(sequence) - 1)]
        return replace(template, client_order_id=client_order_id)

    def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        _ = asset_symbol
        return Decimal("100.000000")


def _make_executor(
    *,
    submit_responses: list[VenueOrderResponse] | None = None,
    status_sequences: list[list[VenueOrderStatus]] | None = None,
    cancel_responses: list[VenueCancelResponse] | None = None,
    poll_interval_ms: int = 50,
) -> tuple[CtfLiveExecutor, _ScriptedVenueAdapter, _RecordingUnwindExecutor]:
    adapter = _ScriptedVenueAdapter(
        submit_responses=submit_responses,
        status_sequences=status_sequences,
        cancel_responses=cancel_responses,
    )
    unwind_executor = _RecordingUnwindExecutor()
    executor = CtfLiveExecutor(
        adapter,
        unwind_executor,
        poll_interval_ms=poll_interval_ms,
        sleep_fn=lambda _: None,
    )
    return executor, adapter, unwind_executor


def test_execute_full_fill_submits_anchor_then_second_leg() -> None:
    manifest = _manifest()
    executor, adapter, unwind_executor = _make_executor(
        status_sequences=[
            [_status("FILLED", filled_size=Decimal("8.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.380000"))],
            [_status("FILLED", filled_size=Decimal("8.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.400000"))],
        ]
    )

    result = executor.execute(manifest, 1000)

    assert result.execution_receipt.execution_outcome == "FULL_FILL"
    assert adapter.submit_calls[0]["side"] == "YES"
    assert adapter.submit_calls[1]["side"] == "NO"
    assert adapter.submit_calls[1]["size"] == Decimal("8.000000")
    assert result.unwind_manifest is None
    assert unwind_executor.manifests == []


def test_execute_uses_no_leg_as_anchor_when_no_is_cheaper() -> None:
    manifest = _manifest(yes_price=Decimal("0.420000"), no_price=Decimal("0.390000"))
    executor, adapter, _ = _make_executor(
        status_sequences=[
            [_status("FILLED", filled_size=Decimal("8.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.390000"))],
            [_status("FILLED", filled_size=Decimal("8.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.420000"))],
        ]
    )

    executor.execute(manifest, 1000)

    assert adapter.submit_calls[0]["side"] == "NO"
    assert adapter.submit_calls[1]["side"] == "YES"


def test_anchor_rejection_aborts_without_second_leg() -> None:
    executor, adapter, unwind_executor = _make_executor(
        submit_responses=[_submit_response(status="REJECTED", rejection_reason="PRICE_BAND", venue_order_id=None)],
        status_sequences=[[_status("UNKNOWN", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None, venue_order_id=None)]],
    )

    result = executor.execute(_manifest(), 1000)

    assert result.execution_receipt.execution_outcome == "ANCHOR_REJECTED"
    assert len(adapter.submit_calls) == 1
    assert result.execution_receipt.no_receipt.fill_status == "SUPPRESSED"
    assert unwind_executor.manifests == []


def test_anchor_timeout_cancels_and_aborts_without_second_leg() -> None:
    executor, adapter, _ = _make_executor(
        status_sequences=[[
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None),
            _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None),
            _status("CANCELLED", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None),
        ]]
    )

    result = executor.execute(_manifest(), 1000)

    assert result.execution_receipt.execution_outcome == "ANCHOR_REJECTED"
    assert len(adapter.submit_calls) == 1
    assert len(adapter.cancel_calls) == 1
    assert result.execution_receipt.yes_receipt.dispatch_receipt.guard_reason == "FILL_TIMEOUT"


def test_anchor_partial_fill_sizes_second_leg_to_actual_filled_size() -> None:
    executor, adapter, unwind_executor = _make_executor(
        status_sequences=[
            [
                _status("PARTIAL", filled_size=Decimal("3.000000"), remaining_size=Decimal("5.000000"), average_fill_price=Decimal("0.380000")),
                _status("PARTIAL", filled_size=Decimal("3.000000"), remaining_size=Decimal("5.000000"), average_fill_price=Decimal("0.380000")),
            ],
            [_status("FILLED", filled_size=Decimal("3.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.400000"))],
        ]
    )

    result = executor.execute(_manifest(), 1000)

    assert result.execution_receipt.execution_outcome == "PARTIAL_FILL"
    assert adapter.submit_calls[1]["size"] == Decimal("3.000000")
    assert len(adapter.cancel_calls) == 1
    assert unwind_executor.manifests == []


def test_anchor_partial_fill_after_timeout_still_hedges_actual_size() -> None:
    executor, adapter, _ = _make_executor(
        status_sequences=[
            [
                _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None),
                _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None),
                _status("PARTIAL", filled_size=Decimal("2.000000"), remaining_size=Decimal("6.000000"), average_fill_price=Decimal("0.380000")),
            ],
            [_status("FILLED", filled_size=Decimal("2.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.400000"))],
        ]
    )

    result = executor.execute(_manifest(), 1000)

    assert result.execution_receipt.execution_outcome == "PARTIAL_FILL"
    assert adapter.submit_calls[1]["size"] == Decimal("2.000000")
    assert len(adapter.cancel_calls) == 1


def test_second_leg_rejection_triggers_unwind_executor() -> None:
    executor, _, unwind_executor = _make_executor(
        submit_responses=[
            _submit_response(status="ACCEPTED"),
            _submit_response(status="REJECTED", rejection_reason="SECOND_REJECTED", venue_order_id=None),
        ],
        status_sequences=[
            [_status("FILLED", filled_size=Decimal("8.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.380000"))],
            [_status("UNKNOWN", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None, venue_order_id=None)],
        ],
    )

    result = executor.execute(_manifest(), 1000)

    assert result.execution_receipt.execution_outcome == "SECOND_LEG_REJECTED"
    assert result.unwind_manifest is not None
    assert result.unwind_manifest.hanging_legs[0].filled_size == Decimal("8.000000")
    assert result.unwind_execution_receipt is not None
    assert len(unwind_executor.manifests) == 1


def test_second_leg_timeout_after_anchor_fill_triggers_unwind() -> None:
    executor, adapter, unwind_executor = _make_executor(
        status_sequences=[
            [_status("FILLED", filled_size=Decimal("8.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.380000"))],
            [
                _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None),
                _status("OPEN", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None),
                _status("CANCELLED", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None),
            ],
        ]
    )

    result = executor.execute(_manifest(), 1000)

    assert result.execution_receipt.execution_outcome == "SECOND_LEG_REJECTED"
    assert len(adapter.cancel_calls) == 1
    assert result.unwind_manifest is not None
    assert unwind_executor.manifests[0].recommended_action == "MARKET_SELL"


def test_second_leg_partial_fill_returns_partial_receipt_and_unwinds_residual() -> None:
    executor, adapter, unwind_executor = _make_executor(
        status_sequences=[
            [_status("FILLED", filled_size=Decimal("8.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.380000"))],
            [
                _status("PARTIAL", filled_size=Decimal("3.000000"), remaining_size=Decimal("5.000000"), average_fill_price=Decimal("0.400000")),
                _status("PARTIAL", filled_size=Decimal("3.000000"), remaining_size=Decimal("5.000000"), average_fill_price=Decimal("0.400000")),
            ],
        ]
    )

    result = executor.execute(_manifest(), 1000)

    assert result.execution_receipt.execution_outcome == "PARTIAL_FILL"
    assert result.execution_receipt.no_receipt.fill_status == "PARTIAL"
    assert len(adapter.cancel_calls) == 1
    assert result.unwind_manifest is not None
    assert result.unwind_manifest.hanging_legs[0].filled_size == Decimal("5.000000")
    assert len(unwind_executor.manifests) == 1


def test_unwind_manifest_uses_anchor_leg_details_for_residual_exposure() -> None:
    manifest = _manifest(yes_price=Decimal("0.420000"), no_price=Decimal("0.390000"))
    executor, _, _ = _make_executor(
        submit_responses=[
            _submit_response(status="ACCEPTED"),
            _submit_response(status="REJECTED", rejection_reason="SECOND_REJECTED", venue_order_id=None),
        ],
        status_sequences=[
            [_status("FILLED", filled_size=Decimal("8.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.390000"))],
            [_status("UNKNOWN", filled_size=Decimal("0"), remaining_size=Decimal("8.000000"), average_fill_price=None, venue_order_id=None)],
        ],
    )

    result = executor.execute(manifest, 1000)

    assert result.unwind_manifest is not None
    assert result.unwind_manifest.hanging_legs[0].side == "NO"
    assert result.unwind_manifest.hanging_legs[0].leg_index == 0


def test_client_order_ids_are_deterministic_within_execution() -> None:
    executor, adapter, _ = _make_executor(
        status_sequences=[
            [_status("FILLED", filled_size=Decimal("8.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.380000"))],
            [_status("FILLED", filled_size=Decimal("8.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.400000"))],
        ]
    )

    executor.execute(_manifest(), 1000)

    assert [call["client_order_id"] for call in adapter.submit_calls] == ["CTF-000001-0", "CTF-000001-1"]


def test_poll_interval_validation_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="poll_interval_ms"):
        CtfLiveExecutor(_ScriptedVenueAdapter(status_sequences=[]), _RecordingUnwindExecutor(), poll_interval_ms=0)


def test_second_leg_size_matches_anchor_partial_when_no_leg_is_anchor() -> None:
    manifest = _manifest(yes_price=Decimal("0.420000"), no_price=Decimal("0.390000"))
    executor, adapter, _ = _make_executor(
        status_sequences=[
            [
                _status("PARTIAL", filled_size=Decimal("4.000000"), remaining_size=Decimal("4.000000"), average_fill_price=Decimal("0.390000")),
                _status("PARTIAL", filled_size=Decimal("4.000000"), remaining_size=Decimal("4.000000"), average_fill_price=Decimal("0.390000")),
            ],
            [_status("FILLED", filled_size=Decimal("4.000000"), remaining_size=Decimal("0"), average_fill_price=Decimal("0.420000"))],
        ]
    )

    result = executor.execute(manifest, 1000)

    assert adapter.submit_calls[0]["side"] == "NO"
    assert adapter.submit_calls[1]["side"] == "YES"
    assert adapter.submit_calls[1]["size"] == Decimal("4.000000")
    assert result.execution_receipt.execution_outcome == "PARTIAL_FILL"