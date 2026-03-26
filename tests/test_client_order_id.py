from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal

import pytest

from src.execution.alpha_adapters import ofi_to_context
from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.live_wallet_balance import LiveWalletBalanceProvider
from src.execution.mev_router import MevExecutionRouter
from src.execution.priority_dispatcher import PriorityDispatcher
from src.execution.venue_adapter_interface import VenueAdapter, VenueOrderResponse, VenueOrderStatus


def _make_router() -> MevExecutionRouter:
    return MevExecutionRouter(
        lambda market_id: {
            "yes_bid": 0.45,
            "yes_ask": 0.55,
            "no_bid": 0.45,
            "no_ask": 0.55,
        }
    )


def _make_context(market_id: str = "MKT_PRIORITY"):
    return ofi_to_context(
        market_id=market_id,
        side="YES",
        target_price=Decimal("0.640000"),
        anchor_volume=Decimal("50.000000"),
        max_capital=Decimal("100.000000"),
        conviction_scalar=Decimal("0.850000"),
    )


class RecordingVenueAdapter(VenueAdapter):
    def __init__(self) -> None:
        self.submit_calls: list[str] = []
        self.status_calls: list[str] = []
        self._submit_response = VenueOrderResponse(
            client_order_id="UNUSED",
            venue_order_id="VENUE-1",
            status="ACCEPTED",
            rejection_reason=None,
            venue_timestamp_ms=1234,
            latency_ms=9,
        )
        self._status_response = VenueOrderStatus(
            client_order_id="UNUSED",
            venue_order_id="VENUE-1",
            fill_status="OPEN",
            filled_size=Decimal("0"),
            remaining_size=Decimal("42.500000"),
            average_fill_price=None,
        )

    def submit_order(
        self,
        market_id: str,
        side: str,
        price: Decimal,
        size: Decimal,
        order_type: str,
        client_order_id: str,
    ) -> VenueOrderResponse:
        self.submit_calls.append(client_order_id)
        return replace(self._submit_response, client_order_id=client_order_id)

    def cancel_order(self, client_order_id: str, market_id: str):
        raise NotImplementedError

    def get_order_status(self, client_order_id: str) -> VenueOrderStatus:
        self.status_calls.append(client_order_id)
        return replace(self._status_response, client_order_id=client_order_id)

    def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        _ = asset_symbol
        return Decimal("100.000000")


def test_client_order_id_valid_construction_passes() -> None:
    generator = ClientOrderIdGenerator("OFI", "session-123")

    assert isinstance(generator, ClientOrderIdGenerator)


def test_generated_id_contains_source_session_market_side_and_timestamp() -> None:
    generator = ClientOrderIdGenerator("OFI", "a3f9b2c1-session")

    generated = generator.generate("deadbeef-market", "YES", 1711234567890)

    assert generated == "OFI-a3f9b2c1-deadbeef-Y-1711234567890"


def test_different_timestamps_produce_different_ids() -> None:
    generator = ClientOrderIdGenerator("OFI", "session-123")

    assert generator.generate("MKT-1", "YES", 100) != generator.generate("MKT-1", "YES", 101)


def test_different_market_ids_produce_different_ids() -> None:
    generator = ClientOrderIdGenerator("OFI", "session-123")

    assert generator.generate("MARKET-AAA", "YES", 100) != generator.generate("MARKET-BBB", "YES", 100)


def test_different_session_ids_produce_different_ids() -> None:
    first = ClientOrderIdGenerator("OFI", "a3f9b2c1-session")
    second = ClientOrderIdGenerator("OFI", "b7d4e8f0-session")

    assert first.generate("MKT-1", "YES", 100) != second.generate("MKT-1", "YES", 100)


def test_same_inputs_are_deterministic() -> None:
    generator = ClientOrderIdGenerator("OFI", "session-123")

    first = generator.generate("MKT-1", "NO", 100)
    second = generator.generate("MKT-1", "NO", 100)

    assert first == second


def test_generated_id_length_is_under_64_characters() -> None:
    generator = ClientOrderIdGenerator("CONTAGION", "very-long-session-id")

    generated = generator.generate("very-long-market-id", "YES", 1711234567890)

    assert len(generated) < 64


def test_market_id_longer_than_eight_chars_is_truncated() -> None:
    generator = ClientOrderIdGenerator("OFI", "session-123")

    assert generator.generate("ABCDEFGHIJK", "YES", 100) == "OFI-session--ABCDEFGH-Y-100"


def test_session_id_longer_than_eight_chars_uses_first_eight() -> None:
    generator = ClientOrderIdGenerator("OFI", "12345678-extra")

    assert generator.generate("MKT-1", "YES", 100) == "OFI-12345678-MKT-1-Y-100"


def test_live_dispatcher_without_client_order_id_generator_raises_value_error() -> None:
    with pytest.raises(ValueError, match="client_order_id_generator"):
        PriorityDispatcher(
            _make_router(),
            "live",
            venue_adapter=RecordingVenueAdapter(),
            wallet_balance_provider=LiveWalletBalanceProvider(
                RecordingVenueAdapter(),
                tracked_assets=["USDC"],
                initial_balances={"USDC": Decimal("100.000000")},
            ),
        )


def test_live_dispatcher_with_generator_produces_receipt_with_non_none_order_id() -> None:
    adapter = RecordingVenueAdapter()
    dispatcher = PriorityDispatcher(
        _make_router(),
        "live",
        venue_adapter=adapter,
        client_order_id_generator=ClientOrderIdGenerator("OFI", "a3f9b2c1-session"),
        wallet_balance_provider=LiveWalletBalanceProvider(
            adapter,
            tracked_assets=["USDC"],
            initial_balances={"USDC": Decimal("100.000000")},
        ),
    )

    receipt = dispatcher.dispatch(_make_context(), 10)

    assert receipt.order_id is not None
    assert receipt.order_id == "OFI-a3f9b2c1-MKT_PRIO-Y-10"


def test_live_dispatcher_order_id_matches_generated_client_order_id_format() -> None:
    adapter = RecordingVenueAdapter()
    dispatcher = PriorityDispatcher(
        _make_router(),
        "live",
        venue_adapter=adapter,
        client_order_id_generator=ClientOrderIdGenerator("OFI", "a3f9b2c1-session"),
        wallet_balance_provider=LiveWalletBalanceProvider(
            adapter,
            tracked_assets=["USDC"],
            initial_balances={"USDC": Decimal("100.000000")},
        ),
    )

    receipt = dispatcher.dispatch(_make_context(), 10)

    assert re.fullmatch(r"OFI-a3f9b2c1-MKT_PRIO-Y-10", receipt.order_id or "")
    assert adapter.submit_calls == [receipt.order_id]
    assert adapter.status_calls == [receipt.order_id]


def test_dispatcher_prefers_context_signal_source_when_generator_session_is_reused() -> None:
    adapter = RecordingVenueAdapter()
    dispatcher = PriorityDispatcher(
        _make_router(),
        "live",
        venue_adapter=adapter,
        client_order_id_generator=ClientOrderIdGenerator("MANUAL", "a3f9b2c1-session"),
        wallet_balance_provider=LiveWalletBalanceProvider(
            adapter,
            tracked_assets=["USDC"],
            initial_balances={"USDC": Decimal("100.000000")},
        ),
    )

    receipt = dispatcher.dispatch(_make_context(), 10)

    assert receipt.order_id == "OFI-a3f9b2c1-MKT_PRIO-Y-10"