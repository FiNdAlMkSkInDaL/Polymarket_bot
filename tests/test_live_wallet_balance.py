from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from src.execution.clob_transport import (
    ClobTransportCircuitOpenError,
    ClobTransportRateLimitError,
    ClobTransportTimeoutError,
)
from src.execution.live_wallet_balance import LiveWalletBalanceProvider
from src.execution.venue_adapter_interface import VenueAdapter, VenueCancelResponse, VenueOrderResponse, VenueOrderStatus


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def warning(self, event: str, **kwargs: object) -> None:
        self.events.append((event, kwargs))


class _BalanceVenueAdapter(VenueAdapter):
    def __init__(self, balances: list[Decimal | Exception] | None = None) -> None:
        self._balances = list(balances or [Decimal("100.000000")])
        self.balance_calls: list[str] = []

    def submit_order(
        self,
        market_id: str,
        side: str,
        price: Decimal,
        size: Decimal,
        order_type: str,
        client_order_id: str,
    ) -> VenueOrderResponse:
        raise NotImplementedError

    def cancel_order(self, client_order_id: str, market_id: str) -> VenueCancelResponse:
        raise NotImplementedError

    def get_order_status(self, client_order_id: str) -> VenueOrderStatus:
        raise NotImplementedError

    def get_wallet_balance(self, asset_symbol: str) -> Decimal:
        self.balance_calls.append(asset_symbol)
        if not self._balances:
            raise AssertionError("no balances left")
        next_value = self._balances.pop(0)
        if isinstance(next_value, Exception):
            raise next_value
        return next_value


def _now_factory(start: int = 1_000):
    state = {"value": start}

    def _now() -> int:
        return state["value"]

    def _advance(delta: int) -> None:
        state["value"] += delta

    return _now, _advance


def test_get_available_margin_returns_zero_for_uncached_asset() -> None:
    provider = LiveWalletBalanceProvider(_BalanceVenueAdapter(), tracked_assets=["USDC"])

    assert provider.get_available_margin("USDC") == Decimal("0")


def test_get_available_margin_returns_seeded_decimal_value() -> None:
    provider = LiveWalletBalanceProvider(
        _BalanceVenueAdapter(),
        tracked_assets=["USDC"],
        initial_balances={"usdc": Decimal("42.500000")},
    )

    assert provider.get_available_margin("USDC") == Decimal("42.500000")


def test_get_available_margin_normalizes_asset_symbol_lookup() -> None:
    provider = LiveWalletBalanceProvider(
        _BalanceVenueAdapter(),
        tracked_assets=["USDC"],
        initial_balances={"USDC": Decimal("5.000000")},
    )

    assert provider.get_available_margin(" usdc ") == Decimal("5.000000")


def test_get_available_margin_is_pure_cache_lookup_without_adapter_calls() -> None:
    adapter = _BalanceVenueAdapter()
    provider = LiveWalletBalanceProvider(
        adapter,
        tracked_assets=["USDC"],
        initial_balances={"USDC": Decimal("7.000000")},
    )

    balance = provider.get_available_margin("USDC")

    assert balance == Decimal("7.000000")
    assert adapter.balance_calls == []


def test_get_last_updated_ms_is_none_before_poll() -> None:
    provider = LiveWalletBalanceProvider(
        _BalanceVenueAdapter(),
        tracked_assets=["USDC"],
        initial_balances={"USDC": Decimal("7.000000")},
    )

    assert provider.get_last_updated_ms("USDC") is None


@pytest.mark.asyncio
async def test_poll_balance_loop_updates_cached_balance() -> None:
    adapter = _BalanceVenueAdapter([Decimal("11.250000")])
    logger = _RecordingLogger()
    now_ms, _ = _now_factory(5_000)

    async def _sleep(_: float) -> None:
        raise asyncio.CancelledError

    provider = LiveWalletBalanceProvider(
        adapter,
        tracked_assets=["USDC"],
        now_ms=now_ms,
        sleep_fn=_sleep,
        logger=logger,
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.poll_balance_loop(250)

    assert provider.get_available_margin("USDC") == Decimal("11.250000")
    assert provider.get_last_updated_ms("USDC") == 5_000
    assert logger.events == []


@pytest.mark.asyncio
async def test_poll_balance_loop_updates_multiple_assets_in_one_cycle() -> None:
    adapter = _BalanceVenueAdapter([Decimal("10.000000"), Decimal("6.500000")])

    async def _sleep(_: float) -> None:
        raise asyncio.CancelledError

    provider = LiveWalletBalanceProvider(
        adapter,
        tracked_assets=["USDC", "COLLATERAL"],
        sleep_fn=_sleep,
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.poll_balance_loop(100)

    assert provider.get_available_margin("USDC") == Decimal("10.000000")
    assert provider.get_available_margin("COLLATERAL") == Decimal("6.500000")
    assert adapter.balance_calls == ["USDC", "COLLATERAL"]


@pytest.mark.asyncio
async def test_poll_balance_loop_retains_previous_balance_on_timeout() -> None:
    adapter = _BalanceVenueAdapter([ClobTransportTimeoutError("slow venue")])
    logger = _RecordingLogger()

    async def _sleep(_: float) -> None:
        raise asyncio.CancelledError

    provider = LiveWalletBalanceProvider(
        adapter,
        tracked_assets=["USDC"],
        initial_balances={"USDC": Decimal("12.000000")},
        sleep_fn=_sleep,
        logger=logger,
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.poll_balance_loop(100)

    assert provider.get_available_margin("USDC") == Decimal("12.000000")
    assert logger.events[0][0] == "wallet_balance_poll_failed"
    assert logger.events[0][1]["exception_type"] == "ClobTransportTimeoutError"


@pytest.mark.asyncio
async def test_poll_balance_loop_retains_previous_balance_on_open_rate_limit_circuit() -> None:
    adapter = _BalanceVenueAdapter([ClobTransportCircuitOpenError(backoff_until_ms=9_000)])
    logger = _RecordingLogger()

    async def _sleep(_: float) -> None:
        raise asyncio.CancelledError

    provider = LiveWalletBalanceProvider(
        adapter,
        tracked_assets=["USDC"],
        initial_balances={"USDC": Decimal("18.000000")},
        sleep_fn=_sleep,
        logger=logger,
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.poll_balance_loop(100)

    assert provider.get_available_margin("USDC") == Decimal("18.000000")
    assert logger.events[0][1]["exception_type"] == "ClobTransportCircuitOpenError"


@pytest.mark.asyncio
async def test_poll_balance_loop_retains_previous_balance_on_direct_429_rate_limit() -> None:
    adapter = _BalanceVenueAdapter([ClobTransportRateLimitError("rate limited", backoff_until_ms=9_000)])
    logger = _RecordingLogger()

    async def _sleep(_: float) -> None:
        raise asyncio.CancelledError

    provider = LiveWalletBalanceProvider(
        adapter,
        tracked_assets=["USDC"],
        initial_balances={"USDC": Decimal("19.000000")},
        sleep_fn=_sleep,
        logger=logger,
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.poll_balance_loop(100)

    assert provider.get_available_margin("USDC") == Decimal("19.000000")
    assert logger.events[0][1]["exception_type"] == "ClobTransportRateLimitError"


@pytest.mark.asyncio
async def test_poll_balance_loop_sleeps_for_interval_seconds() -> None:
    adapter = _BalanceVenueAdapter([Decimal("9.000000")])
    sleep_calls: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise asyncio.CancelledError

    provider = LiveWalletBalanceProvider(
        adapter,
        tracked_assets=["USDC"],
        sleep_fn=_sleep,
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.poll_balance_loop(250)

    assert sleep_calls == [0.25]


@pytest.mark.asyncio
async def test_poll_balance_loop_preserves_last_updated_on_failure() -> None:
    now_ms, advance = _now_factory(1_000)
    adapter = _BalanceVenueAdapter([Decimal("3.000000"), ClobTransportTimeoutError("slow venue")])
    logger = _RecordingLogger()
    sleep_calls = 0

    async def _sleep(_: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        advance(500)
        if sleep_calls >= 2:
            raise asyncio.CancelledError

    provider = LiveWalletBalanceProvider(
        adapter,
        tracked_assets=["USDC"],
        now_ms=now_ms,
        sleep_fn=_sleep,
        logger=logger,
    )

    with pytest.raises(asyncio.CancelledError):
        await provider.poll_balance_loop(100)

    assert provider.get_available_margin("USDC") == Decimal("3.000000")
    assert provider.get_last_updated_ms("USDC") == 1_000
    assert logger.events[0][0] == "wallet_balance_poll_failed"


def test_constructor_rejects_empty_asset_symbol() -> None:
    with pytest.raises(ValueError, match="asset_symbol"):
        LiveWalletBalanceProvider(_BalanceVenueAdapter(), tracked_assets=[""])


@pytest.mark.asyncio
async def test_poll_balance_loop_rejects_non_positive_interval() -> None:
    provider = LiveWalletBalanceProvider(_BalanceVenueAdapter(), tracked_assets=["USDC"])

    with pytest.raises(ValueError, match="interval_ms"):
        await provider.poll_balance_loop(0)