"""
Tests for the OrderStatusPoller — verifies CLOB order-status polling,
fill detection, callback invocation, and error handling.
"""

import asyncio
import time

import pytest

from src.trading.executor import (
    Order,
    OrderExecutor,
    OrderSide,
    OrderStatus,
    OrderStatusPoller,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


class _FakeClobClient:
    """Fake CLOB client that returns configurable responses for get_order()."""

    def __init__(self):
        self.responses: dict[str, dict] = {}

    def set_response(self, clob_id: str, resp: dict) -> None:
        self.responses[clob_id] = resp

    def get_order(self, clob_id: str) -> dict | None:
        return self.responses.get(clob_id)


class _ErrorClobClient:
    """CLOB client that always raises on get_order()."""

    def get_order(self, clob_id: str):
        raise ConnectionError("network down")


# ── Tests ───────────────────────────────────────────────────────────────────


class TestOrderStatusPollerUnit:
    """Unit tests for _apply_update and _record_error (no I/O)."""

    @pytest.fixture
    def executor(self):
        return OrderExecutor(paper_mode=True)

    def _make_poller(self, executor, on_fill=None):
        return OrderStatusPoller(
            executor,
            on_fill=on_fill,
            poll_interval_s=0.05,
            max_retries=2,
        )

    def test_apply_update_fills_order(self, executor):
        poller = self._make_poller(executor)
        order = Order(
            order_id="LIVE-1",
            market_id="M",
            asset_id="A",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            status=OrderStatus.LIVE,
            clob_order_id="CLOB-1",
        )

        poller._apply_update(order, {
            "status": "matched",
            "size_matched": 10.0,
            "average_price": 0.44,
        })

        assert order.status == OrderStatus.FILLED
        assert order.filled_size == 10.0
        assert order.filled_avg_price == 0.44

    def test_apply_update_partial_fill(self, executor):
        poller = self._make_poller(executor)
        order = Order(
            order_id="LIVE-2",
            market_id="M",
            asset_id="A",
            side=OrderSide.BUY,
            price=0.50,
            size=20.0,
            status=OrderStatus.LIVE,
            clob_order_id="CLOB-2",
        )

        poller._apply_update(order, {
            "status": "live",
            "size_matched": 5.0,
            "average_price": 0.49,
        })

        # Status stays LIVE but filled_size updates
        assert order.status == OrderStatus.LIVE
        assert order.filled_size == 5.0
        assert order.filled_avg_price == 0.49

    def test_apply_update_cancelled(self, executor):
        poller = self._make_poller(executor)
        order = Order(
            order_id="LIVE-3",
            market_id="M",
            asset_id="A",
            side=OrderSide.SELL,
            price=0.60,
            size=10.0,
            status=OrderStatus.LIVE,
            clob_order_id="CLOB-3",
        )

        poller._apply_update(order, {"status": "canceled"})
        assert order.status == OrderStatus.CANCELLED

    def test_apply_update_unknown_status_ignored(self, executor):
        poller = self._make_poller(executor)
        order = Order(
            order_id="LIVE-4",
            market_id="M",
            asset_id="A",
            side=OrderSide.BUY,
            price=0.40,
            size=5.0,
            status=OrderStatus.LIVE,
            clob_order_id="CLOB-4",
        )

        poller._apply_update(order, {"status": "some_weird_state"})
        assert order.status == OrderStatus.LIVE  # unchanged

    def test_apply_update_no_regression_on_filled_size(self, executor):
        """filled_size should never decrease (CLOB might temporarily lag)."""
        poller = self._make_poller(executor)
        order = Order(
            order_id="LIVE-5",
            market_id="M",
            asset_id="A",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            status=OrderStatus.LIVE,
            clob_order_id="CLOB-5",
            filled_size=5.0,  # already partially filled
        )

        poller._apply_update(order, {
            "status": "live",
            "size_matched": 3.0,  # lower than current — should NOT regress
            "average_price": 0.44,
        })

        assert order.filled_size == 5.0  # unchanged

    def test_record_error_tracks_consecutive_failures(self, executor):
        poller = self._make_poller(executor, on_fill=None)
        poller._record_error("CLOB-X")
        assert poller._consecutive_errors.get("CLOB-X") == 1
        poller._record_error("CLOB-X")
        assert poller._consecutive_errors.get("CLOB-X") == 2
        # max_retries=2 → warning logged but no crash

    def test_apply_update_with_object_response(self, executor):
        """Support legacy SDKs that return objects instead of dicts."""
        poller = self._make_poller(executor)
        order = Order(
            order_id="LIVE-6",
            market_id="M",
            asset_id="A",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            status=OrderStatus.LIVE,
            clob_order_id="CLOB-6",
        )

        class _Resp:
            status = "matched"
            size_matched = 10.0
            average_price = 0.44

        poller._apply_update(order, _Resp())
        assert order.status == OrderStatus.FILLED


class TestOrderStatusPollerAsync:
    """Async integration tests — verify polling loop and callbacks."""

    @pytest.fixture
    def executor(self):
        """Non-paper executor with a fake CLOB client injected."""
        ex = OrderExecutor(paper_mode=False)
        ex._clob_client = _FakeClobClient()
        return ex

    @pytest.mark.asyncio
    async def test_poll_detects_fill_and_calls_back(self, executor):
        filled_orders: list[Order] = []

        async def on_fill(order: Order):
            filled_orders.append(order)

        # Place a live order (paper_mode=False, fake client)
        order = Order(
            order_id="LIVE-10",
            market_id="M",
            asset_id="A",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            status=OrderStatus.LIVE,
            clob_order_id="CLOB-10",
        )
        executor._orders["LIVE-10"] = order

        # Configure fake CLOB to return filled
        executor._clob_client.set_response("CLOB-10", {
            "status": "matched",
            "size_matched": 10.0,
            "average_price": 0.44,
        })

        poller = OrderStatusPoller(
            executor, on_fill=on_fill, poll_interval_s=0.05, max_retries=3,
        )

        # Run one poll cycle
        await poller._poll_once()

        assert order.status == OrderStatus.FILLED
        # Give the callback task a chance to run
        await asyncio.sleep(0.05)
        assert len(filled_orders) == 1
        assert filled_orders[0].order_id == "LIVE-10"

    @pytest.mark.asyncio
    async def test_poll_skips_orders_without_clob_id(self, executor):
        """Orders that never reached the CLOB should not be polled."""
        order = Order(
            order_id="LIVE-11",
            market_id="M",
            asset_id="A",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            status=OrderStatus.LIVE,
            clob_order_id="",  # no clob id
        )
        executor._orders["LIVE-11"] = order

        poller = OrderStatusPoller(
            executor, poll_interval_s=0.05, max_retries=3,
        )

        # Should not crash
        await poller._poll_once()
        assert order.status == OrderStatus.LIVE  # unchanged

    @pytest.mark.asyncio
    async def test_poll_handles_api_errors_gracefully(self):
        """If the CLOB client raises, the poller logs and continues."""
        ex = OrderExecutor(paper_mode=False)
        ex._clob_client = _ErrorClobClient()

        order = Order(
            order_id="LIVE-12",
            market_id="M",
            asset_id="A",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            status=OrderStatus.LIVE,
            clob_order_id="CLOB-12",
        )
        ex._orders["LIVE-12"] = order

        poller = OrderStatusPoller(
            ex, poll_interval_s=0.05, max_retries=2,
        )

        # Should not raise
        await poller._poll_once()
        assert order.status == OrderStatus.LIVE  # unchanged
        assert poller._consecutive_errors.get("CLOB-12") == 1

    @pytest.mark.asyncio
    async def test_paper_mode_run_is_noop(self):
        """In paper mode, run() returns immediately."""
        ex = OrderExecutor(paper_mode=True)
        poller = OrderStatusPoller(ex, poll_interval_s=0.01)
        # Should return nearly instantly
        await asyncio.wait_for(poller.run(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_poller_clears_errors_on_success(self, executor):
        """After a successful poll, consecutive error counter resets."""
        order = Order(
            order_id="LIVE-13",
            market_id="M",
            asset_id="A",
            side=OrderSide.BUY,
            price=0.45,
            size=10.0,
            status=OrderStatus.LIVE,
            clob_order_id="CLOB-13",
        )
        executor._orders["LIVE-13"] = order

        poller = OrderStatusPoller(
            executor, poll_interval_s=0.05, max_retries=3,
        )

        # Simulate prior errors
        poller._consecutive_errors["CLOB-13"] = 2

        # Now a successful response
        executor._clob_client.set_response("CLOB-13", {
            "status": "live",
            "size_matched": 0,
        })
        await poller._poll_once()

        assert "CLOB-13" not in poller._consecutive_errors

    @pytest.mark.asyncio
    async def test_run_loop_can_be_stopped(self, executor):
        """Verify stop() terminates the run loop."""
        poller = OrderStatusPoller(
            executor, poll_interval_s=0.05, max_retries=3,
        )

        async def stop_soon():
            await asyncio.sleep(0.15)
            poller.stop()

        asyncio.create_task(stop_soon())
        await asyncio.wait_for(poller.run(), timeout=2.0)
        # If we reach here, the poller exited cleanly
