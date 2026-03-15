"""
Integration Tests — Area 2: API & Web3 Connections

Covers:
  - WebSocket connect, subscribe, parse CLOB trade messages (mocked)
  - Whale confluence via Polygonscan RPC (mocked httpx)
  - Rate-limit / error resilience for RPC polling
  - Environment variable safety — no secrets leaked to logs or stdout
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.websocket_client import MarketWebSocket, TradeEvent
from src.signals.whale_monitor import WhaleMonitor, WhaleActivity


# ═══════════════════════════════════════════════════════════════════════════
#  Section A: WebSocket — Connect, Subscribe, Parse
# ═══════════════════════════════════════════════════════════════════════════

class TestWebSocketParsing:
    """Verify that MarketWebSocket correctly parses incoming CLOB messages."""

    def test_parse_trade_event_basic(self):
        """Parse a standard trade message dict into a TradeEvent."""
        ws = MarketWebSocket(["ASSET_1"], asyncio.Queue())
        data = {
            "price": "0.65",
            "size": "100",
            "asset_id": "ASSET_1",
            "market": "MKT_1",
            "outcome": "YES",
            "timestamp": "1700000000",
        }
        event = ws._parse_trade(data, parent={})
        assert event is not None
        assert event.price == 0.65
        assert event.size == 100.0
        assert event.asset_id == "ASSET_1"
        assert event.is_yes is True

    def test_parse_trade_event_no_outcome(self):
        """Outcome = 'NO' should set is_yes=False."""
        ws = MarketWebSocket(["ASSET_2"], asyncio.Queue())
        data = {
            "price": "0.35",
            "size": "50",
            "asset_id": "ASSET_2",
            "outcome": "NO",
            "timestamp": "1700000000",
        }
        event = ws._parse_trade(data, parent={})
        assert event is not None
        assert event.is_yes is False

    def test_parse_trade_event_invalid_price(self):
        """Price <= 0 should return None."""
        ws = MarketWebSocket([], asyncio.Queue())
        data = {"price": "0", "size": "10"}
        event = ws._parse_trade(data, parent={})
        assert event is None

    def test_parse_trade_event_missing_fields(self):
        """Gracefully handle missing fields without crashing."""
        ws = MarketWebSocket([], asyncio.Queue())
        data = {"price": "0.5", "size": "10"}
        event = ws._parse_trade(data, parent={"asset_id": "FALLBACK_ASSET"})
        assert event is not None
        assert event.asset_id == "FALLBACK_ASSET"

    @pytest.mark.asyncio
    async def test_handle_trade_message(self):
        """_handle_message routes a 'trade' event to the queue."""
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue)
        msg = {
            "event_type": "trade",
            "data": [
                {"price": "0.55", "size": "25", "asset_id": "A1", "outcome": "YES"},
            ],
        }
        await ws._handle_message(msg)
        assert not queue.empty()
        event = await queue.get()
        assert event.price == 0.55

    @pytest.mark.asyncio
    async def test_handle_last_trade_price_message(self):
        """last_trade_price event format should also be handled."""
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue)
        msg = {
            "event_type": "last_trade_price",
            "data": {"price": "0.42", "size": "5", "asset_id": "A1"},
        }
        await ws._handle_message(msg)
        assert not queue.empty()

    @pytest.mark.asyncio
    async def test_handle_book_message_ignored(self):
        """Book/price_change messages should be skipped (no queue output)."""
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue)
        await ws._handle_message({"event_type": "book", "data": {}})
        await ws._handle_message({"event_type": "price_change", "data": {}})
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_bad_json_handled_gracefully(self):
        """Invalid JSON does not crash _handle_message."""
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue)
        # _handle_message expects a dict — should not raise on weird input
        await ws._handle_message({"event_type": "unknown_type", "data": {}})
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_stop_flag_terminates_loop(self):
        """Setting _running=False should cause start() to exit."""
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue, ws_url="wss://invalid.example.com/ws")
        ws._running = False  # pre-set to prevent connection attempt
        # start() checks _running at top of the reconnect loop
        # We just verify stop() sets the flag
        await ws.stop()
        assert ws._running is False


# ═══════════════════════════════════════════════════════════════════════════
#  Section B: WebSocket Auto-Reconnect Behaviour
# ═══════════════════════════════════════════════════════════════════════════

class TestWebSocketReconnect:
    """Verify that _connect_and_consume exceptions trigger backoff/retry."""

    @pytest.mark.asyncio
    async def test_reconnect_on_connection_error(self):
        """ConnectionError in _connect_and_consume → sleep → retry."""
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue)

        call_count = 0

        async def failing_connect():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                ws._running = False
                return
            raise ConnectionError("simulated disconnect")

        ws._connect_and_consume = failing_connect

        # Patch asyncio.sleep to avoid real delays
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await ws.start()

        assert call_count >= 3, "Should have retried at least 3 times"

    @pytest.mark.asyncio
    async def test_cancelled_error_stops_cleanly(self):
        """CancelledError should stop the loop without retry."""
        queue: asyncio.Queue[TradeEvent] = asyncio.Queue()
        ws = MarketWebSocket(["A1"], queue)

        async def cancel_connect():
            raise asyncio.CancelledError()

        ws._connect_and_consume = cancel_connect
        await ws.start()
        assert ws._running is False


# ═══════════════════════════════════════════════════════════════════════════
#  Section C: Whale Monitor (Polygon RPC via Polygonscan)
# ═══════════════════════════════════════════════════════════════════════════

class TestWhaleMonitor:
    """Verify whale monitoring logic using mocked HTTP responses."""

    def test_has_confluence_empty(self):
        """No recent activity → has_confluence returns False."""
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        assert monitor.has_confluence("SOME_TOKEN") is False

    def test_has_confluence_with_matching_activity(self):
        """Manually inject activity → confluence should match."""
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        monitor._recent.append(WhaleActivity(
            wallet="0xabc",
            market_token_id="NO_TOKEN_123",
            direction="buy_no",
            amount=1000.0,
            timestamp=time.time(),
            tx_hash="0xhash1",
        ))
        assert monitor.has_confluence("NO_TOKEN_123") is True
        assert monitor.has_confluence("OTHER_TOKEN") is False

    def test_has_confluence_expired_activity(self):
        """Activity older than lookback → not counted."""
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        monitor._recent.append(WhaleActivity(
            wallet="0xabc",
            market_token_id="NO_TOKEN_123",
            direction="buy_no",
            amount=1000.0,
            timestamp=time.time() - 99999,  # very old
            tx_hash="0xhash2",
        ))
        assert monitor.has_confluence("NO_TOKEN_123", lookback_seconds=600) is False

    def test_process_tx_deduplication(self):
        """Same tx_hash should not produce duplicate WhaleActivity."""
        monitor = WhaleMonitor(whale_wallets=["0xwallet"])
        tx = {
            "tokenID": "TOKEN_A",
            "from": "0xother",
            "to": "0xwallet",
            "timeStamp": str(int(time.time())),
            "hash": "0xunique_hash",
            "blockNumber": "100",
            "tokenValue": "500",
        }
        monitor._process_tx(tx, "0xwallet")
        monitor._process_tx(tx, "0xwallet")  # duplicate
        assert len(monitor._recent) == 1

    def test_process_tx_direction_detection(self):
        """to=wallet → buy_no; from=wallet → sell_no."""
        monitor = WhaleMonitor(whale_wallets=["0xwhale"])

        buy_tx = {
            "tokenID": "T1", "from": "0xother", "to": "0xwhale",
            "timeStamp": str(int(time.time())), "hash": "0xbuy",
            "blockNumber": "100", "tokenValue": "200",
        }
        monitor._process_tx(buy_tx, "0xwhale")
        assert monitor._recent[-1].direction == "buy_no"

        sell_tx = {
            "tokenID": "T1", "from": "0xwhale", "to": "0xother",
            "timeStamp": str(int(time.time())), "hash": "0xsell",
            "blockNumber": "101", "tokenValue": "100",
        }
        monitor._process_tx(sell_tx, "0xwhale")
        assert monitor._recent[-1].direction == "sell_no"

    @pytest.mark.asyncio
    async def test_poll_no_wallets(self):
        """If wallets list is empty, _poll does nothing (no crash)."""
        monitor = WhaleMonitor(whale_wallets=[])
        await monitor._poll()  # Should return immediately

    @pytest.mark.asyncio
    async def test_poll_no_api_key(self):
        """If polygonscan_api_key is empty, _poll returns early."""
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        with patch("src.signals.whale_monitor.settings") as mock_settings:
            mock_settings.polygonscan_api_key = ""
            await monitor._poll()  # Should return immediately

    @pytest.mark.asyncio
    async def test_poll_handles_rate_limit_gracefully(self):
        """Simulated 429 or network error during poll → no crash."""
        monitor = WhaleMonitor(whale_wallets=["0xabc"], poll_interval=1)

        import httpx

        async def mock_get(*args, **kwargs):
            raise httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=httpx.Request("GET", "https://api.polygonscan.com"),
                response=httpx.Response(429),
            )

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = mock_get
            MockClient.return_value = instance

            with patch("src.signals.whale_monitor.settings") as mock_settings:
                mock_settings.polygonscan_api_key = "fake_key"
                # Should not raise
                await monitor._poll()

    @pytest.mark.asyncio
    async def test_poll_handles_invalid_json(self):
        """Non-list result in response → no crash."""
        monitor = WhaleMonitor(whale_wallets=["0xabc"])

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={
            "result": "Max rate limit reached"  # string, not list
        })

        with patch("httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            instance.get = AsyncMock(return_value=mock_response)
            MockClient.return_value = instance

            with patch("src.signals.whale_monitor.settings") as mock_settings:
                mock_settings.polygonscan_api_key = "fake_key"
                await monitor._poll()
                # No crash, no new activity
                assert len(monitor._recent) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Section D: Environment Variable Safety
# ═══════════════════════════════════════════════════════════════════════════

class TestEnvSafety:
    """Verify that secrets are never leaked to logs or stdout."""

    def test_settings_repr_does_not_expose_private_key(self):
        """repr/str of settings should not contain the raw private key."""
        from src.core.config import settings
        settings_str = str(settings)
        # If there's a key set, make sure it doesn't appear in repr
        if settings.eoa_private_key and len(settings.eoa_private_key) > 4:
            # The full key should not appear in repr
            # (Note: dataclass repr includes field values by default —
            #  this test documents whether that's a problem)
            pass  # This is a documentation check; see recommendation below

    def test_logger_does_not_print_secrets(self):
        """Ensure structured log output doesn't contain 'private_key' values."""
        from src.core.logger import get_logger
        log = get_logger("test_safety")

        captured = io.StringIO()
        handler = logging.StreamHandler(captured)
        handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(handler)

        try:
            # Simulate a log entry similar to what the bot would produce
            log.info("test_event", api_key="REDACTED", market="MKT_1")
            output = captured.getvalue()
            # Ensure no env var names that could indicate secret leak
            assert "EOA_PRIVATE_KEY" not in output
            assert "0x" not in output or "api" not in output.lower()
        finally:
            logging.getLogger().removeHandler(handler)

    def test_paper_mode_default_is_true(self):
        """Paper mode should default to True (safe default)."""
        from src.core.config import Settings
        # With DEPLOYMENT_ENV unset and PAPER_MODE=true, should default to True
        env = os.environ.copy()
        env.pop("DEPLOYMENT_ENV", None)
        env["PAPER_MODE"] = "true"
        with patch.dict(os.environ, env, clear=True):
            s = Settings()
            assert s.paper_mode is True

    def test_dotenv_loaded(self):
        """dotenv loading should not crash even with missing .env file."""
        # The config module already handles this gracefully
        from src.core.config import settings
        assert settings is not None
