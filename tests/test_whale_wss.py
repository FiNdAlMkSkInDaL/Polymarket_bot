"""
Tests for the WebSocket-based whale monitor — event decoding, heartbeat,
reconnection, and interface preservation.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.signals.whale_monitor import (
    CTF_CONTRACT,
    TRANSFER_BATCH_TOPIC,
    TRANSFER_SINGLE_TOPIC,
    WhaleActivity,
    WhaleMonitor,
    _topic_to_address,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _encode_uint256(value: int) -> bytes:
    """ABI-encode a uint256 as 32 bytes, big-endian."""
    return value.to_bytes(32, "big")


def _addr_to_topic(addr: str) -> str:
    """Left-pad a 20-byte address to a 32-byte topic hex."""
    raw = addr.lower().replace("0x", "")
    return "0x" + raw.rjust(64, "0")


def _make_transfer_single_log(
    operator: str,
    from_addr: str,
    to_addr: str,
    token_id: int,
    value: int,
    tx_hash: str = "0xabc123",
) -> dict:
    """Build a raw EVM log dict for TransferSingle."""
    data = _encode_uint256(token_id) + _encode_uint256(value)
    return {
        "transactionHash": tx_hash,
        "topics": [
            TRANSFER_SINGLE_TOPIC,
            _addr_to_topic(operator),
            _addr_to_topic(from_addr),
            _addr_to_topic(to_addr),
        ],
        "data": "0x" + data.hex(),
    }


def _make_transfer_batch_log(
    operator: str,
    from_addr: str,
    to_addr: str,
    token_ids: list[int],
    values: list[int],
    tx_hash: str = "0xbatch456",
) -> dict:
    """Build a raw EVM log dict for TransferBatch with ABI-encoded arrays."""
    # ABI layout: offset_ids (32) | offset_values (32) | ids_data | vals_data
    n = len(token_ids)

    # offsets: ids starts at byte 64, vals at 64 + 32 + n*32
    offset_ids = 64
    offset_vals = offset_ids + 32 + n * 32

    parts = bytearray()
    parts += _encode_uint256(offset_ids)
    parts += _encode_uint256(offset_vals)
    # ids array: length + elements
    parts += _encode_uint256(n)
    for tid in token_ids:
        parts += _encode_uint256(tid)
    # values array: length + elements
    parts += _encode_uint256(n)
    for v in values:
        parts += _encode_uint256(v)

    return {
        "transactionHash": tx_hash,
        "topics": [
            TRANSFER_BATCH_TOPIC,
            _addr_to_topic(operator),
            _addr_to_topic(from_addr),
            _addr_to_topic(to_addr),
        ],
        "data": "0x" + bytes(parts).hex(),
    }


# ── Unit tests ──────────────────────────────────────────────────────────


class TestTopicToAddress:
    def test_extracts_address(self):
        topic = "0x000000000000000000000000d8da6bf26964af9d7eed9e03e53415d37aa96045"
        assert _topic_to_address(topic) == "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"

    def test_short_topic(self):
        """Short/malformed topics still produce a 0x-prefixed string."""
        topic = "0xabcdef1234567890"
        result = _topic_to_address(topic)
        assert result.startswith("0x")


class TestTransferSingleDecoding:
    """Verify _process_log correctly decodes TransferSingle events."""

    def test_tracked_wallet_buy(self):
        """Transfer TO a tracked wallet → buy activity."""
        wallet = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
        monitor = WhaleMonitor(whale_wallets=[wallet])
        monitor._whale_threshold = 999_999_999  # only trigger on tracked wallets

        log_entry = _make_transfer_single_log(
            operator="0x1111111111111111111111111111111111111111",
            from_addr="0x2222222222222222222222222222222222222222",
            to_addr=wallet,
            token_id=42,
            value=1000,
            tx_hash="0xtx_buy_1",
        )

        monitor._process_log(log_entry)
        assert len(monitor._recent) == 1
        act = monitor._recent[0]
        assert act.wallet == wallet.lower()
        assert act.direction == "buy_no"
        assert act.amount == 1000.0
        assert act.market_token_id == "42"

    def test_tracked_wallet_sell(self):
        """Transfer FROM a tracked wallet → sell activity."""
        wallet = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
        monitor = WhaleMonitor(whale_wallets=[wallet])
        monitor._whale_threshold = 999_999_999

        log_entry = _make_transfer_single_log(
            operator="0x1111111111111111111111111111111111111111",
            from_addr=wallet,
            to_addr="0x3333333333333333333333333333333333333333",
            token_id=99,
            value=500,
            tx_hash="0xtx_sell_1",
        )

        monitor._process_log(log_entry)
        assert len(monitor._recent) == 1
        act = monitor._recent[0]
        assert act.direction == "sell_no"
        assert act.amount == 500.0

    def test_large_transfer_anonymous(self):
        """Transfer exceeding threshold from unknown wallet → detected."""
        monitor = WhaleMonitor(whale_wallets=["0xaaaa"])
        monitor._whale_threshold = 100_000

        log_entry = _make_transfer_single_log(
            operator="0x1111111111111111111111111111111111111111",
            from_addr="0x4444444444444444444444444444444444444444",
            to_addr="0x5555555555555555555555555555555555555555",
            token_id=7,
            value=200_000,
            tx_hash="0xtx_anon_1",
        )

        monitor._process_log(log_entry)
        assert len(monitor._recent) == 1
        assert monitor._recent[0].amount == 200_000.0

    def test_small_transfer_anonymous_ignored(self):
        """Transfer below threshold from unknown wallet → ignored."""
        monitor = WhaleMonitor(whale_wallets=["0xaaaa"])
        monitor._whale_threshold = 100_000

        log_entry = _make_transfer_single_log(
            operator="0x1111111111111111111111111111111111111111",
            from_addr="0x4444444444444444444444444444444444444444",
            to_addr="0x5555555555555555555555555555555555555555",
            token_id=7,
            value=50,
            tx_hash="0xtx_small_1",
        )

        monitor._process_log(log_entry)
        assert len(monitor._recent) == 0

    def test_deduplication(self):
        """Same tx_hash + token + wallet → only one activity."""
        wallet = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
        monitor = WhaleMonitor(whale_wallets=[wallet])
        monitor._whale_threshold = 999_999_999

        log_entry = _make_transfer_single_log(
            operator="0x1111111111111111111111111111111111111111",
            from_addr="0x2222222222222222222222222222222222222222",
            to_addr=wallet,
            token_id=42,
            value=1000,
            tx_hash="0xdup_hash",
        )

        monitor._process_log(log_entry)
        monitor._process_log(log_entry)
        assert len(monitor._recent) == 1

    def test_market_map_direction(self):
        """When market map is set, direction uses yes/no correctly."""
        wallet = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
        monitor = WhaleMonitor(whale_wallets=[wallet])
        monitor._whale_threshold = 999_999_999
        monitor.set_market_map({"42": ("cond_abc", "yes")})

        log_entry = _make_transfer_single_log(
            operator="0x1111111111111111111111111111111111111111",
            from_addr="0x2222222222222222222222222222222222222222",
            to_addr=wallet,
            token_id=42,
            value=1000,
            tx_hash="0xtx_mapped",
        )

        monitor._process_log(log_entry)
        assert monitor._recent[0].direction == "buy_yes"


class TestTransferBatchDecoding:
    """Verify _process_log correctly decodes TransferBatch events."""

    def test_batch_tracked_wallet(self):
        """Batch transfer TO tracked wallet → multiple activities."""
        wallet = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
        monitor = WhaleMonitor(whale_wallets=[wallet])
        monitor._whale_threshold = 999_999_999

        log_entry = _make_transfer_batch_log(
            operator="0x1111111111111111111111111111111111111111",
            from_addr="0x2222222222222222222222222222222222222222",
            to_addr=wallet,
            token_ids=[10, 20, 30],
            values=[100, 200, 300],
            tx_hash="0xbatch_1",
        )

        monitor._process_log(log_entry)
        assert len(monitor._recent) == 3
        amounts = [a.amount for a in monitor._recent]
        assert amounts == [100.0, 200.0, 300.0]
        token_ids = [a.market_token_id for a in monitor._recent]
        assert token_ids == ["10", "20", "30"]

    def test_batch_large_anonymous(self):
        """Batch with one transfer above threshold → only that one captured."""
        monitor = WhaleMonitor(whale_wallets=["0xaaaa"])
        monitor._whale_threshold = 150

        log_entry = _make_transfer_batch_log(
            operator="0x1111111111111111111111111111111111111111",
            from_addr="0x4444444444444444444444444444444444444444",
            to_addr="0x5555555555555555555555555555555555555555",
            token_ids=[10, 20],
            values=[100, 200],
            tx_hash="0xbatch_anon",
        )

        monitor._process_log(log_entry)
        # Only the 200-value transfer exceeds threshold
        assert len(monitor._recent) == 1
        assert monitor._recent[0].amount == 200.0


class TestInterfacePreservation:
    """Verify that the public API surface is unchanged."""

    def test_has_confluence_still_works(self):
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        monitor._recent.append(WhaleActivity(
            wallet="0xabc",
            market_token_id="TOKEN_X",
            direction="buy_no",
            amount=5000.0,
            timestamp=time.time(),
            tx_hash="0xtest1",
        ))
        assert monitor.has_confluence("TOKEN_X") is True
        assert monitor.has_confluence("OTHER") is False

    def test_has_strong_confluence(self):
        monitor = WhaleMonitor(whale_wallets=["0xa", "0xb"])
        now = time.time()
        for wallet, tx in [("0xa", "0xt1"), ("0xb", "0xt2")]:
            monitor._recent.append(WhaleActivity(
                wallet=wallet,
                market_token_id="TOKEN_Y",
                direction="buy_no",
                amount=1000.0,
                timestamp=now,
                tx_hash=tx,
            ))
        assert monitor.has_strong_confluence("TOKEN_Y") is True

    def test_has_whale_sells(self):
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        monitor._recent.append(WhaleActivity(
            wallet="0xabc",
            market_token_id="TOKEN_Z",
            direction="sell_yes",
            amount=2000.0,
            timestamp=time.time(),
            tx_hash="0xsell1",
        ))
        assert monitor.has_whale_sells("TOKEN_Z") is True

    def test_get_whale_tokens(self):
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        monitor._recent.append(WhaleActivity(
            wallet="0xabc",
            market_token_id="TOKEN_A",
            direction="buy_no",
            amount=1000.0,
            timestamp=time.time(),
            tx_hash="0xt",
        ))
        tokens = monitor.get_whale_tokens()
        assert "token_a" in tokens

    def test_recent_activity_property(self):
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        assert monitor.recent_activity == []
        monitor._recent.append(WhaleActivity(
            wallet="0xabc",
            market_token_id="T",
            direction="buy_no",
            amount=1.0,
            timestamp=time.time(),
            tx_hash="0x1",
        ))
        assert len(monitor.recent_activity) == 1

    def test_set_market_map(self):
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        monitor.set_market_map({"asset1": ("cond1", "yes"), "asset2": ("cond2", "no")})
        assert monitor._token_to_market["asset1"] == "yes"
        assert monitor._asset_to_condition["asset1"] == "cond1"


class TestFallbackPolling:
    """Verify the REST polling fallback path is preserved."""

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

    def test_process_tx_direction(self):
        """to=wallet → buy_no; from=wallet → sell_no."""
        monitor = WhaleMonitor(whale_wallets=["0xwhale"])

        buy_tx = {
            "tokenID": "T1", "from": "0xother", "to": "0xwhale",
            "timeStamp": str(int(time.time())), "hash": "0xbuy",
            "blockNumber": "100", "tokenValue": "200",
        }
        monitor._process_tx(buy_tx, "0xwhale")
        assert monitor._recent[-1].direction == "buy_no"

    @pytest.mark.asyncio
    async def test_poll_no_wallets(self):
        monitor = WhaleMonitor(whale_wallets=[])
        await monitor._poll()  # No crash

    @pytest.mark.asyncio
    async def test_poll_no_api_key(self):
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        with patch("src.signals.whale_monitor.settings") as mock_settings:
            mock_settings.polygonscan_api_key = ""
            await monitor._poll()


class TestStartModeSelection:
    """Verify start() picks WSS vs poll based on config."""

    @pytest.mark.asyncio
    async def test_starts_wss_when_url_set(self):
        monitor = WhaleMonitor(whale_wallets=["0xabc"], wss_url="wss://fake")
        monitor._running = False  # prevent actual loop

        with patch.object(monitor, "_run_wss_loop", new_callable=AsyncMock) as mock_wss:
            with patch.object(monitor, "_run_poll_loop", new_callable=AsyncMock) as mock_poll:
                with patch.object(monitor, "_maybe_rebuild_clusters", new_callable=AsyncMock):
                    await monitor.start()
                    mock_wss.assert_awaited_once()
                    mock_poll.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_starts_poll_when_no_url(self):
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        monitor._wss_url = ""
        monitor._running = False

        with patch.object(monitor, "_run_wss_loop", new_callable=AsyncMock) as mock_wss:
            with patch.object(monitor, "_run_poll_loop", new_callable=AsyncMock) as mock_poll:
                with patch.object(monitor, "_maybe_rebuild_clusters", new_callable=AsyncMock):
                    await monitor.start()
                    mock_poll.assert_awaited_once()
                    mock_wss.assert_not_awaited()


class TestTrimStale:
    """Verify stale entry cleanup."""

    def test_trim_removes_old_entries(self):
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        monitor._recent.append(WhaleActivity(
            wallet="0xabc",
            market_token_id="T",
            direction="buy_no",
            amount=1.0,
            timestamp=time.time() - 7200,  # 2 hours ago
            tx_hash="0xold",
        ))
        monitor._recent_tx_hashes.add("0xold")
        monitor._trim_stale()
        assert len(monitor._recent) == 0
        assert "0xold" not in monitor._recent_tx_hashes

    def test_trim_keeps_recent(self):
        monitor = WhaleMonitor(whale_wallets=["0xabc"])
        monitor._recent.append(WhaleActivity(
            wallet="0xabc",
            market_token_id="T",
            direction="buy_no",
            amount=1.0,
            timestamp=time.time() - 60,  # 1 minute ago
            tx_hash="0xrecent",
        ))
        monitor._recent_tx_hashes.add("0xrecent")
        monitor._trim_stale()
        assert len(monitor._recent) == 1
