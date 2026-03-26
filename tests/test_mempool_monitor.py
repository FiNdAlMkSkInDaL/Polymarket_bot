from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from src.execution.mempool_monitor import (
    DEFAULT_PENDING_VOLUME_THRESHOLD,
    MempoolMonitor,
    PendingVolumeStateMachine,
    POLYMARKET_CTF_CONTRACT,
    POLYGON_USDC_CONTRACTS,
)


def _encode_uint256(value: int) -> bytes:
    return value.to_bytes(32, "big")


def _encode_address(address: str) -> bytes:
    raw = bytes.fromhex(address.lower().replace("0x", ""))
    return b"\x00" * 12 + raw


def _make_split_position_input(amount_raw: int) -> str:
    selector = bytes.fromhex("c9ff79aa")
    parent_collection_id = b"\x00" * 32
    condition_id = bytes.fromhex("11" * 32)
    partition_offset = _encode_uint256(32 * 5)
    amount = _encode_uint256(amount_raw)
    partition = _encode_uint256(2) + _encode_uint256(1) + _encode_uint256(2)
    return "0x" + (selector + _encode_address(next(iter(POLYGON_USDC_CONTRACTS))) + parent_collection_id + condition_id + partition_offset + amount + partition).hex()


def _make_approve_input(spender: str, amount_raw: int) -> str:
    selector = bytes.fromhex("095ea7b3")
    return "0x" + (selector + _encode_address(spender) + _encode_uint256(amount_raw)).hex()


class _FakePendingTxRpcClient:
    def __init__(self, pending_hashes: list[str], tx_map: dict[str, dict]) -> None:
        self._pending_hashes = list(pending_hashes)
        self._tx_map = tx_map
        self.closed = False

    async def subscribe_pending_transactions(self) -> AsyncIterator[str]:
        for tx_hash in self._pending_hashes:
            yield tx_hash

    async def get_transaction_by_hash(self, tx_hash: str):
        return self._tx_map.get(tx_hash)

    async def close(self) -> None:
        self.closed = True


class TestPendingVolumeStateMachine:
    def test_threshold_flips_and_expiry_removes_volume(self):
        machine = PendingVolumeStateMachine(volume_threshold=100.0, ttl_s=10.0)
        machine._tracked["0x1"] = (90.0, 10.0)
        machine._expiry_queue.append((10.0, "0x1"))
        machine._pending_volume = 90.0
        assert machine.is_whale_incoming is False

        machine._tracked["0x2"] = (20.0, 11.0)
        machine._expiry_queue.append((11.0, "0x2"))
        machine._pending_volume = 110.0
        assert machine.is_whale_incoming is True

        machine.sweep_expired(now=10.5)
        assert machine.pending_volume == 20.0
        assert machine.is_whale_incoming is False


class TestMempoolMonitor:
    @pytest.mark.asyncio
    async def test_start_tracks_split_position_pending_volume(self):
        fake_now = [1000.0]
        condition_id = "0x" + "11" * 32
        tx_hash = "0xsplit1"
        tx_map = {
            tx_hash: {
                "hash": tx_hash,
                "from": "0xabc0000000000000000000000000000000000001",
                "to": POLYMARKET_CTF_CONTRACT,
                "input": _make_split_position_input(150_000_000_000),
            }
        }
        monitor = MempoolMonitor(
            _FakePendingTxRpcClient([tx_hash], tx_map),
            volume_threshold=DEFAULT_PENDING_VOLUME_THRESHOLD,
            clock=lambda: fake_now[0],
        )

        await monitor.start()

        assert pytest.approx(monitor.pending_volume) == 150_000.0
        assert monitor.is_whale_incoming is True
        assert monitor.recent_matches[-1].method_name == "splitPosition"
        assert monitor.recent_matches[-1].metadata["condition_id"] == condition_id

    def test_ingest_tracks_usdc_approve_only_when_spender_is_ctf(self):
        usdc_contract = next(iter(POLYGON_USDC_CONTRACTS))
        fake_now = [1000.0]
        monitor = MempoolMonitor(
            _FakePendingTxRpcClient([], {}),
            volume_threshold=50_000.0,
            clock=lambda: fake_now[0],
        )

        ignored = monitor.ingest_transaction({
            "hash": "0xignore",
            "from": "0xabc0000000000000000000000000000000000001",
            "to": usdc_contract,
            "input": _make_approve_input("0x1234000000000000000000000000000000009999", 90_000_000_000),
        }, seen_at=1000.0)
        assert ignored is None
        assert monitor.pending_volume == 0.0

        tracked = monitor.ingest_transaction({
            "hash": "0xapprove",
            "from": "0xabc0000000000000000000000000000000000001",
            "to": usdc_contract,
            "input": _make_approve_input(POLYMARKET_CTF_CONTRACT, 90_000_000_000),
        }, seen_at=1001.0)
        assert tracked is not None
        assert tracked.method_name == "approve"
        assert pytest.approx(monitor.pending_volume) == 90_000.0
        assert monitor.is_whale_incoming is True

    def test_duplicate_hash_replaces_pending_volume_in_o1_state(self):
        fake_now = [1000.0]
        monitor = MempoolMonitor(
            _FakePendingTxRpcClient([], {}),
            volume_threshold=100_000.0,
            clock=lambda: fake_now[0],
        )
        transaction = {
            "hash": "0xdup",
            "from": "0xabc0000000000000000000000000000000000001",
            "to": POLYMARKET_CTF_CONTRACT,
            "input": _make_split_position_input(60_000_000_000),
        }
        updated_transaction = {
            **transaction,
            "input": _make_split_position_input(80_000_000_000),
        }

        monitor.ingest_transaction(transaction, seen_at=1000.0)
        monitor.ingest_transaction(updated_transaction, seen_at=1001.0)

        assert pytest.approx(monitor.pending_volume) == 80_000.0
        assert monitor.is_whale_incoming is False

    def test_remove_pending_and_ttl_cleanup(self):
        fake_now = [1000.0]
        monitor = MempoolMonitor(
            _FakePendingTxRpcClient([], {}),
            volume_threshold=100_000.0,
            pending_ttl_s=10.0,
            clock=lambda: fake_now[0],
        )
        monitor.ingest_transaction({
            "hash": "0xttl",
            "from": "0xabc0000000000000000000000000000000000001",
            "to": POLYMARKET_CTF_CONTRACT,
            "input": _make_split_position_input(70_000_000_000),
        }, seen_at=1000.0)
        monitor.ingest_transaction({
            "hash": "0xother",
            "from": "0xabc0000000000000000000000000000000000001",
            "to": POLYMARKET_CTF_CONTRACT,
            "input": _make_split_position_input(50_000_000_000),
        }, seen_at=1002.0)

        monitor.remove_pending("0xother", now=1003.0)
        assert pytest.approx(monitor.pending_volume) == 70_000.0

        fake_now[0] = 1011.0
        monitor.ingest_transaction(None, seen_at=1011.0)
        assert monitor.pending_volume == 0.0
