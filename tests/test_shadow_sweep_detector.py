from __future__ import annotations

import pytest

from src.events.mev_events import ShadowSweepSignal
from src.execution.mempool_monitor import MempoolMonitor, POLYMARKET_CTF_CONTRACT
from src.signals.shadow_sweep_detector import ShadowSweepDetector


def _encode_uint256(value: int) -> bytes:
    return value.to_bytes(32, "big")


def _encode_address(address: str) -> bytes:
    raw = bytes.fromhex(address.lower().replace("0x", ""))
    return b"\x00" * 12 + raw


def _make_split_position_input(collateral_token: str, condition_id_hex: str, amount_raw: int) -> str:
    selector = bytes.fromhex("c9ff79aa")
    parent_collection_id = b"\x00" * 32
    condition_id = bytes.fromhex(condition_id_hex.lower().replace("0x", ""))
    partition_offset = _encode_uint256(32 * 5)
    amount = _encode_uint256(amount_raw)
    partition = _encode_uint256(2) + _encode_uint256(1) + _encode_uint256(2)
    return "0x" + (
        selector
        + _encode_address(collateral_token)
        + parent_collection_id
        + condition_id
        + partition_offset
        + amount
        + partition
    ).hex()


class _StubPendingTxClient:
    async def subscribe_pending_transactions(self):
        if False:
            yield ""

    async def get_transaction_by_hash(self, tx_hash: str):
        return None

    async def close(self) -> None:
        return None


def test_shadow_sweep_detector_translates_mempool_spike_to_signal() -> None:
    condition_id = "0x" + "42" * 32
    monitor = MempoolMonitor(
        _StubPendingTxClient(),
        volume_threshold=100_000.0,
        clock=lambda: 1000.0,
    )
    detector = ShadowSweepDetector(
        monitor,
        direction_resolver=lambda match: "NO",
        max_capital=75.0,
        premium_pct=0.04,
    )

    signal = detector.ingest_transaction(
        {
            "hash": "0xshadow",
            "from": "0xabc0000000000000000000000000000000000001",
            "to": POLYMARKET_CTF_CONTRACT,
            "input": _make_split_position_input(
                "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
                condition_id,
                150_000_000_000,
            ),
        },
        seen_at=1000.0,
    )

    assert signal is not None
    assert isinstance(signal, ShadowSweepSignal)
    assert signal.target_market_id == condition_id
    assert signal.direction == "NO"
    assert signal.max_capital == 75.0
    assert signal.premium_pct == 0.04


def test_shadow_sweep_detector_deduplicates_same_trigger_hash() -> None:
    condition_id = "0x" + "24" * 32
    monitor = MempoolMonitor(
        _StubPendingTxClient(),
        volume_threshold=50_000.0,
        clock=lambda: 1000.0,
    )
    detector = ShadowSweepDetector(monitor, max_capital=50.0)
    transaction = {
        "hash": "0xdedup",
        "from": "0xabc0000000000000000000000000000000000001",
        "to": POLYMARKET_CTF_CONTRACT,
        "input": _make_split_position_input(
            "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
            condition_id,
            80_000_000_000,
        ),
    }

    first = detector.ingest_transaction(transaction, seen_at=1000.0)
    second = detector.evaluate()

    assert first is not None
    assert second is None


def test_shadow_sweep_detector_returns_none_when_market_id_unresolved() -> None:
    monitor = MempoolMonitor(
        _StubPendingTxClient(),
        volume_threshold=50_000.0,
        clock=lambda: 1000.0,
    )
    detector = ShadowSweepDetector(
        monitor,
        market_id_resolver=lambda match: "",
    )

    signal = detector.ingest_transaction(
        {
            "hash": "0xnomarket",
            "from": "0xabc0000000000000000000000000000000000001",
            "to": POLYMARKET_CTF_CONTRACT,
            "input": _make_split_position_input(
                "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
                "0x" + "11" * 32,
                90_000_000_000,
            ),
        },
        seen_at=1000.0,
    )

    assert signal is None