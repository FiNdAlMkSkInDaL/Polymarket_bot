#!/usr/bin/env python3
"""
Verification script — test that the refactored WhaleMonitor correctly
captures and decodes real-time CTF TransferSingle / TransferBatch events
from the Polygon WebSocket RPC.

Usage:
    # Set your WSS endpoint in .env or environment:
    #   POLYGON_RPC_WSS_URL=wss://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY
    #
    # Then run:
    python scripts/test_whale_wss.py

    # Or with a custom duration (seconds):
    python scripts/test_whale_wss.py --duration 60

The script will:
  1. Connect to the Polygon WSS and subscribe to CTF logs.
  2. Print every decoded whale transfer in real-time.
  3. After --duration seconds, print a summary and exit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.signals.whale_monitor import (
    CTF_CONTRACT,
    TRANSFER_BATCH_TOPIC,
    TRANSFER_SINGLE_TOPIC,
    WhaleActivity,
    WhaleMonitor,
    _topic_to_address,
)


def _print_activity(activity: WhaleActivity) -> None:
    """Pretty-print a whale activity event."""
    elapsed = time.time() - activity.timestamp
    print(
        f"  [{activity.direction:>10}]  "
        f"wallet={activity.wallet[:12]}…  "
        f"token={activity.market_token_id[:20]}  "
        f"amount={activity.amount:,.0f}  "
        f"tx={activity.tx_hash[:16]}…  "
        f"age={elapsed:.1f}s"
    )


async def run_live_test(duration: float) -> None:
    """Run the whale monitor for *duration* seconds, printing events."""
    wss_url = os.getenv("POLYGON_RPC_WSS_URL", "")
    if not wss_url:
        print("ERROR: POLYGON_RPC_WSS_URL not set.  Add it to .env or export it.")
        print("  Example: POLYGON_RPC_WSS_URL=wss://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY")
        sys.exit(1)

    print(f"CTF contract : {CTF_CONTRACT}")
    print(f"WSS endpoint : {wss_url[:60]}…")
    print(f"Duration     : {duration:.0f}s")
    print(f"Threshold    : {os.getenv('WHALE_THRESHOLD_SHARES', '50000')} shares")
    print()

    # Use a low threshold so we see many events during the test
    monitor = WhaleMonitor(
        whale_wallets=["0x0000000000000000000000000000000000000000"],  # dummy
        wss_url=wss_url,
    )

    # Override threshold to 0 to capture ALL transfers for verification
    monitor._whale_threshold = 0

    # Patch _emit_activity to print live
    original_emit = monitor._emit_activity

    def _patched_emit(from_addr, to_addr, token_id, value, tx_hash):
        original_emit(from_addr, to_addr, token_id, value, tx_hash)
        if monitor._recent:
            _print_activity(monitor._recent[-1])

    monitor._emit_activity = _patched_emit

    # Start as background task, stop after duration
    task = asyncio.create_task(monitor.start())
    print(f"Listening for CTF transfer events…\n")

    await asyncio.sleep(duration)
    await monitor.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total events captured : {len(monitor._recent)}")
    print(f"Reconnect count       : {monitor.reconnect_count}")

    if monitor._recent:
        buys = [a for a in monitor._recent if a.direction.startswith("buy")]
        sells = [a for a in monitor._recent if a.direction.startswith("sell")]
        tokens = {a.market_token_id for a in monitor._recent}
        wallets = {a.wallet for a in monitor._recent}
        print(f"  Buys  : {len(buys)}")
        print(f"  Sells : {len(sells)}")
        print(f"  Unique tokens  : {len(tokens)}")
        print(f"  Unique wallets : {len(wallets)}")

        # Show top volume events
        top = sorted(monitor._recent, key=lambda a: a.amount, reverse=True)[:5]
        print(f"\nTop 5 by volume:")
        for a in top:
            _print_activity(a)
    else:
        print("  No events captured.  This may mean there was no CTF")
        print("  activity on Polygon during the test window, or the")
        print("  WebSocket subscription failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whale WSS streaming")
    parser.add_argument(
        "--duration", type=float, default=30,
        help="How many seconds to listen (default: 30)",
    )
    args = parser.parse_args()
    asyncio.run(run_live_test(args.duration))


if __name__ == "__main__":
    main()
