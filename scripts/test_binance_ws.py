from __future__ import annotations

import asyncio
import time

from src.data.adapters.binance_adapter import BinanceWebSocketAdapter
from src.data.oracle_adapter import OracleMarketConfig, OracleSnapshot


async def main() -> None:
    queue: asyncio.Queue[OracleSnapshot] = asyncio.Queue(maxsize=1000)
    cfg = OracleMarketConfig(
        market_id="dryrun_binance_btcusdt",
        oracle_type="crypto",
        external_id="btcusdt",
    )
    adapter = BinanceWebSocketAdapter(cfg)
    task = asyncio.create_task(adapter.start(queue), name="test_binance_ws")

    start = time.monotonic()
    print("starting_binance_ws_dry_run symbol=btcusdt duration_s=5")

    tick_count = 0
    try:
        while True:
            elapsed = time.monotonic() - start
            remaining = 5.0 - elapsed
            if remaining <= 0:
                break

            timeout = min(1.0, remaining)
            try:
                snapshot = await asyncio.wait_for(queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                continue

            price = snapshot.raw_state.get("price")
            qty = snapshot.raw_state.get("quantity")
            event_time_ms = snapshot.raw_state.get("event_time_ms")
            tick_count += 1
            print(f"tick symbol=btcusdt price={price} qty={qty} event_time_ms={event_time_ms}")

        if tick_count == 0:
            print("warning no_ticks_received_in_5s_window")
    finally:
        adapter.stop()
        try:
            await asyncio.wait_for(task, timeout=1.5)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    print("binance_ws_dry_run_complete")


if __name__ == "__main__":
    asyncio.run(main())
