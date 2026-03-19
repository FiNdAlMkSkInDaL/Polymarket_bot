#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.data.adapters.websocket_adapter_base import WebSocketOracleAdapter
from src.data.oracle_adapter import OracleAdapterRegistry, OracleMarketConfig, OracleSnapshot

SUPPORTED_TYPES = {"odds_api_ws", "tree_news_ws"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test SI-8 websocket oracle adapters against live feeds.",
    )
    parser.add_argument(
        "--oracle-type",
        choices=sorted(SUPPORTED_TYPES),
        action="append",
        help="Restrict to one or more websocket adapter types.",
    )
    parser.add_argument(
        "--market-id",
        action="append",
        help="Restrict to one or more market_id values from ORACLE_MARKET_CONFIGS.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Maximum runtime in seconds before a clean shutdown. Default: 30.",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=0,
        help="Stop after printing this many snapshots across all adapters. 0 means unlimited.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print OracleSnapshot JSON instead of compact single-line output.",
    )
    return parser.parse_args(argv)


def load_ws_market_configs(args: argparse.Namespace) -> list[OracleMarketConfig]:
    raw = settings.strategy.oracle_market_configs
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid ORACLE_MARKET_CONFIGS JSON: {exc}") from exc

    if not isinstance(rows, list):
        raise SystemExit("ORACLE_MARKET_CONFIGS must decode to a list of objects")

    allowed_types = set(args.oracle_type or SUPPORTED_TYPES)
    allowed_market_ids = set(args.market_id or [])
    configs: list[OracleMarketConfig] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("oracle_type") not in allowed_types:
            continue
        if allowed_market_ids and row.get("market_id") not in allowed_market_ids:
            continue
        configs.append(
            OracleMarketConfig(
                market_id=str(row.get("market_id", "")),
                oracle_type=str(row.get("oracle_type", "")),
                oracle_params=dict(row.get("oracle_params", {})),
                external_id=str(row.get("external_id", "")),
                target_outcome=str(row.get("target_outcome", "")),
                market_type=str(row.get("market_type", "winner")),
                goal_line=float(row.get("goal_line", 2.5) or 2.5),
                yes_asset_id=str(row.get("yes_asset_id", "")),
                no_asset_id=str(row.get("no_asset_id", "")),
                event_id=str(row.get("event_id", "")),
            )
        )

    return configs


def validate_env_for_configs(configs: list[OracleMarketConfig]) -> None:
    missing: list[str] = []
    for cfg in configs:
        if cfg.oracle_type == "odds_api_ws" and not settings.strategy.oracle_odds_api_ws_url:
            missing.append("ORACLE_ODDS_API_WS_URL")
        if cfg.oracle_type == "tree_news_ws" and not settings.strategy.oracle_tree_news_ws_url:
            missing.append("ORACLE_TREE_NEWS_WS_URL")

    if missing:
        names = ", ".join(sorted(set(missing)))
        raise SystemExit(f"Missing required websocket env var(s): {names}")


def render_snapshot(snapshot: OracleSnapshot, pretty: bool) -> str:
    payload = asdict(snapshot)
    if pretty:
        return json.dumps(payload, indent=2, sort_keys=True)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


async def stream_adapter(
    adapter: WebSocketOracleAdapter,
    output_queue: asyncio.Queue[tuple[str, OracleSnapshot]],
) -> None:
    async for snapshot in adapter.stream_snapshots():
        await output_queue.put((adapter.name, snapshot))


async def run(args: argparse.Namespace) -> int:
    configs = load_ws_market_configs(args)
    if not configs:
        print("No websocket oracle configs matched the requested filters.")
        return 1

    validate_env_for_configs(configs)
    registry = OracleAdapterRegistry()
    output_queue: asyncio.Queue[tuple[str, OracleSnapshot]] = asyncio.Queue()

    adapters: list[WebSocketOracleAdapter] = []
    for cfg in configs:
        adapter = registry.create(cfg.oracle_type, cfg)
        if not isinstance(adapter, WebSocketOracleAdapter):
            raise SystemExit(f"Configured oracle_type {cfg.oracle_type!r} is not a websocket adapter")
        adapters.append(adapter)

    print(f"Loaded {len(adapters)} websocket oracle adapter(s)")
    for adapter in adapters:
        print(f"  - {adapter.name} market_id={adapter._config.market_id} external_id={adapter._config.external_id}")
    print()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    tasks = [asyncio.create_task(stream_adapter(adapter, output_queue)) for adapter in adapters]
    deadline = loop.time() + max(args.duration, 0.0)
    printed = 0

    try:
        while not stop_event.is_set():
            timeout = max(0.0, deadline - loop.time()) if args.duration > 0 else None
            if args.duration > 0 and timeout == 0.0:
                break
            try:
                adapter_name, snapshot = await asyncio.wait_for(output_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break

            print(f"[{adapter_name}] {render_snapshot(snapshot, args.pretty)}")
            printed += 1
            if args.max_snapshots > 0 and printed >= args.max_snapshots:
                break
    finally:
        for adapter in adapters:
            adapter.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    print()
    print(f"Captured {printed} OracleSnapshot payload(s)")
    return 0 if printed > 0 else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())