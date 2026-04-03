#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.core.config import settings
from src.core.logger import get_logger, setup_logging
from src.data.alchemy_rpc_client import AlchemyRpcClient
from src.data.l2_book import L2OrderBook
from src.data.l2_websocket import L2WebSocket
from src.data.market_discovery import MarketInfo
from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.live_execution_boundary import LiveExecutionBoundary, build_live_execution_boundary
from src.execution.mev_router import MevExecutionRouter, MevMarketSnapshot
from src.execution.priority_dispatcher import PriorityDispatcher
from src.signals.hybrid_arb_maker import HybridArbMaker


log = get_logger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "hybrid_arb_universe_balanced.json"


def _normalize_condition_id(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError("condition_id is required")
    if not normalized.startswith("0x"):
        normalized = "0x" + normalized
    body = normalized[2:]
    if len(body) != 64:
        raise ValueError(f"condition_id must be 32 bytes / 64 hex chars; got {value!r}")
    int(body, 16)
    return normalized


def _extract_condition_ids(payload: Any) -> list[str]:
    discovered: list[str] = []
    seen: set[str] = set()

    def _append(raw_value: Any) -> None:
        if not isinstance(raw_value, str):
            return
        try:
            condition_id = _normalize_condition_id(raw_value)
        except ValueError:
            return
        if condition_id in seen:
            return
        seen.add(condition_id)
        discovered.append(condition_id)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            _append(node.get("condition_id"))
            _append(node.get("conditionId"))
            for value in node.values():
                _walk(value)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        _append(node)

    _walk(payload)
    return discovered


def _load_condition_ids(config_path: Path) -> list[str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Hybrid arb config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    markets = payload.get("markets") if isinstance(payload, dict) else None
    if isinstance(markets, list):
        condition_ids = []
        seen: set[str] = set()
        for row in markets:
            if not isinstance(row, dict):
                continue
            raw_condition_id = row.get("condition_id") or row.get("market_id")
            if not isinstance(raw_condition_id, str):
                continue
            condition_id = _normalize_condition_id(raw_condition_id)
            if condition_id in seen:
                continue
            seen.add(condition_id)
            condition_ids.append(condition_id)
    else:
        condition_ids = _extract_condition_ids(payload)
    if not condition_ids:
        raise ValueError(f"No condition_ids found in {config_path}")
    return condition_ids


def _require_alchemy_rpc_url() -> str:
    rpc_url = os.getenv("ALCHEMY_POLYGON_RPC_URL", "").strip()
    if not rpc_url:
        raise EnvironmentError("ALCHEMY_POLYGON_RPC_URL is required for launch_hybrid_arb")
    return rpc_url


def _session_id(prefix: str = "hybrid-arb") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}"


def _top_levels(book: L2OrderBook, side: str, depth: int = 1) -> list[dict[str, float]]:
    return [
        {"price": float(level.price), "size": float(level.size)}
        for level in book.levels(side, n=depth)
    ]


def _complementary_price(value: float) -> float:
    if value <= 0.0:
        return 0.0
    return round(min(0.99, max(0.01, 1.0 - value)), 2)


def _build_clob_client() -> Any:
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
    except ModuleNotFoundError as exc:
        raise RuntimeError("py_clob_client is required for live hybrid arb dispatch mode") from exc

    creds = ApiCreds(
        api_key=settings.polymarket_api_key,
        api_secret=settings.polymarket_secret,
        api_passphrase=settings.polymarket_passphrase,
    )
    return ClobClient(
        settings.clob_http_url,
        key=settings.eoa_private_key,
        chain_id=137,
        creds=creds,
    )


@dataclass(frozen=True, slots=True)
class ResolvedHybridMarket:
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    metadata_market_id: str
    market_maker_address: str

    def to_market_info(self) -> MarketInfo:
        return MarketInfo(
            condition_id=self.condition_id,
            question=self.question,
            yes_token_id=self.yes_token_id,
            no_token_id=self.no_token_id,
            daily_volume_usd=0.0,
            end_date=None,
            active=True,
            event_id="",
            liquidity_usd=0.0,
            score=0.0,
            accepting_orders=True,
            tags="hybrid_arb",
            neg_risk=False,
        )


class HybridArbRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._stop_event = asyncio.Event()
        self._strategy_lock = threading.Lock()
        self._session_id = args.session_id or _session_id()
        self._alchemy_client: AlchemyRpcClient | None = None
        self._dispatch_boundary: LiveExecutionBoundary | None = None
        self._dispatcher: PriorityDispatcher | None = None
        self._strategy: HybridArbMaker | None = None
        self._l2_ws: L2WebSocket | None = None
        self._market_by_yes_asset: dict[str, ResolvedHybridMarket] = {}
        self._resolved_market_by_condition: dict[str, ResolvedHybridMarket] = {}
        self._yes_books_by_asset: dict[str, L2OrderBook] = {}
        self._condition_id_by_yes_asset: dict[str, str] = {}
        self._market_by_condition: dict[str, MarketInfo] = {}
        self._tasks: list[asyncio.Task[Any]] = []
        self._reserve_fetch_successes = 0
        self._reserve_fetch_failures = 0
        self._last_reserve_error = ""
        self._inactive_condition_ids: set[str] = set()
        self._preflight_reserve_cache: dict[str, dict[str, Decimal]] = {}

    async def start(self) -> None:
        _require_alchemy_rpc_url()
        normalized_env = str(self._args.env).strip().upper()
        if normalized_env not in {"PAPER", "LIVE"}:
            raise RuntimeError(f"Unsupported env for launch_hybrid_arb: {self._args.env!r}")
        if self._args.dispatch_mode == "live":
            missing = settings.validate_credentials()
            if missing:
                raise RuntimeError("Live dispatch mode is missing required credentials: " + "; ".join(missing))

        condition_ids = _load_condition_ids(self._args.config)
        log.info(
            "hybrid_arb_config_loaded",
            config_path=str(self._args.config),
            condition_count=len(condition_ids),
            dispatch_mode=self._args.dispatch_mode,
        )

        self._alchemy_client = AlchemyRpcClient()
        resolved_markets = await asyncio.to_thread(self._resolve_markets, condition_ids)
        live_ready_markets = await asyncio.to_thread(self._filter_live_ready_markets, resolved_markets)
        if not live_ready_markets:
            log.warning(
                "hybrid_arb_no_live_ready_markets",
                configured_markets=len(condition_ids),
                metadata_ready_markets=len(resolved_markets),
                inactive_markets=len(self._inactive_condition_ids),
            )
            await self.stop()
            return
        self._resolved_market_by_condition = {
            market.condition_id: market for market in live_ready_markets
        }
        self._market_by_yes_asset = {
            market.yes_token_id: market for market in live_ready_markets
        }
        self._market_by_condition = {
            market.condition_id: market.to_market_info() for market in live_ready_markets
        }

        self._dispatcher, self._dispatch_boundary = self._build_dispatcher()
        self._strategy = HybridArbMaker(
            dispatcher=self._dispatcher,
            market_catalog=self._market_by_condition,
            amm_model="CPMM",
            order_size_shares=self._args.order_size_shares,
            capital_cap_usd=self._args.capital_cap_usd,
            max_trade_size_usd=self._args.max_trade_size_usd,
            min_trade_notional_usd=self._args.min_trade_notional_usd,
            gas_and_fee_buffer_cents=self._args.gas_and_fee_buffer_cents,
            cooldown_ms=self._args.cooldown_ms,
            signal_source="MANUAL",
            reserve_provider=self._reserve_provider,
        )

        self._yes_books_by_asset = {}
        self._condition_id_by_yes_asset = {}
        for market in live_ready_markets:
            self._condition_id_by_yes_asset[market.yes_token_id] = market.condition_id
            self._yes_books_by_asset[market.yes_token_id] = L2OrderBook(
                market.yes_token_id,
                on_bbo_change=self._on_yes_bbo_change,
            )

        self._l2_ws = L2WebSocket(self._yes_books_by_asset)
        self._install_signal_handlers()

        if self._dispatch_boundary is not None and self._dispatch_boundary.wallet_balance_provider is not None:
            self._prime_wallet_balance(self._dispatch_boundary)
            self._create_task(
                self._dispatch_boundary.wallet_balance_provider.poll_balance_loop(self._args.wallet_poll_interval_ms),
                name="hybrid_arb_wallet_balance",
            )

        self._create_task(self._l2_ws.start(), name="hybrid_arb_l2_ws")
        self._create_task(self._tick_loop(), name="hybrid_arb_tick_loop")
        if self._args.auto_sigint_after_s > 0:
            self._create_task(self._auto_sigint_after_delay(), name="hybrid_arb_auto_sigint")

        log.info(
            "hybrid_arb_runtime_started",
            markets=len(live_ready_markets),
            yes_assets=len(self._yes_books_by_asset),
            inactive_markets=len(self._inactive_condition_ids),
            tick_interval_ms=self._args.tick_interval_ms,
            dispatch_mode=self._args.dispatch_mode,
            env=self._args.env,
            session_id=self._session_id,
        )

        try:
            await self._stop_event.wait()
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self._stop_event.is_set() is False:
            self._stop_event.set()

        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

        if self._l2_ws is not None:
            await self._l2_ws.stop()
            self._l2_ws = None

        if self._dispatch_boundary is not None:
            await self._dispatch_boundary.close()
            self._dispatch_boundary = None

        if self._alchemy_client is not None:
            self._alchemy_client.close()
            self._alchemy_client = None

        log.info(
            "hybrid_arb_runtime_stopped",
            reserve_fetch_successes=self._reserve_fetch_successes,
            reserve_fetch_failures=self._reserve_fetch_failures,
            last_reserve_error=self._last_reserve_error or None,
        )

    def request_stop(self) -> None:
        if not self._stop_event.is_set():
            self._stop_event.set()

    def _resolve_markets(self, condition_ids: Sequence[str]) -> list[ResolvedHybridMarket]:
        assert self._alchemy_client is not None
        resolved: list[ResolvedHybridMarket] = []
        for condition_id in condition_ids:
            try:
                metadata = self._alchemy_client.get_market_metadata(condition_id)
            except LookupError as exc:
                log.warning(
                    "hybrid_arb_condition_unresolved",
                    condition_id=condition_id,
                    error=str(exc),
                )
                continue
            resolved.append(
                ResolvedHybridMarket(
                    condition_id=metadata.condition_id,
                    question=metadata.question,
                    yes_token_id=metadata.yes_token_id,
                    no_token_id=metadata.no_token_id,
                    metadata_market_id=metadata.market_id,
                    market_maker_address=metadata.market_maker_address,
                )
            )
        if not resolved:
            raise RuntimeError("No hybrid arb markets could be resolved from the configured universe")
        return resolved

    def _filter_live_ready_markets(
        self,
        markets: Sequence[ResolvedHybridMarket],
    ) -> list[ResolvedHybridMarket]:
        assert self._alchemy_client is not None
        live_ready_markets: list[ResolvedHybridMarket] = []
        for market in markets:
            try:
                reserves = self._alchemy_client.get_pool_reserves(market.condition_id)
            except Exception as exc:
                self._inactive_condition_ids.add(market.condition_id)
                log.warning(
                    "hybrid_arb_market_preflight_failed",
                    condition_id=market.condition_id,
                    market_id=market.metadata_market_id,
                    market_maker_address=market.market_maker_address,
                    error=str(exc),
                )
                continue
            if reserves.yes_reserve <= 0 and reserves.no_reserve <= 0:
                self._inactive_condition_ids.add(market.condition_id)
                log.warning(
                    "hybrid_arb_market_inactive",
                    condition_id=market.condition_id,
                    market_id=market.metadata_market_id,
                    market_maker_address=market.market_maker_address,
                    yes_reserve=str(reserves.yes_reserve),
                    no_reserve=str(reserves.no_reserve),
                    reason="zero_zero_initial_reserves",
                )
                continue
            self._preflight_reserve_cache[market.condition_id] = {
                "yes_reserve": reserves.yes_reserve,
                "no_reserve": reserves.no_reserve,
            }
            live_ready_markets.append(market)

        log.info(
            "hybrid_arb_preflight_complete",
            metadata_ready_markets=len(markets),
            live_ready_markets=len(live_ready_markets),
            inactive_markets=len(self._inactive_condition_ids),
        )
        return live_ready_markets

    def _reserve_provider(self, condition_id: str) -> dict[str, Decimal]:
        assert self._alchemy_client is not None
        cached = self._preflight_reserve_cache.pop(condition_id, None)
        if cached is not None:
            return cached
        if condition_id in self._inactive_condition_ids:
            raise RuntimeError(f"inactive hybrid arb market: {condition_id}")
        try:
            reserves = self._alchemy_client.get_pool_reserves(condition_id)
        except Exception as exc:
            self._reserve_fetch_failures += 1
            self._last_reserve_error = f"{type(exc).__name__}: {exc}"
            raise
        if reserves.yes_reserve <= 0 and reserves.no_reserve <= 0:
            self._inactive_condition_ids.add(condition_id)
            self._reserve_fetch_failures += 1
            self._last_reserve_error = (
                f"ValueError: non-positive pool reserves for {condition_id}: "
                f"yes={reserves.yes_reserve} no={reserves.no_reserve}"
            )
            self._deactivate_market(condition_id, reason="zero_zero_runtime_reserves")
            raise ValueError(
                f"non-positive pool reserves for {condition_id}: yes={reserves.yes_reserve} no={reserves.no_reserve}"
            )
        self._reserve_fetch_successes += 1
        return {
            "yes_reserve": reserves.yes_reserve,
            "no_reserve": reserves.no_reserve,
        }

    async def _auto_sigint_after_delay(self) -> None:
        await asyncio.sleep(self._args.auto_sigint_after_s)
        log.info("hybrid_arb_auto_sigint_requested", after_seconds=self._args.auto_sigint_after_s)
        signal.raise_signal(signal.SIGINT)

    def _build_dispatcher(self) -> tuple[PriorityDispatcher, LiveExecutionBoundary | None]:
        router = MevExecutionRouter(self._mev_snapshot_provider)
        if self._args.dispatch_mode != "live":
            dispatcher = PriorityDispatcher(
                router=router,
                mode=self._args.dispatch_mode,
                guard_enabled=False,
            )
            return dispatcher, None

        boundary = build_live_execution_boundary(
            deployment_phase="LIVE",
            session_id=self._session_id,
            market_by_condition=self._market_by_condition,
            now_ms=lambda: int(time.time() * 1000),
            clob_client=_build_clob_client(),
        )
        if boundary.venue_adapter is None or boundary.wallet_balance_provider is None:
            raise RuntimeError("Live hybrid arb boundary did not initialize venue adapter and wallet balance provider")

        dispatcher = PriorityDispatcher(
            router=router,
            mode="live",
            guard_enabled=False,
            venue_adapter=boundary.venue_adapter,
            client_order_id_generator=ClientOrderIdGenerator("MANUAL", self._session_id),
            wallet_balance_provider=boundary.wallet_balance_provider,
        )
        return dispatcher, boundary

    def _prime_wallet_balance(self, boundary: LiveExecutionBoundary) -> None:
        venue_adapter = boundary.venue_adapter
        provider = boundary.wallet_balance_provider
        if venue_adapter is None or provider is None:
            return
        balance = venue_adapter.get_wallet_balance("USDC")
        provider._set_cached_balance("USDC", balance, int(time.time() * 1000))

    def _mev_snapshot_provider(self, market_id: str) -> MevMarketSnapshot:
        market = self._market_by_condition.get(str(market_id).strip())
        if market is None:
            return MevMarketSnapshot(yes_bid=0.45, yes_ask=0.55, no_bid=0.45, no_ask=0.55)

        yes_book = self._yes_books_by_asset.get(market.yes_token_id)
        if yes_book is None:
            return MevMarketSnapshot(yes_bid=0.45, yes_ask=0.55, no_bid=0.45, no_ask=0.55)

        yes_bid = float(yes_book.best_bid)
        yes_ask = float(yes_book.best_ask)
        if yes_bid <= 0.0 or yes_ask <= 0.0:
            return MevMarketSnapshot(yes_bid=0.45, yes_ask=0.55, no_bid=0.45, no_ask=0.55)

        return MevMarketSnapshot(
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=_complementary_price(yes_ask),
            no_ask=_complementary_price(yes_bid),
        )

    async def _on_yes_bbo_change(self, asset_id: str, _score: Any) -> None:
        strategy = self._strategy
        if strategy is None:
            return

        condition_id = self._condition_id_by_yes_asset.get(asset_id)
        book = self._yes_books_by_asset.get(asset_id)
        if condition_id is None or book is None:
            return
        if condition_id in self._inactive_condition_ids:
            return

        top_bids = _top_levels(book, "bid", depth=1)
        top_asks = _top_levels(book, "ask", depth=1)
        if not top_bids or not top_asks:
            return

        with self._strategy_lock:
            strategy.on_bbo_update(condition_id, top_bids, top_asks)

    async def _tick_loop(self) -> None:
        strategy = self._strategy
        if strategy is None:
            return
        sleep_seconds = self._args.tick_interval_ms / 1000.0
        while not self._stop_event.is_set():
            await asyncio.to_thread(self._run_strategy_tick)
            await asyncio.sleep(sleep_seconds)

    def _run_strategy_tick(self) -> None:
        strategy = self._strategy
        if strategy is None:
            return
        with self._strategy_lock:
            strategy.on_tick()

    def _deactivate_market(self, condition_id: str, *, reason: str) -> None:
        market = self._resolved_market_by_condition.pop(condition_id, None)
        self._market_by_condition.pop(condition_id, None)
        if market is None:
            return

        self._preflight_reserve_cache.pop(condition_id, None)
        self._market_by_yes_asset.pop(market.yes_token_id, None)
        self._condition_id_by_yes_asset.pop(market.yes_token_id, None)
        self._yes_books_by_asset.pop(market.yes_token_id, None)
        if self._strategy is not None:
            self._strategy._market_state.pop(condition_id, None)

        log.warning(
            "hybrid_arb_market_deactivated",
            condition_id=condition_id,
            market_id=market.metadata_market_id,
            market_maker_address=market.market_maker_address,
            reason=reason,
        )

        if self._l2_ws is not None:
            self._create_task(
                self._l2_ws.remove_assets([market.yes_token_id]),
                name=f"hybrid_arb_remove_{market.yes_token_id}",
            )
        if not self._yes_books_by_asset:
            self.request_stop()

    def _create_task(self, coro: Any, *, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        task.add_done_callback(self._make_task_done_callback(name))
        self._tasks.append(task)
        return task

    def _make_task_done_callback(self, name: str):
        def _callback(task: asyncio.Task[Any]) -> None:
            if task in self._tasks:
                self._tasks.remove(task)
            if task.cancelled():
                return
            exc = task.exception()
            if exc is None:
                return
            log.error("hybrid_arb_task_failed", task_name=name, error=repr(exc), exc_info=exc)
            self.request_stop()

        return _callback

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for signame in ("SIGINT", "SIGTERM"):
            signum = getattr(signal, signame, None)
            if signum is None:
                continue
            try:
                loop.add_signal_handler(signum, self.request_stop)
            except NotImplementedError:
                signal.signal(signum, lambda _signum, _frame: self.request_stop())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the Hybrid Venue Dislocation Arbitrage runtime against a condition-id universe.",
    )
    parser.add_argument(
        "--env",
        default="PAPER",
        choices=("PAPER", "LIVE"),
        help="Compatibility launch environment selector. PAPER maps to paper dispatch mode; LIVE maps to live dispatch mode.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the hybrid arb universe JSON file.",
    )
    parser.add_argument(
        "--dispatch-mode",
        choices=("paper", "dry_run", "live"),
        default="dry_run",
        help="PriorityDispatcher mode. Default is dry_run; use live only when credentials are ready.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Optional explicit dispatcher session id. Defaults to a UTC timestamped hybrid-arb session id.",
    )
    parser.add_argument(
        "--capital-cap-usd",
        type=Decimal,
        default=Decimal("5000"),
        help="Global capital cap used by HybridArbMaker safe sizing.",
    )
    parser.add_argument(
        "--max-trade-size-usd",
        type=Decimal,
        default=Decimal("50"),
        help="Per-trade hard notional cap.",
    )
    parser.add_argument(
        "--min-trade-notional-usd",
        type=Decimal,
        default=Decimal("10"),
        help="Dust filter for emitted intents.",
    )
    parser.add_argument(
        "--order-size-shares",
        type=Decimal,
        default=Decimal("10"),
        help="Reference order size used for AMM dislocation pricing.",
    )
    parser.add_argument(
        "--gas-and-fee-buffer-cents",
        type=Decimal,
        default=Decimal("1.5"),
        help="Net spread buffer required before signaling an arbitrage.",
    )
    parser.add_argument(
        "--cooldown-ms",
        type=int,
        default=1000,
        help="Per-market signal cooldown in milliseconds.",
    )
    parser.add_argument(
        "--tick-interval-ms",
        type=int,
        default=250,
        help="Periodic HybridArbMaker on_tick cadence in milliseconds.",
    )
    parser.add_argument(
        "--wallet-poll-interval-ms",
        type=int,
        default=1000,
        help="Live wallet-balance polling interval in milliseconds.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Structured log output directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--auto-sigint-after-s",
        type=float,
        default=float(os.getenv("AUTO_SIGINT_AFTER_S", "0") or "0"),
        help="Raise SIGINT after the given number of seconds for bounded smoke tests.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    runtime = HybridArbRuntime(args)
    await runtime.start()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.env == "PAPER" and args.dispatch_mode == "dry_run":
        args.dispatch_mode = "paper"
    if args.env == "LIVE" and args.dispatch_mode == "dry_run":
        args.dispatch_mode = "live"
    setup_logging(
        log_dir=args.log_dir,
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        log_file="hybrid_arb_launch.jsonl",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        log.warning("hybrid_arb_keyboard_interrupt")
        return 130
    except Exception as exc:
        log.error("hybrid_arb_launch_failed", error=str(exc), exc_info=exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())