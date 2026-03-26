from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from src.core.config import settings
from src.data.market_discovery import MarketInfo
from src.execution.client_order_id import ClientOrderIdGenerator
from src.execution.clob_signer import ClobSigner
from src.execution.clob_transport import AiohttpClobTransport
from src.execution.live_wallet_balance import LiveWalletBalanceProvider
from src.execution.nonce_manager import ClobNonceManager
from src.execution.ofi_exit_router import OfiExitRouter
from src.execution.polymarket_clob_adapter import PolymarketClobAdapter
from src.execution.polymarket_clob_translator import ClobPayloadBuilder, ClobReceiptParser
from src.execution.venue_adapter_interface import VenueAdapter


class _BoundaryClobClientAdapter:
    def __init__(self, clob_client: Any, owner_id: str, market_by_condition: Mapping[str, MarketInfo]) -> None:
        self._clob_client = clob_client
        self.owner_id = owner_id
        self._market_by_condition = dict(market_by_condition)

    def resolve_market_token(self, payload: Mapping[str, str]) -> dict[str, str]:
        condition_id = str(payload.get("conditionId", "") or "").strip()
        outcome = str(payload.get("outcome", "") or "").strip().upper()
        market = self._market_by_condition.get(condition_id)
        if market is None:
            raise ValueError(f"Unknown condition_id: {condition_id!r}")
        if outcome == "YES":
            return {"token_id": market.yes_token_id}
        if outcome == "NO":
            return {"token_id": market.no_token_id}
        raise ValueError(f"Unsupported outcome: {outcome!r}")


class _AsyncTransportLoopRunner:
    def __init__(self, *, thread_name: str = "clob-transport-loop") -> None:
        self._thread_name = thread_name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def run(self, coro: Any) -> Any:
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def shutdown(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
            self._ready.clear()
        if loop is None or thread is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None:
                return self._loop
            self._thread = threading.Thread(target=self._run_loop, name=self._thread_name, daemon=True)
            self._thread.start()
        self._ready.wait(timeout=5)
        if self._loop is None:
            raise RuntimeError("failed to start async transport loop")
        return self._loop

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with self._lock:
            self._loop = loop
            self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


@dataclass(slots=True)
class LiveExecutionBoundary:
    venue_adapter: VenueAdapter | None
    wallet_balance_provider: LiveWalletBalanceProvider | None
    ofi_exit_router: OfiExitRouter | None
    transport: AiohttpClobTransport | None = None
    _transport_runner: _AsyncTransportLoopRunner | None = None

    async def close(self) -> None:
        if self.transport is not None and self._transport_runner is not None:
            await asyncio.to_thread(self._transport_runner.run, self.transport.close())
        if self._transport_runner is not None:
            await asyncio.to_thread(self._transport_runner.shutdown)


def build_live_execution_boundary(
    *,
    deployment_phase: str,
    session_id: str,
    market_by_condition: Mapping[str, MarketInfo],
    now_ms: Callable[[], int],
    clob_client: Any | None,
) -> LiveExecutionBoundary:
    phase = str(deployment_phase or "").strip().upper()
    if phase != "LIVE":
        return LiveExecutionBoundary(
            venue_adapter=None,
            wallet_balance_provider=None,
            ofi_exit_router=None,
        )

    if clob_client is None:
        raise RuntimeError("live execution boundary requires an initialized CLOB client")

    owner_id = str(settings.polymarket_api_key or "").strip()
    if not owner_id:
        raise RuntimeError("POLYMARKET_API_KEY is required for live orchestrator startup")

    from py_clob_client.config import get_contract_config

    contract_config = get_contract_config(137)
    transport_runner = _AsyncTransportLoopRunner()
    transport = AiohttpClobTransport(
        base_url=settings.clob_http_url,
        now_ms=now_ms,
    )
    venue_adapter = PolymarketClobAdapter(
        client=_BoundaryClobClientAdapter(clob_client, owner_id, market_by_condition),
        transport=transport,
        payload_builder=ClobPayloadBuilder(),
        receipt_parser=ClobReceiptParser(),
        nonce_manager=ClobNonceManager(),
        signer=ClobSigner(
            private_key=settings.eoa_private_key,
            chain_id=137,
            exchange_address=contract_config.exchange,
        ),
        transport_runner=transport_runner.run,
    )
    wallet_balance_provider = LiveWalletBalanceProvider(
        venue_adapter,
        tracked_assets=("USDC",),
    )
    ofi_exit_router = OfiExitRouter(
        venue_adapter,
        ClientOrderIdGenerator("OFI", session_id),
    )
    return LiveExecutionBoundary(
        venue_adapter=venue_adapter,
        wallet_balance_provider=wallet_balance_provider,
        ofi_exit_router=ofi_exit_router,
        transport=transport,
        _transport_runner=transport_runner,
    )