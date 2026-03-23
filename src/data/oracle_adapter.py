"""
Off-Chain Oracle Adapter — abstract base class and registry for real-world
API adapters that feed the Generalised Oracle Latency Arbitrage system (SI-8).

Each adapter polls an external data source (election feeds, sports APIs, etc.)
and emits ``OracleSnapshot`` objects into a shared ``asyncio.Queue`` consumed
by the bot's ``_oracle_polling_loop``.

Architecture
────────────
    OffChainOracleAdapter (ABC)
        ├── APElectionAdapter       — Associated Press election race-call feed
        ├── SportsAdapter           — Live sports match state (football-data.org, etc.)
        ├── OddsAPIWebSocketAdapter — Live sports WebSocket feed
        └── TreeNewsWebSocketAdapter — News/event-resolution WebSocket feed

    OracleAdapterRegistry
        Maps oracle_type strings → adapter classes for config-driven instantiation.

Design decisions
────────────────
* **Dynamic polling interval** — Each adapter computes its own interval based
  on ``event_phase``.  Defaults: 1 000 ms normal, throttles to 200 ms during
  ``"critical"`` windows (final minutes of a match, race-call imminent),
  relaxes to 30 000 ms during ``"idle"`` (halftime, polls not yet open).
  WebSocket-based adapters bypass polling entirely.

* **Per-adapter circuit breaker** — Each adapter's internal polling loop owns
  its own ``ExceptionCircuitBreaker(threshold=5, window_s=60.0)``.  On trip
  the adapter stops itself without crashing the bot; the consumer-level
  breaker in ``bot.py`` handles escalation to ``_suspend_and_reset()``.

* **Silent exception swallowing is forbidden** — All exceptions are logged
  via structlog before being recorded in the circuit breaker.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from src.core.config import settings
from src.core.exception_circuit_breaker import ExceptionCircuitBreaker
from src.core.logger import get_logger

log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Data types
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class OracleSnapshot:
    """Structured state from a single off-chain API poll.

    Fields
    ------
    adapter_name:
        Identifier of the adapter that produced this snapshot.
    market_id:
        Polymarket condition_id this snapshot maps to.
    raw_state:
        Adapter-specific raw API response data (for logging/debugging).
    resolved_outcome:
        ``"YES"`` / ``"NO"`` if the off-chain source indicates a
        definitive result, or ``None`` if the event is still uncertain.
    confidence:
        Graduated confidence ∈ [0.0, 1.0] derived from API status
        metadata.  E.g. 0.85 for a confirmed sports play, 0.97 for an
        AP race call, 0.99 for multi-desk election consensus.
    event_phase:
        Current phase of the real-world event.  Drives the adapter's
        dynamic polling interval.  One of ``"pre_event"``,
        ``"in_progress"``, ``"critical"``, ``"idle"``, ``"final"``.
    timestamp:
        ``time.monotonic()`` at the moment of the poll.
    """

    adapter_name: str
    market_id: str
    raw_state: dict = field(default_factory=dict)
    resolved_outcome: str | None = None
    confidence: float = 0.0
    event_phase: str = "pre_event"
    timestamp: float = 0.0


@dataclass
class OracleMarketConfig:
    """Per-market oracle binding parsed from ``oracle_market_configs`` JSON.

    Fields
    ------
    market_id:
        Polymarket condition_id to target.
    oracle_type:
        Adapter type key (e.g. ``"ap_election"``, ``"sports"``).
    oracle_params:
        Legacy adapter-specific nested parameters (deprecated).
    external_id:
        Flat-schema external event identifier (e.g. match/race id).
    target_outcome:
        Flat-schema target outcome label mapped to YES.
    market_type:
        Flat-schema market shape (e.g. ``"winner"`` or ``"over_goals"``).
    goal_line:
        Flat-schema total-goals threshold for over/under style markets.
    yes_asset_id, no_asset_id:
        Token IDs for the YES/NO outcome tokens.
    event_id:
        Polymarket event grouping ID.
    """

    market_id: str = ""
    oracle_type: str = ""
    oracle_params: dict = field(default_factory=dict)
    external_id: str = ""
    target_outcome: str = ""
    market_type: str = "winner"
    goal_line: float = 2.5
    yes_asset_id: str = ""
    no_asset_id: str = ""
    event_id: str = ""


# ═══════════════════════════════════════════════════════════════════════════
#  Abstract base class
# ═══════════════════════════════════════════════════════════════════════════


class OffChainOracleAdapter(ABC):
    """Abstract base for off-chain real-world API adapters.

    Subclasses implement :meth:`poll` to fetch structured state from an
    external API, and optionally override :meth:`_compute_interval` to
    customise the dynamic polling cadence per event phase.

    Parameters
    ----------
    market_config:
        The oracle market binding for this adapter instance.
    on_trip:
        Async callback invoked when the adapter's circuit breaker trips.
        Typically wired to log a critical alert.  Must never raise.
    """

    def __init__(
        self,
        market_config: OracleMarketConfig,
        *,
        on_trip: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._config = market_config
        self._on_trip = on_trip
        self._breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)
        self._running = False
        self._last_phase: str = "pre_event"

    # ── Abstract interface ─────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique adapter type identifier (e.g. ``"ap_election"``)."""
        ...

    @abstractmethod
    async def poll(self) -> OracleSnapshot:
        """Fetch current state from the off-chain API.

        Must return an ``OracleSnapshot`` populated with the latest data.
        Implementations should use ``aiohttp`` with a reasonable timeout.
        """
        ...

    # ── Dynamic polling interval ───────────────────────────────────────

    @property
    def current_polling_interval_ms(self) -> int:
        """Dynamic polling interval based on event phase.

        Subclasses may override :meth:`_compute_interval` for custom
        phase → interval mappings.
        """
        return self._compute_interval(self._last_phase)

    def _compute_interval(self, phase: str) -> int:
        """Map event phase to polling interval in milliseconds.

        Default implementation uses the SI-8 config values.  Subclasses
        override this for adapter-specific cadence schedules.
        """
        strat = settings.strategy
        _PHASE_MAP = {
            "critical": strat.oracle_critical_poll_ms,
            "idle": strat.oracle_idle_poll_ms,
        }
        return _PHASE_MAP.get(phase, strat.oracle_default_poll_ms)

    # ── Polling loop ───────────────────────────────────────────────────

    async def start(self, queue: asyncio.Queue) -> None:
        """Run the polling loop, pushing snapshots into *queue*.

        Each iteration:
        1. Calls :meth:`poll` to fetch the latest API state.
        2. Pushes the resulting ``OracleSnapshot`` into the shared queue.
        3. Sleeps for :attr:`current_polling_interval_ms`.

        On exception: logs via structlog, records in the circuit breaker.
        If the breaker trips, calls ``on_trip`` and exits.
        """
        self._running = True
        log.info(
            "oracle_adapter_started",
            adapter=self.name,
            market_id=self._config.market_id,
        )

        while self._running:
            try:
                snapshot = await self.poll()
                snapshot.timestamp = time.monotonic()
                self._last_phase = snapshot.event_phase
                queue.put_nowait(snapshot)

            except asyncio.CancelledError:
                raise  # never swallow cancellation

            except Exception:
                log.error(
                    "oracle_adapter_poll_error",
                    adapter=self.name,
                    market_id=self._config.market_id,
                    exc_info=True,
                )
                if self._breaker.record():
                    log.critical(
                        "oracle_adapter_breaker_tripped",
                        adapter=self.name,
                        market_id=self._config.market_id,
                        errors_in_window=self._breaker.recent_errors,
                    )
                    if self._on_trip is not None:
                        try:
                            await self._on_trip()
                        except Exception:
                            log.error("oracle_adapter_on_trip_error", exc_info=True)
                    self._running = False
                    return

            interval_s = self.current_polling_interval_ms / 1000.0
            await asyncio.sleep(interval_s)

        log.info(
            "oracle_adapter_stopped",
            adapter=self.name,
            market_id=self._config.market_id,
        )

    def stop(self) -> None:
        """Signal the polling loop to exit on next iteration."""
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════
#  Adapter registry
# ═══════════════════════════════════════════════════════════════════════════


class OracleAdapterRegistry:
    """Maps ``oracle_type`` strings to adapter classes.

    Usage::

        registry = OracleAdapterRegistry()
        registry.register("ap_election", APElectionAdapter)
        adapter = registry.create("ap_election", market_config, on_trip=callback)
    """

    def __init__(self) -> None:
        self._adapters: dict[str, type[OffChainOracleAdapter]] = {}
        self._register_builtin_adapters()

    def _register_builtin_adapters(self) -> None:
        from src.data.adapters.ap_election_adapter import APElectionAdapter
        from src.data.adapters.odds_api_websocket_adapter import OddsAPIWebSocketAdapter
        from src.data.adapters.sports_adapter import SportsAdapter
        from src.data.adapters.tree_news_websocket_adapter import TreeNewsWebSocketAdapter

        self.register("ap_election", APElectionAdapter)
        self.register("sports", SportsAdapter)
        self.register("odds_api_ws", OddsAPIWebSocketAdapter)
        self.register("tree_news_ws", TreeNewsWebSocketAdapter)

    def register(self, oracle_type: str, cls: type[OffChainOracleAdapter]) -> None:
        self._adapters[oracle_type] = cls

    def create(
        self,
        oracle_type: str,
        market_config: OracleMarketConfig,
        *,
        on_trip: Callable[[], Awaitable[None]] | None = None,
    ) -> OffChainOracleAdapter:
        """Instantiate an adapter by type key.

        Raises ``KeyError`` if the oracle_type is not registered.
        """
        resolved_type = market_config.oracle_type or oracle_type
        if resolved_type == "crypto":
            from src.data.adapters.binance_adapter import BinanceWebSocketAdapter

            return BinanceWebSocketAdapter(market_config, on_trip=on_trip)

        cls = self._adapters[resolved_type]
        return cls(market_config, on_trip=on_trip)

    @property
    def registered_types(self) -> list[str]:
        return list(self._adapters.keys())
