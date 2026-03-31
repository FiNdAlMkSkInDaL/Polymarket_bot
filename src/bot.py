"""
Main bot orchestrator — wires the data pipeline, signal detection,
order execution, and monitoring into a single async event loop.

V2 — Institutional-grade lifecycle:
  - Three-tier market lifecycle (observing → active → draining)
  - Live orderbook tracking (spread, depth, mid-price)
  - Composite market scoring with dynamic re-evaluation
  - Per-market/event position limits, daily loss circuit breaker,
    trailing stop-loss, signal cooldown
  - Automatic resolution detection and market eviction
  - Memory cleanup for stale positions, orders, aggregators
"""

from __future__ import annotations

import asyncio
from collections import deque
from decimal import Decimal
import os
import queue as _queue_mod
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any

from src.core.config import settings, DeploymentEnv
from src.core.exception_circuit_breaker import ExceptionCircuitBreaker
from src.core.guard import DeploymentGuard
from src.core.heartbeat import BookHeartbeat, PolygonHeadLagChecker
from src.core.latency_guard import LatencyGuard, LatencyState
from src.core.logger import get_logger, setup_logging
from src.core.process_manager import ProcessManager
from src.data.market_discovery import MarketInfo, fetch_active_markets
from src.data.market_lifecycle import MarketLifecycleManager
from src.data.market_scorer import compute_score
from src.data.ohlcv import OHLCVAggregator
from src.data.orderbook import L2OrderBookAdapter, OrderbookTracker, SharedBookReaderAdapter
from src.data.websocket_client import MarketWebSocket, MarketWebSocketPool, TradeEvent
from src.data.l2_book import BookState, L2OrderBook
from src.data.l2_websocket import L2WebSocket
from src.monitoring.telegram import TelegramAlerter
from src.monitoring.telemetry import SyncGateTelemetry
from src.monitoring.trade_store import TradeStore
from src.signals.adverse_selection_guard import AdverseSelectionGuard
from src.signals.edge_filter import compute_edge_score
from src.signals.ofi_momentum import OFIMomentumDetector, OFIMomentumSignal
from src.signals.panic_detector import PanicDetector, PanicSignal
from src.signals.resolution_probability import (
    CryptoPriceModel,
    GenericBayesianModel,
    RPECalibrationTracker,
    ResolutionProbabilityEngine,
)
from src.signals.signal_framework import (
    BaseSignal,
    CompositeSignalEvaluator,
    MetaStrategyController,
    OrderbookImbalanceSignal,
    SignalResult,
    SpreadCompressionSignal,
    VacuumSignal,
)
from src.signals.whale_monitor import WhaleMonitor
from src.trading.chaser import ChaserState, OrderChaser
from src.trading.ensemble_risk import EnsembleRiskManager
from src.trading.executor import OrderExecutor, OrderSide, OrderStatus, OrderStatusPoller
from src.trading.fee_cache import FeeCache
from src.trading.fees import compute_adaptive_stop_loss_cents, compute_net_pnl_cents, get_fee_rate
from src.trading.position_manager import ComboPosition, ComboState, PositionManager, PositionState, ShadowExecutionTracker
from src.trading.portfolio_correlation import PortfolioCorrelationEngine
from src.trading.stop_loss import StopLossMonitor
from src.trading.stealth_executor import StealthExecutor
from src.trading.take_profit import compute_dynamic_spread, compute_take_profit
from src.signals.regime_detector import RegimeDetector
from src.signals.iceberg_detector import IcebergDetector
from src.signals.contagion_arb import ContagionArbDetector, ContagionArbSignal
from src.signals.cross_market import CrossMarketSignal, CrossMarketSignalGenerator
from src.signals.combinatorial_arb import ComboArbDetector, ComboArbSignal, ComboSizer
from src.signals.bayesian_arb import BayesianArbDetector, BayesianArbRelationshipManager
from src.data.arb_clusters import ArbCluster, ArbitrageClusterManager
from src.signals.drift_signal import DriftSignal, MeanReversionDrift
from src.signals.oracle_signal import OracleSignalEngine
from src.data.oracle_adapter import (
    OracleAdapterRegistry,
    OracleMarketConfig,
    OracleSnapshot,
    OffChainOracleAdapter,
)
from src.data.adapters.ap_election_adapter import APElectionAdapter
from src.data.adapters.sports_adapter import SportsAdapter
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.live_execution_boundary import LiveExecutionBoundary, build_live_execution_boundary
from src.execution.live_orchestrator_config import LiveOrchestratorConfig
from src.execution.mev_serializer import deserialize_envelope
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.ofi_signal_bridge import OfiEntrySignal
from src.execution.priority_context import PriorityOrderContext
from src.execution.orchestrator_factory import build_live_orchestrator
from src.execution.orchestrator_health_monitor import HealthMonitorConfig, OrchestratorHealthMonitor
from src.execution.si9_unwind_manifest import Si9UnwindConfig
from src.execution.ofi_local_exit_monitor import OfiExitDecision
from src.rewards.reward_poster_sidecar import RewardPosterSidecar
from src.rewards.reward_selector import RewardSelector
from src.trading.adverse_selection_monitor import AdverseSelectionMonitor, make_fill_record
from src.strategies.pure_market_maker import PureMarketMaker

# Data recording (lazy import — only used when RECORD_DATA=true)
try:
    from src.backtest.data_recorder import MarketDataRecorder
except ImportError:
    MarketDataRecorder = None  # type: ignore[misc,assignment]

log = get_logger(__name__)


_SI9_HARD_EXCLUDE_FEEDBACK_REASONS = {
    "infeasible_size",
    "thin_bid_depth",
}
_SI9_STRUCTURAL_GUARDRAIL_DISTANCE = 0.05
_SI9_OVERSIZED_EVENT_CAPACITY_SHARE = 0.5
_CROSS_MARKET_SHADOW_SOURCE = "SI-3_CrossMarket"
_OFI_REVERSE_SHADOW_SOURCE = "OFI_REVERSE_SHADOW"
_PANIC_SHADOW_SOURCE = "PANIC_STRICT_SHADOW"


def _decimal_from_number(value: Any) -> Decimal:
    return Decimal(str(value))


def _contagion_dispatch_context(
    *,
    market_id: str,
    side: str,
    target_price: Decimal,
    anchor_volume: Decimal,
    max_capital: Decimal,
    conviction_scalar: Decimal,
) -> PriorityOrderContext:
    return PriorityOrderContext(
        market_id=market_id,
        side=side,
        signal_source="CONTAGION",
        conviction_scalar=conviction_scalar,
        target_price=target_price,
        anchor_volume=anchor_volume,
        max_capital=max_capital,
    )


def _safe_task_done_callback(task: asyncio.Task) -> None:
    """Callback attached to fire-and-forget tasks to log unhandled exceptions.

    Without this, exceptions in tasks created via ``asyncio.ensure_future``
    or ``asyncio.create_task`` are silently dropped until the task is GC'd,
    producing the infamous *"Task exception was never retrieved"* message.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.error(
            "unhandled_task_exception",
            task_name=task.get_name(),
            error=repr(exc),
            exc_info=exc,
        )


def _safe_fire_and_forget(coro, *, name: str | None = None) -> asyncio.Task:
    """Schedule a coroutine as a fire-and-forget task with error logging.

    Replaces bare ``asyncio.ensure_future`` calls throughout the bot to
    ensure that exceptions are always captured and logged.
    """
    task = asyncio.ensure_future(coro)
    if name:
        task.set_name(name)
    task.add_done_callback(_safe_task_done_callback)
    return task


class TradingBot:
    """Top-level orchestrator for the live quoting and execution stack."""

    def __init__(
        self,
        paper_mode: bool | None = None,
        *,
        deployment_env: DeploymentEnv | None = None,
        session_id: str | None = None,
        confirmed_production: bool = False,
    ):
        # Deployment env is the canonical source of truth.
        # Legacy paper_mode kwarg is supported for backward compatibility.
        if deployment_env is not None:
            self.deployment_env = deployment_env
        elif paper_mode is not None:
            self.deployment_env = (
                DeploymentEnv.PAPER if paper_mode else DeploymentEnv.PRODUCTION
            )
        else:
            self.deployment_env = settings.deployment_env

        self.guard = DeploymentGuard(
            self.deployment_env,
            confirmed_production=confirmed_production,
        )
        injected_session_id = str(session_id or os.getenv("POLYBOT_SESSION_ID", "") or "").strip()
        if not injected_session_id and self.deployment_env == DeploymentEnv.PAPER:
            injected_session_id = "paper-session"
        self._session_id = injected_session_id
        self.paper_mode = self.guard.is_paper

        # Components (initialised in start())
        self.executor = OrderExecutor(paper_mode=self.paper_mode)
        self.trade_store = TradeStore()
        self._shadow_tracker = ShadowExecutionTracker(self.trade_store)

        # Portfolio Correlation Engine (Pillar 15)
        self.pce = PortfolioCorrelationEngine(
            data_dir=settings.record_data_dir,
        )
        self.pce.load_state()

        # Per-market state — initialise book_trackers dict early so the
        # PositionManager (which receives a reference to it for paper-mode
        # fill simulation) sees subsequent additions.
        self._book_trackers: dict[str, OrderbookTracker] = {}  # asset_id → tracker

        # SI-2: Per-asset iceberg detectors (initialised before PositionManager
        # so the dict reference is shared — new detectors added later are visible).
        self._iceberg_detectors: dict[str, IcebergDetector] = {}  # asset_id → detector
        self.ensemble_risk = EnsembleRiskManager()

        self.positions = PositionManager(
            self.executor, trade_store=self.trade_store, guard=self.guard,
            pce=self.pce,
            book_trackers=self._book_trackers,
            iceberg_detectors=self._iceberg_detectors,
            ensemble_risk=self.ensemble_risk,
        )
        self.whale_monitor = WhaleMonitor(zscore_fn=self._latest_zscore)
        self.telegram = TelegramAlerter()
        self.sync_telemetry = SyncGateTelemetry()
        self.executor.configure_runtime_hooks(
            telegram_alerter=self.telegram,
            on_shutdown=self._schedule_stop,
        )
        self.lifecycle = MarketLifecycleManager()
        self.latency_guard = LatencyGuard()

        # Shared fast-kill event for adverse-selection guard
        self._fast_kill_event = asyncio.Event()
        self._fast_kill_event.set()  # start SET — chasers may proceed (clear = paused)

        # Per-market state
        self._markets: list[MarketInfo] = []
        self._yes_aggs: dict[str, OHLCVAggregator] = {}  # keyed by yes_token_id
        self._no_aggs: dict[str, OHLCVAggregator] = {}   # keyed by no_token_id
        self._detectors: dict[str, PanicDetector] = {}    # keyed by condition_id
        self._market_map: dict[str, MarketInfo] = {}      # asset_id → MarketInfo
        # _book_trackers initialised above (before PositionManager)
        self._l2_books: dict[str, L2OrderBook] = {}  # asset_id → L2 book (when L2 enabled)
        self._trade_counts: dict[str, float] = {}  # asset_id → trades/min
        self._taker_counts: dict[str, int] = {}    # asset_id → taker-initiated trades
        self._total_counts: dict[str, int] = {}    # asset_id → total classified trades
        self._recent_trade_volume: dict[str, list[tuple[float, float]]] = {}  # cond_id → [(ts, size)]
        self._recent_contagion_matrix: deque[dict[str, Any]] = deque(maxlen=12)

        # Spread-based signal evaluators (Problem 3)
        self._spread_evaluators: dict[str, CompositeSignalEvaluator] = {}  # condition_id → evaluator
        self._spread_cooldowns: dict[str, float] = {}  # condition_id → last fire timestamp

        # SI-1: Per-market regime detectors
        self._regime_detectors: dict[str, RegimeDetector] = {}  # condition_id → detector

        # SI-6: Meta-strategy hybrid controller (regime-weighted master switch)
        self._meta_controller = MetaStrategyController()

        # V3: Per-market drift signal detectors
        self._drift_detectors: dict[str, MeanReversionDrift] = {}  # condition_id → detector
        self._drift_cooldowns: dict[str, float] = {}  # condition_id → last fire timestamp
        self._ofi_detectors: dict[str, OFIMomentumDetector] = {}  # condition_id → detector

        # Maker adverse-selection monitor (V1/V4 calibration)
        self._maker_monitor: AdverseSelectionMonitor | None = None

        # SI-2: _iceberg_detectors initialised above (before PositionManager)

        # SI-3: Cross-market signal generator
        self._cross_market = (
            CrossMarketSignalGenerator(self.pce)
            if self._cross_market_lane_enabled()
            else None
        )

        # SI-4: Stealth executor wrapper
        self._stealth = StealthExecutor(self.executor) if settings.strategy.stealth_enabled else None

        # Lifecycle
        self._running = False
        self._start_time: float = 0.0       # set in _run() for uptime tracking
        self._latest_z: float = 0.0        # tracks most recent panic Z-score
        self._trade_queue: asyncio.Queue[TradeEvent] = asyncio.Queue(maxsize=1000)
        self._book_queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._ws: MarketWebSocketPool | None = None
        self._l2_ws: L2WebSocket | None = None
        self._tasks: list[asyncio.Task] = []
        self._stop_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Multi-core ProcessManager — orchestrates L2 and PCE workers
        self._process_manager: ProcessManager | None = None
        self._multicore_enabled: bool = settings.strategy.l2_enabled  # gate on L2
        self._pce_queue_drops: int = 0  # rate-limited counter for PCE input queue drops

        # Singleton RPE (Pillar 14) — shared across all markets.
        # The CryptoPriceModel and GenericBayesianModel are stateless
        # per call; the per-market estimate cache inside the RPE already
        # keys by market_id, so one instance suffices.
        self._rpe: ResolutionProbabilityEngine | None = None
        if self._rpe_lane_enabled() or self._contagion_lane_enabled():
            self._rpe = ResolutionProbabilityEngine(
                models=[
                    CryptoPriceModel(
                        price_fn=lambda: self._get_crypto_spot(),
                    ),
                    GenericBayesianModel(),
                ],
            )
        self._rpe_calibration: RPECalibrationTracker | None = (
            RPECalibrationTracker() if self._rpe is not None else None
        )
        self._contagion_arb: ContagionArbDetector | None = (
            ContagionArbDetector(
                self.pce,
                self._rpe,
                on_sync_block=lambda _assessment: self.sync_telemetry.record_contagion_block(),
            )
            if self._contagion_lane_enabled() and self._rpe is not None
            else None
        )
        self._rpe_last_signal: dict[str, float] = {}
        self._rpe_last_spot: float | None = None
        self._l2_active_set: set[str] = set()
        self._warm_market_ids: set[str] = set()
        self._tracked_single_market_ids: set[str] = set()
        self._tracked_combo_market_ids: set[str] = set()
        self._tracked_combo_event_ids: tuple[str, ...] = ()
        self._si9_scan_summary: dict[str, int] = {
            "candidate_events": 0,
            "admitted_events": 0,
            "excluded_mixed_neg_risk_event": 0,
            "excluded_too_few_neg_risk_legs": 0,
        }
        self._single_name_rejection_counts: dict[str, dict[str, int]] = {}
        self._fee_category_set: set[str] = {
            part.strip().lower()
            for part in (settings.strategy.fee_enabled_categories or "").split(",")
            if part.strip()
        }
        self._fee_cache = FeeCache()

        self._trade_loop_breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)
        self._stale_bar_breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)
        self._retrigger_breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)
        self._tp_rescale_breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)
        self._combo_breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)
        self._ghost_breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)
        self._timeout_breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)
        self._oracle_breaker = ExceptionCircuitBreaker(threshold=5, window_s=60.0)

        self._cluster_mgr = ArbitrageClusterManager()
        self._combo_positions: dict[str, ComboPosition] = {}
        self._combo_ofi_alerted: set[str] = set()
        self._combo_detector: ComboArbDetector | None = None
        self._bayesian_cluster_mgr = BayesianArbRelationshipManager()
        self._bayesian_ofi_alerted: set[str] = set()
        self._bayesian_detector: BayesianArbDetector | None = None

        self._oracle_registry = OracleAdapterRegistry()
        self._oracle_signal_engine = OracleSignalEngine()
        self._oracle_adapter_tasks: list[asyncio.Task] = []
        self._oracle_last_signal: dict[str, float] = {}

        self._recorder = (
            MarketDataRecorder(settings.record_data_dir)
            if settings.record_data and MarketDataRecorder is not None
            else None
        )
        self._ws = MarketWebSocketPool(
            [],
            self._trade_queue,
            book_queue=self._book_queue,
            recorder=self._recorder,
        )
        self._l2_ws = (
            L2WebSocket(self._l2_books, recorder=self._recorder)
            if settings.strategy.l2_enabled
            else None
        )

        self._adverse_guard = AdverseSelectionGuard(
            self.executor,
            self._book_trackers,
            self._fast_kill_event,
            taker_counts=self._taker_counts,
            total_counts=self._total_counts,
            trade_counts=self._trade_counts,
            get_position_assets=self._positioned_asset_ids,
            telegram=self.telegram,
            on_shutdown=self._schedule_stop,
        )

        self._heartbeat: BookHeartbeat | None = None
        self._order_poller: OrderStatusPoller | None = None
        self._stop_loss_monitor: StopLossMonitor | None = None
        self._pure_mm: PureMarketMaker | None = None
        self._reward_sidecar: RewardPosterSidecar | None = None
        self._live_orchestrator: MultiSignalOrchestrator | None = None
        self._live_execution_boundary: LiveExecutionBoundary | None = None
        self._orchestrator_health_monitor: OrchestratorHealthMonitor | None = None
        self._orchestrator_tick_interval_ms: int = settings.strategy.heartbeat_check_ms

    def _panic_lane_enabled(self) -> bool:
        return settings.strategy.lane_is_active("panic") or settings.strategy.panic_shadow_mode

    def _ofi_lane_enabled(self) -> bool:
        return settings.strategy.lane_is_live("ofi_momentum")

    def _contagion_lane_enabled(self) -> bool:
        return settings.strategy.lane_is_active("contagion")

    def _contagion_shadow_runtime_enabled(self) -> bool:
        return settings.strategy.contagion_shadow_runtime_enabled()

    def _reward_lane_enabled(self) -> bool:
        return settings.strategy.lane_is_active("reward")

    def _contagion_live_enabled(self) -> bool:
        return settings.strategy.lane_is_live("contagion")

    def _rpe_lane_enabled(self) -> bool:
        return settings.strategy.lane_is_active("rpe")

    def _oracle_lane_enabled(self) -> bool:
        return settings.strategy.lane_is_active("oracle")

    def _pure_mm_lane_enabled(self) -> bool:
        return settings.strategy.lane_is_live("pure_market_maker")

    def _si9_lane_enabled(self) -> bool:
        return settings.strategy.lane_is_live("si9_combo")

    def _cross_market_lane_enabled(self) -> bool:
        return settings.strategy.lane_is_active("cross_market")

    def _si10_lane_enabled(self) -> bool:
        return settings.strategy.lane_is_live("si10_bayesian")

    def _tracked_market_budgets(self, l2_limit: int) -> tuple[int, int]:
        """Split tracked-market capacity between single-name and SI-9."""
        if l2_limit <= 0:
            return 0, 0

        strat = settings.strategy
        if not self._si9_lane_enabled():
            return l2_limit, 0

        single_override = int(strat.tracked_single_name_markets)
        combo_override = int(strat.si9_tracked_market_budget)

        if single_override < 0 and combo_override < 0:
            combo_budget = max(2, round(l2_limit * 0.4)) if l2_limit >= 4 else max(0, l2_limit - 1)
            return max(0, l2_limit - combo_budget), combo_budget

        if single_override < 0:
            combo_budget = max(0, min(combo_override, l2_limit))
            return max(0, l2_limit - combo_budget), combo_budget

        if combo_override < 0:
            single_budget = max(0, min(single_override, l2_limit))
            return single_budget, max(0, l2_limit - single_budget)

        single_budget = max(0, min(single_override, l2_limit))
        combo_budget = max(0, min(combo_override, l2_limit - single_budget))
        return single_budget, combo_budget

    def _select_tracked_markets(self, all_tracked: list[MarketInfo]) -> list[MarketInfo]:
        """Prefer full SI-9 events and then backfill with top single-name markets."""
        l2_limit = settings.strategy.max_active_l2_markets
        single_budget, combo_budget = self._tracked_market_budgets(l2_limit)
        tracked_capacity = max(1, single_budget + combo_budget)
        sorted_markets = sorted(
            all_tracked,
            key=lambda market: market.daily_volume_usd,
            reverse=True,
        )

        selected_combo_clusters: list[ArbCluster] = []
        selected_combo_ids: set[str] = set()
        combo_markets: list[MarketInfo] = []
        remaining_combo_budget = combo_budget
        remaining_single_budget = single_budget
        ranked_clusters: list[ArbCluster] = []
        selector_summary: dict[str, int] = {
            "candidate_events": 0,
            "admitted_events": 0,
            "excluded_mixed_neg_risk_event": 0,
            "excluded_too_few_neg_risk_legs": 0,
        }
        if combo_budget > 0 and self._si9_lane_enabled():
            event_ids = {market.event_id for market in all_tracked if market.event_id}
            self._cluster_mgr.scan_clusters(all_tracked)
            ranked_clusters = sorted(
                self._cluster_mgr.active_clusters,
                key=lambda cluster: (
                    sum(
                        float(getattr(market, "daily_volume_usd", 0.0) or 0.0)
                        + float(getattr(market, "liquidity_usd", 0.0) or 0.0)
                        for market in cluster.legs
                    ),
                    -cluster.n_legs,
                ),
                reverse=True,
            )
            selector_summary = {
                "candidate_events": len(event_ids),
                "admitted_events": len(ranked_clusters),
                "excluded_mixed_neg_risk_event": 0,
                "excluded_too_few_neg_risk_legs": max(0, len(event_ids) - len(ranked_clusters)),
            }
        self._si9_scan_summary = selector_summary

        if combo_budget > 0:
            for cluster in ranked_clusters:
                leg_count = len(cluster.legs)
                if self._exclude_low_quality_oversized_si9_cluster(
                    cluster,
                    tracked_capacity=tracked_capacity,
                ):
                    log.info(
                        "si9_selector_excluded",
                        event_id=cluster.event_id,
                        reason="oversized_low_quality_event",
                        n_legs=leg_count,
                        score=self._si9_cluster_score(cluster),
                        score_floor=settings.strategy.min_market_score,
                        capacity_share=round(leg_count / tracked_capacity, 4),
                    )
                    continue

                cluster_legs = sorted(
                    cluster.legs,
                    key=lambda market: market.daily_volume_usd,
                    reverse=True,
                )
                total_remaining_capacity = remaining_combo_budget + remaining_single_budget
                if leg_count > total_remaining_capacity:
                    log.info(
                        "si9_selector_excluded",
                        event_id=cluster.event_id,
                        reason="insufficient_total_capacity",
                        n_legs=leg_count,
                        remaining_combo_budget=remaining_combo_budget,
                        remaining_single_budget=remaining_single_budget,
                        score=self._si9_cluster_score(cluster),
                    )
                    continue

                borrowed_single_budget = 0
                if leg_count > remaining_combo_budget:
                    borrowed_single_budget = leg_count - remaining_combo_budget
                    remaining_single_budget = max(0, remaining_single_budget - borrowed_single_budget)
                    remaining_combo_budget = 0
                else:
                    remaining_combo_budget -= leg_count

                combo_markets.extend(cluster_legs)
                selected_combo_clusters.append(cluster)
                selected_combo_ids.update(market.condition_id for market in cluster_legs)
                log.info(
                    "si9_selector_selected",
                    event_id=cluster.event_id,
                    n_legs=leg_count,
                    score=self._si9_cluster_score(cluster),
                    borrowed_single_budget=borrowed_single_budget,
                    remaining_combo_budget=remaining_combo_budget,
                    remaining_single_budget=remaining_single_budget,
                )
                if remaining_combo_budget == 0:
                    if remaining_single_budget == 0:
                        break

        single_candidates = [
            market for market in sorted_markets if market.condition_id not in selected_combo_ids
        ]
        single_slots = min(len(single_candidates), remaining_single_budget + remaining_combo_budget)
        single_markets = single_candidates[:single_slots]

        self._tracked_single_market_ids = {market.condition_id for market in single_markets}
        self._tracked_combo_market_ids = {market.condition_id for market in combo_markets}
        self._tracked_combo_event_ids = tuple(
            cluster.event_id for cluster in selected_combo_clusters
        )

        log.info(
            "tracked_market_selection",
            total_selected=len(combo_markets) + len(single_markets),
            l2_limit=l2_limit,
            single_name_budget=single_budget,
            single_name_selected=len(single_markets),
            si9_market_budget=combo_budget,
            si9_selected=len(combo_markets),
            si9_events=list(self._tracked_combo_event_ids),
            si9_candidate_events=selector_summary.get("candidate_events", 0),
            si9_admitted_events=selector_summary.get("admitted_events", 0),
        )
        return (combo_markets + single_markets)[:l2_limit]

    def _si9_cluster_score(self, cluster: ArbCluster) -> float:
        return sum(
            float(getattr(market, "daily_volume_usd", 0.0) or 0.0)
            + float(getattr(market, "liquidity_usd", 0.0) or 0.0)
            for market in cluster.legs
        )

    def _warm_market_limit(self) -> int:
        return max(
            int(settings.strategy.max_active_l2_markets),
            int(settings.strategy.warm_market_observation_limit),
        )

    def _select_warm_markets(
        self,
        all_tracked: list[MarketInfo],
        tracked_markets: list[MarketInfo],
        *,
        open_market_ids: set[str] | None = None,
    ) -> list[MarketInfo]:
        warm_limit = self._warm_market_limit()
        if warm_limit <= 0:
            return list(tracked_markets)

        open_market_ids = open_market_ids or set()
        active_ids = set(self.lifecycle.active)
        required_ids = {
            *(market.condition_id for market in tracked_markets),
            *active_ids,
            *open_market_ids,
        }
        market_by_id = {market.condition_id: market for market in all_tracked}

        warm_markets: list[MarketInfo] = []
        seen: set[str] = set()

        def include(market: MarketInfo | None) -> None:
            if market is None or market.condition_id in seen:
                return
            seen.add(market.condition_id)
            warm_markets.append(market)

        for market in tracked_markets:
            include(market)

        for condition_id in required_ids:
            include(market_by_id.get(condition_id))

        ranked_candidates = sorted(
            all_tracked,
            key=lambda market: (
                market.condition_id in active_ids,
                market.condition_id in open_market_ids,
                float(getattr(market, "score", 0.0) or 0.0),
                float(market.daily_volume_usd or 0.0),
            ),
            reverse=True,
        )
        for market in ranked_candidates:
            if len(warm_markets) >= warm_limit:
                break
            include(market)

        return warm_markets

    def _exclude_low_quality_oversized_si9_cluster(
        self,
        cluster: ArbCluster,
        *,
        tracked_capacity: int,
    ) -> bool:
        configured_leg_cap = max(int(settings.strategy.si9_max_legs), 0)
        leg_count = max(cluster.n_legs, 0)
        if leg_count <= 0:
            return False
        if configured_leg_cap > 0 and leg_count <= configured_leg_cap:
            return False

        capacity_share = leg_count / max(tracked_capacity, 1)
        if capacity_share < _SI9_OVERSIZED_EVENT_CAPACITY_SHARE:
            return False

        return self._si9_cluster_score(cluster) < settings.strategy.min_market_score

    def _record_single_name_rejection(
        self,
        strategy: str,
        market: MarketInfo,
        reason: str,
        *,
        log_event: bool = True,
        **details: Any,
    ) -> None:
        strategy_key = (strategy or "unknown").lower()
        strategy_counts = self._single_name_rejection_counts.setdefault(strategy_key, {})
        strategy_counts[reason] = strategy_counts.get(reason, 0) + 1
        if log_event:
            log.info(
                "single_name_signal_rejected",
                strategy=strategy_key,
                market=market.condition_id,
                reason=reason,
                **details,
            )

    async def _refresh_markets_once(self, *, decay_counters: bool) -> None:
        """Refresh the tracked market universe once."""
        if decay_counters:
            decay_mins = settings.strategy.market_refresh_minutes
            for aid in self._trade_counts:
                self._trade_counts[aid] = self._trade_counts[aid] / max(decay_mins, 1)

            for aid in list(self._taker_counts):
                self._taker_counts[aid] = max(0, self._taker_counts[aid] // 2)
            for aid in list(self._total_counts):
                self._total_counts[aid] = max(0, self._total_counts[aid] // 2)

        open_markets = self.positions.get_open_market_ids()
        whale_tokens = self.whale_monitor.get_whale_tokens()

        prev_active = len(self.lifecycle.active)
        prev_observing = len(self.lifecycle.observing)
        prev_draining = len(self.lifecycle.draining)

        newly_added, evicted = await self.lifecycle.refresh(
            orderbook_trackers=self._book_trackers,
            trade_counts=self._trade_counts,
            whale_tokens=whale_tokens,
            open_position_markets=open_markets,
            taker_counts=self._taker_counts,
            total_counts=self._total_counts,
        )

        for cid in evicted:
            self._l2_active_set.discard(cid)
            for m in list(self._markets):
                if m.condition_id == cid:
                    if self._ws:
                        await self._ws.remove_assets([m.yes_token_id, m.no_token_id])
                    if self._l2_ws:
                        await self._l2_ws.remove_assets([m.yes_token_id, m.no_token_id])
                    self._unwire_market(m)
                    break

        all_tracked = self.lifecycle.get_all_tracked()
        tracked_markets = self._select_tracked_markets(all_tracked)
        self._l2_active_set = {market.condition_id for market in tracked_markets}
        warm_markets = self._select_warm_markets(
            all_tracked,
            tracked_markets,
            open_market_ids=open_markets,
        )
        self._warm_market_ids = {market.condition_id for market in warm_markets}
        await self._reconcile_warm_markets(warm_markets)
        if self._reward_sidecar is not None:
            self._reward_sidecar.replace_market_universe(list(self._markets), self._current_timestamp_ms())

        stale_evicted = self.lifecycle.check_stale_markets(
            yes_aggs=self._yes_aggs,
            open_position_markets=open_markets,
            stale_threshold_s=settings.strategy.stale_market_eviction_s,
        )
        for cid in stale_evicted:
            for m in list(self._markets):
                if m.condition_id == cid:
                    if self._ws:
                        await self._ws.remove_assets([m.yes_token_id, m.no_token_id])
                    if self._l2_ws:
                        await self._l2_ws.remove_assets([m.yes_token_id, m.no_token_id])
                    self._unwire_market(m)
                    break

        for cid, am in self.lifecycle.active.items():
            tid = am.info.yes_token_id
            agg = self._yes_aggs.get(tid)
            bt = self._book_trackers.get(tid)
            tpm = self._trade_counts.get(tid, 0.0)
            last_trade_age = (
                round(time.time() - agg.last_trade_time, 1)
                if agg and agg.last_trade_time > 0
                else -1
            )
            bars = len(agg.bars) if agg else 0
            spread = bt.spread_cents if bt and bt.has_data else -1
            log.info(
                "market_health",
                condition_id=cid,
                score=round(am.score.total, 1),
                trades_per_min=round(tpm, 2),
                last_trade_age_s=last_trade_age,
                bars=bars,
                spread_cents=spread,
            )

        total = len(self.lifecycle.active)
        obs = len(self.lifecycle.observing)
        drn = len(self.lifecycle.draining)
        universe = total + obs + drn
        discovered = len(newly_added)
        dropped = len(evicted) + len(stale_evicted)

        if newly_added or evicted or stale_evicted:
            log.info(
                "market_refresh_done",
                discovered=discovered,
                dropped=dropped,
                active=total,
                observing=obs,
                draining=drn,
            )
            promoted = max(0, total - prev_active)
            await self.telegram.send(
                f"🔄 <b>Market refresh</b>\n"
                f"Universe: {universe} tracked "
                f"(+{discovered} discovered, -{dropped} dropped)\n"
                f"Active: {total} (+{promoted} promoted)  |  "
                f"Observing: {obs}  |  Draining: {drn}"
            )

    async def start(self) -> None:
        """Start background tasks and live runtime services for the trading bot."""
        self._running = True
        self._start_time = time.monotonic()
        mode_label = self.deployment_env.value

        # Polygon head-lag checker (moved from adverse-selection guard)
        polygon_checker = (
            PolygonHeadLagChecker(
                rpc_url=settings.polygon_rpc_url,
                lag_threshold_ms=settings.strategy.adverse_sel_polygon_head_lag_ms,
            )
            if settings.polygon_rpc_url
            else None
        )

        # Book heartbeat (Pillar 8)
        self._heartbeat = BookHeartbeat(
            book_trackers=self._book_trackers,
            latency_guard=self.latency_guard,
            fast_kill_event=self._fast_kill_event,
            executor=self.executor,
            ws_transport=self._l2_ws,
            get_position_assets=self._positioned_asset_ids,
            telegram=self.telegram,
            polygon_checker=polygon_checker,
            process_manager=self._process_manager,
        )

        # Order status poller (Pillar 10) — live mode only
        self._order_poller = OrderStatusPoller(
            self.executor,
            on_fill=self._on_clob_fill,
        )

        self._live_execution_boundary = self._build_live_execution_boundary_runtime()
        self._live_orchestrator = self._build_live_orchestrator_runtime(self._live_execution_boundary)
        self._orchestrator_health_monitor = OrchestratorHealthMonitor(
            self._live_orchestrator,
            HealthMonitorConfig(
                max_release_failures_before_halt=max(1, self._live_orchestrator_config().max_position_release_failures),
                stale_snapshot_threshold_ms=max(self._orchestrator_tick_interval_ms * 3, 1),
                min_heartbeat_interval_ms=self._orchestrator_tick_interval_ms,
            ),
        )
        self._initialize_wallet_balance()
        self._apply_live_market_target_map()

        # Event-driven stop-loss monitor (Pillar 11) — no polling task
        # V4: on_probe_breakeven wires back to PositionManager.scale_probe_to_full
        def _probe_breakeven_cb(pos):
            return self.positions.scale_probe_to_full(
                pos,
                get_mid_price_fn=lambda asset_id: (
                    self._stop_loss_monitor._get_mid_price(asset_id)
                    if hasattr(self, "_stop_loss_monitor")
                    else 0.0
                ),
            )

        self._stop_loss_monitor = StopLossMonitor(
            position_manager=self.positions,
            no_aggs=self._no_aggs,
            book_trackers=self._book_trackers,
            trade_store=self.trade_store,
            telegram=self.telegram,
            on_probe_breakeven=_probe_breakeven_cb,
        )
        self._stop_loss_monitor.start()  # mark active, no coroutine

        # Maker adverse-selection monitor — tracks T+5/15/60 PnL for POST_ONLY fills
        async def _mid_price_for_monitor(asset_id: str) -> float | None:
            mid = self._stop_loss_monitor._get_mid_price(asset_id)
            return mid if mid > 0 else None

        def _vol_provider_for_monitor(market_id: str) -> float | None:
            """Return current EWMA vol for *market_id* from the NO aggregator."""
            agg = self._no_aggs.get(market_id)
            if agg is None:
                return None
            return agg.rolling_volatility_ewma

        self._vol_provider_for_monitor = _vol_provider_for_monitor

        self._maker_monitor = AdverseSelectionMonitor(
            mid_price_fn=_mid_price_for_monitor,
            alpha=settings.strategy.adverse_monitor_alpha_base,
            vol_provider=self._vol_provider_for_monitor,
            vol_ref=settings.strategy.adverse_monitor_vol_ref,
            alpha_gamma=settings.strategy.adverse_monitor_alpha_gamma,
            alpha_min=settings.strategy.adverse_monitor_alpha_min,
            alpha_max=settings.strategy.adverse_monitor_alpha_max,
        )
        # Wire monitor into PositionManager for exec-mode downgrade on suspension
        self.positions._maker_monitor = self._maker_monitor
        if self._reward_lane_enabled() and self._live_orchestrator is not None and self._orchestrator_health_monitor is not None:
            self._reward_sidecar = RewardPosterSidecar(
                orchestrator=self._live_orchestrator,
                selector=RewardSelector(),
                markets_provider=lambda: list(self._markets),
                market_by_asset_provider=lambda asset_id: self._market_map.get(asset_id),
                book_provider=lambda asset_id: self._book_trackers.get(asset_id),
                health_report_provider=lambda current_timestamp_ms: self._orchestrator_health_monitor.check(current_timestamp_ms),
                maker_monitor=self._maker_monitor,
                now_ms=self._current_timestamp_ms,
                shadow_persist_callback=self._persist_reward_shadow_trade,
            )
        if self._pure_mm_lane_enabled():
            self._pure_mm = PureMarketMaker(
                executor=self.executor,
                get_active_markets=lambda: [am.info for am in self.lifecycle.active.values()],
                get_l2_books=lambda: self._l2_books,
                get_book_trackers=lambda: self._book_trackers,
                get_l2_active_set=lambda: self._l2_active_set,
                latency_guard=self.latency_guard,
                fast_kill_event=self._fast_kill_event,
                maker_monitor=self._maker_monitor,
                iceberg_detectors=self._iceberg_detectors,
            )
        # Wire stealth executor into PositionManager for sliced probe scale-ups
        if self._stealth is not None:
            self.positions._stealth = self._stealth
        # Wire OHLCV aggregators for POV volume lookups in scale_probe_to_full
        self.positions._ohlcv_aggs = self._no_aggs

        await self._refresh_markets_once(decay_counters=False)

        self._tasks = [
            asyncio.create_task(self._ws.start(), name="ws"),
            asyncio.create_task(self._process_trades(), name="trade_processor"),
            asyncio.create_task(self._process_book_updates(), name="book_processor"),
            asyncio.create_task(self._timeout_loop(), name="timeout_loop"),
            asyncio.create_task(self._stats_loop(), name="stats_loop"),
            asyncio.create_task(self.whale_monitor.start(), name="whale_monitor"),
            asyncio.create_task(self._market_refresh_loop(), name="market_refresh"),
            asyncio.create_task(self._cleanup_loop(), name="cleanup"),
            asyncio.create_task(self._tp_rescale_loop(), name="tp_rescale"),
            asyncio.create_task(self._adverse_guard.start(), name="adverse_sel"),
            asyncio.create_task(self._heartbeat.run(), name="heartbeat"),
            asyncio.create_task(self._orchestrator_tick_loop(), name="orchestrator_tick"),
            asyncio.create_task(self._ghost_liquidity_loop(), name="ghost_liquidity"),
            asyncio.create_task(self._order_poller.run(), name="order_status_poller"),
            asyncio.create_task(self._health_reporter(), name="health_reporter"),
            asyncio.create_task(self._stale_bar_flush_loop(), name="stale_bar_flush"),
            asyncio.create_task(self._paper_summary_loop(), name="paper_summary"),
        ]

        if self._rpe_lane_enabled() and self._rpe is not None:
            self._tasks.append(
                asyncio.create_task(self._rpe_crypto_retrigger_loop(), name="rpe_retrigger")
            )

        if self._live_orchestrator is not None and self._live_orchestrator.wallet_balance_provider is not None:
            self._tasks.append(
                asyncio.create_task(self._wallet_balance_poll_loop(), name="wallet_balance_poll")
            )

        if self._pure_mm is not None:
            self._tasks.append(
                asyncio.create_task(self._pure_mm.run(), name="pure_market_maker")
            )

        # SI-8: Oracle Latency Arbitrage polling loop (if enabled)
        if self._oracle_lane_enabled():
            self._tasks.append(
                asyncio.create_task(self._oracle_polling_loop(), name="oracle_poll")
            )

        # SI-9: Combinatorial Arbitrage cluster scanning + execution
        if self._si9_lane_enabled():
            # Build initial clusters from discovered markets
            self._cluster_mgr.scan_clusters(self._markets)
            # Ensure YES-token books exist for cluster legs
            for tid in self._cluster_mgr.all_cluster_yes_token_ids():
                if tid not in self._book_trackers:
                    self._book_trackers[tid] = OrderbookTracker(tid)
            self._sync_live_orchestrator_clusters()
            self._apply_live_market_target_map()
            self._combo_detector = ComboArbDetector(
                self._book_trackers,
                aggregators=self._yes_aggs,
                on_sync_block=lambda _assessment: self.sync_telemetry.record_si9_block(),
            )
            self._tasks.append(
                asyncio.create_task(self._combo_arbitrage_loop(), name="combo_arb")
            )

        if self._si10_lane_enabled():
            self._bayesian_cluster_mgr.scan_clusters(self._markets)
            for tid in self._bayesian_cluster_mgr.all_cluster_asset_ids():
                if tid not in self._book_trackers:
                    self._book_trackers[tid] = OrderbookTracker(tid)
            await self._fee_cache.prefetch(sorted(self._bayesian_cluster_mgr.all_cluster_asset_ids()))
            self._bayesian_detector = BayesianArbDetector(
                self._book_trackers,
                fee_enabled_resolver=self._is_fee_enabled,
                fee_rate_bps_lookup=self._fee_cache.get_fee_rate_sync,
                on_sync_block=lambda _assessment: self.sync_telemetry.record_si10_block(),
            )
            self._tasks.append(
                asyncio.create_task(self._bayesian_arb_loop(), name="bayesian_arb")
            )

        # Multi-core mode: add worker consumer tasks, skip in-process PCE loop
        if self._process_manager is not None:
            self._tasks.append(
                asyncio.create_task(self._consume_bbo_events(), name="bbo_consumer")
            )
            self._tasks.append(
                asyncio.create_task(self._consume_pce_results(), name="pce_consumer")
            )
            self._tasks.append(
                asyncio.create_task(
                    self._process_manager.health_check_loop(), name="worker_health"
                )
            )
        else:
            # Single-process fallback: PCE refresh runs in-process
            self._tasks.append(
                asyncio.create_task(self._pce_refresh_loop(), name="pce_refresh")
            )

        # Launch data recorder if enabled
        if self._recorder is not None:
            self._tasks.append(
                asyncio.create_task(self._recorder.run(), name="data_recorder")
            )
            log.info("data_recorder_enabled", data_dir=settings.record_data_dir)

        # Launch L2 WebSocket task if enabled
        if self._l2_ws is not None:
            self._tasks.append(
                asyncio.create_task(self._l2_ws.start(), name="l2_ws")
            )

        # ── Global asyncio exception handler (last line of defence) ─────
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(self._asyncio_exception_handler)

        # Handle graceful shutdown via SIGINT / SIGTERM
        self._loop = loop
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._schedule_stop)
        else:
            # On Windows, loop.add_signal_handler is not supported.
            # Register via the signal module instead (handles Ctrl-C).
            signal.signal(
                signal.SIGINT,
                lambda *_: loop.call_soon_threadsafe(self._schedule_stop),
            )

        log.info("bot_running", markets=len(self._markets), mode=mode_label)

        # Wait for all tasks (they run indefinitely until stop())
        try:
            await asyncio.gather(*self._tasks, return_exceptions=False)
        except asyncio.CancelledError:
            pass

        # Ensure stop() runs to completion before start() returns
        # (otherwise asyncio.run() tears down the loop while stop()
        #  is still doing cleanup — the real cause of the hang).
        if self._stop_task is not None and not self._stop_task.done():
            await self._stop_task

    # ── Global asyncio exception handler ────────────────────────────────────
    def _asyncio_exception_handler(self, loop: asyncio.AbstractEventLoop, context: dict) -> None:
        """Catch-all for unhandled exceptions in asyncio tasks/callbacks.

        Prevents the silent *"Task exception was never retrieved"* problem
        by logging with full context instead of relying on GC finalisation.
        """
        exc = context.get("exception")
        msg = context.get("message", "Unhandled exception in event loop")
        task = context.get("future")
        task_name = task.get_name() if task and hasattr(task, "get_name") else str(task)
        log.error(
            "asyncio_unhandled_exception",
            message=msg,
            task=task_name,
            error=repr(exc) if exc else "unknown",
            exc_info=exc,
        )

    # ── Market wiring / unwiring ───────────────────────────────────────────
    def _wire_market(self, m: MarketInfo) -> None:
        """Create aggregators, detectors, and book trackers for a market."""
        if m.yes_token_id in self._market_map:
            return  # already wired

        self._market_map[m.yes_token_id] = m
        self._market_map[m.no_token_id] = m

        yes_agg = OHLCVAggregator(m.yes_token_id)
        no_agg = OHLCVAggregator(m.no_token_id)
        self._yes_aggs[m.yes_token_id] = yes_agg
        self._no_aggs[m.no_token_id] = no_agg

        # RPE: Resolution Probability Engine (Pillar 14)
        # Wiring is handled by the singleton self._rpe;
        # no per-market instance needed.

        # SI-1: Per-market regime detector
        if settings.strategy.regime_enabled:
            self._regime_detectors[m.condition_id] = RegimeDetector(m.condition_id)

        self._detectors[m.condition_id] = PanicDetector(
            market_id=m.condition_id,
            yes_asset_id=m.yes_token_id,
            no_asset_id=m.no_token_id,
            yes_aggregator=yes_agg,
            no_aggregator=no_agg,
            zscore_threshold=settings.strategy.panic_zscore_threshold,
            volume_ratio_threshold=settings.strategy.panic_volume_ratio_threshold,
            trend_guard_pct=settings.strategy.panic_trend_guard_pct,
            trend_guard_bars=settings.strategy.panic_trend_guard_bars,
            ofi_veto_threshold=settings.strategy.panic_ofi_veto_threshold,
        )

        if self._ofi_lane_enabled():
            self._ofi_detectors[m.condition_id] = OFIMomentumDetector(
                market_id=m.condition_id,
                no_asset_id=m.no_token_id,
                window_ms=settings.strategy.window_ms,
                threshold=settings.strategy.ofi_threshold,
                tvi_kappa=settings.strategy.ofi_tvi_kappa,
            )
        if self._contagion_arb is not None:
            self._contagion_arb.register_market(m)

        # PCE: register market for correlation tracking (Pillar 15)
        self.pce.register_market(
            m.condition_id, m.event_id, getattr(m, 'tags', '') or '',
            yes_agg,
        )

        # Orderbook trackers for both tokens
        # Load shedding: only markets in _l2_active_set get full L2 book
        # reconstruction.  Others get the lightweight OrderbookTracker.
        l2_eligible = (
            settings.strategy.l2_enabled
            and hasattr(self, '_l2_active_set')
            and m.condition_id in self._l2_active_set
        )
        if l2_eligible:
            for token_id in (m.yes_token_id, m.no_token_id):
                # SI-2: Create iceberg detector and wire as level-change callback
                ice_cb = None
                if settings.strategy.iceberg_enabled:
                    ice_det = IcebergDetector(token_id)
                    self._iceberg_detectors[token_id] = ice_det
                    ice_cb = ice_det.on_level_change

                l2_book = L2OrderBook(
                    token_id,
                    on_bbo_change=self._on_l2_bbo_change,
                    on_level_change=ice_cb,
                )
                self._l2_books[token_id] = l2_book
                self._book_trackers[token_id] = L2OrderBookAdapter(l2_book)
        else:
            self._book_trackers[m.yes_token_id] = OrderbookTracker(
                m.yes_token_id,
                on_bbo_change=self._on_orderbook_bbo_change,
            )
            self._book_trackers[m.no_token_id] = OrderbookTracker(
                m.no_token_id,
                on_bbo_change=self._on_orderbook_bbo_change,
            )

    def _unwire_market(self, m: MarketInfo) -> None:
        """Remove all state for an evicted market."""
        self._market_map.pop(m.yes_token_id, None)
        self._market_map.pop(m.no_token_id, None)
        self._yes_aggs.pop(m.yes_token_id, None)
        self._no_aggs.pop(m.no_token_id, None)
        self._detectors.pop(m.condition_id, None)
        self._spread_evaluators.pop(m.condition_id, None)
        self._spread_cooldowns.pop(m.condition_id, None)
        self._regime_detectors.pop(m.condition_id, None)
        self._drift_detectors.pop(m.condition_id, None)
        self._drift_cooldowns.pop(m.condition_id, None)
        self._ofi_detectors.pop(m.condition_id, None)
        if self._contagion_arb is not None:
            self._contagion_arb.unregister_market(m.condition_id)
        if self._rpe is not None:
            self._rpe.clear_market(m.condition_id)
        self.pce.unregister_market(m.condition_id)
        self._book_trackers.pop(m.yes_token_id, None)
        self._book_trackers.pop(m.no_token_id, None)
        # Clean up L2 book instances
        for token_id in (m.yes_token_id, m.no_token_id):
            l2_book = self._l2_books.pop(token_id, None)
            if l2_book is not None:
                l2_book.reset()
            ice = self._iceberg_detectors.pop(token_id, None)
            if ice is not None:
                ice.reset()
        self._trade_counts.pop(m.yes_token_id, None)
        self._trade_counts.pop(m.no_token_id, None)
        self._taker_counts.pop(m.yes_token_id, None)
        self._taker_counts.pop(m.no_token_id, None)
        self._total_counts.pop(m.yes_token_id, None)
        self._total_counts.pop(m.no_token_id, None)
        self._recent_trade_volume.pop(m.condition_id, None)
        self._markets = [x for x in self._markets if x.condition_id != m.condition_id]

    def _market_uses_full_l2(self, market: MarketInfo) -> bool:
        return market.yes_token_id in self._l2_books and market.no_token_id in self._l2_books

    async def _reconcile_warm_markets(self, target_markets: list[MarketInfo]) -> None:
        """Keep wired/subscribed markets aligned to the warm universe and L2 tier."""
        target_by_id = {market.condition_id: market for market in target_markets}
        current_by_id = {market.condition_id: market for market in self._markets}

        for market in list(self._markets):
            if market.condition_id in target_by_id:
                continue
            if self._ws:
                await self._ws.remove_assets([market.yes_token_id, market.no_token_id])
            if self._l2_ws:
                await self._l2_ws.remove_assets([market.yes_token_id, market.no_token_id])
            self._unwire_market(market)

        for condition_id, market in list(current_by_id.items()):
            if condition_id not in target_by_id:
                continue
            wants_full_l2 = market.condition_id in self._l2_active_set and settings.strategy.l2_enabled
            if wants_full_l2 == self._market_uses_full_l2(market):
                continue
            if self._ws:
                await self._ws.remove_assets([market.yes_token_id, market.no_token_id])
            if self._l2_ws:
                await self._l2_ws.remove_assets([market.yes_token_id, market.no_token_id])
            self._unwire_market(market)
            current_by_id.pop(condition_id, None)

        new_asset_ids: list[str] = []
        new_l2_books: dict[str, L2OrderBook] = {}
        current_ids = set(current_by_id)
        for market in target_markets:
            if market.condition_id in current_ids:
                continue
            self._wire_market(market)
            new_asset_ids.extend([market.yes_token_id, market.no_token_id])
            for token_id in (market.yes_token_id, market.no_token_id):
                if token_id in self._l2_books:
                    new_l2_books[token_id] = self._l2_books[token_id]

        if self._ws and new_asset_ids:
            await self._ws.add_assets(new_asset_ids)
        if self._l2_ws and new_l2_books:
            await self._l2_ws.add_assets(new_l2_books)

        self._markets = list(target_markets)
        self._sync_live_orchestrator_clusters()
        self._apply_live_market_target_map()

    def _positioned_asset_ids(self) -> set[str]:
        """Return asset IDs that currently have open positions.

        Used by the heartbeat to decide which books need fresh data.
        Includes trade_asset_id (YES or NO token) for bidirectional
        RPE positions.
        """
        ids: set[str] = set()
        for pos in self.positions.get_open_positions():
            ids.add(pos.trade_asset_id or pos.no_asset_id)
            ids.add(pos.no_asset_id)
        return ids

    def _deployment_phase_for_orchestrator(self) -> str:
        if self.deployment_env == DeploymentEnv.PAPER:
            return "PAPER"
        return "LIVE"

    def _require_orchestrator_session_id(self) -> str:
        if self._session_id:
            return self._session_id
        raise RuntimeError("POLYBOT_SESSION_ID is required for live orchestrator startup")

    def _si9_cluster_configs(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        if not self._si9_lane_enabled():
            return ()
        self._cluster_mgr.scan_clusters(self._markets)
        return tuple(
            (cluster.event_id, tuple(leg.condition_id for leg in cluster.legs))
            for cluster in self._cluster_mgr.active_clusters
        )

    def _orchestrator_market_books(self) -> dict[str, tuple[OrderbookTracker, OrderbookTracker]]:
        market_books: dict[str, tuple[OrderbookTracker, OrderbookTracker]] = {}
        for market in self._tracked_l2_markets():
            yes_tracker = self._book_trackers.get(market.yes_token_id)
            no_tracker = self._book_trackers.get(market.no_token_id)
            if yes_tracker is None or no_tracker is None:
                continue
            market_books[market.condition_id] = (yes_tracker, no_tracker)
        return market_books

    def _si9_unwind_config(self) -> Si9UnwindConfig:
        return Si9UnwindConfig(
            market_sell_threshold=(
                _decimal_from_number(settings.strategy.si9_emergency_taker_max_cents) / Decimal("100")
            ),
            passive_unwind_threshold=Decimal("0.010000"),
            max_hold_recovery_ms=max(int(settings.strategy.si9_max_leg_delay_ms), 1),
            min_best_bid=Decimal("0.010000"),
        )

    def _live_orchestrator_config(self) -> LiveOrchestratorConfig:
        heartbeat_interval_ms = max(int(settings.strategy.heartbeat_check_ms), 1)
        self._orchestrator_tick_interval_ms = heartbeat_interval_ms
        signal_sources: set[str] = set()
        if self._ofi_lane_enabled():
            signal_sources.add("OFI")
        if self._contagion_arb is not None:
            signal_sources.add("CONTAGION")
        if self._reward_lane_enabled():
            signal_sources.add("REWARD")
        return LiveOrchestratorConfig(
            orchestrator_config=OrchestratorConfig(
                tick_interval_ms=heartbeat_interval_ms,
                max_pending_unwinds=0,
                max_concurrent_clusters=1,
                signal_sources_enabled=frozenset(signal_sources),
            ),
            guard_config=DispatchGuardConfig(
                dedup_window_ms=heartbeat_interval_ms,
                max_dispatches_per_source_per_window=max(1, min(max(int(settings.strategy.max_active_l2_markets), 1), 10)),
                rate_window_ms=max(heartbeat_interval_ms * 2, 1),
                circuit_breaker_threshold=2,
                circuit_breaker_reset_ms=max(heartbeat_interval_ms * 6, 1),
                max_open_positions_per_market=max(int(getattr(self.positions, "max_open", 1)), 1),
            ),
            deployment_phase=self._deployment_phase_for_orchestrator(),
            session_id=self._require_orchestrator_session_id(),
            max_position_release_failures=2,
            heartbeat_interval_ms=heartbeat_interval_ms,
        )

    def _build_live_execution_boundary_runtime(self) -> LiveExecutionBoundary:
        return build_live_execution_boundary(
            deployment_phase=self._deployment_phase_for_orchestrator(),
            session_id=self._require_orchestrator_session_id(),
            market_by_condition={market.condition_id: market for market in self._tracked_l2_markets()},
            now_ms=self._current_timestamp_ms,
            clob_client=None if self.deployment_env == DeploymentEnv.PAPER else self.executor._get_clob_client(),
        )

    def _build_live_orchestrator_runtime(self, execution_boundary: LiveExecutionBoundary) -> MultiSignalOrchestrator:
        config = self._live_orchestrator_config()
        market_books = self._orchestrator_market_books()
        return build_live_orchestrator(
            config=config,
            orderbook_tracker=market_books,
            position_manager=self.positions,
            execution_boundary=execution_boundary,
            ofi_exit_trackers=self._book_trackers,
        )

    def _ranked_live_market_ids(self) -> list[str]:
        return [
            market.condition_id
            for market in sorted(
                self._tracked_l2_markets(),
                key=lambda row: row.daily_volume_usd,
                reverse=True,
            )
        ]

    def _apply_live_market_target_map(self) -> None:
        if self._live_orchestrator is None or self._live_orchestrator.load_shedder is None:
            return
        self._live_orchestrator.load_shedder.update_target_map(self._ranked_live_market_ids())

    def _sync_live_orchestrator_clusters(self) -> None:
        return

    def _current_timestamp_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def _persist_reward_shadow_trade(self, payload: dict[str, object]) -> None:
        task = asyncio.create_task(self._record_reward_shadow_trade(payload), name="reward_shadow_persist")
        task.add_done_callback(_safe_task_done_callback)

    async def _record_reward_shadow_trade(self, payload: dict[str, object]) -> None:
        try:
            await self.trade_store.record_shadow_trade(**payload)
        except Exception:
            log.warning(
                "reward_shadow_persist_failed",
                trade_id=str(payload.get("trade_id", "")),
                exc_info=True,
            )

    def _initialize_wallet_balance(self) -> None:
        provider = None if self._live_orchestrator is None else self._live_orchestrator.wallet_balance_provider
        if provider is not None:
            provider.set_balance_update_callback(self._on_wallet_balance_update)
            self.positions.set_wallet_balance(provider.get_available_margin("USDC"))
            return

        if self.paper_mode and float(self.positions._wallet_balance_usd) <= 0.0:
            self.positions.set_wallet_balance(settings.strategy.paper_starting_balance_usd)

    def _on_wallet_balance_update(self, asset_symbol: str, balance: Decimal) -> None:
        if str(asset_symbol or "").strip().upper() != "USDC":
            return
        self.positions.set_wallet_balance(balance)

    async def _wallet_balance_poll_loop(self) -> None:
        if self._live_orchestrator is None or self._live_orchestrator.wallet_balance_provider is None:
            return
        await self._live_orchestrator.wallet_balance_provider.poll_balance_loop(self._orchestrator_tick_interval_ms)

    async def _orchestrator_tick_loop(self) -> None:
        interval_s = self._orchestrator_tick_interval_ms / 1000.0
        while self._running:
            await asyncio.sleep(interval_s)
            if self._live_orchestrator is None or self._orchestrator_health_monitor is None:
                continue
            current_timestamp_ms = self._current_timestamp_ms()
            try:
                if self.deployment_env == DeploymentEnv.PAPER:
                    events = self._live_orchestrator.on_tick(current_timestamp_ms)
                else:
                    events = await asyncio.to_thread(self._live_orchestrator.on_tick, current_timestamp_ms)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._orchestrator_health_monitor.record_position_release_failure()
                log.error("orchestrator_tick_error", exc_info=True)
                continue
            if any(event.event_type == "UNWIND_COMPLETE" for event in events):
                self._orchestrator_health_monitor.reset_release_failure_count()
            self._orchestrator_health_monitor.check(current_timestamp_ms)
            if self._reward_sidecar is not None:
                self._reward_sidecar.on_tick(current_timestamp_ms)

    def _fan_out_live_best_yes_ask(self, asset_id: str) -> None:
        _ = asset_id
        return

    def _has_live_ofi_exit_runtime(self) -> bool:
        return (
            self._ofi_lane_enabled()
            and self._live_orchestrator is not None
            and self._live_orchestrator.ofi_exit_router is not None
        )

    def _ofi_exit_positions_for_asset(self, asset_id: str) -> list[PositionState]:
        return [
            pos
            for pos in self.positions.get_open_positions()
            if getattr(pos, "signal_type", "") == "ofi_momentum"
            and pos.state == PositionState.EXIT_PENDING
            and ((pos.trade_asset_id or pos.no_asset_id) == asset_id or pos.no_asset_id == asset_id)
        ]

    def _current_decimal_bbo(self, asset_id: str) -> dict[str, Decimal]:
        tracker = self._book_trackers.get(asset_id)
        if tracker is None:
            return {"best_bid": Decimal("0"), "best_ask": Decimal("0")}
        try:
            snapshot = tracker.snapshot()
            best_bid = getattr(snapshot, "best_bid", 0.0)
            best_ask = getattr(snapshot, "best_ask", 0.0)
        except Exception:
            best_bid = getattr(tracker, "best_bid", 0.0)
            best_ask = getattr(tracker, "best_ask", 0.0)
        return {
            "best_bid": _decimal_from_number(max(0.0, best_bid)),
            "best_ask": _decimal_from_number(max(0.0, best_ask)),
        }

    @staticmethod
    def _dispatch_fill_from_receipt(receipt: Any) -> tuple[float, float] | None:
        fill_price = getattr(receipt, "fill_price", None)
        fill_size = getattr(receipt, "fill_size", None)
        if fill_price is not None and fill_size is not None:
            resolved_price = float(fill_price)
            resolved_size = float(fill_size)
            if resolved_price > 0.0 and resolved_size > 0.0:
                return resolved_price, resolved_size

        partial_fill_price = getattr(receipt, "partial_fill_price", None)
        partial_fill_size = getattr(receipt, "partial_fill_size", None)
        if partial_fill_price is not None and partial_fill_size is not None:
            resolved_price = float(partial_fill_price)
            resolved_size = float(partial_fill_size)
            if resolved_price > 0.0 and resolved_size > 0.0:
                return resolved_price, resolved_size

        serialized_envelope = str(getattr(receipt, "serialized_envelope", "") or "").strip()
        if not serialized_envelope:
            return None

        try:
            envelope = deserialize_envelope(serialized_envelope)
        except Exception:
            return None

        payloads = envelope.get("payloads")
        if not isinstance(payloads, list) or not payloads:
            return None

        first_payload = payloads[0]
        if not isinstance(first_payload, dict):
            return None

        try:
            planned_price = float(first_payload.get("price") or 0.0)
        except (TypeError, ValueError):
            return None

        metadata = first_payload.get("metadata")
        effective_size_raw = metadata.get("effective_size") if isinstance(metadata, dict) else None
        if effective_size_raw is None:
            effective_size_raw = first_payload.get("size")

        try:
            planned_size = float(effective_size_raw or 0.0)
        except (TypeError, ValueError):
            return None

        if planned_price <= 0.0 or planned_size <= 0.0:
            return None
        return planned_price, planned_size

    def _build_live_ofi_exit_position_state(self, pos: Any, current_timestamp_ms: int) -> dict[str, Any]:
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        max_hold_seconds = pos.drawn_time or pos.max_hold_seconds or 0.0
        baseline_spread = max(0.01, pos.entry_price * pos.drawn_stop_pct) if pos.drawn_stop_pct > 0 else 0.01
        current_bbo = self._current_decimal_bbo(exit_asset)
        return {
            "position_id": pos.id,
            "market_id": pos.market_id,
            "side": pos.trade_side or "NO",
            "size": _decimal_from_number(pos.effective_size),
            "drawn_tp": _decimal_from_number(max(0.0, pos.drawn_tp if pos.drawn_tp > 0 else pos.target_price)),
            "drawn_stop": _decimal_from_number(max(0.0, pos.drawn_stop if pos.drawn_stop > 0 else pos.stop_price)),
            "drawn_time_ms": int((pos.entry_time + max(max_hold_seconds, 0.0)) * 1000),
            "baseline_spread": _decimal_from_number(baseline_spread),
            "current_timestamp_ms": int(current_timestamp_ms),
            "current_best_bid": current_bbo["best_bid"],
            "current_best_ask": current_bbo["best_ask"],
        }

    async def _apply_live_ofi_exit_receipt(
        self,
        pos: Any,
        receipt: Any,
        *,
        fallback_price: Decimal,
        reason: str,
        post_only: bool,
    ) -> None:
        from src.trading.executor import Order, OrderSide, OrderStatus

        if not receipt.executed:
            if self._live_orchestrator is not None:
                self._live_orchestrator.clear_ofi_exit(pos.id)
            return

        prior_order = getattr(pos, "exit_order", None)
        if prior_order is not None and prior_order.order_id != (receipt.order_id or ""):
            prior_order.status = OrderStatus.CANCELLED
            self.executor.register_external_order(prior_order)

        order_status = OrderStatus.LIVE
        filled_size = Decimal("0")
        filled_avg_price = Decimal("0")
        if receipt.fill_status == "FULL":
            order_status = OrderStatus.FILLED
            filled_size = receipt.fill_size or Decimal("0")
            filled_avg_price = receipt.fill_price or Decimal("0")
        elif receipt.fill_status == "PARTIAL":
            order_status = OrderStatus.PARTIALLY_FILLED
            filled_size = receipt.partial_fill_size or Decimal("0")
            filled_avg_price = receipt.partial_fill_price or Decimal("0")

        exit_order = Order(
            order_id=receipt.order_id or f"OFI-EXIT-{pos.id}",
            market_id=pos.market_id,
            asset_id=pos.trade_asset_id or pos.no_asset_id,
            side=OrderSide.SELL,
            price=receipt.fill_price or fallback_price or Decimal("0.01"),
            size=_decimal_from_number(pos.effective_size),
            status=order_status,
            filled_size=filled_size,
            filled_avg_price=filled_avg_price,
            clob_order_id=str(receipt.execution_id or receipt.order_id or ""),
            post_only=post_only,
        )
        self.executor.register_external_order(exit_order)
        pos.exit_order = exit_order
        pos.exit_reason = reason
        if exit_order.status == OrderStatus.FILLED:
            self._handle_exit_fill(pos)

    async def _evaluate_live_ofi_exit_path(self, asset_id: str) -> None:
        if not self._has_live_ofi_exit_runtime() or self._live_orchestrator is None:
            return

        current_timestamp_ms = self._current_timestamp_ms()
        for pos in self._ofi_exit_positions_for_asset(asset_id):
            position_state = self._build_live_ofi_exit_position_state(pos, current_timestamp_ms)
            current_bbo = {
                "best_bid": position_state["current_best_bid"],
                "best_ask": position_state["current_best_ask"],
            }

            if pos.exit_order is not None and pos.exit_reason == "time_stop":
                receipt = self._live_orchestrator.evaluate_ofi_exit_promotion(
                    position_id=pos.id,
                    current_timestamp_ms=current_timestamp_ms,
                    current_bbo=current_bbo,
                )
                if receipt is not None:
                    await self._apply_live_ofi_exit_receipt(
                        pos,
                        receipt,
                        fallback_price=current_bbo["best_bid"],
                        reason="time_stop",
                        post_only=False,
                    )
                continue

            decision = self._live_orchestrator.evaluate_ofi_exit(
                asset_id=asset_id,
                position_state=position_state,
                current_timestamp_ms=current_timestamp_ms,
            )
            if decision.action in {"HOLD", "SUPPRESSED_BY_VACUUM"}:
                continue

            receipt = self._live_orchestrator.route_ofi_exit(position_state, decision)
            if receipt is None:
                continue
            await self._apply_live_ofi_exit_receipt(
                pos,
                receipt,
                fallback_price=current_bbo["best_ask"] if decision.action == "TIME_STOP_TRIGGERED" else decision.trigger_price,
                reason="target" if decision.action == "TARGET_HIT" else "stop_loss" if decision.action == "STOP_HIT" else "time_stop",
                post_only=decision.action == "TIME_STOP_TRIGGERED",
            )

    def _ensemble_allows_entry(
        self,
        *,
        strategy_source: str,
        market_id: str,
        direction: str,
        log_event: str,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        allowed, reason = self.ensemble_risk.can_enter(
            market_id=market_id,
            strategy_source=strategy_source,
            direction=direction,
        )
        if allowed:
            return True
        payload = {
            "market_id": market_id,
            "direction": (reason or {}).get("direction", direction),
            "strategy_source": strategy_source,
            "blocking_strategy": (reason or {}).get("blocking_strategy", ""),
        }
        if extra:
            payload.update(extra)
        log.info(log_event, **payload)
        return False

    def _ensemble_allows_combo_entry(
        self,
        *,
        strategy_source: str,
        entry_id: str,
        exposures: list[tuple[str, str]],
        log_event: str,
    ) -> bool:
        allowed, reason = self.ensemble_risk.can_enter_batch(
            strategy_source=strategy_source,
            exposures=exposures,
        )
        if allowed:
            return True
        log.info(
            log_event,
            entry_id=entry_id,
            market_id=(reason or {}).get("market_id", ""),
            direction=(reason or {}).get("direction", ""),
            strategy_source=strategy_source,
            blocking_strategy=(reason or {}).get("blocking_strategy", ""),
        )
        return False

    def _is_fee_enabled(self, market: MarketInfo) -> bool:
        """Check if a market's tags match any fee-enabled category."""
        market_tags = (getattr(market, 'tags', '') or '').lower()
        if not market_tags:
            return True
        return any(cat in market_tags for cat in self._fee_category_set)

    def _get_crypto_spot(self) -> float | None:
        """Return latest BTC spot price from the adverse-selection guard.

        Returns None when the guard hasn't started or has no ticks.
        Used as the ``price_fn`` for the shared CryptoPriceModel.
        """
        guard = getattr(self, "_adverse_guard", None)
        if guard is None:
            return None
        # AdverseSelectionGuard doesn't expose get_latest_price();
        # it doesn't maintain external Binance ticks.  The RPE crypto
        # model will return None (no hallucination) until a real
        # external price source is wired.
        fn = getattr(guard, "get_latest_price", None)
        if fn is not None:
            return fn()
        return None

    def _schedule_stop(self) -> None:
        """Schedule stop() as an independent task (idempotent).

        MUST be used instead of calling stop() directly from within any
        of self._tasks — calling stop() inline creates a circular future
        chain (task → stop → gather(tasks) → task) that causes
        RecursionError on cancellation.
        """
        if self._stop_task is None:
            loop = self._loop or asyncio.get_running_loop()
            self._stop_task = loop.create_task(self.stop())

    async def _suspend_and_reset(self) -> None:
        """Non-fatal circuit-breaker response: pause chasers, reset WS, resume.

        Unlike ``stop()``, this keeps the main process and all risk-management
        state alive.  Only the WebSocket transport is torn down and rebuilt
        after a 30-second cooldown.

        IMPORTANT: This method must NEVER raise — it is called from inside
        ``except Exception`` blocks in hot-path loops.  An unhandled
        exception here would escape the except handler and kill the
        asyncio task, crashing the bot via ``gather(return_exceptions=False)``.
        """
        try:
            log.warning("suspend_and_reset_begin")

            # 1. Pause order chasers via the fast-kill event
            self._fast_kill_event.clear()

            # 2. Cleanly stop the WebSocket pool
            if self._ws is not None:
                try:
                    await self._ws.stop()
                except Exception:
                    log.warning("ws_stop_error_during_reset", exc_info=True)

            # Let the event loop process the WS task completion naturally.
            # DO NOT cancel the WS task explicitly — it is part of the
            # top-level asyncio.gather() in start().  Cancelling it causes
            # gather to propagate CancelledError, which terminates start()
            # and crashes the entire bot process.
            await asyncio.sleep(0)  # yield so the WS task can finalise

            # Wait briefly for the WS task to finish after stop()
            for task in list(self._tasks):
                if task.get_name() == "ws" and not task.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(task), timeout=5,
                        )
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass

            # 3. Cooldown penalty
            log.info("suspend_cooldown_start", cooldown_s=30)
            await asyncio.sleep(30)

            # 4. Re-initialise WebSocket connections with the same asset set
            if self._ws is not None:
                asset_ids = self._ws.asset_ids
                self._ws = MarketWebSocketPool(
                    asset_ids,
                    self._trade_queue,
                    book_queue=self._book_queue,
                    recorder=self._recorder,
                )
                ws_task = asyncio.create_task(self._ws.start(), name="ws")
                # Replace the old WS task in self._tasks
                self._tasks = [
                    t for t in self._tasks if t.get_name() != "ws" or not t.done()
                ]
                self._tasks.append(ws_task)

            # 5. Reset the circuit breakers so they can trip again on new errors
            self._trade_loop_breaker.reset()
            self._stale_bar_breaker.reset()
            self._retrigger_breaker.reset()
            self._tp_rescale_breaker.reset()
            self._ghost_breaker.reset()
            self._timeout_breaker.reset()
            self._oracle_breaker.reset()

            # 6. Resume chasers
            self._fast_kill_event.set()

            log.warning("suspend_and_reset_complete")
            try:
                await self.telegram.send(
                    "✅ <b>WS Hard-Reset complete</b> — chasers resumed, "
                    "risk state preserved."
                )
            except Exception:
                pass

        except asyncio.CancelledError:
            raise  # never swallow task cancellation
        except Exception:
            log.error("suspend_and_reset_error", exc_info=True)

    async def stop(self) -> None:
        """Graceful shutdown: cancel orders, flatten, stop tasks."""
        if not self._running:
            return
        self._running = False
        log.info("bot_stopping")

        cancelled = await self.executor.cancel_all()
        log.info("orders_cancelled", count=cancelled)

        # Cancel all active chaser tasks on open positions
        for pos in self.positions.get_open_positions():
            if pos.entry_chaser_task and not pos.entry_chaser_task.done():
                pos.entry_chaser_task.cancel()
            if pos.exit_chaser_task and not pos.exit_chaser_task.done():
                pos.exit_chaser_task.cancel()

        if self._ws:
            await self._ws.stop()
        if self._l2_ws:
            await self._l2_ws.stop()

        # Cancel all tasks BEFORE stopping process manager so that
        # queue-consumer threads (bbo_consumer, pce_consumer) exit
        # before their underlying queues are closed.
        for task in self._tasks:
            task.cancel()
        # Allow cancelled tasks (especially to_thread consumers) to
        # observe the cancellation and finish their current iteration.
        # Timeout prevents hanging if a thread is stuck on a closed queue.
        if self._tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._tasks, return_exceptions=True),
                    timeout=5,
                )
            except asyncio.TimeoutError:
                log.warning("task_cancel_timeout", stuck_tasks=[
                    t.get_name() for t in self._tasks if not t.done()
                ])

        # Stop multi-core worker processes (queues safe to close now)
        if self._process_manager is not None:
            await asyncio.to_thread(self._process_manager.stop_all)
            self._process_manager = None

        await self.whale_monitor.stop()

        # Stop new guard and heartbeat
        if hasattr(self, "_adverse_guard"):
            await self._adverse_guard.stop()
        if hasattr(self, "_heartbeat"):
            self._heartbeat.stop()
        if hasattr(self, "_order_poller"):
            self._order_poller.stop()
        if hasattr(self, "_stop_loss_monitor"):
            self._stop_loss_monitor.stop()
        if hasattr(self, "_pure_mm") and self._pure_mm is not None:
            await self._pure_mm.stop()

        try:
            stats = self._augment_trade_stats(
                await asyncio.wait_for(self.trade_store.get_stats(), timeout=5)
            )
            log.info("final_stats", **stats)
            await asyncio.wait_for(self.telegram.notify_stats(stats), timeout=5)
            await asyncio.wait_for(
                self.telegram.send(
                    f"🛑 <b>Bot stopped</b>\n"
                    f"Trades: {int(stats.get('total_trades', 0) or 0)}\n"
                    f"Expectancy: {float(stats.get('expectancy_cents', 0.0) or 0.0):+.2f}¢/trade"
                    f"{self._smart_passive_operator_block()}"
                    f"{self._sync_gate_operator_block()}"
                ),
                timeout=5,
            )
        except Exception:
            log.warning("final_stats_skipped")
        try:
            await asyncio.wait_for(self.trade_store.clear_live_state(), timeout=5)
            await asyncio.wait_for(self.trade_store.close(), timeout=5)
        except Exception:
            log.warning("trade_store_close_error", exc_info=True)
        try:
            await asyncio.wait_for(self._fee_cache.close(), timeout=5)
        except Exception:
            log.warning("fee_cache_close_error", exc_info=True)
        if self._live_execution_boundary is not None:
            try:
                await asyncio.wait_for(self._live_execution_boundary.close(), timeout=5)
            except Exception:
                log.warning("orchestrator_transport_close_error", exc_info=True)
            finally:
                self._live_execution_boundary = None

        log.info("bot_stopped")

    # ═══════════════════════════════════════════════════════════════════════
    #  Kill switch
    # ═══════════════════════════════════════════════════════════════════════
    async def kill(self) -> None:
        """Emergency shutdown — cancel everything and exit."""
        log.warning("KILL_SWITCH_ACTIVATED")
        await self.telegram.notify_kill()
        await self.stop()

    # ═══════════════════════════════════════════════════════════════════════
    #  Pillar 10 — CLOB fill callback for order-status poller
    # ═══════════════════════════════════════════════════════════════════════
    async def _on_clob_fill(self, order: "Order") -> None:
        """Called by :class:`OrderStatusPoller` when a LIVE order fills.

        Routes the fill to the correct position lifecycle handler:
        entry fill → :meth:`_handle_entry_fill`, exit fill → :meth:`_handle_exit_fill`.
        """
        from src.trading.executor import Order  # avoid circular at module-level

        if self._reward_sidecar is not None and self._reward_sidecar.on_fill(order, current_timestamp_ms=self._current_timestamp_ms()):
            return

        if hasattr(self, "_pure_mm") and self._pure_mm is not None:
            if await self._pure_mm.on_order_fill(order):
                if (
                    self._maker_monitor is not None
                    and getattr(order, "post_only", False)
                    and order.filled_avg_price > 0
                    and order.filled_size > 0
                ):
                    fill_rec = make_fill_record(
                        market_id=order.market_id,
                        asset_id=order.asset_id,
                        fill_price=order.filled_avg_price,
                        fill_side=order.side.value,
                        size_usd=order.filled_avg_price * order.filled_size,
                    )
                    self._maker_monitor.record_maker_fill(fill_rec)
                return

        if await self.positions.on_combo_order_update(order, self._combo_positions):
            return

        for pos in self.positions.get_open_positions():
            if pos.entry_order and pos.entry_order.order_id == order.order_id:
                if pos.state == PositionState.ENTRY_PENDING and order.status == OrderStatus.FILLED:
                    await self._handle_entry_fill(pos)
                    # V1 Adverse-selection monitor: record POST_ONLY entry fills
                    if (
                        self._maker_monitor is not None
                        and getattr(order, "post_only", False)
                        and order.filled_avg_price > 0
                        and order.filled_size > 0
                    ):
                        fill_rec = make_fill_record(
                            market_id=pos.market_id,
                            asset_id=pos.trade_asset_id or pos.no_asset_id,
                            fill_price=order.filled_avg_price,
                            fill_side="BUY",
                            size_usd=order.filled_avg_price * order.filled_size,
                        )
                        self._maker_monitor.record_maker_fill(fill_rec)
                return
            if pos.exit_order and pos.exit_order.order_id == order.order_id:
                if pos.state == PositionState.EXIT_PENDING and order.status == OrderStatus.FILLED:
                    self._handle_exit_fill(pos)
                return

        log.debug("clob_fill_no_position", order_id=order.order_id)

    # ═══════════════════════════════════════════════════════════════════════
    #  Adaptive polling helper
    # ═══════════════════════════════════════════════════════════════════════
    def _latest_zscore(self) -> float:
        """Callback for WhaleMonitor — returns the most recent panic Z-score."""
        return self._latest_z

    # ═══════════════════════════════════════════════════════════════════════
    #  Core processing loops
    # ═══════════════════════════════════════════════════════════════════════
    async def _process_trades(self) -> None:
        """Main loop: consume TradeEvents, update aggregators, check signals.

        Pillar 4 integration: every message is latency-checked.  If the
        guard is BLOCKED, aggregators still receive the price (so bars
        don't gap on recovery) but signal evaluation and fill checking
        are skipped.
        """
        while self._running:
            try:
                event = await asyncio.wait_for(self._trade_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                # ── Pillar 4: Latency gate ─────────────────────────────────
                latency_state = self.latency_guard.check(event.timestamp)
                is_blocked = latency_state == LatencyState.BLOCKED

                if is_blocked and latency_state != getattr(self, "_prev_latency", None):
                    await self.telegram.send(
                        "🚨 <b>Latency BLOCKED</b> — execution halted until WS stabilises.\n"
                        f"Delta: {self.latency_guard.last_delta_ms:.0f}ms"
                    )
                if not is_blocked and getattr(self, "_prev_latency", None) == LatencyState.BLOCKED:
                    await self.telegram.send("✅ <b>Latency recovered</b> — execution resumed.")
                self._prev_latency = latency_state

                # Track trade frequency
                self._trade_counts[event.asset_id] = (
                    self._trade_counts.get(event.asset_id, 0.0) + 1.0
                )

                # ── MTI: classify trade as taker or maker ──────────────────
                book = self._book_trackers.get(event.asset_id)
                if book and book.has_data:
                    snap = book.snapshot()
                    is_taker = False
                    if event.side == "buy" and snap.best_ask > 0 and event.price >= snap.best_ask:
                        is_taker = True
                    elif event.side == "sell" and snap.best_bid > 0 and event.price <= snap.best_bid:
                        is_taker = True
                    event.is_taker = is_taker

                    self._total_counts[event.asset_id] = (
                        self._total_counts.get(event.asset_id, 0) + 1
                    )
                    if is_taker:
                        self._taker_counts[event.asset_id] = (
                            self._taker_counts.get(event.asset_id, 0) + 1
                        )

                # ── Ghost liquidity: record recent trade volume ────────────
                market_info = self._market_map.get(event.asset_id)
                if market_info:
                    cid = market_info.condition_id
                    if cid not in self._recent_trade_volume:
                        self._recent_trade_volume[cid] = []
                    self._recent_trade_volume[cid].append(
                        (event.timestamp, event.price * event.size)
                    )

                # Route to the correct aggregator (always, even when blocked)
                yes_agg = self._yes_aggs.get(event.asset_id)
                no_agg = self._no_aggs.get(event.asset_id)

                if yes_agg:
                    yes_agg.on_trade(event)

                if no_agg:
                    no_agg.on_trade(event)
                    if self.paper_mode and not is_blocked:
                        self._check_paper_fills(event)

                if self.paper_mode and yes_agg and not is_blocked:
                    self._check_paper_fills(event)
            except asyncio.CancelledError:
                raise
            except KeyError:
                # Stale / missing asset data — non-fatal, skip this event
                log.warning(
                    "trade_processing_stale_key",
                    asset_id=event.asset_id,
                    exc_info=True,
                )
            except Exception:
                log.error(
                    "trade_processing_error",
                    asset_id=event.asset_id,
                    exc_info=True,
                )
                if self._trade_loop_breaker.record():
                    log.critical(
                        "trade_processor_circuit_breaker_tripped",
                        errors_in_window=self._trade_loop_breaker.recent_errors,
                        msg="Too many unexpected errors in trade processor — suspending & resetting WS",
                    )
                    await self.telegram.send(
                        "🟡 <b>CIRCUIT BREAKER</b>: trade_processor tripped "
                        "(5 unexpected errors in 60s) — suspend & hard-reset WS."
                    )
                    await self._suspend_and_reset()
                    # After reset, continue processing

    async def _process_book_updates(self) -> None:
        """Consume orderbook WS events and update trackers.

        When the latency guard is BLOCKED, snapshots are tagged as
        ``fresh=False`` so downstream consumers (sizer, chaser) know
        the data may be stale.

        When L2 is enabled, trackers backed by L2OrderBookAdapter are
        fed directly by the L2WebSocket — ``on_price_change`` /
        ``on_book_snapshot`` are no-ops on adapters.  This loop still
        runs for any remaining non-L2 trackers and as a fallback.
        """
        while self._running:
            try:
                msg = await asyncio.wait_for(self._book_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if not isinstance(msg, dict):
                continue

            event_type = msg.get("event_type") or msg.get("type", "")
            asset_id = msg.get("asset_id") or ""

            tracker = self._book_trackers.get(asset_id)
            if not tracker:
                continue

            # L2-backed trackers are no-ops for these calls, but we
            # still route for backward compat and non-L2 fallback.
            try:
                if event_type == "price_change":
                    tracker.on_price_change(msg)
                elif event_type == "book":
                    tracker.on_book_snapshot(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.error(
                    "book_update_error",
                    asset_id=asset_id,
                    event_type=event_type,
                    exc_info=True,
                )

    # ── Multi-core consumer loops ─────────────────────────────────────────
    async def _consume_bbo_events(self) -> None:
        """Consume BBO change notifications from L2 worker processes.

        Reads ``(\"bbo\", asset_id, seq)`` tuples from the shared
        ``multiprocessing.Queue``, then drives the same callbacks the
        in-process ``_on_l2_bbo_change`` would have fired.
        """
        if self._process_manager is None:
            return
        bbo_queue = self._process_manager.bbo_queue

        while self._running:
            try:
                # Poll the multiprocessing Queue from a thread to avoid
                # blocking the event loop.
                try:
                    msg = await asyncio.to_thread(bbo_queue.get, timeout=0.5)
                except Exception:
                    # Empty queue or timeout
                    continue

                if msg is None:
                    continue

                cmd = msg[0]
                if cmd == "bbo":
                    _, asset_id, seq = msg
                    # Read spread score from shared memory for the callback
                    reader = self._process_manager.get_reader(asset_id)
                    if reader is None:
                        continue
                    snap = reader.read_header()
                    # Build a minimal SpreadScore-like object
                    from src.data.spread_score import SpreadScore
                    score = SpreadScore(
                        raw_spread_cents=round(snap.spread * 100, 2) if snap.spread else 0.0,
                        depth_weighted_spread_cents=0.0,
                        score=snap.spread_score,
                        timestamp=snap.timestamp,
                    )
                    await self._on_l2_bbo_change(asset_id, score)
            except asyncio.CancelledError:
                break
            except Exception:
                log.error("bbo_consumer_error", exc_info=True)

    async def _consume_pce_results(self) -> None:
        """Consume results from the PCE computation worker.

        Dispatches dashboard data, cross-market signals, etc.
        """
        if self._process_manager is None:
            return
        output_queue = self._process_manager.pce_output_queue

        while self._running:
            try:
                try:
                    msg = await asyncio.to_thread(output_queue.get, timeout=1.0)
                except Exception:
                    continue

                if msg is None:
                    continue

                cmd = msg[0]
                if cmd == "pce_refreshed":
                    _, dashboard = msg
                    await self.telegram.notify_pce_dashboard(dashboard)
                    log.info(
                        "pce_dashboard_received",
                        portfolio_var=dashboard.get("portfolio_var", 0.0),
                        pairs_tracked=dashboard.get("total_pairs_tracked", 0),
                    )
                elif cmd == "cm_signals":
                    _, signal_dicts = msg
                    if signal_dicts:
                        log.info("cross_market_signals_received", count=len(signal_dicts))
                        await self._consume_cross_market_signals(signal_dicts)
                elif cmd == "prior_validation":
                    _, summary = msg
                    log.info("pce_prior_validation_received", **summary)
            except asyncio.CancelledError:
                break
            except Exception:
                log.error("pce_consumer_error", exc_info=True)

    # ── L2 callbacks ──────────────────────────────────────────────────────
    async def _on_l2_bbo_change(self, asset_id: str, score: Any) -> None:
        """Callback from L2OrderBook when the BBO changes.

        Logs the live spread score for monitoring and drives the
        event-driven stop-loss engine.  Also evaluates the spread-based
        signal path (Problem 3) on NO-token BBO ticks.

        Fully exception-guarded: any failure is logged and contained
        so that a single bad tick cannot kill the callback pipeline.
        """
        try:
            await self._on_l2_bbo_change_inner(asset_id, score)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error(
                "l2_bbo_callback_error",
                asset_id=asset_id,
                exc_info=True,
            )

    async def _on_l2_bbo_change_inner(self, asset_id: str, score: Any) -> None:
        """Inner implementation — called by the guarded wrapper."""
        self._fan_out_live_best_yes_ask(asset_id)
        log.debug(
            "l2_bbo_update",
            asset_id=asset_id,
            spread_score=round(score.score, 1),
            raw_spread_cents=round(score.raw_spread_cents, 2),
        )
        await self._tick_shadow_tracker_for_asset(asset_id)
        await self._evaluate_live_ofi_exit_path(asset_id)
        # Drive event-driven stop-loss evaluation
        await self._stop_loss_monitor.on_bbo_update(
            asset_id,
            exclude_signal_types={"ofi_momentum"} if self._has_live_ofi_exit_runtime() else None,
        )

        # Tick the maker adverse-selection monitor (schedules T+5/15/60 marks)
        if self._maker_monitor is not None:
            await self._maker_monitor.tick()
        if self._reward_sidecar is not None:
            self._reward_sidecar.on_book_update(asset_id, current_timestamp_ms=self._current_timestamp_ms())

        if self._ofi_lane_enabled():
            await self._evaluate_ofi_momentum(asset_id)
        if self._contagion_arb is not None:
            await self._evaluate_contagion_arb(asset_id)

    def _on_orderbook_bbo_change(self, asset_id: str, snapshot: Any) -> None:
        """Callback from basic OrderbookTracker when the BBO changes.

        Schedules an async stop-loss evaluation for this asset.
        """
        del snapshot
        self._fan_out_live_best_yes_ask(asset_id)
        _safe_fire_and_forget(
            self._evaluate_live_ofi_exit_path(asset_id),
            name=f"ofi_exit_bbo_{asset_id[:12]}",
        )
        _safe_fire_and_forget(
            self._tick_shadow_tracker_for_asset(asset_id),
            name=f"shadow_bbo_{asset_id[:12]}",
        )
        if self._reward_sidecar is not None:
            self._reward_sidecar.on_book_update(asset_id, current_timestamp_ms=self._current_timestamp_ms())
        _safe_fire_and_forget(
            self._stop_loss_monitor.on_bbo_update(
                asset_id,
                exclude_signal_types={"ofi_momentum"} if self._has_live_ofi_exit_runtime() else None,
            ),
            name=f"stop_loss_bbo_{asset_id[:12]}",
        )
        if self._ofi_lane_enabled():
            _safe_fire_and_forget(
                self._evaluate_ofi_momentum(asset_id),
                name=f"ofi_bbo_{asset_id[:12]}",
            )
        if self._contagion_arb is not None:
            _safe_fire_and_forget(
                self._evaluate_contagion_arb(asset_id),
                name=f"contagion_bbo_{asset_id[:12]}",
            )

    async def _evaluate_ofi_momentum(self, asset_id: str) -> None:
        """Evaluate OFI momentum on book updates and route BUY signals."""
        if not self._ofi_lane_enabled():
            return

        market_info = self._market_map.get(asset_id)
        if market_info is None or asset_id != market_info.no_token_id:
            return

        if not self.lifecycle.is_tradeable(market_info.condition_id):
            assessment = self.lifecycle.tradeability_assessment(market_info.condition_id)
            self._record_single_name_rejection(
                "ofi_momentum",
                market_info,
                f"not_tradeable_{assessment.reason}",
                log_event=False,
                tier=assessment.tier,
                **assessment.live_metrics,
            )
            return
        if not market_info.accepting_orders:
            self._record_single_name_rejection("ofi_momentum", market_info, "not_accepting_orders", log_event=False)
            return
        if not self.lifecycle.is_cooled_down(market_info.condition_id):
            self._record_single_name_rejection("ofi_momentum", market_info, "signal_cooldown", log_event=False)
            return

        current_timestamp_ms = self._current_timestamp_ms()
        is_live_ofi_runtime = (
            self._live_orchestrator is not None
            and self._orchestrator_health_monitor is not None
        )
        if not is_live_ofi_runtime:
            self._record_single_name_rejection("ofi_momentum", market_info, "runtime_disabled", log_event=False)
            return

        detector = self._ofi_detectors.get(market_info.condition_id)
        no_book = self._book_trackers.get(market_info.no_token_id)
        no_agg = self._no_aggs.get(market_info.no_token_id)
        if detector is None or no_book is None or no_agg is None or not no_book.has_data:
            reason = "missing_detector"
            if no_book is None or not getattr(no_book, "has_data", False):
                reason = "missing_no_book"
            elif no_agg is None:
                reason = "missing_no_aggregator"
            self._record_single_name_rejection("ofi_momentum", market_info, reason, log_event=False)
            return

        l2_no = self._l2_books.get(market_info.no_token_id)
        if l2_no is not None and not l2_no.is_reliable:
            self._record_single_name_rejection("ofi_momentum", market_info, "l2_unreliable", log_event=False)
            return

        sig = detector.generate_signal(no_book=no_book, trade_aggregator=no_agg)
        if sig is None:
            self._record_single_name_rejection("ofi_momentum", market_info, "no_signal", log_event=False)
            return

        trade_price = float(sig.no_best_ask or 0.0)
        if trade_price >= 0.97 or trade_price <= 0.03:
            self._record_single_name_rejection("ofi_momentum", market_info, "near_resolved_price", log_event=False)
            self.lifecycle.drain_market(market_info.condition_id, reason="near_resolved_price")
            return

        if not (settings.strategy.min_tradeable_price < trade_price < settings.strategy.max_tradeable_price):
            self._record_single_name_rejection("ofi_momentum", market_info, "price_out_of_band", log_event=False)
            return

        if sig.trade_flow_imbalance < settings.strategy.ofi_min_trade_flow_imbalance:
            self._record_single_name_rejection(
                "ofi_momentum",
                market_info,
                "weak_trade_flow_confirmation",
                log_event=False,
            )
            return

        if sig.tvi_multiplier < settings.strategy.ofi_min_tvi_multiplier:
            self._record_single_name_rejection(
                "ofi_momentum",
                market_info,
                "weak_tvi_confirmation",
                log_event=False,
            )
            return

        if sig.toxicity_index >= settings.strategy.ofi_toxicity_veto_threshold:
            self._record_single_name_rejection("ofi_momentum", market_info, "toxicity_veto", log_event=False)
            return

        regime_det = self._regime_detectors.get(market_info.condition_id)
        regime_score = regime_det.regime_score if regime_det else 0.5
        meta_decision = self._meta_controller.evaluate("ofi_momentum", regime_score)
        if meta_decision.vetoed:
            self._record_single_name_rejection("ofi_momentum", market_info, "meta_controller_veto", log_event=False)
            log.info(
                "meta_controller_veto",
                market_id=market_info.condition_id[:16],
                signal_type="ofi_momentum",
                regime_score=round(regime_score, 3),
                reason=meta_decision.veto_reason,
            )
            return

        if sig.direction != "BUY":
            self._record_single_name_rejection("ofi_momentum", market_info, "non_buy_signal", log_event=False)
            log.debug(
                "ofi_momentum_sell_observed",
                market_id=market_info.condition_id,
                ofi=round(sig.ofi, 4),
            )
            return

        if not self.positions.is_stop_loss_cooled_down(market_info.condition_id):
            self._record_single_name_rejection("ofi_momentum", market_info, "stop_loss_cooldown", log_event=False)
            log.info(
                "stop_loss_cooldown_suppressed_ofi",
                market_id=market_info.condition_id[:16],
            )
            return

        self._open_ofi_reverse_shadow_position(market_info, sig, reference_price=trade_price)

        live_signal = OfiEntrySignal(
            market_id=market_info.condition_id,
            side="NO",
            target_price=_decimal_from_number(sig.no_best_ask),
            anchor_volume=_decimal_from_number(sig.top_ask_size),
            conviction_scalar=_decimal_from_number(min(max(abs(sig.rolling_vi), 0.0), 1.0)),
            signal_timestamp_ms=int(sig.timestamp_ms),
            tvi_kappa=_decimal_from_number(max(sig.tvi_multiplier - 1.0, 0.0)),
            ofi_window_ms=max(int(sig.window_ms), 1),
        )
        self._live_orchestrator.on_ofi_signal(
            live_signal,
            _decimal_from_number(settings.strategy.max_trade_size_usd),
            int(sig.timestamp_ms),
        )
        self.lifecycle.record_signal(market_info.condition_id)
        return

    async def _evaluate_contagion_arb(self, asset_id: str) -> None:
        """Evaluate the toxicity contagion detector on every BBO update."""
        if self._contagion_arb is None:
            return

        market_info = self._market_map.get(asset_id)
        if market_info is None:
            return
        if not self.lifecycle.is_tradeable(market_info.condition_id):
            return
        if not market_info.accepting_orders:
            return

        yes_book = self._book_trackers.get(market_info.yes_token_id)
        no_book = self._book_trackers.get(market_info.no_token_id)
        if yes_book is None or no_book is None:
            return
        if not getattr(yes_book, "has_data", False) or not getattr(no_book, "has_data", False):
            return

        l2_yes = self._l2_books.get(market_info.yes_token_id)
        l2_no = self._l2_books.get(market_info.no_token_id)
        if l2_yes is not None and not l2_yes.is_reliable:
            return
        if l2_no is not None and not l2_no.is_reliable:
            return

        yes_snapshot = yes_book.snapshot()
        no_snapshot = no_book.snapshot()
        yes_price = 0.0
        if yes_snapshot.best_bid > 0 and yes_snapshot.best_ask > 0:
            yes_price = round((yes_snapshot.best_bid + yes_snapshot.best_ask) / 2.0, 4)
        elif yes_snapshot.best_ask > 0:
            yes_price = round(yes_snapshot.best_ask, 4)
        else:
            yes_agg = self._yes_aggs.get(market_info.yes_token_id)
            yes_price = round(yes_agg.current_price, 4) if yes_agg else 0.0

        if yes_price <= 0 or yes_price >= 1:
            return

        yes_buy_toxicity = float(yes_book.toxicity_index("BUY"))
        no_buy_toxicity = float(no_book.toxicity_index("BUY"))
        signals = self._contagion_arb.evaluate_market(
            market=market_info,
            yes_price=yes_price,
            yes_buy_toxicity=yes_buy_toxicity,
            no_buy_toxicity=no_buy_toxicity,
            universe=self._tracked_l2_markets(),
            book_snapshots=(yes_snapshot, no_snapshot),
        )
        for signal in signals:
            lagging_market = self._find_market(signal.lagging_market_id)
            if lagging_market is None:
                continue
            await self._on_contagion_signal(signal, lagging_market)

    async def _on_l2_desync(self, asset_id: str) -> None:
        """Callback from L2OrderBook on sequence gap detection.

        Routes to the L2WebSocket for snapshot re-fetch.
        Exception-guarded so a desync error cannot kill the callback.
        """
        try:
            if self._l2_ws is not None:
                await self._l2_ws._on_book_desync(asset_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error("l2_desync_callback_error", asset_id=asset_id, exc_info=True)

    async def _on_yes_bar_closed(self, yes_asset_id: str, bar: Any) -> None:
        """A 1-min YES bar just closed — evaluate the panic detector."""
        market_info = self._market_map.get(yes_asset_id)
        if not market_info:
            return

        # Only trade active-tier markets
        if not self.lifecycle.is_tradeable(market_info.condition_id):
            assessment = self.lifecycle.tradeability_assessment(market_info.condition_id)
            rejection_reason = f"not_tradeable_{assessment.reason}"
            details = {
                "tier": assessment.tier,
                **assessment.live_metrics,
            }
            self._record_single_name_rejection("panic", market_info, rejection_reason, log_event=False, **details)
            self._record_single_name_rejection("rpe", market_info, rejection_reason, log_event=False, **details)
            return

        # ── Fix 4: Real-time drain if market stopped accepting orders ──
        if not market_info.accepting_orders:
            self._record_single_name_rejection("panic", market_info, "not_accepting_orders", log_event=False)
            self._record_single_name_rejection("rpe", market_info, "not_accepting_orders", log_event=False)
            self.lifecycle.drain_market(market_info.condition_id, reason="not_accepting_orders")
            return

        # Signal cooldown check
        if not self.lifecycle.is_cooled_down(market_info.condition_id):
            self._record_single_name_rejection("panic", market_info, "signal_cooldown", log_event=False)
            self._record_single_name_rejection("rpe", market_info, "signal_cooldown", log_event=False)
            return

        # ── L2 book reliability gate ───────────────────────────────
        # Chronically desyncing books produce unreliable BBO/depth data.
        # Skip signal evaluation but do NOT evict — let recovery continue.
        l2_yes = self._l2_books.get(market_info.yes_token_id)
        if l2_yes is not None and not l2_yes.is_reliable:
            self._record_single_name_rejection("panic", market_info, "l2_unreliable", log_event=False)
            self._record_single_name_rejection("rpe", market_info, "l2_unreliable", log_event=False)
            log.info(
                "l2_book_unreliable",
                asset_id=market_info.yes_token_id,
                seq_gap_rate=round(l2_yes.seq_gap_rate, 4),
                delta_count=l2_yes.delta_count,
            )
            return

        detector = self._detectors.get(market_info.condition_id)
        if not detector:
            self._record_single_name_rejection("panic", market_info, "missing_detector", log_event=False)
            return

        no_agg = self._no_aggs.get(market_info.no_token_id)
        if not no_agg:
            self._record_single_name_rejection("panic", market_info, "missing_no_aggregator", log_event=False)
            self._record_single_name_rejection("rpe", market_info, "missing_no_aggregator", log_event=False)
            return

        # Use real orderbook best_ask if available, else fall back to last trade
        no_book = self._book_trackers.get(market_info.no_token_id)
        if no_book and no_book.has_data:
            snap = no_book.snapshot()
            no_best_ask = snap.best_ask if snap.best_ask > 0 else no_agg.current_price
        else:
            no_best_ask = no_agg.current_price

        if no_best_ask <= 0:
            self._record_single_name_rejection("panic", market_info, "invalid_no_best_ask", log_event=False)
            self._record_single_name_rejection("rpe", market_info, "invalid_no_best_ask", log_event=False)
            return

        # ── Derive YES close price for price-band checks ───────────────
        yes_price = bar.close if hasattr(bar, 'close') else 0.0

        # ── Fix 2: Near-resolved price auto-drain ──────────────────────
        if yes_price >= 0.97 or yes_price <= 0.03:
            self._record_single_name_rejection("panic", market_info, "near_resolved_price", log_event=False)
            self._record_single_name_rejection("rpe", market_info, "near_resolved_price", log_event=False)
            self.lifecycle.drain_market(market_info.condition_id, reason="near_resolved_price")
            return

        # ── Fix 1: Tradeable price band guard ──────────────────────────
        min_price = settings.strategy.min_tradeable_price
        max_price = settings.strategy.max_tradeable_price
        price_in_band = min_price < yes_price < max_price
        if not price_in_band:
            self._record_single_name_rejection("panic", market_info, "price_out_of_band", log_event=False)
            self._record_single_name_rejection("rpe", market_info, "price_out_of_band", log_event=False)

        # Whale confluence check
        whale = self.whale_monitor.has_confluence(market_info.no_token_id)

        if price_in_band:
            # ── SI-1: Update regime detector and gate entries ───────────
            regime_det = self._regime_detectors.get(market_info.condition_id)
            yes_agg_regime = self._yes_aggs.get(market_info.yes_token_id)
            if regime_det and yes_agg_regime:
                # Compute log-return from the bar
                closes = [b.close for b in yes_agg_regime.bars]
                if len(closes) >= 2 and closes[-2] > 0:
                    import math as _math
                    lr = _math.log(closes[-1] / closes[-2])
                else:
                    lr = 0.0
                regime_det.update(
                    log_return=lr,
                    ewma_vol=yes_agg_regime.rolling_volatility_ewma,
                    ew_vol=yes_agg_regime.rolling_volatility,
                )
                # SI-3: record return for cross-market signal generator
                if self._multicore_enabled and self._process_manager is not None:
                    try:
                        self._process_manager.pce_input_queue.put_nowait(
                            ("bar_return", market_info.condition_id, lr,
                             market_info.yes_token_id, market_info.no_token_id),
                        )
                    except _queue_mod.Full:
                        self._pce_queue_drops += 1
                        if self._pce_queue_drops % 100 == 1:
                            log.warning("queue_full_drop", queue="pce_input", total_drops=self._pce_queue_drops)
                elif self._cross_market is not None:
                    self._cross_market.record_return(
                        market_info.condition_id,
                        lr,
                        yes_asset_id=market_info.yes_token_id,
                        no_asset_id=market_info.no_token_id,
                    )

            sig = detector.evaluate(bar, no_best_ask=no_best_ask, whale_confluence=whale)
            _regime_score = regime_det.regime_score if regime_det else 0.5
            if sig:
                # SI-6: Meta-strategy controller — regime-weighted master switch
                meta_decision = self._meta_controller.evaluate("panic", _regime_score)
                if meta_decision.vetoed:
                    self._record_single_name_rejection("panic", market_info, "meta_controller_veto", log_event=False)
                    log.info(
                        "meta_controller_veto",
                        market_id=market_info.condition_id[:16],
                        signal_type="panic",
                        regime_score=round(_regime_score, 3),
                        reason=meta_decision.veto_reason,
                    )
                elif not self.positions.is_stop_loss_cooled_down(market_info.condition_id):
                    self._record_single_name_rejection("panic", market_info, "stop_loss_cooldown", log_event=False)
                    log.info(
                        "stop_loss_cooldown_suppressed",
                        market_id=market_info.condition_id[:16],
                    )
                elif not self.positions.is_panic_loss_cooled_down(market_info.condition_id):
                    self._record_single_name_rejection("panic", market_info, "panic_post_loss_cooldown", log_event=False)
                    log.info(
                        "panic_post_loss_cooldown_suppressed",
                        market_id=market_info.condition_id[:16],
                    )
                else:
                    self._latest_z = sig.zscore
                    self.lifecycle.record_signal(market_info.condition_id)
                    await self._on_panic_signal(
                        sig, no_agg, market_info,
                        signal_metadata={
                            "regime_mean_revert": (
                                regime_det.is_mean_revert if regime_det else False
                            ),
                            "meta_weight": meta_decision.weight,
                        },
                    )
            else:
                self._record_single_name_rejection("panic", market_info, "no_signal", log_event=False)

            # ── V3: Drift signal (low-volatility mean-reversion) ───────────
            # Only evaluate when PanicDetector did NOT fire — ensures
            # uncorrelated signal source.  Requires MR regime.
            if not sig and False:
                drift_det = self._drift_detectors.get(market_info.condition_id)
                if drift_det and no_agg:
                    # Drift cooldown
                    now_drift = time.time()
                    last_drift = self._drift_cooldowns.get(market_info.condition_id, 0.0)
                    if now_drift - last_drift >= 60.0:
                        # Check L2 reliability
                        l2_no = self._l2_books.get(market_info.no_token_id)
                        l2_ok = l2_no is None or l2_no.is_reliable

                        drift_sig = drift_det.evaluate(
                            no_agg,
                            no_asset_id=market_info.no_token_id,
                            no_best_ask=no_best_ask,
                            regime_is_mean_revert=(
                                regime_det.is_mean_revert if regime_det else False
                            ),
                            l2_reliable=l2_ok,
                        )
                        if drift_sig and drift_sig.direction == "BUY_NO":
                            drift_meta = self._meta_controller.evaluate(
                                "drift", _regime_score,
                            )
                            if drift_meta.vetoed:
                                log.info(
                                    "meta_controller_veto",
                                    market_id=market_info.condition_id[:16],
                                    signal_type="drift",
                                    regime_score=round(_regime_score, 3),
                                    reason=drift_meta.veto_reason,
                                )
                            elif not self.positions.is_stop_loss_cooled_down(market_info.condition_id):
                                log.info(
                                    "stop_loss_cooldown_suppressed_drift",
                                    market_id=market_info.condition_id[:16],
                                )
                            else:
                                self._drift_cooldowns[market_info.condition_id] = now_drift
                                self.lifecycle.record_signal(market_info.condition_id)
                                await self._on_panic_signal(
                                    drift_sig, no_agg, market_info,
                                    signal_metadata={
                                        "signal_source": "drift",
                                        "source": "drift",
                                        "drift_score": drift_sig.score,
                                        "regime_mean_revert": True,
                                        "spread_compressed": False,
                                        "meta_weight": drift_meta.weight,
                                    },
                                )

        # ── RPE evaluation (Pillar 14) ───────────────────────────────────
        rpe = self._rpe
        if self._rpe_lane_enabled() and rpe and price_in_band:
            # Deliverable D: Data freshness gate
            # Use last_trade_time (updated on every trade) instead of
            # bars[-1].open_time (only updated when a bar closes).
            # This prevents false stale-data rejections between bars
            # on healthy markets with moderate trade frequency.
            yes_agg_rpe = self._yes_aggs.get(market_info.yes_token_id)
            max_age = settings.strategy.rpe_max_data_age_seconds
            if yes_agg_rpe:
                if yes_agg_rpe.last_trade_time <= 0:
                    self._record_single_name_rejection("rpe", market_info, "no_trade_data", log_event=False)
                    # No trades received at all — genuinely stale
                    log.info(
                        "rpe_no_trade_data",
                        market=market_info.condition_id,
                    )
                    return
                data_age = time.time() - yes_agg_rpe.last_trade_time
                if data_age > max_age:
                    self._record_single_name_rejection("rpe", market_info, "stale_data", log_event=False)
                    log.info(
                        "rpe_stale_data",
                        market=market_info.condition_id,
                        data_age_s=round(data_age, 1),
                        max_age_s=max_age,
                    )
                    return

            days = 30
            total_duration_days = 90.0
            if market_info.end_date:
                days = max(1, (market_info.end_date - datetime.now(timezone.utc)).days)
                # Estimate total market duration from creation to end.
                # Polymarket markets typically run 30-180 days.  If we
                # don't know the creation date, assume 90 days total.
                total_duration_days = max(float(days), 90.0)

            # Thread L2 order-book data into RPE for continuous
            # observation updates (Dynamic Prior Generation Engine).
            no_book_rpe = self._book_trackers.get(market_info.no_token_id)
            book_ratio = None
            l2_ok = False
            if no_book_rpe and no_book_rpe.has_data:
                book_ratio = no_book_rpe.book_depth_ratio
                l2_ok = no_book_rpe.is_reliable

            try:
                rpe_signal = rpe.evaluate(
                    market=market_info,
                    market_price=yes_price,
                    days_to_resolution=days,
                    total_duration_days=total_duration_days,
                    book_depth_ratio=book_ratio,
                    l2_reliable=l2_ok,
                )
                if rpe_signal:
                    await self._on_rpe_signal(rpe_signal, market_info, days, current_price=yes_price)
                else:
                    self._record_single_name_rejection("rpe", market_info, "no_signal", log_event=False)
            except Exception:
                log.warning("rpe_evaluation_error", market=market_info.condition_id, exc_info=True)

    async def _on_panic_signal(
        self, sig: BaseSignal, no_agg: OHLCVAggregator, market: MarketInfo,
        signal_metadata: dict | None = None,
    ) -> None:
        """Handle a confirmed entry signal routed into PositionManager."""
        strategy_source = (signal_metadata or {}).get("signal_source") or getattr(sig, "signal_source", "") or "panic"
        if not self._ensemble_allows_entry(
            strategy_source=strategy_source,
            market_id=market.condition_id,
            direction="NO",
            log_event="ensemble_risk_blocked_bot_signal",
            extra={"signal_type": strategy_source},
        ):
            self._record_single_name_rejection(strategy_source, market, "ensemble_veto", log_event=False)
            return

        # Fire-and-forget — Telegram notification must NOT block the
        # alpha-critical execution path (saves 50-500ms per trade).
        if isinstance(sig, PanicSignal):
            _notify_zscore = sig.zscore
            _notify_vratio = sig.volume_ratio
        elif isinstance(sig, OFIMomentumSignal):
            _notify_zscore = abs(sig.rolling_vi or sig.current_vi)
            _notify_vratio = 1.0
        else:
            _notify_zscore = abs(getattr(sig, "displacement", 0.0))
            _notify_vratio = 1.0
        asyncio.ensure_future(self.telegram.notify_signal(
            sig.market_id, _notify_zscore, _notify_vratio
        ))

        # Compute days to resolution
        days = 30
        if market.end_date:
            days = max(1, (market.end_date - datetime.now(timezone.utc)).days)

        # Get real book depth ratio if available
        book_depth = 1.0
        no_book = self._book_trackers.get(market.no_token_id)
        if no_book and no_book.has_data:
            book_depth = no_book.book_depth_ratio

            # ── Minimum ask-depth gate ─────────────────────────────────
            snap = no_book.snapshot()
            min_depth = (
                settings.strategy.panic_min_ask_depth_usd
                if strategy_source == "panic"
                else settings.strategy.min_ask_depth_usd
            )
            if snap.ask_depth_usd < min_depth:
                self._record_single_name_rejection(strategy_source, market, "thin_asks", log_event=False)
                log.info(
                    "panic_rejected_thin_asks",
                    market=market.condition_id,
                    ask_depth_usd=round(snap.ask_depth_usd, 2),
                    min_required=min_depth,
                    strategy=strategy_source,
                )
                return

            if strategy_source == "panic":
                spread_cents = float(getattr(no_book, "spread_cents", 0.0) or 0.0)
                max_spread_cents = float(settings.strategy.panic_max_spread_cents)
                if max_spread_cents > 0 and spread_cents > max_spread_cents:
                    self._record_single_name_rejection(strategy_source, market, "wide_spread", log_event=False)
                    log.info(
                        "panic_rejected_wide_spread",
                        market=market.condition_id,
                        spread_cents=round(spread_cents, 2),
                        max_allowed=max_spread_cents,
                    )
                    return

                min_support_ratio = float(settings.strategy.panic_min_book_depth_ratio)
                if min_support_ratio > 0 and book_depth < min_support_ratio:
                    self._record_single_name_rejection(strategy_source, market, "weak_bid_support", log_event=False)
                    log.info(
                        "panic_rejected_weak_bid_support",
                        market=market.condition_id,
                        book_depth_ratio=round(book_depth, 3),
                        min_required=min_support_ratio,
                    )
                    return

        # Fetch fee rates for this token
        # Determine if market is fee-enabled based on category
        fee_enabled = self._is_fee_enabled(market)

        if strategy_source == "panic" and settings.strategy.panic_shadow_runtime_enabled():
            self._open_panic_shadow_position(
                sig,
                no_agg,
                market,
                fee_enabled=fee_enabled,
                days_to_resolution=days,
            )
            return

        pos = await self.positions.open_position(
            sig,
            no_agg,
            no_book=no_book,
            event_id=market.event_id,
            days_to_resolution=days,
            book_depth_ratio=book_depth,
            fee_enabled=fee_enabled,
            signal_metadata=signal_metadata,
        )
        if pos:
            if pos.signal_type == "ofi_momentum":
                log.info(
                    "momentum_signal_routed_taker",
                    pos_id=pos.id,
                    market=market.condition_id,
                    entry=pos.entry_price,
                    target=pos.target_price,
                    stop=pos.stop_price,
                    max_hold_seconds=pos.max_hold_seconds,
                )
                return

            # Launch entry chaser as a child task (Pillar 1)
            # Panic signals use urgent=True to escalate immediately
            # and capture the edge before it decays.
            if no_book and no_book.has_data:
                chaser_task = asyncio.create_task(
                    self._entry_chaser_flow(pos, no_book, urgent=True),
                    name=f"chaser_entry_{pos.id}",
                )
                chaser_task.add_done_callback(_safe_task_done_callback)
                pos.entry_chaser_task = chaser_task
            # else: order already placed directly by PositionManager

        if pos is None:
            rejection_reason = self.positions.pop_last_entry_rejection_reason(strategy_source, market.condition_id)
            self._record_single_name_rejection(
                strategy_source,
                market,
                rejection_reason or "position_manager_rejected",
                log_event=False,
            )

        if pos and self.positions.circuit_breaker_active:
            await self.telegram.send(
                "🛑 <b>Circuit breaker tripped</b> — pausing new positions."
            )

    async def _on_rpe_signal(
        self,
        signal: SignalResult,
        market: MarketInfo,
        days_to_resolution: int,
        *,
        current_price: float | None = None,
    ) -> None:
        """Handle an RPE divergence signal — may be shadow or live.

        Shadow mode control flow
        ────────────────────────
        RPE shadow mode is controlled by the ``RPE_SHADOW_MODE`` env var
        (default ``False`` → **live** mode).  When ``rpe_shadow_mode=True``:

        1. The RPE still evaluates markets normally and fires signals.
        2. Signals pass through EQS filtering and cooldown checks.
        3. Signals are recorded in ``RPECalibrationTracker`` with
           ``shadow=True`` for offline Brier/log-loss scoring.
        4. A Telegram notification is sent (prefixed "👻 SHADOW").
        5. **No position is opened** — the method returns early at the
           ``if shadow: return`` gate below the notification block.

        This allows the operator to collect calibration data on RPE
        accuracy without risking real (or paper) capital, then switch
        to live by setting ``RPE_SHADOW_MODE=false``.
        """
        meta = signal.metadata
        direction = meta.get("direction", "buy_no")
        model_prob = meta.get("model_probability", 0.5)
        confidence = meta.get("confidence", 0.0)
        shadow = meta.get("shadow_mode", True)
        strat = settings.strategy

        # Fix 3: Use actual market price, not the normalised divergence score
        display_price = current_price if current_price is not None else meta.get("market_price", signal.score)

        # ── Deliverable C: Per-market cooldown ─────────────────────────
        now = time.monotonic()
        last_fire = self._rpe_last_signal.get(market.condition_id, 0.0)
        if now - last_fire < strat.rpe_cooldown_seconds:
            self._record_single_name_rejection("rpe", market, "signal_cooldown", log_event=False)
            log.debug(
                "rpe_cooldown_active",
                market=market.condition_id,
                seconds_remaining=round(strat.rpe_cooldown_seconds - (now - last_fire), 1),
            )
            return

        # ── Deliverable B: EQS gate (all paths, before Telegram) ────────
        # Determine entry price for EQS calculation
        yes_agg = self._yes_aggs.get(market.yes_token_id)
        no_agg = self._no_aggs.get(market.no_token_id)
        if direction == "buy_no":
            book = self._book_trackers.get(market.no_token_id)
            if book and book.has_data:
                snap = book.snapshot()
                eqs_entry = snap.best_ask - 0.01 if snap.best_ask > 0 else 0.0
            else:
                eqs_entry = no_agg.current_price if no_agg else 0.0
        else:
            book = self._book_trackers.get(market.yes_token_id)
            if book and book.has_data:
                snap = book.snapshot()
                eqs_entry = snap.best_ask - 0.01 if snap.best_ask > 0 else 0.0
            else:
                eqs_entry = yes_agg.current_price if yes_agg else 0.0

        if eqs_entry > 0:
            fee_enabled = self._is_fee_enabled(market)

            edge = compute_edge_score(
                entry_price=eqs_entry,
                no_vwap=model_prob,
                zscore=0.0,
                volume_ratio=0.0,
                whale_confluence=False,
                fee_enabled=fee_enabled,
                model_confidence=confidence,
                min_score=strat.rpe_min_eqs,
            )
            log.debug(
                "rpe_eqs_assessment",
                market=market.condition_id,
                eqs=edge.score,
                viable=edge.viable,
                regime=edge.regime_quality,
                fee_eff=edge.fee_efficiency,
                signal_q=edge.signal_quality,
                reason=edge.rejection_reason,
            )
            if not edge.viable:
                self._record_single_name_rejection("rpe", market, "eqs_rejected", log_event=False)
                log.info(
                    "rpe_eqs_rejected",
                    market=market.condition_id,
                    score=edge.score,
                    min_required=strat.rpe_min_eqs,
                    reason=edge.rejection_reason,
                    direction=direction,
                    confidence=round(confidence, 3),
                )
                # Do NOT stamp the cooldown on EQS rejection — market
                # conditions (price, spread, depth) can change rapidly and
                # we want to re-evaluate as soon as a new bar arrives.
                # The per-source cooldown on *successful* fires prevents
                # log spam from repeated passes.
                return

        # ── Record cooldown timestamp (after EQS passes) ─────────────
        self._rpe_last_signal[market.condition_id] = now

        # ── Record in calibration tracker (Deliverable A) ────────────
        model_meta = meta.get("model_metadata", {})
        if self._rpe_calibration is not None:
            self._rpe_calibration.record_signal(
                market_id=market.condition_id,
                model_prob=model_prob,
                market_price=display_price,
                direction=direction,
                timestamp=time.time(),
                shadow=shadow,
                prior_source=model_meta.get("prior_source", ""),
                l2_active=model_meta.get("l2_active", False),
                theta_w_prior=model_meta.get("w_prior", 0.0),
            )

        # ── Telegram notification (with calibration footer) ───────────
        cal_footer = ""
        cal_stats = (
            self._rpe_calibration.calibration_summary()
            if self._rpe_calibration is not None
            else {}
        )
        if cal_stats.get("resolved", 0) >= 20:
            brier = cal_stats.get("brier_score", "n/a")
            logloss = cal_stats.get("log_loss", "n/a")
            acc = cal_stats.get("direction_accuracy", "n/a")
            cal_footer = f"\nCalibration (N={cal_stats['resolved']}): Brier={brier} LL={logloss} Acc={acc}"

        # Fire-and-forget — keep Telegram off the execution hot path.
        asyncio.ensure_future(self.telegram.notify_rpe_signal(
            market_id=market.condition_id,
            model_prob=model_prob,
            market_price=display_price,
            direction=direction,
            confidence=confidence,
            shadow=shadow,
            question=market.question,
            calibration_footer=cal_footer,
        ))

        # Record in data recorder for backtesting
        if self._recorder is not None:
            self._recorder.enqueue("rpe_signal", {
                "market": market.condition_id,
                "asset_id": market.yes_token_id,
                "model_probability": model_prob,
                "confidence": confidence,
                "direction": direction,
                "shadow": shadow,
                "model_metadata": meta.get("model_metadata", {}),
            })

        if shadow:
            log.info(
                "rpe_shadow_signal",
                market=market.condition_id,
                model_prob=round(model_prob, 4),
                confidence=round(confidence, 3),
                direction=direction,
            )
            return

        # ── Live-mode safety gates (conservative RPE activation) ───────
        # Gate 1: Require higher confidence than the base threshold
        if confidence < 0.15:
            self._record_single_name_rejection("rpe", market, "confidence_too_low", log_event=False)
            log.info(
                "rpe_live_confidence_too_low",
                market=market.condition_id,
                confidence=round(confidence, 3),
                min_required=0.15,
            )
            return

        # Gate 2: Require >= 10 OHLCV bars of history for meaningful data
        bar_count = max(
            len(yes_agg.bars) if yes_agg else 0,
            len(no_agg.bars) if no_agg else 0,
        )
        if bar_count < 10:
            self._record_single_name_rejection("rpe", market, "insufficient_bars", log_event=False)
            log.info(
                "rpe_live_insufficient_bars",
                market=market.condition_id,
                bar_count=bar_count,
                min_required=10,
            )
            return

        # ── Deliverable C: Position deduplication ────────────────────
        positioned = self._positioned_asset_ids()
        if market.yes_token_id in positioned or market.no_token_id in positioned:
            self._record_single_name_rejection("rpe", market, "already_positioned", log_event=False)
            log.info(
                "rpe_already_positioned",
                market=market.condition_id,
                yes_token=market.yes_token_id,
                no_token=market.no_token_id,
            )
            return

        # Live mode — open a position
        if not fee_enabled:
            fee_enabled = self._is_fee_enabled(market)

        # Determine entry price from the appropriate book
        if direction == "buy_no":
            book = self._book_trackers.get(market.no_token_id)
            if book and book.has_data:
                snap = book.snapshot()
                entry_price = round(snap.best_ask - 0.01, 2) if snap.best_ask > 0 else 0.0
            else:
                entry_price = round(no_agg.current_price, 2) if no_agg else 0.0
        else:
            book = self._book_trackers.get(market.yes_token_id)
            if book and book.has_data:
                snap = book.snapshot()
                entry_price = round(snap.best_ask - 0.01, 2) if snap.best_ask > 0 else 0.0
            else:
                entry_price = round(yes_agg.current_price, 2) if yes_agg else 0.0

        if entry_price <= 0:
            self._record_single_name_rejection("rpe", market, "invalid_entry_price", log_event=False)
            return

        # ── Minimum ask-depth gate (RPE path) ────────────────────────
        if book and book.has_data:
            snap_depth = book.snapshot()
            min_depth = settings.strategy.min_ask_depth_usd
            if snap_depth.ask_depth_usd < min_depth:
                self._record_single_name_rejection("rpe", market, "thin_asks", log_event=False)
                log.info(
                    "rpe_rejected_thin_asks",
                    market=market.condition_id,
                    ask_depth_usd=round(snap_depth.ask_depth_usd, 2),
                    min_required=min_depth,
                    direction=direction,
                )
                return

        # ── SI-6: Meta-strategy controller for RPE ───────────────────
        rpe_regime_det = self._regime_detectors.get(market.condition_id)
        _rpe_regime_score = rpe_regime_det.regime_score if rpe_regime_det else 0.5
        rpe_meta = self._meta_controller.evaluate("rpe", _rpe_regime_score)
        if rpe_meta.vetoed:
            self._record_single_name_rejection("rpe", market, "meta_controller_veto", log_event=False)
            log.info(
                "meta_controller_veto",
                market_id=market.condition_id[:16],
                signal_type="rpe",
                regime_score=round(_rpe_regime_score, 3),
                reason=rpe_meta.veto_reason,
            )
            return

        rpe_signal_meta = dict(meta)
        rpe_signal_meta["meta_weight"] = rpe_meta.weight

        # ── Fast-Strike: pass latency health for taker path ──────
        _lat_healthy = (
            self.latency_guard.state == LatencyState.HEALTHY
        )

        pos = await self.positions.open_rpe_position(
            market_id=market.condition_id,
            yes_asset_id=market.yes_token_id,
            no_asset_id=market.no_token_id,
            direction=direction,
            model_probability=model_prob,
            confidence=confidence,
            entry_price=entry_price,
            event_id=market.event_id,
            days_to_resolution=days_to_resolution,
            fee_enabled=fee_enabled,
            book=book,
            signal_metadata=rpe_signal_meta,
            latency_healthy=_lat_healthy,
        )

        # ── Model-only probe routing (Dynamic Prior Engine) ──────────
        # When GenericBayesianModel fires a massive divergence but
        # PanicDetector is silent, route through V4 probe sizing for
        # bounded-risk model exploration.
        if pos is None and meta.get("model_name") == "generic_bayesian":
            model_meta = meta.get("model_metadata", {})
            divergence_abs = abs(meta.get("divergence", 0.0))
            probe_min = settings.strategy.rpe_probe_divergence_min

            # Check PanicDetector is truly silent for this market
            panic_det = self._detectors.get(market.condition_id)
            panic_silent = True
            if panic_det and no_agg and no_agg.bars:
                latest_bar = no_agg.bars[-1]
                no_book_probe = self._book_trackers.get(market.no_token_id)
                no_ask_probe = 0.0
                if no_book_probe and no_book_probe.has_data:
                    no_ask_probe = no_book_probe.snapshot().best_ask
                try:
                    panic_sig = panic_det.evaluate(latest_bar, no_ask_probe)
                    if panic_sig is not None:
                        panic_silent = False
                except Exception:
                    pass

            if (
                divergence_abs >= probe_min
                and panic_silent
                and model_meta.get("dynamic_prior_enabled", False)
            ):
                probe_meta = dict(meta)
                probe_meta["force_probe"] = True
                probe_meta["probe_reason"] = "model_divergence_no_panic"
                log.info(
                    "rpe_model_probe_attempt",
                    market=market.condition_id,
                    divergence=round(divergence_abs, 4),
                    probe_min=probe_min,
                    prior_source=model_meta.get("prior_source", "unknown"),
                )
                pos = await self.positions.open_rpe_position(
                    market_id=market.condition_id,
                    yes_asset_id=market.yes_token_id,
                    no_asset_id=market.no_token_id,
                    direction=direction,
                    model_probability=model_prob,
                    confidence=confidence,
                    entry_price=entry_price,
                    event_id=market.event_id,
                    days_to_resolution=days_to_resolution,
                    fee_enabled=fee_enabled,
                    book=book,
                    signal_metadata=probe_meta,
                )

        if pos is None:
            self._record_single_name_rejection("rpe", market, "position_manager_rejected", log_event=False)

        if pos:
            # Fast-strike positions skip the chaser — order was placed
            # as a taker to immediately hit stale liquidity.
            if getattr(pos, 'fast_strike', False):
                log.info(
                    "rpe_fast_strike_no_chaser",
                    pos_id=pos.id,
                    market=market.condition_id,
                )
            elif book and book.has_data:
                chaser_task = asyncio.create_task(
                    self._entry_chaser_flow(pos, book),
                    name=f"chaser_entry_{pos.id}",
                )
                chaser_task.add_done_callback(_safe_task_done_callback)
                pos.entry_chaser_task = chaser_task

    async def _rpe_crypto_retrigger_loop(self) -> None:
        """Periodically check if BTC spot has moved enough to re-evaluate RPE.

        Runs every 200ms to maximise latency edge against stale CLOB
        orders.  When the spot price changes by more than
        ``rpe_crypto_retrigger_cents`` since the last evaluation, re-runs
        the RPE for all active crypto-tagged markets.  This catches
        mispricings that appear between 1-minute bar closes.

        Does NOT open a new WebSocket — reuses the same ``price_fn``
        that the CryptoPriceModel already reads from.
        """
        interval_s = 0.2
        while self._running:
            await asyncio.sleep(interval_s)
            try:
                spot = self._get_crypto_spot()
                if spot is None:
                    continue

                threshold = settings.strategy.rpe_crypto_retrigger_cents
                last = self._rpe_last_spot
                if last is not None and abs(spot - last) < threshold:
                    continue

                # Spot moved enough — re-evaluate crypto markets
                self._rpe_last_spot = spot
                for m in list(self._markets):
                    if not self.lifecycle.is_tradeable(m.condition_id):
                        continue
                    # Quick check: is this a crypto market?
                    tags = (getattr(m, "tags", "") or "").lower()
                    question_lower = m.question.lower()
                    is_crypto = (
                        "crypto" in tags
                        or any(
                            kw in question_lower
                            for kw in ("bitcoin", "btc", "ethereum", "eth")
                        )
                    )
                    if not is_crypto:
                        continue

                    # Use latest YES price from aggregator or book
                    yes_agg = self._yes_aggs.get(m.yes_token_id)
                    yes_book = self._book_trackers.get(m.yes_token_id)
                    if yes_book and yes_book.has_data:
                        snap = yes_book.snapshot()
                        yes_price = snap.mid_price if snap.mid_price > 0 else (
                            yes_agg.current_price if yes_agg else 0.0
                        )
                    else:
                        yes_price = yes_agg.current_price if yes_agg else 0.0

                    if not (
                        settings.strategy.min_tradeable_price
                        < yes_price
                        < settings.strategy.max_tradeable_price
                    ):
                        continue

                    # L2 book reliability gate (retrigger path)
                    l2_yes = self._l2_books.get(m.yes_token_id)
                    if l2_yes is not None and not l2_yes.is_reliable:
                        log.info(
                            "l2_book_unreliable",
                            asset_id=m.yes_token_id,
                            seq_gap_rate=round(l2_yes.seq_gap_rate, 4),
                            delta_count=l2_yes.delta_count,
                            path="retrigger",
                        )
                        continue

                    # Deliverable D: Data freshness gate (retrigger path)
                    # Use last_trade_time for accurate freshness.
                    max_age = settings.strategy.rpe_max_data_age_seconds
                    if yes_agg:
                        if yes_agg.last_trade_time <= 0:
                            continue
                        data_age = time.time() - yes_agg.last_trade_time
                        if data_age > max_age:
                            log.info(
                                "rpe_stale_data",
                                market=m.condition_id,
                                data_age_s=round(data_age, 1),
                                max_age_s=max_age,
                                path="retrigger",
                            )
                            continue

                    days = 30
                    total_dur = 90.0
                    if m.end_date:
                        days = max(1, (m.end_date - datetime.now(timezone.utc)).days)
                        total_dur = max(float(days), 90.0)

                    # Thread L2 data for retrigger path
                    no_book_rt = self._book_trackers.get(m.no_token_id)
                    rt_ratio = None
                    rt_l2_ok = False
                    if no_book_rt and no_book_rt.has_data:
                        rt_ratio = no_book_rt.book_depth_ratio
                        rt_l2_ok = no_book_rt.is_reliable

                    rpe_signal = self._rpe.evaluate(
                        market=m,
                        market_price=yes_price,
                        days_to_resolution=days,
                        total_duration_days=total_dur,
                        book_depth_ratio=rt_ratio,
                        l2_reliable=rt_l2_ok,
                    )
                    if rpe_signal:
                        await self._on_rpe_signal(rpe_signal, m, days, current_price=yes_price)

            except asyncio.CancelledError:
                raise
            except Exception:
                log.error("rpe_retrigger_error", exc_info=True)
                if self._retrigger_breaker.record():
                    log.critical(
                        "rpe_retrigger_circuit_breaker_tripped",
                        errors_in_window=self._retrigger_breaker.recent_errors,
                    )
                    await self.telegram.send(
                        "🟡 <b>CIRCUIT BREAKER</b>: rpe_retrigger tripped "
                        "(5 errors in 60s) — suspend & hard-reset WS."
                    )
                    await self._suspend_and_reset()

    # ── SI-8: Oracle Latency Arbitrage ─────────────────────────────────────

    async def _oracle_polling_loop(self) -> None:
        """Poll off-chain oracle APIs and fire fast-strike taker signals.

        Parses ``oracle_market_configs`` JSON from StrategyParams,
        instantiates adapters via the registry, spawns each adapter's
        polling loop as a sub-task, and consumes ``OracleSnapshot``
        objects from the shared queue.

        Each snapshot is evaluated by ``OracleSignalEngine`` against
        the current CLOB price; if a signal fires it is routed into
        ``_on_oracle_signal()`` which calls
        ``PositionManager.open_rpe_position()`` with
        ``latency_healthy=True`` to enable the fast-strike taker path.

        Circuit breaker: ``self._oracle_breaker`` (threshold=5,
        window=60 s).  On trip → ``_suspend_and_reset()``.
        """
        import json as _json

        strat = settings.strategy
        try:
            configs_raw = _json.loads(strat.oracle_market_configs)
        except _json.JSONDecodeError:
            log.error("oracle_market_configs_invalid_json", raw=strat.oracle_market_configs)
            return

        if not configs_raw:
            log.info("oracle_polling_loop_no_configs")
            return

        oracle_queue: asyncio.Queue[OracleSnapshot] = asyncio.Queue(maxsize=500)

        # Instantiate adapters and spawn their polling sub-tasks
        for cfg_dict in configs_raw:
            try:
                mc = OracleMarketConfig(
                    market_id=cfg_dict.get("market_id", ""),
                    oracle_type=cfg_dict.get("oracle_type", ""),
                    oracle_params=cfg_dict.get("oracle_params", {}),
                    external_id=cfg_dict.get(
                        "external_id",
                        cfg_dict.get("oracle_params", {}).get("match_id", ""),
                    ),
                    target_outcome=cfg_dict.get(
                        "target_outcome",
                        cfg_dict.get("oracle_params", {}).get("team", ""),
                    ),
                    market_type=cfg_dict.get(
                        "market_type",
                        cfg_dict.get("oracle_params", {}).get("market_type", "winner"),
                    ),
                    goal_line=float(
                        cfg_dict.get(
                            "goal_line",
                            cfg_dict.get("oracle_params", {}).get("goal_line", 2.5),
                        )
                    ),
                    yes_asset_id=cfg_dict.get("yes_asset_id", ""),
                    no_asset_id=cfg_dict.get("no_asset_id", ""),
                    event_id=cfg_dict.get("event_id", ""),
                )
                adapter = self._oracle_registry.create(
                    mc.oracle_type,
                    mc,
                    on_trip=self._oracle_adapter_tripped,
                )
                task = asyncio.create_task(
                    adapter.start(oracle_queue),
                    name=f"oracle_{mc.oracle_type}_{mc.market_id[:12]}",
                )
                task.add_done_callback(_safe_task_done_callback)
                self._oracle_adapter_tasks.append(task)
                log.info(
                    "oracle_adapter_spawned",
                    oracle_type=mc.oracle_type,
                    market_id=mc.market_id,
                )
            except KeyError:
                log.error(
                    "oracle_adapter_unknown_type",
                    oracle_type=cfg_dict.get("oracle_type", ""),
                )
            except Exception:
                log.error("oracle_adapter_init_error", exc_info=True)

        if not self._oracle_adapter_tasks:
            log.warning("oracle_polling_loop_no_adapters_created")
            return

        log.info(
            "oracle_polling_loop_started",
            adapter_count=len(self._oracle_adapter_tasks),
        )

        # ── Consumer loop ──────────────────────────────────────────────
        while self._running:
            try:
                try:
                    snapshot = await asyncio.wait_for(oracle_queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue

                # Look up oracle config for asset IDs
                oc = self._find_oracle_config(snapshot.market_id, configs_raw)
                if oc is None:
                    continue

                # Look up market info in lifecycle
                market = self._find_market(snapshot.market_id)
                if market is None:
                    log.debug("oracle_market_not_tracked", market_id=snapshot.market_id)
                    continue

                # Gate: market must be tradeable and accepting orders
                if not self.lifecycle.is_tradeable(market.condition_id):
                    continue
                if not market.accepting_orders:
                    continue

                # Get current YES price from book or aggregator
                yes_agg = self._yes_aggs.get(market.yes_token_id)
                yes_price = yes_agg.current_price if yes_agg else 0.0
                if yes_price <= 0:
                    continue

                # Get current spread for spread-width gate
                spread_cents = 0.0
                no_book = self._book_trackers.get(market.no_token_id)
                if no_book and no_book.has_data:
                    snap = no_book.snapshot()
                    spread_cents = (snap.best_ask - snap.best_bid) * 100.0 if snap.best_ask > 0 and snap.best_bid > 0 else 0.0

                # Evaluate signal
                signal = self._oracle_signal_engine.evaluate(
                    snapshot,
                    market_price=yes_price,
                    spread_cents=spread_cents,
                )

                if signal:
                    await self._on_oracle_signal(signal, market, oc)

            except asyncio.CancelledError:
                raise
            except Exception:
                log.error("oracle_polling_loop_error", exc_info=True)
                if self._oracle_breaker.record():
                    log.critical(
                        "oracle_polling_circuit_breaker_tripped",
                        errors_in_window=self._oracle_breaker.recent_errors,
                    )
                    await self.telegram.send(
                        "🟡 <b>CIRCUIT BREAKER</b>: oracle_poll tripped "
                        "(5 errors in 60s) — suspend & hard-reset WS."
                    )
                    await self._suspend_and_reset()
                    return

        # Cleanup adapter sub-tasks on shutdown
        for t in self._oracle_adapter_tasks:
            if not t.done():
                t.cancel()

    async def _oracle_adapter_tripped(self) -> None:
        """Callback invoked when an individual oracle adapter's breaker trips."""
        log.critical("oracle_adapter_breaker_tripped_callback")
        try:
            await self.telegram.send(
                "🟡 <b>Oracle adapter circuit breaker tripped</b> — "
                "adapter stopped, bot continues."
            )
        except Exception:
            pass

    def _find_oracle_config(
        self, market_id: str, configs_raw: list[dict],
    ) -> dict | None:
        """Find the oracle config dict for a given market_id."""
        for cfg in configs_raw:
            if cfg.get("market_id") == market_id:
                return cfg
        return None

    def _find_market(self, market_id: str) -> MarketInfo | None:
        """Find tracked MarketInfo by condition_id."""
        for m in list(self._markets):
            if m.condition_id == market_id:
                return m
        return None

    def _find_market_for_cross_market_signal(
        self,
        market_id: str,
        asset_id: str,
    ) -> MarketInfo | None:
        market = self._find_market(market_id)
        if market is not None:
            return market
        mapped_market = self._market_map.get(asset_id)
        if mapped_market is not None and mapped_market.condition_id == market_id:
            return mapped_market
        return None

    def _get_reliable_book_bbo(self, asset_id: str) -> tuple[float, float, str | None]:
        book = self._book_trackers.get(asset_id)
        if book is None or not getattr(book, "has_data", False):
            return 0.0, 0.0, "missing_book"

        l2_book = self._l2_books.get(asset_id)
        if l2_book is not None and not l2_book.is_reliable:
            return 0.0, 0.0, "l2_unreliable"

        try:
            snap = book.snapshot()
        except Exception:
            return 0.0, 0.0, "snapshot_failed"

        best_bid = float(getattr(snap, "best_bid", 0.0) or 0.0)
        best_ask = float(getattr(snap, "best_ask", 0.0) or 0.0)
        if best_bid <= 0.0 or best_ask <= 0.0:
            return 0.0, 0.0, "missing_bbo"
        if best_bid > best_ask:
            return 0.0, 0.0, "crossed_book"
        return best_bid, best_ask, None

    def _plan_panic_shadow_trade(
        self,
        *,
        signal: BaseSignal,
        no_agg: OHLCVAggregator,
        no_book: OrderbookTracker | None,
        days_to_resolution: int,
        fee_enabled: bool,
    ) -> tuple[dict[str, float], str | None]:
        """Mirror the live panic plan without opening capital."""
        strat = settings.strategy
        entry_price = round(float(getattr(signal, "no_best_ask", 0.0) or 0.0) - 0.01, 2)
        if entry_price <= 0.0 or entry_price >= 1.0:
            return {}, "invalid_entry_price"

        actual_depth_ratio = 1.0
        if no_book is not None and no_book.has_data:
            actual_depth_ratio = float(getattr(no_book, "book_depth_ratio", 1.0) or 1.0)

        iceberg_active = False
        iceberg_detector = self._iceberg_detectors.get(getattr(signal, "no_asset_id", ""))
        if iceberg_detector is not None:
            strongest = iceberg_detector.strongest_iceberg("BUY")
            if (
                strongest is not None
                and strongest.confidence >= strat.iceberg_peg_min_confidence
            ):
                iceberg_active = True

        entry_fee_frac = get_fee_rate(entry_price, fee_enabled=fee_enabled)
        entry_fee_bps = round(entry_fee_frac * 10000)
        tp = compute_take_profit(
            entry_price=entry_price,
            no_vwap=no_agg.rolling_vwap,
            realised_vol=no_agg.rolling_volatility,
            whale_confluence=bool(getattr(signal, "whale_confluence", False)),
            iceberg_active=iceberg_active,
            book_depth_ratio=actual_depth_ratio,
            days_to_resolution=days_to_resolution,
            entry_fee_bps=entry_fee_bps,
            exit_fee_bps=0,
            fee_enabled=fee_enabled,
        )
        if not tp.viable:
            return {}, "low_spread"

        exec_mode = "maker" if strat.maker_routing_enabled else "taker"
        if exec_mode == "maker":
            min_viable_spread = float(strat.desired_margin_cents)
        else:
            exit_fee_cents = get_fee_rate(tp.target_price, fee_enabled=True) * 100.0
            min_viable_spread = (
                2.0 * float(strat.paper_slippage_cents)
                + entry_fee_frac * 100.0
                + exit_fee_cents
                + float(strat.desired_margin_cents)
            )
        if tp.spread_cents < min_viable_spread:
            return {
                "target_price": tp.target_price,
                "min_viable_spread": float(min_viable_spread),
            }, "insufficient_edge"

        sl_trigger = compute_adaptive_stop_loss_cents(
            sl_base_cents=strat.stop_loss_cents,
            entry_price=entry_price,
            fee_enabled=fee_enabled,
            ewma_vol=no_agg.rolling_downside_vol_ewma or None,
            ref_vol=strat.sl_vol_ref,
            is_adaptive=strat.sl_vol_adaptive,
            max_multiplier=strat.sl_vol_multiplier_max,
        )
        stop_price = round(max(0.01, entry_price - sl_trigger / 100.0), 4)
        expected_net_target_per_share_cents = compute_net_pnl_cents(
            entry_price=entry_price,
            exit_price=tp.target_price,
            size=1.0,
            fee_enabled=fee_enabled,
            is_maker_entry=(exec_mode == "maker"),
            is_maker_exit=(exec_mode == "maker"),
        )
        expected_net_target_minus_one_tick_per_share_cents = compute_net_pnl_cents(
            entry_price=entry_price,
            exit_price=max(0.01, round(tp.target_price - 0.01, 4)),
            size=1.0,
            fee_enabled=fee_enabled,
            is_maker_entry=(exec_mode == "maker"),
            is_maker_exit=(exec_mode == "maker"),
        )
        if (
            expected_net_target_per_share_cents
            < float(strat.panic_min_expected_net_target_per_share_cents)
            or expected_net_target_minus_one_tick_per_share_cents
            < float(strat.panic_min_expected_net_target_minus_one_tick_per_share_cents)
        ):
            return {
                "target_price": tp.target_price,
                "expected_net_target_per_share_cents": (
                    expected_net_target_per_share_cents
                ),
                "expected_net_target_minus_one_tick_per_share_cents": (
                    expected_net_target_minus_one_tick_per_share_cents
                ),
            }, "insufficient_expected_edge"

        return {
            "entry_price": entry_price,
            "target_price": tp.target_price,
            "stop_price": stop_price,
            "sl_trigger_cents": float(sl_trigger),
            "expected_net_target_per_share_cents": (
                expected_net_target_per_share_cents
            ),
            "expected_net_target_minus_one_tick_per_share_cents": (
                expected_net_target_minus_one_tick_per_share_cents
            ),
        }, None

    def _open_panic_shadow_position(
        self,
        signal: BaseSignal,
        no_agg: OHLCVAggregator,
        market: MarketInfo,
        *,
        fee_enabled: bool,
        days_to_resolution: int,
    ) -> bool:
        best_bid, best_ask, rejection_reason = self._get_reliable_book_bbo(
            getattr(signal, "no_asset_id", "")
        )
        if rejection_reason is not None:
            self._record_single_name_rejection(
                "panic",
                market,
                rejection_reason,
                log_event=False,
            )
            log.info(
                "panic_shadow_rejected",
                market_id=market.condition_id,
                asset_id=getattr(signal, "no_asset_id", ""),
                reason=rejection_reason,
            )
            return False

        no_book = self._book_trackers.get(getattr(signal, "no_asset_id", ""))
        plan, rejection_reason = self._plan_panic_shadow_trade(
            signal=signal,
            no_agg=no_agg,
            no_book=no_book,
            days_to_resolution=days_to_resolution,
            fee_enabled=fee_enabled,
        )
        if rejection_reason is not None:
            self._record_single_name_rejection(
                "panic",
                market,
                rejection_reason,
                log_event=False,
            )
            log.info(
                "panic_shadow_rejected",
                market_id=market.condition_id,
                asset_id=getattr(signal, "no_asset_id", ""),
                reason=rejection_reason,
                target_price=round(float(plan.get("target_price", 0.0) or 0.0), 4),
                expected_target_cents=round(
                    float(plan.get("expected_net_target_per_share_cents", 0.0) or 0.0),
                    4,
                ),
                expected_target_minus_one_tick_cents=round(
                    float(
                        plan.get(
                            "expected_net_target_minus_one_tick_per_share_cents", 0.0
                        )
                        or 0.0
                    ),
                    4,
                ),
            )
            return False

        entry_size = round(
            float(settings.strategy.max_trade_size_usd)
            / max(float(plan["entry_price"]), 0.01),
            2,
        )
        if entry_size < 1.0:
            self._record_single_name_rejection(
                "panic",
                market,
                "insufficient_size",
                log_event=False,
            )
            log.info(
                "panic_shadow_rejected",
                market_id=market.condition_id,
                asset_id=getattr(signal, "no_asset_id", ""),
                reason="insufficient_size",
            )
            return False

        zscore = abs(float(getattr(signal, "zscore", 0.0) or 0.0))
        threshold = max(1e-9, float(settings.strategy.panic_zscore_threshold))
        confidence = min(1.0, max(0.0, (zscore - threshold) / threshold))
        reference_price = float(getattr(signal, "yes_price", 0.0) or 0.0)

        pos = self._shadow_tracker.open_shadow_position(
            signal_source=_PANIC_SHADOW_SOURCE,
            market_id=market.condition_id,
            asset_id=getattr(signal, "no_asset_id", ""),
            direction="NO",
            best_ask=best_ask,
            best_bid=best_bid,
            entry_price=float(plan["entry_price"]),
            target_price=float(plan["target_price"]),
            stop_price=float(plan["stop_price"]),
            entry_size=entry_size,
            fee_enabled=fee_enabled,
            zscore=zscore,
            confidence=confidence,
            reference_price=reference_price,
            toxicity_index=0.0,
        )
        if pos is None:
            self._record_single_name_rejection(
                "panic",
                market,
                "invalid_entry_price",
                log_event=False,
            )
            log.info(
                "panic_shadow_rejected",
                market_id=market.condition_id,
                asset_id=getattr(signal, "no_asset_id", ""),
                reason="invalid_entry_price",
            )
            return False

        log.info(
            "panic_shadow_opened",
            market_id=market.condition_id,
            asset_id=getattr(signal, "no_asset_id", ""),
            entry_price=round(float(plan["entry_price"]), 4),
            target_price=round(float(plan["target_price"]), 4),
            stop_price=round(float(plan["stop_price"]), 4),
            entry_size=round(entry_size, 4),
            zscore=round(zscore, 6),
            confidence=round(confidence, 6),
            expected_target_cents=round(
                float(plan["expected_net_target_per_share_cents"]), 4
            ),
            expected_target_minus_one_tick_cents=round(
                float(plan["expected_net_target_minus_one_tick_per_share_cents"]),
                4,
            ),
        )
        return True

    async def _consume_cross_market_signals(
        self,
        signals: list[dict[str, Any] | CrossMarketSignal],
    ) -> None:
        opened = 0
        rejected = 0

        for signal in signals:
            if self._open_cross_market_shadow_position(signal):
                opened += 1
            else:
                rejected += 1

        log.info(
            "cross_market_shadow_batch_processed",
            received=len(signals),
            opened=opened,
            rejected=rejected,
        )

    def _open_cross_market_shadow_position(
        self,
        signal: dict[str, Any] | CrossMarketSignal,
    ) -> bool:
        def _value(key: str, default: Any = "") -> Any:
            if isinstance(signal, dict):
                return signal.get(key, default)
            return getattr(signal, key, default)

        lagging_market_id = str(_value("lagging_market_id", "") or "")
        direction = str(_value("direction", "") or "").upper()
        asset_id = str(_value("lagging_asset_id", "") or "")
        signal_source = str(_value("signal_source", _CROSS_MARKET_SHADOW_SOURCE) or _CROSS_MARKET_SHADOW_SOURCE)
        zscore = float(_value("z_score", 0.0) or 0.0)
        confidence = float(_value("confidence", 0.0) or 0.0)

        market = self._find_market_for_cross_market_signal(lagging_market_id, asset_id)
        if not asset_id and market is not None:
            asset_id = market.yes_token_id if direction == "YES" else market.no_token_id

        if not lagging_market_id and market is not None:
            lagging_market_id = market.condition_id

        if direction not in {"YES", "NO"} or not lagging_market_id or not asset_id:
            if market is not None:
                self._record_single_name_rejection("cross_market", market, "invalid_signal_payload", log_event=False)
            log.info(
                "cross_market_shadow_rejected",
                market_id=lagging_market_id,
                asset_id=asset_id,
                direction=direction,
                reason="invalid_signal_payload",
            )
            return False

        best_bid, best_ask, rejection_reason = self._get_reliable_book_bbo(asset_id)
        if rejection_reason is not None:
            if market is not None:
                self._record_single_name_rejection("cross_market", market, rejection_reason, log_event=False)
            log.info(
                "cross_market_shadow_rejected",
                market_id=lagging_market_id,
                asset_id=asset_id,
                direction=direction,
                reason=rejection_reason,
            )
            return False

        pos = self._shadow_tracker.open_shadow_position(
            signal_source=signal_source,
            market_id=lagging_market_id,
            asset_id=asset_id,
            direction=direction,
            best_ask=best_ask,
            best_bid=best_bid,
            zscore=zscore,
            confidence=confidence,
        )
        if pos is None:
            if market is not None:
                self._record_single_name_rejection("cross_market", market, "invalid_entry_price", log_event=False)
            log.info(
                "cross_market_shadow_rejected",
                market_id=lagging_market_id,
                asset_id=asset_id,
                direction=direction,
                reason="invalid_entry_price",
            )
            return False

        return True

    def _open_ofi_reverse_shadow_position(
        self,
        market: MarketInfo,
        sig: OFIMomentumSignal,
        *,
        reference_price: float,
    ) -> bool:
        if not settings.strategy.ofi_reverse_shadow_enabled:
            return False

        min_reference_price = float(settings.strategy.ofi_reverse_shadow_min_reference_price)
        max_reference_price = float(settings.strategy.ofi_reverse_shadow_max_reference_price)
        if not (min_reference_price <= reference_price < max_reference_price):
            self._record_single_name_rejection(
                "ofi_reverse_shadow",
                market,
                "reference_price_out_of_band",
                log_event=False,
            )
            return False

        asset_id = market.yes_token_id
        best_bid, best_ask, rejection_reason = self._get_reliable_book_bbo(asset_id)
        if rejection_reason is not None:
            self._record_single_name_rejection(
                "ofi_reverse_shadow",
                market,
                rejection_reason,
                log_event=False,
            )
            return False

        if best_ask >= 0.97 or best_ask <= 0.03:
            self._record_single_name_rejection(
                "ofi_reverse_shadow",
                market,
                "reverse_near_resolved_price",
                log_event=False,
            )
            return False

        if not (settings.strategy.min_tradeable_price < best_ask < settings.strategy.max_tradeable_price):
            self._record_single_name_rejection(
                "ofi_reverse_shadow",
                market,
                "reverse_price_out_of_band",
                log_event=False,
            )
            return False

        confidence = min(
            max(abs(float(sig.rolling_vi or sig.current_vi or sig.ofi or 0.0)), 0.0),
            1.0,
        )
        entry_size = round(
            float(settings.strategy.max_trade_size_usd) / max(best_ask, 0.01),
            2,
        )
        if entry_size < 1.0:
            self._record_single_name_rejection(
                "ofi_reverse_shadow",
                market,
                "reverse_insufficient_size",
                log_event=False,
            )
            return False

        toxicity_index = 0.0
        book = self._book_trackers.get(asset_id)
        metrics_fn = getattr(book, "toxicity_metrics", None)
        if callable(metrics_fn):
            try:
                metrics = metrics_fn("BUY")
                if isinstance(metrics, dict):
                    toxicity_index = float(metrics.get("toxicity_index", 0.0) or 0.0)
            except Exception:
                log.warning(
                    "ofi_reverse_shadow_toxicity_snapshot_failed",
                    market_id=market.condition_id,
                    asset_id=asset_id,
                    exc_info=True,
                )

        pos = self._shadow_tracker.open_shadow_position(
            signal_source=_OFI_REVERSE_SHADOW_SOURCE,
            market_id=market.condition_id,
            asset_id=asset_id,
            direction="YES",
            best_ask=best_ask,
            best_bid=best_bid,
            entry_price=best_ask,
            entry_size=entry_size,
            fee_enabled=self._is_fee_enabled(market),
            zscore=float(sig.rolling_vi or sig.ofi or 0.0),
            confidence=confidence,
            reference_price=reference_price,
            reference_price_band=f"{min_reference_price:.2f}-{max_reference_price:.2f}",
            toxicity_index=toxicity_index,
        )
        if pos is None:
            self._record_single_name_rejection(
                "ofi_reverse_shadow",
                market,
                "invalid_entry_price",
                log_event=False,
            )
            return False

        log.info(
            "ofi_reverse_shadow_opened",
            market_id=market.condition_id,
            asset_id=asset_id,
            direction="YES",
            reference_price=round(reference_price, 4),
            reference_price_band=f"{min_reference_price:.2f}-{max_reference_price:.2f}",
            entry_price=round(pos.entry_price, 4),
            entry_size=round(pos.entry_size, 4),
            rolling_vi=round(float(sig.rolling_vi or sig.ofi or 0.0), 6),
            confidence=round(confidence, 6),
            toxicity_index=round(toxicity_index, 6),
        )
        return True

    async def _tick_shadow_tracker_for_asset(self, asset_id: str) -> None:
        if self._shadow_tracker.open_count <= 0:
            return

        best_bid, best_ask, rejection_reason = self._get_reliable_book_bbo(asset_id)
        if rejection_reason is not None:
            return

        await self._shadow_tracker.tick(asset_id, best_bid, best_ask)

    def _build_contagion_matrix_entry(
        self,
        signal: ContagionArbSignal,
        market: MarketInfo,
        *,
        book: OrderbookTracker | None,
        agg: OHLCVAggregator | None,
        suppressed: bool = False,
        suppression_reason: str | None = None,
    ) -> dict[str, Any]:
        fair_value = signal.implied_probability
        market_price = signal.lagging_market_price
        best_bid = 0.0
        best_ask = 0.0
        last_trade_age_s = -1.0

        if signal.direction == "buy_no":
            fair_value = max(0.01, min(0.99, 1.0 - signal.implied_probability))
            market_price = max(0.01, min(0.99, 1.0 - signal.lagging_market_price))

        if book is not None and getattr(book, "has_data", False):
            snap = book.snapshot()
            best_bid = float(getattr(snap, "best_bid", 0.0) or 0.0)
            best_ask = float(getattr(snap, "best_ask", 0.0) or 0.0)
            if signal.direction == "buy_no" and best_bid > 0 and best_ask > 0:
                best_bid = max(0.01, min(0.99, 1.0 - float(getattr(snap, "best_ask", 0.0) or 0.0)))
                best_ask = max(0.01, min(0.99, 1.0 - float(getattr(snap, "best_bid", 0.0) or 0.0)))

        if agg is not None and getattr(agg, "last_trade_time", 0.0) > 0:
            last_trade_age_s = max(0.0, time.time() - float(agg.last_trade_time))

        mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else market_price
        cross_spread_slip_cents = max(0.0, (best_ask - mid_price) * 100.0) if best_ask > 0 else 0.0
        edge_cents = (fair_value - (best_ask or market_price)) * 100.0
        leader_market = self._find_market(signal.leading_market_id)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "shadow": signal.is_shadow,
            "suppressed": suppressed,
            "suppression_reason": suppression_reason or "",
            "leader_market_id": signal.leading_market_id,
            "lagging_market_id": signal.lagging_market_id,
            "leader_question": getattr(leader_market, "question", "") if leader_market else "",
            "lagging_question": getattr(market, "question", ""),
            "leader_direction": signal.direction,
            "leader_toxicity_excess": round(signal.leader_toxicity - signal.toxicity_percentile, 6),
            "leader_price_shift_cents": round(signal.leader_price_shift * 100.0, 3),
            "expected_shift_cents": round(signal.expected_probability_shift * 100.0, 3),
            "correlation": round(signal.correlation, 6),
            "thematic_group": signal.thematic_group,
            "fair_value": round(fair_value, 6),
            "market_price": round(market_price, 6),
            "edge_cents": round(edge_cents, 3),
            "cross_spread_slip_cents": round(cross_spread_slip_cents, 3),
            "last_trade_age_s": round(last_trade_age_s, 3),
        }

    async def _record_contagion_matrix(
        self,
        signal: ContagionArbSignal,
        market: MarketInfo,
        *,
        book: OrderbookTracker | None,
        agg: OHLCVAggregator | None,
        suppressed: bool = False,
        suppression_reason: str | None = None,
    ) -> None:
        entry = self._build_contagion_matrix_entry(
            signal,
            market,
            book=book,
            agg=agg,
            suppressed=suppressed,
            suppression_reason=suppression_reason,
        )
        self._recent_contagion_matrix.appendleft(entry)
        await self.telegram.notify_contagion_matrix(entry)

    async def _on_contagion_signal(
        self,
        signal: ContagionArbSignal,
        market: MarketInfo,
    ) -> None:
        """Route a contagion arb signal into the RPE fast-strike entry path."""
        if not self._contagion_lane_enabled():
            return

        shadow_mode = signal.is_shadow or self._contagion_shadow_runtime_enabled()

        if not self.lifecycle.is_tradeable(market.condition_id):
            return
        if not market.accepting_orders:
            return
        if not self.lifecycle.is_cooled_down(market.condition_id):
            return
        if not self.positions.is_stop_loss_cooled_down(market.condition_id):
            log.info(
                "stop_loss_cooldown_suppressed_contagion",
                market_id=market.condition_id[:16],
            )
            return

        positioned = self._positioned_asset_ids()
        if market.yes_token_id in positioned or market.no_token_id in positioned:
            log.info("contagion_already_positioned", market=market.condition_id)
            return

        if signal.direction == "buy_no":
            book = self._book_trackers.get(market.no_token_id)
            agg = self._no_aggs.get(market.no_token_id)
        else:
            book = self._book_trackers.get(market.yes_token_id)
            agg = self._yes_aggs.get(market.yes_token_id)

        if book is not None or agg is not None:
            await self._record_contagion_matrix(signal, market, book=book, agg=agg)

        if book and book.has_data:
            snap = book.snapshot()
            if snap.best_bid > 0 and snap.best_ask > 0:
                spread_pct = ((snap.best_ask - snap.best_bid) / snap.best_ask) * 100.0
                if spread_pct > settings.strategy.contagion_arb_max_lagging_spread_pct:
                    log.info(
                        "contagion_suppressed_wide_lagger",
                        market=market.condition_id,
                        spread_pct=round(spread_pct, 3),
                        limit=settings.strategy.contagion_arb_max_lagging_spread_pct,
                    )
                    await self._record_contagion_matrix(
                        signal,
                        market,
                        book=book,
                        agg=agg,
                        suppressed=True,
                        suppression_reason="lagger_spread_wide",
                    )
                    return
            entry_price = round(snap.best_ask - 0.01, 2) if snap.best_ask > 0 else 0.0
            min_depth = settings.strategy.min_ask_depth_usd
            if snap.ask_depth_usd < min_depth:
                log.info(
                    "contagion_rejected_thin_asks",
                    market=market.condition_id,
                    ask_depth_usd=round(snap.ask_depth_usd, 2),
                    min_required=min_depth,
                    direction=signal.direction,
                )
                return
        else:
            entry_price = round(agg.current_price, 2) if agg else 0.0

        if agg is None or agg.last_trade_time <= 0:
            log.info("contagion_suppressed_stale_lagger", market=market.condition_id, reason="no_last_trade")
            await self._record_contagion_matrix(
                signal,
                market,
                book=book,
                agg=agg,
                suppressed=True,
                suppression_reason="lagger_no_trade_timestamp",
            )
            return

        last_trade_age_s = time.time() - agg.last_trade_time
        if last_trade_age_s > settings.strategy.contagion_arb_max_last_trade_age_s:
            log.info(
                "contagion_suppressed_stale_lagger",
                market=market.condition_id,
                last_trade_age_s=round(last_trade_age_s, 2),
                limit=settings.strategy.contagion_arb_max_last_trade_age_s,
            )
            await self._record_contagion_matrix(
                signal,
                market,
                book=book,
                agg=agg,
                suppressed=True,
                suppression_reason="lagger_trade_stale",
            )
            return

        if entry_price <= 0:
            return

        fee_enabled = self._is_fee_enabled(market)
        days_to_resolution = 1
        if market.end_date:
            delta = market.end_date - datetime.now(tz=timezone.utc)
            days_to_resolution = max(1, delta.days)

        contagion_meta = dict(signal.metadata)
        contagion_meta.update({
            "signal_source": signal.signal_source,
            "leading_market_id": signal.leading_market_id,
            "thematic_group": signal.thematic_group,
            "leader_toxicity": signal.leader_toxicity,
            "toxicity_percentile": signal.toxicity_percentile,
            "correlation": signal.correlation,
        })

        if shadow_mode:
            if signal.direction == "buy_no":
                asset_id = market.no_token_id
                shadow_direction = "NO"
            else:
                asset_id = market.yes_token_id
                shadow_direction = "YES"

            best_bid, best_ask, rejection_reason = self._get_reliable_book_bbo(asset_id)
            if rejection_reason is not None:
                self._record_single_name_rejection(
                    "contagion",
                    market,
                    rejection_reason,
                    log_event=False,
                )
                log.info(
                    "contagion_shadow_rejected",
                    market_id=market.condition_id,
                    asset_id=asset_id,
                    direction=shadow_direction,
                    reason=rejection_reason,
                )
                return

            entry_fill: tuple[float, float] | None = None
            if self.deployment_env == DeploymentEnv.PAPER and self._live_orchestrator is not None:
                dispatch_receipt = self._live_orchestrator.dispatcher.dispatch(
                    _contagion_dispatch_context(
                        market_id=market.condition_id,
                        side=shadow_direction,
                        target_price=_decimal_from_number(entry_price),
                        anchor_volume=_decimal_from_number(
                            float(settings.strategy.max_trade_size_usd) / max(entry_price, 0.01)
                        ),
                        max_capital=_decimal_from_number(settings.strategy.max_trade_size_usd),
                        conviction_scalar=Decimal("1"),
                    ),
                    self._current_timestamp_ms(),
                    enforce_guard=True,
                )
                if not dispatch_receipt.executed and dispatch_receipt.guard_reason is not None:
                    self._record_single_name_rejection(
                        "contagion",
                        market,
                        str(dispatch_receipt.guard_reason).lower(),
                        log_event=False,
                    )
                    log.info(
                        "contagion_shadow_rejected",
                        market_id=market.condition_id,
                        asset_id=asset_id,
                        direction=shadow_direction,
                        reason=dispatch_receipt.guard_reason,
                    )
                    return
                entry_fill = self._dispatch_fill_from_receipt(dispatch_receipt)

            if entry_fill is not None:
                entry_price, entry_size = entry_fill
            else:
                entry_size = round(
                    float(settings.strategy.max_trade_size_usd) / max(entry_price, 0.01),
                    2,
                )
            if entry_size < 1.0:
                self._record_single_name_rejection(
                    "contagion",
                    market,
                    "insufficient_size",
                    log_event=False,
                )
                log.info(
                    "contagion_shadow_rejected",
                    market_id=market.condition_id,
                    asset_id=asset_id,
                    direction=shadow_direction,
                    reason="insufficient_size",
                )
                return

            pos = self._shadow_tracker.open_shadow_position(
                signal_source=signal.signal_source,
                market_id=market.condition_id,
                asset_id=asset_id,
                direction=shadow_direction,
                best_ask=best_ask,
                best_bid=best_bid,
                entry_price=entry_price,
                entry_size=entry_size,
                fee_enabled=fee_enabled,
                zscore=signal.score,
                confidence=signal.confidence,
                reference_price=signal.implied_probability,
                toxicity_index=signal.leader_toxicity,
            )
            if pos is None:
                self._record_single_name_rejection(
                    "contagion",
                    market,
                    "invalid_entry_price",
                    log_event=False,
                )
                log.info(
                    "contagion_shadow_rejected",
                    market_id=market.condition_id,
                    asset_id=asset_id,
                    direction=shadow_direction,
                    reason="invalid_entry_price",
                )
                return

            self.lifecycle.record_signal(market.condition_id)
            log.info(
                "contagion_signal_shadow",
                leading_market=signal.leading_market_id,
                lagging_market=signal.lagging_market_id,
                direction=signal.direction,
                implied_probability=round(signal.implied_probability, 4),
                correlation=round(signal.correlation, 4),
                thematic_group=signal.thematic_group,
                shadow_position_id=pos.id,
            )
            return

        trade_direction = "NO" if signal.direction == "buy_no" else "YES"
        if not self._ensemble_allows_entry(
            strategy_source=signal.signal_source,
            market_id=market.condition_id,
            direction=trade_direction,
            log_event="ensemble_risk_blocked_contagion_signal",
            extra={"leading_market_id": signal.leading_market_id},
        ):
            return

        dispatch_timestamp_ms = self._current_timestamp_ms()
        dispatch_context = None
        if self._live_orchestrator is not None:
            anchor_volume = Decimal("1")
            if book is not None and getattr(book, "has_data", False):
                try:
                    anchor_volume = _decimal_from_number(max(getattr(book.snapshot(), "ask_depth_usd", 1.0), 1.0))
                except Exception:
                    anchor_volume = Decimal("1")
            dispatch_context = _contagion_dispatch_context(
                market_id=market.condition_id,
                side=trade_direction,
                target_price=_decimal_from_number(entry_price),
                anchor_volume=anchor_volume,
                max_capital=_decimal_from_number(settings.strategy.max_trade_size_usd),
                conviction_scalar=_decimal_from_number(min(max(signal.confidence, 0.0), 1.0)),
            )
            intent = self._live_orchestrator.dispatcher.evaluate_intent(
                dispatch_context,
                dispatch_timestamp_ms,
                enforce_guard=True,
            )
            if not intent.allowed:
                log.info(
                    "contagion_dispatcher_rejected",
                    market=market.condition_id,
                    reason=intent.reason,
                    leader_market=signal.leading_market_id,
                )
                await self._record_contagion_matrix(
                    signal,
                    market,
                    book=book,
                    agg=agg,
                    suppressed=True,
                    suppression_reason=f"dispatcher_{intent.reason or 'rejected'}",
                )
                return

        pos = await self.positions.open_rpe_position(
            market_id=market.condition_id,
            yes_asset_id=market.yes_token_id,
            no_asset_id=market.no_token_id,
            direction=signal.direction,
            model_probability=signal.implied_probability,
            confidence=signal.confidence,
            entry_price=entry_price,
            event_id=market.event_id,
            days_to_resolution=days_to_resolution,
            fee_enabled=fee_enabled,
            book=book,
            signal_metadata=contagion_meta,
            latency_healthy=True,
        )
        if pos:
            if dispatch_context is not None and self._live_orchestrator is not None:
                self._live_orchestrator.dispatcher.record_external_dispatch(
                    dispatch_context,
                    dispatch_timestamp_ms,
                    enforce_guard=True,
                )
            self.lifecycle.record_signal(market.condition_id)
            if getattr(pos, "fast_strike", False):
                log.info(
                    "contagion_fast_strike_no_chaser",
                    pos_id=pos.id,
                    market=market.condition_id,
                    leader_market=signal.leading_market_id,
                )
            elif book and book.has_data:
                chaser_task = asyncio.create_task(
                    self._entry_chaser_flow(pos, book),
                    name=f"chaser_entry_contagion_{pos.id}",
                )
                chaser_task.add_done_callback(_safe_task_done_callback)
                pos.entry_chaser_task = chaser_task

    async def _on_oracle_signal(
        self,
        signal: SignalResult,
        market: MarketInfo,
        oracle_config: dict,
    ) -> None:
        """Handle an oracle divergence signal — routes into the RPE execution funnel.

        Mirrors ``_on_rpe_signal()`` with oracle-specific gates, then
        calls ``PositionManager.open_rpe_position()`` with
        ``latency_healthy=True`` to enable the SI-7 fast-strike taker
        path.  The existing ``_fair_vs_clob > 0.02`` check in
        ``_open_rpe_position_inner()`` still gates whether fast-strike
        actually fires.
        """
        meta = signal.metadata
        direction = meta.get("direction", "buy_no")
        model_prob = meta.get("model_probability", 0.5)
        confidence = meta.get("confidence", 0.0)
        shadow = settings.strategy.oracle_shadow_runtime_enabled()
        strat = settings.strategy

        # ── Per-market cooldown ────────────────────────────────────
        now = time.monotonic()
        last_fire = self._oracle_last_signal.get(market.condition_id, 0.0)
        if now - last_fire < strat.oracle_cooldown_seconds:
            log.debug(
                "oracle_cooldown_active",
                market=market.condition_id,
                seconds_remaining=round(strat.oracle_cooldown_seconds - (now - last_fire), 1),
            )
            return

        # ── Stamp cooldown ─────────────────────────────────────────
        self._oracle_last_signal[market.condition_id] = now

        # ── Telegram notification ──────────────────────────────────
        mode_label = "👻 SHADOW" if shadow else "🔴 LIVE"
        _safe_fire_and_forget(
            self.telegram.send(
                f"{mode_label} <b>Oracle Signal</b>\n"
                f"Market: <code>{market.condition_id[:16]}…</code>\n"
                f"Question: {market.question[:80]}\n"
                f"Direction: {direction}\n"
                f"Model P: {model_prob:.4f}\n"
                f"Confidence: {confidence:.3f}\n"
                f"Phase: {meta.get('oracle_event_phase', '?')}\n"
                f"Adapter: {meta.get('model_name', '?')}"
            ),
            name="oracle_telegram_notify",
        )

        # ── Data recorder ──────────────────────────────────────────
        if self._recorder is not None:
            self._recorder.enqueue("oracle_signal", {
                "market": market.condition_id,
                "model_probability": model_prob,
                "confidence": confidence,
                "direction": direction,
                "shadow": shadow,
                "event_phase": meta.get("oracle_event_phase"),
                "adapter": meta.get("model_name"),
                "resolved_outcome": meta.get("resolved_outcome"),
            })

        if shadow:
            log.info(
                "oracle_signal_shadow",
                market=market.condition_id,
                model_prob=round(model_prob, 4),
                confidence=round(confidence, 3),
                direction=direction,
                event_phase=meta.get("oracle_event_phase"),
            )
            return

        # ── Live-mode gates ────────────────────────────────────────

        # Gate: Position deduplication
        positioned = self._positioned_asset_ids()
        if market.yes_token_id in positioned or market.no_token_id in positioned:
            log.info(
                "oracle_already_positioned",
                market=market.condition_id,
            )
            return

        # Gate: Determine entry price from appropriate book
        if direction == "buy_no":
            book = self._book_trackers.get(market.no_token_id)
            if book and book.has_data:
                snap = book.snapshot()
                entry_price = round(snap.best_ask - 0.01, 2) if snap.best_ask > 0 else 0.0
            else:
                no_agg = self._no_aggs.get(market.no_token_id)
                entry_price = round(no_agg.current_price, 2) if no_agg else 0.0
        else:
            book = self._book_trackers.get(market.yes_token_id)
            if book and book.has_data:
                snap = book.snapshot()
                entry_price = round(snap.best_ask - 0.01, 2) if snap.best_ask > 0 else 0.0
            else:
                yes_agg = self._yes_aggs.get(market.yes_token_id)
                entry_price = round(yes_agg.current_price, 2) if yes_agg else 0.0

        if entry_price <= 0:
            return

        # Gate: Minimum ask depth
        if book and book.has_data:
            snap_depth = book.snapshot()
            min_depth = strat.min_ask_depth_usd
            if snap_depth.ask_depth_usd < min_depth:
                log.info(
                    "oracle_rejected_thin_asks",
                    market=market.condition_id,
                    ask_depth_usd=round(snap_depth.ask_depth_usd, 2),
                    min_required=min_depth,
                )
                return

        fee_enabled = self._is_fee_enabled(market)

        # Days to resolution
        days_to_resolution = 1
        if market.end_date:
            delta = market.end_date - datetime.now(tz=timezone.utc)
            days_to_resolution = max(1, delta.days)

        oracle_signal_meta = dict(meta)
        oracle_signal_meta["signal_source"] = "si8_oracle"

        # ── Route into the RPE execution funnel with latency_healthy=True ──
        # Pass latency_healthy=True to enable the fast-strike taker gate.
        # The existing _fair_vs_clob > 0.02 check in _open_rpe_position_inner()
        # still gates whether the +1¢ bump and taker execution actually fires.
        pos = await self.positions.open_rpe_position(
            market_id=market.condition_id,
            yes_asset_id=oracle_config.get("yes_asset_id", market.yes_token_id),
            no_asset_id=oracle_config.get("no_asset_id", market.no_token_id),
            direction=direction,
            model_probability=model_prob,
            confidence=confidence,
            entry_price=entry_price,
            event_id=oracle_config.get("event_id", market.event_id),
            days_to_resolution=days_to_resolution,
            fee_enabled=fee_enabled,
            book=book,
            signal_metadata=oracle_signal_meta,
            latency_healthy=True,
        )

        if pos:
            # Fast-strike positions skip the OrderChaser — order placed as
            # a taker to immediately consume stale CLOB liquidity.
            if getattr(pos, 'fast_strike', False):
                log.info(
                    "oracle_fast_strike_no_chaser",
                    pos_id=pos.id,
                    market=market.condition_id,
                )
            elif book and book.has_data:
                chaser_task = asyncio.create_task(
                    self._entry_chaser_flow(pos, book),
                    name=f"chaser_entry_oracle_{pos.id}",
                )
                chaser_task.add_done_callback(_safe_task_done_callback)
                pos.entry_chaser_task = chaser_task

    def _check_paper_fills(self, event: TradeEvent) -> None:
        """Check if any paper orders should fill based on this trade."""
        filled_orders = self.executor.check_paper_fill(
            event.asset_id,
            event.price,
            trade_size=event.size,
            trade_side=event.side,
            is_taker=event.is_taker,
        )
        for order in filled_orders:
            if hasattr(self, "_pure_mm") and self._pure_mm is not None:
                _safe_fire_and_forget(
                    self._pure_mm.on_order_fill(order),
                    name=f"pure_mm_fill_{order.order_id}",
                )

            if self.positions.is_combo_order(order.order_id):
                _safe_fire_and_forget(
                    self.positions.on_combo_order_update(order, self._combo_positions),
                    name=f"combo_fill_{order.order_id}",
                )
                continue

            # Find the position that owns this order
            for pos in self.positions.get_open_positions():
                if pos.entry_order and pos.entry_order.order_id == order.order_id:
                    # Entry filled — schedule exit placement
                    _safe_fire_and_forget(
                        self._handle_entry_fill(pos),
                        name=f"entry_fill_{pos.id}",
                    )
                elif pos.exit_order and pos.exit_order.order_id == order.order_id:
                    # Exit filled — close position
                    self._handle_exit_fill(pos)

    async def _handle_entry_fill(self, pos: Any) -> None:
        """Entry order filled — compute target and place exit.

        Exception-guarded: failures are logged but do not propagate
        (the task was launched via fire-and-forget).
        """
        try:
            await self.positions.on_entry_filled(pos)
            await self.telegram.notify_entry(
                pos.id,
                pos.market_id,
                pos.entry_price,
                pos.entry_size,
                pos.target_price,
            )
            if pos.signal_type == "ofi_momentum":
                return

            # Launch exit chaser (Pillar 1)
            # Use trade_asset_id to route to the correct book —
            # panic positions trade NO, RPE positions may trade YES or NO.
            exit_asset = pos.trade_asset_id or pos.no_asset_id
            exit_book = self._book_trackers.get(exit_asset)
            if exit_book and exit_book.has_data:
                exit_task = asyncio.create_task(
                    self._exit_chaser_flow(pos, exit_book),
                    name=f"chaser_exit_{pos.id}",
                )
                exit_task.add_done_callback(_safe_task_done_callback)
                pos.exit_chaser_task = exit_task
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error("handle_entry_fill_error", pos_id=pos.id, exc_info=True)

    def _handle_exit_fill(self, pos: Any) -> None:
        """Exit order filled — close position and record."""
        if pos.state != PositionState.EXIT_PENDING:
            log.warning(
                "duplicate_exit_fill_ignored",
                pos_id=getattr(pos, "id", "?"),
                state=str(getattr(pos, "state", "?")),
            )
            return

        if pos.signal_type == "ofi_momentum" and self._live_orchestrator is not None:
            self._live_orchestrator.clear_ofi_exit(pos.id)
        self.positions.on_exit_filled(pos, reason=pos.exit_reason or "target")
        # Refresh lifecycle cooldown from close time (not signal fire time)
        self.lifecycle.record_signal(pos.market_id)
        _safe_fire_and_forget(
            self._record_and_notify_exit(pos),
            name=f"record_exit_{pos.id}",
        )

    async def _record_and_notify_exit(self, pos: Any) -> None:
        try:
            await self._persist_closed_position(pos)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.error("record_notify_exit_error", pos_id=pos.id, exc_info=True)

    async def _persist_closed_position(self, pos: Any) -> None:
        if not getattr(pos, "_recorded", False):
            await self.trade_store.record(pos)
            setattr(pos, "_recorded", True)
            self.positions._invalidate_stats_cache()
        if not getattr(pos, "_exit_notified", False):
            await self.telegram.notify_exit(
                pos.id,
                pos.entry_price,
                pos.exit_price,
                pos.pnl_cents,
                pos.exit_reason,
                self.positions.smart_passive_counters,
            )
            setattr(pos, "_exit_notified", True)

    # ═══════════════════════════════════════════════════════════════════════
    #  Pillar 1 — Passive-aggressive chaser flows
    # ═══════════════════════════════════════════════════════════════════════
    async def _entry_chaser_flow(
        self, pos: Any, no_book: OrderbookTracker, *, urgent: bool = False,
    ) -> None:
        """Run the entry-side chaser for a position.

        If the chaser fills, triggers the entry-fill handler.
        If abandoned, cancels the position.

        Parameters
        ----------
        urgent:
            When True the chaser escalates immediately on the first
            POST_ONLY rejection, capturing fast-decaying panic edges
            by accepting a taker fill.
        """
        # Resolve the correct asset_id — RPE positions may trade YES.
        entry_asset = pos.trade_asset_id or pos.no_asset_id
        try:
            if (
                pos.entry_order is not None
                and pos.entry_order.status == OrderStatus.LIVE
            ):
                await self.executor.cancel_order(pos.entry_order)
                pos.entry_order = None

            chaser = OrderChaser(
                executor=self.executor,
                book=no_book,
                market_id=pos.market_id,
                asset_id=entry_asset,
                side=OrderSide.BUY,
                target_size=pos.entry_size,
                anchor_price=pos.entry_price,
                latency_guard=self.latency_guard,
                tp_target_price=pos.target_price,
                fee_rate_bps=pos.entry_fee_bps,
                fast_kill_event=self._fast_kill_event,
                urgent=urgent,
                iceberg_detector=self._iceberg_detectors.get(entry_asset),
                adverse_monitor=self._maker_monitor,
            )
            result = await chaser.run()

            if chaser.state == ChaserState.FILLED and result:
                # Guard: don't resurrect a position that was cancelled
                # while the chaser was running (e.g. ghost liquidity eviction)
                if pos.state == PositionState.CANCELLED:
                    log.warning(
                        "entry_chaser_filled_but_cancelled",
                        pos_id=pos.id,
                    )
                else:
                    pos.entry_order = result
                    # Propagate actual fill quantity from chaser
                    pos.filled_size = chaser.filled_size
                    await self._handle_entry_fill(pos)
            elif chaser.state == ChaserState.ABANDONED:
                # Cancel the original order placed by PositionManager
                if pos.entry_order:
                    await self.executor.cancel_order(pos.entry_order)
                pos.state = PositionState.CANCELLED
                pos.exit_reason = "chaser_abandoned"
                log.info("entry_chaser_abandoned", pos_id=pos.id)
        except asyncio.CancelledError:
            log.info("entry_chaser_cancelled", pos_id=pos.id)
        except Exception as exc:
            log.error("entry_chaser_error", pos_id=pos.id, error=str(exc))

    async def _exit_chaser_flow(self, pos: Any, no_book: OrderbookTracker) -> None:
        """Run the exit-side chaser for a position.

        If the chaser fills, triggers the exit-fill handler.
        If abandoned, logs but does NOT force-close — the timeout loop
        will handle it via an IOC market sell.
        """
        # Resolve the correct asset_id — RPE positions may trade YES.
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        try:
            chaser = OrderChaser(
                executor=self.executor,
                book=no_book,
                market_id=pos.market_id,
                asset_id=exit_asset,
                side=OrderSide.SELL,
                target_size=pos.effective_size,
                anchor_price=pos.target_price,
                latency_guard=self.latency_guard,
                fee_rate_bps=pos.exit_fee_bps,
                fast_kill_event=self._fast_kill_event,
                iceberg_detector=self._iceberg_detectors.get(exit_asset),
                adverse_monitor=self._maker_monitor,
            )
            result = await chaser.run()

            if chaser.state == ChaserState.FILLED and result:
                pos.exit_order = result
                self._handle_exit_fill(pos)
            elif chaser.state == ChaserState.ABANDONED:
                log.info(
                    "exit_chaser_abandoned",
                    pos_id=pos.id,
                    target=pos.target_price,
                )
                # Let timeout_loop handle the force-exit
        except asyncio.CancelledError:
            log.info("exit_chaser_cancelled", pos_id=pos.id)
        except Exception as exc:
            log.error("exit_chaser_error", pos_id=pos.id, error=str(exc))

    # ═══════════════════════════════════════════════════════════════════════
    #  Pillar 3 — Adaptive TP rescaling loop
    # ═══════════════════════════════════════════════════════════════════════
    async def _tp_rescale_loop(self) -> None:
        """Periodically re-evaluate take-profit targets for open positions.

        Runs every ``tp_rescale_interval_s`` seconds.  For each position
        in EXIT_PENDING state, recomputes the target using current σ₃₀
        and VWAP.  If the new target diverges by more than 0.5¢ from the
        current resting exit order, cancels and requotes.
        """
        interval = settings.strategy.tp_rescale_interval_s

        while self._running:
            await asyncio.sleep(interval)
            try:
                for pos in self.positions.get_open_positions():
                    if pos.state != PositionState.EXIT_PENDING:
                        continue
                    if not pos.exit_order:
                        continue
                    if pos.signal_type == "ofi_momentum":
                        continue

                    asset_id = pos.trade_asset_id or pos.no_asset_id
                    no_agg = self._no_aggs.get(asset_id)
                    if not no_agg or no_agg.rolling_vwap <= 0:
                        continue

                    # Current 30-min volatility
                    sigma_30 = no_agg.rolling_volatility_30m

                    # Dynamic spread floor
                    dynamic_spread = compute_dynamic_spread(sigma_30)

                    # Recompute take-profit with fresh market state
                    no_book = self._book_trackers.get(asset_id)
                    depth_ratio = 1.0
                    if no_book and no_book.has_data:
                        depth_ratio = no_book.book_depth_ratio

                    whale = self.whale_monitor.has_confluence(asset_id)

                    new_tp = compute_take_profit(
                        entry_price=pos.entry_price,
                        no_vwap=no_agg.rolling_vwap,
                        realised_vol=sigma_30,
                        book_depth_ratio=depth_ratio,
                        whale_confluence=whale,
                        entry_fee_bps=pos.entry_fee_bps,
                        exit_fee_bps=pos.exit_fee_bps,
                        desired_margin_cents=settings.strategy.desired_margin_cents,
                    )

                    # Tighten if calm: if current BBO is already profitable
                    # and σ is low, lock the scalp
                    current_best_bid = 0.0
                    tp_book = self._book_trackers.get(asset_id)
                    if tp_book and tp_book.has_data:
                        snap = tp_book.snapshot()
                        current_best_bid = snap.best_bid

                    new_target = new_tp.target_price
                    if (
                        dynamic_spread < settings.strategy.min_spread_cents
                        and current_best_bid > pos.entry_price
                    ):
                        scalp_floor = (
                            _decimal_from_number(pos.entry_price)
                            + (_decimal_from_number(dynamic_spread) / Decimal("100"))
                        )
                        scalp_target = max(
                            current_best_bid - 0.01,
                            scalp_floor,
                        )
                        new_target = min(new_target, scalp_target)

                    # Anti-ratchet guard: never move TP FURTHER from entry
                    # than the original target.  Rescaling should only
                    # tighten (lock profits) or track mean-reversion,
                    # not chase a moving VWAP upward indefinitely.
                    original_target = pos.tp_result.target_price if pos.tp_result else pos.target_price
                    if new_target > original_target:
                        new_target = original_target

                    # Only requote if target moved by more than 0.5¢
                    delta_cents = abs(new_target - pos.target_price) * 100
                    if delta_cents < 0.5:
                        continue

                    old_target = pos.target_price
                    pos.target_price = round(new_target, 4)
                    pos.tp_result = new_tp

                    # Cancel exit chaser — it will be restarted with new target
                    if pos.exit_chaser_task and not pos.exit_chaser_task.done():
                        pos.exit_chaser_task.cancel()

                    # Cancel resting exit order
                    if pos.exit_order and pos.exit_order.status in (
                        OrderStatus.LIVE, OrderStatus.PARTIALLY_FILLED,
                    ):
                        await self.executor.cancel_order(pos.exit_order)

                    # Place new exit order (or restart chaser)
                    exit_book = self._book_trackers.get(asset_id)
                    if exit_book and exit_book.has_data:
                        exit_task = asyncio.create_task(
                            self._exit_chaser_flow(pos, exit_book),
                            name=f"chaser_exit_rescale_{pos.id}",
                        )
                        exit_task.add_done_callback(_safe_task_done_callback)
                        pos.exit_chaser_task = exit_task
                    else:
                        exit_order = await self.executor.place_limit_order(
                            market_id=pos.market_id,
                            asset_id=asset_id,
                            side=OrderSide.SELL,
                            price=pos.target_price,
                            size=pos.effective_size,
                            fee_rate_bps=pos.exit_fee_bps,
                        )
                        pos.exit_order = exit_order

                    log.info(
                        "tp_rescaled",
                        pos_id=pos.id,
                        old_target=old_target,
                        new_target=pos.target_price,
                        sigma_30=round(sigma_30, 5),
                        dynamic_spread=dynamic_spread,
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("tp_rescale_error", error=str(exc), exc_info=True)
                if self._tp_rescale_breaker.record():
                    log.critical(
                        "tp_rescale_circuit_breaker_tripped",
                        errors_in_window=self._tp_rescale_breaker.recent_errors,
                    )
                    await self.telegram.send(
                        "🟡 <b>CIRCUIT BREAKER</b>: tp_rescale tripped "
                        "(5 errors in 60s) — suspend & hard-reset WS."
                    )
                    await self._suspend_and_reset()

    # ═══════════════════════════════════════════════════════════════════════
    #  Pillar 9 — Ghost Liquidity Circuit Breaker
    # ═══════════════════════════════════════════════════════════════════════
    def _trade_volume_in_window(self, condition_id: str, window_s: float) -> float:
        """Sum of trade volume (USD) for *condition_id* in the last *window_s* seconds."""
        entries = self._recent_trade_volume.get(condition_id, [])
        if not entries:
            return 0.0
        cutoff = time.time() - window_s
        return sum(vol for ts, vol in entries if ts >= cutoff)

    def _prune_trade_volume(self) -> None:
        """Remove trade-volume entries older than 5 seconds."""
        cutoff = time.time() - 5.0
        for cid in list(self._recent_trade_volume):
            self._recent_trade_volume[cid] = [
                (ts, vol) for ts, vol in self._recent_trade_volume[cid] if ts >= cutoff
            ]
            if not self._recent_trade_volume[cid]:
                del self._recent_trade_volume[cid]

    async def _ghost_liquidity_loop(self) -> None:
        """Fast loop (500ms): detect ghost liquidity and trigger emergency drain.

        Ghost = depth drops > 50% in < 2s WITHOUT matching trade volume.

        Safety guards to prevent false positives:
        - Require book in SYNCED state (no resync artifacts)
        - Require minimum absolute depth ($50) before evaluating
        - Per-market cooldown of 60s between ghost triggers
        - Depth velocity requires >=4 history samples
        """
        interval_s = settings.strategy.ghost_check_interval_ms / 1000.0
        ghost_window = settings.strategy.ghost_window_s
        ghost_threshold = -settings.strategy.ghost_depth_drop_threshold  # e.g. -0.50
        # Minimum absolute depth to evaluate ghost detection (very thin
        # books naturally fluctuate dramatically)
        min_depth_for_ghost = 50.0  # $50 USD
        ghost_cooldown_s = 60.0  # seconds between ghost triggers per market
        ghost_cooldowns: dict[str, float] = {}  # cid -> last trigger time

        while self._running:
            try:
                self._prune_trade_volume()

                # Check active markets for ghost liquidity
                for cid in list(self.lifecycle.active):
                    am = self.lifecycle.active.get(cid)
                    if not am:
                        continue

                    # Per-market cooldown
                    last_ghost = ghost_cooldowns.get(cid, 0.0)
                    if time.time() - last_ghost < ghost_cooldown_s:
                        continue

                    # Check both YES and NO book trackers
                    for token_id in (am.info.yes_token_id, am.info.no_token_id):
                        tracker = self._book_trackers.get(token_id)
                        if not tracker or not tracker.has_data:
                            continue

                        # Require minimum absolute depth to avoid false
                        # positives on naturally thin books
                        current_depth = tracker.current_total_depth()
                        if current_depth < min_depth_for_ghost:
                            continue

                        dv = tracker.depth_velocity(ghost_window)
                        if dv is None:
                            continue

                        if dv <= ghost_threshold:
                            # Check if there were trades to explain the depth drop
                            trade_vol = self._trade_volume_in_window(cid, ghost_window)
                            if trade_vol > 0:
                                continue  # legitimate consumption, not ghost

                            # GHOST DETECTED — emergency drain
                            baseline = tracker.current_total_depth()
                            # Use pre-drop depth approximation
                            if dv < -0.99:
                                baseline_pre = baseline * 100.0  # near-total wipe
                            else:
                                baseline_pre = baseline / (1.0 + dv)

                            ghost_cooldowns[cid] = time.time()
                            self.lifecycle.emergency_drain(cid, baseline_pre)

                            # Force-exit any open positions in this market
                            for pos in self.positions.get_open_positions():
                                if pos.market_id == cid:
                                    try:
                                        if pos.state == PositionState.EXIT_PENDING:
                                            await self.positions.force_stop_loss(pos)
                                        elif pos.state == PositionState.ENTRY_PENDING:
                                            if pos.entry_order:
                                                await self.executor.cancel_order(pos.entry_order)
                                            pos.state = PositionState.CANCELLED
                                            pos.exit_reason = "ghost_liquidity"
                                    except Exception as fex:
                                        log.warning(
                                            "ghost_force_exit_error",
                                            pos_id=pos.id,
                                            error=str(fex),
                                        )

                            log.warning(
                                "ghost_liquidity_detected",
                                condition_id=cid,
                                token_id=token_id,
                                depth_velocity=round(dv, 3),
                                baseline_depth=round(baseline_pre, 2),
                                l2_state=(
                                    self._l2_books[token_id].state.value
                                    if token_id in self._l2_books
                                    else "n/a"
                                ),
                            )
                            await self.telegram.send(
                                f"👻 <b>Ghost Liquidity</b> detected!\n"
                                f"Market: <code>{cid[:16]}</code>\n"
                                f"Depth drop: {abs(dv)*100:.0f}% in {ghost_window}s\n"
                                f"Emergency drain activated."
                            )

                            # ── Vacuum stink-bid exploit ───────────────────
                            # Place deeply OTM POST_ONLY orders on both sides
                            # to catch flash-crash wicks during the vacuum.
                            if settings.strategy.vacuum_stink_bid_enabled and am:
                                no_tracker = self._book_trackers.get(am.info.no_token_id)
                                mid = 0.0
                                no_ask = 0.0
                                if no_tracker and no_tracker.has_data:
                                    snap = no_tracker.snapshot()
                                    mid = snap.mid_price
                                    no_ask = snap.best_ask
                                if mid > 0:
                                    vsig = VacuumSignal(
                                        market_id=cid,
                                        no_asset_id=am.info.no_token_id,
                                        no_best_ask=no_ask,
                                        depth_velocity=dv,
                                        baseline_depth=baseline_pre,
                                        yes_asset_id=am.info.yes_token_id,
                                        mid_price=mid,
                                        signal_source="VACUUM_GhostLiquidity",
                                    )
                                    try:
                                        vac_positions = await self.positions.open_vacuum_stink_bids(
                                            vsig, event_id=am.info.event_id,
                                        )
                                        if vac_positions:
                                            await self.telegram.send(
                                                f"🎯 <b>Vacuum Stink Bids</b> placed: "
                                                f"{len(vac_positions)} orders on "
                                                f"<code>{cid[:16]}</code>"
                                            )
                                    except Exception as vex:
                                        log.error("vacuum_stink_bid_error", error=str(vex))

                            break  # only need one token to trigger per market

                # Check emergency markets for recovery
                recovered = self.lifecycle.check_emergency_recovery(self._book_trackers)
                for cid in recovered:
                    # Cancel any unfilled vacuum stink bids — structural
                    # advantage has disappeared now that depth recovered.
                    try:
                        n_cancelled = await self.positions.cancel_vacuum_stink_bids(cid)
                    except Exception:
                        log.warning("cancel_vacuum_bids_error", condition_id=cid, exc_info=True)
                        n_cancelled = 0
                    log.info(
                        "ghost_liquidity_recovered",
                        condition_id=cid,
                        vacuum_bids_cancelled=n_cancelled,
                    )
                    await self.telegram.send(
                        f"✅ <b>Ghost recovered</b>: <code>{cid[:16]}</code> back to ACTIVE"
                        + (f"\n🗑 Cancelled {n_cancelled} vacuum stink bid(s)" if n_cancelled else "")
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("ghost_liquidity_error", error=str(exc), exc_info=True)
                if self._ghost_breaker.record():
                    log.critical(
                        "ghost_circuit_breaker_tripped",
                        errors_in_window=self._ghost_breaker.recent_errors,
                    )
                    await self.telegram.send(
                        "🟡 <b>CIRCUIT BREAKER</b>: ghost_liquidity tripped "
                        "(5 errors in 60s) — suspend & hard-reset WS."
                    )
                    await self._suspend_and_reset()
            await asyncio.sleep(interval_s)

    # ═══════════════════════════════════════════════════════════════════════
    #  Background loops
    # ═══════════════════════════════════════════════════════════════════════
    async def _timeout_loop(self) -> None:
        """Every 10 seconds: check timeouts and record closed positions.

        Stop-loss checks have been moved to :class:`StopLossMonitor`
        for faster (500ms) reaction time.
        """
        while self._running:
            try:
                await self.positions.check_timeouts(exclude_signal_types={"ofi_momentum"})

                # Backstop any closed position whose first async persist/notify
                # attempt was missed or failed.
                for pos in self.positions.get_all_positions():
                    if (
                        pos.state == PositionState.CLOSED
                        and (
                            not getattr(pos, "_recorded", False)
                            or not getattr(pos, "_exit_notified", False)
                        )
                    ):
                        await self._persist_closed_position(pos)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("timeout_loop_error", error=str(exc), exc_info=True)
                if self._timeout_breaker.record():
                    log.critical(
                        "timeout_circuit_breaker_tripped",
                        errors_in_window=self._timeout_breaker.recent_errors,
                    )
                    await self.telegram.send(
                        "🟡 <b>CIRCUIT BREAKER</b>: timeout_loop tripped "
                        "(5 errors in 60s) — suspend & hard-reset WS."
                    )
                    await self._suspend_and_reset()
            await asyncio.sleep(10)

    async def _stale_bar_flush_loop(self) -> None:
        """Every 30 seconds, flush stale OHLCV bars on all aggregators.

        On low-volume markets, minutes can pass without a single trade.
        Without this loop, bars only close when the next trade arrives,
        causing rolling VWAP and σ to freeze — which in turn makes the
        panic detector and RPE evaluate against stale statistics.

        This loop calls ``flush_stale_bar()`` on every aggregator,
        closing any bar that has been open longer than BAR_INTERVAL
        even if no new trade has arrived.
        """
        while self._running:
            await asyncio.sleep(30)
            try:
                now = time.time()
                yes_flushed = 0
                no_flushed = 0
                for yes_agg in list(self._yes_aggs.values()):
                    bar = yes_agg.flush_stale_bar(now)
                    if bar:
                        yes_flushed += 1
                        # Drive the same signal evaluation as a normal bar close
                        asset_id = yes_agg.asset_id
                        latency_state = self.latency_guard.check(now)
                        if latency_state != LatencyState.BLOCKED:
                            await self._on_yes_bar_closed(asset_id, bar)
                for no_agg in list(self._no_aggs.values()):
                    bar = no_agg.flush_stale_bar(now)
                    if bar:
                        no_flushed += 1

                log.info(
                    "stale_bar_flush_tick",
                    yes_aggs=len(self._yes_aggs),
                    no_aggs=len(self._no_aggs),
                    yes_bars_flushed=yes_flushed,
                    no_bars_flushed=no_flushed,
                )

                # SI-3: Cross-market divergence scan after all bars updated
                # In multicore mode, PCE worker handles scanning autonomously
                if not self._multicore_enabled and self._cross_market is not None:
                    cm_signals = self._cross_market.scan()
                    if cm_signals:
                        log.info("cross_market_scan", signals=len(cm_signals))
                        await self._consume_cross_market_signals(cm_signals)
            except asyncio.CancelledError:
                break
            except (KeyError, ValueError) as exc:
                # Targeted non-fatal: stale aggregator keys, bad bar data
                log.warning("stale_bar_flush_non_fatal", error=str(exc))
            except Exception as exc:
                log.error(
                    "stale_bar_flush_error",
                    error=str(exc),
                    exc_info=True,
                )
                if self._stale_bar_breaker.record():
                    log.critical(
                        "stale_bar_circuit_breaker_tripped",
                        errors_in_window=self._stale_bar_breaker.recent_errors,
                        msg="Too many unexpected errors in stale_bar_flush — suspending & resetting WS",
                    )
                    await self.telegram.send(
                        "🟡 <b>CIRCUIT BREAKER</b>: stale_bar_flush tripped "
                        "(5 unexpected errors in 60s) — suspend & hard-reset WS."
                    )
                    await self._suspend_and_reset()
                    self._stale_bar_breaker.reset()

    async def _stats_loop(self) -> None:
        """Every 15 minutes, log and broadcast aggregate stats."""
        while self._running:
            await asyncio.sleep(900)
            try:
                stats = self._augment_trade_stats(await self.trade_store.get_stats())
                log.info("periodic_stats", **stats)
                await self.telegram.notify_stats(stats)

                if self._combo_detector is not None:
                    paused = self._combo_detector.active_deferrals()
                    log.info(
                        "combo_arb_ofi_status",
                        paused_clusters=len(paused),
                        paused_event_ids=[state.event_id for state in paused],
                        si9_ofi_window_ms=settings.strategy.si9_ofi_window_ms,
                        si9_toxic_ofi_threshold=settings.strategy.si9_toxic_ofi_threshold,
                    )

                all_rankings = self._top_toxicity_rankings(limit=None)
                top_rankings = all_rankings[:5]
                if top_rankings:
                    log.info(
                        "top_market_toxicity_rankings",
                        tracked_markets=len(all_rankings),
                        l2_active=len(getattr(self, "_l2_active_set", set())),
                        top_markets=top_rankings,
                    )

                # Log market scores
                for am in self.lifecycle.active.values():
                    bd = am.score
                    log.info(
                        "market_scored",
                        question=am.info.question[:50],
                        **bd.as_dict(),
                    )

                if self.paper_mode:
                    ready, _ = await self.trade_store.passes_go_live_criteria()
                    if ready:
                        await self.telegram.send(
                            "🟢 <b>Paper-trading criteria MET</b> — ready for live deployment!"
                        )

                shadow_overview = await self.trade_store.get_shadow_source_overview()
                if shadow_overview:
                    log.info(
                        "shadow_strategy_overview",
                        shadow_sources=shadow_overview,
                    )
                    panic_shadow = next(
                        (
                            row
                            for row in shadow_overview
                            if row.get("signal_source") == _PANIC_SHADOW_SOURCE
                        ),
                        None,
                    )
                    if panic_shadow is not None and int(
                        panic_shadow.get("total_trades", 0) or 0
                    ) >= 20:
                        log_method = (
                            log.info
                            if bool(panic_shadow.get("passes_go_live", False))
                            else log.warning
                        )
                        log_method(
                            "panic_shadow_decision_state",
                            total_trades=int(panic_shadow.get("total_trades", 0) or 0),
                            expectancy_cents=round(
                                float(panic_shadow.get("expectancy_cents", 0.0) or 0.0),
                                4,
                            ),
                            win_rate=round(
                                float(panic_shadow.get("win_rate", 0.0) or 0.0),
                                4,
                            ),
                            passes_go_live=bool(
                                panic_shadow.get("passes_go_live", False)
                            ),
                        )
                    await self.telegram.notify_shadow_summary(shadow_overview)

                # ── RPE calibration gate check ────────────────────────
                cal_stats = (
                    self._rpe_calibration.calibration_summary()
                    if self._rpe_calibration is not None
                    else {}
                )
                resolved_n = cal_stats.get("resolved", 0)

                if settings.strategy.rpe_generic_enabled and resolved_n < 30:
                    log.warning(
                        "rpe_generic_insufficient_calibration",
                        resolved=resolved_n,
                        required=30,
                    )

                if (
                    self.paper_mode
                    and resolved_n >= 30
                    and cal_stats.get("direction_accuracy", 0) >= 0.55
                ):
                    await self.telegram.send(
                        "🎯 <b>Generic RPE calibrated</b>\n"
                        f"Direction accuracy: {cal_stats['direction_accuracy']:.1%} "
                        f"on {resolved_n} resolved signals.\n"
                        "Consider enabling <code>RPE_GENERIC_ENABLED=true</code>."
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("stats_loop_error", error=str(exc))

    def _tracked_l2_markets(self) -> list[MarketInfo]:
        condition_ids = getattr(self, "_l2_active_set", set())
        if not condition_ids:
            return []

        tracked: dict[str, MarketInfo] = {}
        for market in self._markets:
            if market.condition_id in condition_ids:
                tracked[market.condition_id] = market
        for active_market in self.lifecycle.active.values():
            market = active_market.info
            if market.condition_id in condition_ids:
                tracked[market.condition_id] = market
        for observing_market in self.lifecycle.observing.values():
            market = observing_market.info
            if market.condition_id in condition_ids:
                tracked[market.condition_id] = market
        return list(tracked.values())

    def _top_toxicity_rankings(self, limit: int | None = 5) -> list[dict[str, Any]]:
        rankings: list[dict[str, Any]] = []
        for market in self._tracked_l2_markets():
            book = self._book_trackers.get(market.no_token_id)
            if book is None:
                book = self._book_trackers.get(market.yes_token_id)
            if book is None or not getattr(book, "has_data", True):
                continue

            buy_metrics = book.toxicity_metrics("BUY")
            sell_metrics = book.toxicity_metrics("SELL")
            dominant_side = "BUY"
            dominant_metrics = buy_metrics
            if sell_metrics["toxicity_index"] > buy_metrics["toxicity_index"]:
                dominant_side = "SELL"
                dominant_metrics = sell_metrics

            toxicity_index = float(dominant_metrics.get("toxicity_index", 0.0))
            if toxicity_index <= 0:
                continue

            rankings.append(
                {
                    "condition_id": market.condition_id,
                    "question": getattr(market, "question", "")[:60],
                    "asset_id": market.no_token_id,
                    "dominant_side": dominant_side,
                    "toxicity_index": round(toxicity_index, 4),
                    "depth_evaporation": round(
                        float(dominant_metrics.get("toxicity_depth_evaporation", 0.0)),
                        4,
                    ),
                    "sweep_ratio": round(
                        float(dominant_metrics.get("toxicity_sweep_ratio", 0.0)),
                        4,
                    ),
                }
            )

        rankings.sort(
            key=lambda row: (
                -row["toxicity_index"],
                -row["depth_evaporation"],
                -row["sweep_ratio"],
                row["condition_id"],
            )
        )
        if limit is None:
            return rankings
        return rankings[:limit]

    def _augment_trade_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(stats)
        enriched["smart_passive_counters"] = self.positions.smart_passive_counters
        sync_gate_counters = self.sync_telemetry.snapshot()
        enriched["sync_gate_counters"] = sync_gate_counters
        enriched.update(sync_gate_counters)
        return enriched

    def _smart_passive_operator_block(self) -> str:
        counters = self.positions.smart_passive_counters
        return (
            "\n"
            f"Smart-passive: {int(counters.get('smart_passive_started', 0) or 0)} started  |  "
            f"{int(counters.get('maker_filled', 0) or 0)} maker-filled  |  "
            f"{int(counters.get('fallback_triggered', 0) or 0)} fallback"
        )

    def _sync_gate_operator_block(self) -> str:
        counters = self.sync_telemetry.snapshot()
        return (
            "\n"
            f"Sync gate: contagion={counters['contagion_sync_blocks']}  |  "
            f"si9={counters['si9_sync_blocks']}  |  "
            f"si10={counters['si10_sync_blocks']}"
        )

    async def _paper_summary_loop(self) -> None:
        """Every 30 minutes, send a formatted paper trade summary to Telegram.

        Gives the operator immediate visibility into paper performance
        without having to inspect the DB or raw log entries.
        """
        interval = 1800  # 30 minutes
        while self._running:
            await asyncio.sleep(interval)
            try:
                stats = self._augment_trade_stats(await self.trade_store.get_stats())
                uptime_h = (time.monotonic() - self._start_time) / 3600.0
                await self.telegram.notify_paper_summary(
                    stats,
                    uptime_h,
                    toxicity_rankings=self._top_toxicity_rankings(limit=5),
                )
                log.info(
                    "paper_summary_sent",
                    total_trades=stats.get("total_trades", 0),
                    uptime_h=round(uptime_h, 1),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("paper_summary_error", error=str(exc))

    async def _health_reporter(self) -> None:
        """Every 300 seconds, write ``system_health.json`` to the log dir.

        Tracks memory usage, WebSocket reconnect counts, SQLite lock
        retries, latency guard state, and heartbeat status — critical
        telemetry for the long-running data-harvesting soak test.

        Load shedding: samples up to 5 random L2 books per cycle instead
        of iterating every book, to reduce IPC overhead on large universes.
        """
        import json as _json
        import os as _os
        import random as _random
        import tempfile
        from pathlib import Path as _Path

        try:
            health_dir = _Path(settings.log_dir)
            health_dir.mkdir(parents=True, exist_ok=True)
            health_path = health_dir / "system_health.json"
        except Exception:
            log.warning("health_reporter_dir_error", exc_info=True)
            health_dir = _Path(tempfile.gettempdir())
            health_path = health_dir / "system_health.json"

        # Try to import psutil for memory tracking; degrade gracefully.
        try:
            import psutil  # type: ignore[import-untyped]
            _process = psutil.Process(_os.getpid())
        except (ImportError, Exception):
            _process = None

        _HEALTH_SAMPLE_SIZE = 5  # max L2 books to sample per cycle

        while self._running:
            await asyncio.sleep(300)
            try:
                ws_reconnects = (
                    getattr(self._ws, "reconnect_count", 0)
                    if self._ws else 0
                )
                l2_reconnects = (
                    getattr(self._l2_ws, "reconnect_count", 0)
                    if self._l2_ws else 0
                )
                memory_bytes = (
                    (await asyncio.to_thread(_process.memory_info)).rss
                    if _process else None
                )
                uptime_s = round(time.monotonic() - self._start_time, 1)

                heartbeat_state = "unknown"
                if hasattr(self, "_heartbeat"):
                    heartbeat_state = (
                        "suspended" if self._heartbeat.is_suspended else "alive"
                    )
                latency_state = self.latency_guard.state.value

                health = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "deployment_env": self.deployment_env.value,
                    "uptime_s": uptime_s,
                    "memory_bytes": memory_bytes,
                    "ws_reconnects": ws_reconnects,
                    "l2_ws_reconnects": l2_reconnects,
                    "db_lock_retries": self.trade_store.db_lock_retries,
                    "latency_guard_state": latency_state,
                    "heartbeat_state": heartbeat_state,
                    "active_positions": len(self.positions.get_open_positions()),
                    "wallet_balance_usd": round(float(self.positions._wallet_balance_usd), 4),
                    "active_markets": len(self.lifecycle.active),
                    "observing_markets": len(self.lifecycle.observing),
                }
                sync_gate_counters = self.sync_telemetry.snapshot()
                health["sync_gate_counters"] = sync_gate_counters
                health.update(sync_gate_counters)

                # Aggregate L2 book health metrics across all tracked books.
                # In multicore mode, the actual L2 books live in worker
                # processes — read health from shared-memory readers instead
                # of the stale main-process book objects.
                l2_total_deltas = 0
                l2_total_desyncs = 0
                l2_synced_count = 0
                l2_unreliable_count = 0
                l2_total_count = len(self._l2_books)

                # Load shedding: sample up to _HEALTH_SAMPLE_SIZE
                # random books instead of reading all of them.
                if (
                    self._multicore_enabled
                    and self._process_manager is not None
                ):
                    readers = self._process_manager.get_all_readers()
                    l2_total_count = max(l2_total_count, len(readers))
                    reader_items = list(readers.values())
                    if len(reader_items) > _HEALTH_SAMPLE_SIZE:
                        reader_items = _random.sample(reader_items, _HEALTH_SAMPLE_SIZE)
                    for reader in reader_items:
                        snap = reader.read_header()
                        l2_total_deltas += snap.delta_count
                        l2_total_desyncs += snap.desync_total
                        if snap.state == 2:  # BookState.SYNCED
                            l2_synced_count += 1
                        if not snap.is_reliable:
                            l2_unreliable_count += 1
                else:
                    book_items = list(self._l2_books.values())
                    if len(book_items) > _HEALTH_SAMPLE_SIZE:
                        book_items = _random.sample(book_items, _HEALTH_SAMPLE_SIZE)
                    for book in book_items:
                        l2_total_deltas += book.delta_count
                        l2_total_desyncs += book.desync_total
                        if book.state == BookState.SYNCED:
                            l2_synced_count += 1
                        if not book.is_reliable:
                            l2_unreliable_count += 1

                health["l2_total_deltas"] = l2_total_deltas
                health["l2_total_desyncs"] = l2_total_desyncs
                health["l2_synced_books"] = l2_synced_count
                health["l2_total_books"] = l2_total_count
                health["l2_seq_gap_rate"] = round(
                    l2_total_desyncs / max(1, l2_total_deltas), 6
                )
                health["l2_unreliable_books"] = l2_unreliable_count
                health["tracked_market_selection"] = {
                    "warm_markets": len(self._warm_market_ids),
                    "warm_market_limit": self._warm_market_limit(),
                    "single_name_markets": len(self._tracked_single_market_ids),
                    "si9_markets": len(self._tracked_combo_market_ids),
                    "si9_events": list(self._tracked_combo_event_ids),
                    "si9_scan_summary": dict(self._si9_scan_summary),
                }
                health["observing_market_blockers"] = {
                    "counts": self.lifecycle.observing_blocker_counts(),
                    "sample": self.lifecycle.observing_blockers_snapshot(limit=10),
                }
                if self._combo_detector is not None:
                    health["tracked_market_selection"]["si9_selection_feedback"] = (
                        self._combo_detector.selection_feedback(self._tracked_combo_event_ids)
                    )
                    health["tracked_market_selection"]["si9_recent_evaluations"] = (
                        self._combo_detector.evaluation_snapshot(self._tracked_combo_event_ids)
                    )
                health["single_name_rejection_counts"] = {
                    strategy: dict(reason_counts)
                    for strategy, reason_counts in self._single_name_rejection_counts.items()
                }
                health["smart_passive_counters"] = self.positions.smart_passive_counters
                health["contagion_matrix"] = list(self._recent_contagion_matrix)

                # Atomic write: offload blocking file I/O to a thread
                await asyncio.to_thread(
                    self._write_health_sync, health, health_dir, health_path,
                )

                log.debug("health_report_written", path=str(health_path))

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("health_reporter_error", error=str(exc))

    @staticmethod
    def _write_health_sync(health: dict, health_dir: Any, health_path: Any) -> None:
        """Synchronous atomic write — called via ``asyncio.to_thread``."""
        import json as _json
        import os as _os
        import tempfile

        fd, tmp_path = tempfile.mkstemp(dir=str(health_dir), suffix=".tmp")
        try:
            with _os.fdopen(fd, "w") as f:
                _json.dump(health, f, indent=2)
            _os.replace(tmp_path, str(health_path))
        except Exception:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _market_refresh_sleep_seconds(
        *,
        full_refresh_interval_s: float,
        observation_period_s: float,
        next_full_refresh_deadline: float,
        now: float,
    ) -> float:
        """Return the next sleep interval for the market refresh loop.

        Full discovery/decay refreshes should remain on the configured
        ``MARKET_REFRESH_MINUTES`` cadence, but observation-tier markets
        should be reconsidered as soon as their observation window expires
        after restart. This helper therefore caps the next wake-up at the
        shorter of the observation period and the remaining time until the
        next full refresh.
        """
        promotion_interval_s = max(1.0, min(full_refresh_interval_s, observation_period_s))
        time_to_full_refresh_s = max(0.0, next_full_refresh_deadline - now)
        if time_to_full_refresh_s <= 0.0:
            return 0.0
        return min(promotion_interval_s, time_to_full_refresh_s)

    async def _market_refresh_loop(self) -> None:
        """Periodically re-discover, re-score, promote/demote/evict.

        Full refreshes stay on the configured discovery cadence so trade-count
        decay and Gamma polling semantics remain unchanged. Between those full
        refreshes, the loop performs earlier non-decaying refreshes whenever
        the observation window is shorter than the discovery cadence, allowing
        observing markets to promote promptly after restart.
        """
        full_refresh_interval_s = max(1.0, settings.strategy.market_refresh_minutes * 60)
        observation_period_s = max(1.0, settings.strategy.observation_period_minutes * 60)
        next_full_refresh_deadline = time.monotonic() + full_refresh_interval_s
        while self._running:
            sleep_s = self._market_refresh_sleep_seconds(
                full_refresh_interval_s=full_refresh_interval_s,
                observation_period_s=observation_period_s,
                next_full_refresh_deadline=next_full_refresh_deadline,
                now=time.monotonic(),
            )
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

            decay_counters = time.monotonic() >= next_full_refresh_deadline
            try:
                await self._refresh_markets_once(decay_counters=decay_counters)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("market_refresh_error", error=str(exc), exc_info=True)
            finally:
                if decay_counters:
                    next_full_refresh_deadline = time.monotonic() + full_refresh_interval_s

    async def _pce_refresh_loop(self) -> None:
        """Periodically recompute correlations, persist, and send dashboard."""
        interval = settings.strategy.pce_correlation_refresh_minutes * 60
        cycle_count = 0
        while self._running:
            await asyncio.sleep(interval)
            try:
                cycle_count += 1
                await asyncio.to_thread(self.pce.refresh_correlations)
                await asyncio.to_thread(self.pce.save_state)

                # Every 12th cycle (~6 hours) validate structural priors
                if cycle_count % 12 == 0:
                    prior_summary = await asyncio.to_thread(
                        self.pce.validate_structural_priors
                    )
                    log.info("pce_prior_validation", **prior_summary)

                # Send Telegram dashboard
                open_positions = self.positions.get_open_positions()
                dashboard = self.pce.get_dashboard_data(open_positions)
                await self.telegram.notify_pce_dashboard(dashboard)

                log.info(
                    "pce_dashboard_sent",
                    portfolio_var=dashboard.get("portfolio_var", 0.0),
                    pairs_tracked=dashboard.get("total_pairs_tracked", 0),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("pce_refresh_error", error=str(exc), exc_info=True)

    async def _cleanup_loop(self) -> None:
        """Every 5 minutes: clean up stale positions, orders, reset daily PnL,
        and checkpoint live state for crash recovery."""
        while self._running:
            await asyncio.sleep(300)
            try:
                try:
                    removed_pos = self.positions.cleanup_closed()
                except Exception:
                    if self._orchestrator_health_monitor is not None:
                        self._orchestrator_health_monitor.record_position_release_failure()
                    raise
                else:
                    if self._orchestrator_health_monitor is not None:
                        self._orchestrator_health_monitor.reset_release_failure_count()
                if removed_pos:
                    log.info("positions_cleaned", count=len(removed_pos))

                removed_orders = self.executor.cleanup_old_orders()
                if removed_orders:
                    log.info("orders_cleaned", count=removed_orders)

                # Check for midnight UTC → reset daily PnL (exactly once)
                now = datetime.now(timezone.utc)
                today = now.date()
                if not hasattr(self, "_last_pnl_reset_date"):
                    self._last_pnl_reset_date = today
                if today != self._last_pnl_reset_date:
                    self.positions.reset_daily_pnl()
                    self._last_pnl_reset_date = today

                # Checkpoint live state for crash recovery
                await self.trade_store.checkpoint_orders(
                    self.executor.get_open_orders()
                )
                await self.trade_store.checkpoint_positions(
                    self.positions.get_open_positions()
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("cleanup_loop_error", error=str(exc), exc_info=True)

    # ═══════════════════════════════════════════════════════════════════════
    #  SI-9: Combinatorial Arbitrage Loop
    # ═══════════════════════════════════════════════════════════════════════

    async def _combo_arbitrage_loop(self) -> None:
        """Scan negRisk clusters for mis-priced arb and place multi-leg orders."""
        strat = settings.strategy
        interval = strat.si9_scan_interval_ms / 1000.0
        max_leg_delay_s = strat.si9_max_leg_delay_ms / 1000.0

        log.info("combo_arb_loop_started", scan_interval_ms=strat.si9_scan_interval_ms)

        while self._running:
            try:
                await asyncio.sleep(interval)

                if self._combo_detector is None:
                    continue
                if (
                    self.deployment_env != DeploymentEnv.PAPER
                    and self._orchestrator_health_monitor is not None
                    and not self._orchestrator_health_monitor.is_safe_to_trade(self._current_timestamp_ms())
                ):
                    continue

                wallet_bal = self.positions._wallet_balance_usd

                for cluster in self._cluster_mgr.active_clusters:
                    existing = self._combo_positions.get(cluster.event_id)
                    if existing and existing.state in (
                        ComboState.ENTRY_PENDING,
                        ComboState.PARTIAL_FILL,
                        ComboState.ALL_FILLED,
                    ):
                        continue

                    signal = self._combo_detector.evaluate_cluster(
                        cluster, wallet_balance=wallet_bal,
                    )
                    resumed = self._combo_detector.pop_recent_resume(cluster.event_id)
                    if resumed is not None:
                        self._combo_ofi_alerted.discard(cluster.event_id)
                        log.info(
                            "combo_arb_resumed_notification",
                            event_id=cluster.event_id,
                            maker_leg=resumed.maker_leg,
                            market_id=resumed.market_id,
                            defer_count=resumed.defer_count,
                        )
                        await self.telegram.notify_combo_arb_resumed(
                            cluster.event_id,
                            resumed.maker_leg,
                            resumed.defer_count,
                            question=resumed.question,
                        )
                    if signal is None:
                        deferred = self._combo_detector.get_active_deferral(cluster.event_id)
                        if (
                            deferred is not None
                            and cluster.event_id not in self._combo_ofi_alerted
                        ):
                            self._combo_ofi_alerted.add(cluster.event_id)
                            log.info(
                                "combo_arb_deferred",
                                event_id=cluster.event_id,
                                reason=deferred.reason,
                                maker_leg=deferred.maker_leg,
                                market_id=deferred.market_id,
                                rolling_vi=round(deferred.rolling_vi, 6),
                                current_vi=round(deferred.current_vi, 6),
                                threshold=round(deferred.threshold, 6),
                                defer_count=deferred.defer_count,
                            )
                            await self.telegram.notify_combo_arb_deferred(
                                cluster.event_id,
                                deferred.maker_leg,
                                deferred.rolling_vi,
                                deferred.threshold,
                                current_vi=deferred.current_vi,
                                question=deferred.question,
                            )
                        continue

                    if not self._ensemble_allows_combo_entry(
                        strategy_source="si9_combo_arb",
                        entry_id=cluster.event_id,
                        exposures=[
                            (getattr(leg, "market_id", ""), getattr(leg, "trade_side", "YES") or "YES")
                            for leg in signal.legs
                        ],
                        log_event="ensemble_risk_blocked_combo_signal",
                    ):
                        continue

                    combo = await self.positions.open_combo_position(
                        signal, self._combo_positions,
                    )
                    if combo is not None:
                        self._combo_positions[cluster.event_id] = combo
                        await self.telegram.send(
                            f"🎯 <b>SI-9 Combo Arb</b>\n"
                            f"Event: {cluster.event_id[:16]}\n"
                            f"Legs: {combo.n_legs} | "
                            f"Shares: {signal.target_shares} | "
                            f"Edge: {signal.edge_cents:.1f}¢ | "
                            f"Σbids: {signal.sum_best_bids:.4f} | "
                            f"Collateral: ${signal.total_collateral:.2f}"
                        )

                now = time.time()
                for event_id, combo in list(self._combo_positions.items()):
                    if combo.state in (ComboState.ABANDONED, ComboState.CLOSED):
                        continue

                    if combo.all_filled:
                        if combo.state != ComboState.ALL_FILLED:
                            combo.state = ComboState.ALL_FILLED
                            log.info(
                                "combo_all_filled",
                                combo_id=combo.combo_id,
                                pnl_cents=round(combo.pnl_cents_if_resolved, 2),
                            )
                            await self.telegram.send(
                                f"✅ <b>SI-9 Combo Filled</b>\n"
                                f"Combo: {combo.combo_id}\n"
                                f"PnL at resolution: "
                                f"{combo.pnl_cents_if_resolved:.1f}¢"
                            )
                    elif len(combo.filled_legs) > 0 and combo.state == ComboState.ENTRY_PENDING:
                        combo.state = ComboState.PARTIAL_FILL

                    elapsed = now - combo.created_at
                    if (
                        elapsed > max_leg_delay_s
                        and combo.state in (ComboState.ENTRY_PENDING, ComboState.PARTIAL_FILL)
                        and not combo.all_filled
                    ):
                        log.warning(
                            "combo_leg_timeout",
                            combo_id=combo.combo_id,
                            elapsed_s=round(elapsed, 1),
                            filled=len(combo.filled_legs),
                            pending=len(combo.pending_legs),
                        )
                        await self.positions.abandon_combo(combo, reason="leg_timeout")
                        await self.telegram.send(
                            f"⚠️ <b>SI-9 Abandon</b>\n"
                            f"Combo: {combo.combo_id}\n"
                            f"State: {combo.state.value}\n"
                            f"Filled: {len(combo.filled_legs)} | "
                            f"Elapsed: {elapsed:.0f}s"
                        )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("combo_arb_loop_error", error=str(exc), exc_info=True)
                if self._combo_breaker.record():
                    log.critical(
                        "combo_breaker_tripped",
                        recent_errors=self._combo_breaker.recent_errors,
                    )
                    await self._suspend_and_reset()
                    break

    async def _bayesian_arb_loop(self) -> None:
        """Scan configured SI-10 relationships for Bayesian Dutch-book arb."""
        strat = settings.strategy
        interval = strat.si10_scan_interval_ms / 1000.0
        max_leg_delay_s = strat.si9_max_leg_delay_ms / 1000.0

        log.info("bayesian_arb_loop_started", scan_interval_ms=strat.si10_scan_interval_ms)

        while self._running:
            try:
                await asyncio.sleep(interval)

                if self._bayesian_detector is None:
                    continue

                self._bayesian_cluster_mgr.scan_clusters(self._markets)
                await self._fee_cache.prefetch(
                    sorted(self._bayesian_cluster_mgr.all_cluster_asset_ids())
                )
                active_ids = {
                    cluster.relationship_id
                    for cluster in self._bayesian_cluster_mgr.active_clusters
                }
                wallet_bal = self.positions._wallet_balance_usd

                for cluster in self._bayesian_cluster_mgr.active_clusters:
                    existing = self._combo_positions.get(cluster.relationship_id)
                    if existing and existing.state in (
                        ComboState.ENTRY_PENDING,
                        ComboState.PARTIAL_FILL,
                        ComboState.ALL_FILLED,
                    ):
                        continue

                    signal = self._bayesian_detector.evaluate_cluster(
                        cluster, wallet_balance=wallet_bal,
                    )
                    resumed = self._bayesian_detector.pop_recent_resume(
                        cluster.relationship_id
                    )
                    if resumed is not None:
                        self._bayesian_ofi_alerted.discard(cluster.relationship_id)
                        log.info(
                            "bayesian_arb_resumed_notification",
                            relationship_id=cluster.relationship_id,
                            maker_leg=resumed.maker_leg,
                            market_id=resumed.market_id,
                            defer_count=resumed.defer_count,
                        )

                    if signal is None:
                        deferred = self._bayesian_detector.get_active_deferral(
                            cluster.relationship_id
                        )
                        if (
                            deferred is not None
                            and cluster.relationship_id not in self._bayesian_ofi_alerted
                        ):
                            self._bayesian_ofi_alerted.add(cluster.relationship_id)
                            log.info(
                                "bayesian_arb_deferred",
                                relationship_id=cluster.relationship_id,
                                reason=deferred.reason,
                                maker_leg=deferred.maker_leg,
                                market_id=deferred.market_id,
                                rolling_vi=round(deferred.rolling_vi, 6),
                                current_vi=round(deferred.current_vi, 6),
                                threshold=round(deferred.threshold, 6),
                                defer_count=deferred.defer_count,
                            )
                        continue

                    if not self._ensemble_allows_combo_entry(
                        strategy_source="si10_bayesian_arb",
                        entry_id=cluster.relationship_id,
                        exposures=[
                            (getattr(leg, "market_id", ""), getattr(leg, "trade_side", "YES") or "YES")
                            for leg in signal.legs
                        ],
                        log_event="ensemble_risk_blocked_bayesian_signal",
                    ):
                        continue

                    combo = await self.positions.open_combo_position(
                        signal, self._combo_positions,
                    )
                    if combo is not None:
                        self._combo_positions[cluster.relationship_id] = combo
                        log.info(
                            "bayesian_arb_signal_notification",
                            relationship_id=cluster.relationship_id,
                            bound_title=signal.bound_title,
                            bound_expression=signal.bound_expression,
                            observed_yes_prices=signal.observed_yes_prices,
                            traded_leg_prices=signal.traded_leg_prices,
                            shares=signal.target_shares,
                            edge_cents=signal.edge_cents,
                            gross_edge_cents=signal.gross_edge_cents,
                            spread_cost_cents=signal.spread_cost_cents,
                            taker_fee_cents=signal.taker_fee_cents,
                            net_ev_usd=signal.net_ev_usd,
                            annualized_yield=signal.annualized_yield,
                            days_to_resolution=signal.days_to_resolution,
                            collateral=signal.total_collateral,
                        )
                        await self.telegram.notify_bayesian_arb_signal(
                            cluster.relationship_id,
                            label=signal.relationship_label,
                            bound_title=signal.bound_title,
                            bound_expression=signal.bound_expression,
                            observed_yes_prices=signal.observed_yes_prices,
                            traded_leg_prices=signal.traded_leg_prices,
                            shares=signal.target_shares,
                            edge_cents=signal.edge_cents,
                            gross_edge_cents=signal.gross_edge_cents,
                            spread_cost_cents=signal.spread_cost_cents,
                            taker_fee_cents=signal.taker_fee_cents,
                            net_ev_usd=signal.net_ev_usd,
                            annualized_yield=signal.annualized_yield,
                            days_to_resolution=signal.days_to_resolution,
                            collateral_usd=signal.total_collateral,
                        )

                now = time.time()
                for relationship_id, combo in list(self._combo_positions.items()):
                    if combo.state in (ComboState.ABANDONED, ComboState.CLOSED):
                        continue
                    if relationship_id not in active_ids:
                        continue
                    elapsed = now - combo.created_at
                    if (
                        elapsed > max_leg_delay_s
                        and combo.state in (ComboState.ENTRY_PENDING, ComboState.PARTIAL_FILL)
                        and not combo.all_filled
                    ):
                        log.warning(
                            "bayesian_arb_leg_timeout",
                            combo_id=combo.combo_id,
                            relationship_id=relationship_id,
                            elapsed_s=round(elapsed, 1),
                            filled=len(combo.filled_legs),
                            pending=len(combo.pending_legs),
                        )
                        await self.positions.abandon_combo(combo, reason="leg_timeout")
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("bayesian_arb_loop_error", error=str(exc), exc_info=True)