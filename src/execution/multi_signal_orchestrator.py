from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from src.execution.alpha_adapters import ofi_to_context
from src.execution.dispatch_guard import DispatchGuard
from src.execution.guard_observability import GuardObservabilityPanel, ObservabilitySnapshot
from src.execution.live_wallet_balance import LiveWalletBalanceProvider
from src.execution.ofi_exit_router import OfiExitRouter
from src.execution.ofi_local_exit_monitor import OfiExitDecision, OfiLocalExitMonitor
from src.execution.ofi_paper_ledger import OfiLedgerSnapshot, OfiPaperLedger
from src.execution.ofi_signal_bridge import OfiEntrySignal
from src.execution.orderbook_best_bid_provider import OrderbookBestBidProvider
from src.execution.orchestrator_load_shedder import OrchestratorLoadShedder
from src.execution.priority_context import PriorityOrderContext
from src.execution.priority_dispatcher import PriorityDispatcher
from src.execution.reward_poster_adapter import RewardPosterAdapter, RewardQuoteState

if TYPE_CHECKING:
    from src.data.orderbook import OrderbookTracker
    from src.rewards.models import RewardPosterIntent


SignalSource = Literal["OFI", "CONTAGION", "REWARD", "SYSTEM"]
OrchestratorHealth = Literal["GREEN", "YELLOW", "RED"]
OrchestratorEventType = Literal[
    "OFI_DISPATCHED",
    "OFI_REJECTED",
    "REWARD_DISPATCHED",
    "REWARD_REJECTED",
    "TICK_PROCESSED",
]


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    tick_interval_ms: int
    max_pending_unwinds: int
    max_concurrent_clusters: int
    signal_sources_enabled: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.tick_interval_ms, int) or self.tick_interval_ms <= 0:
            raise ValueError("tick_interval_ms must be a strictly positive int")
        if not isinstance(self.max_pending_unwinds, int) or self.max_pending_unwinds < 0:
            raise ValueError("max_pending_unwinds must be >= 0")
        if not isinstance(self.max_concurrent_clusters, int) or self.max_concurrent_clusters < 1:
            raise ValueError("max_concurrent_clusters must be >= 1")
        if not isinstance(self.signal_sources_enabled, frozenset):
            raise ValueError("signal_sources_enabled must be a frozenset")
        valid_sources = {"OFI", "CONTAGION", "REWARD"}
        if not self.signal_sources_enabled.issubset(valid_sources):
            raise ValueError("signal_sources_enabled contains unsupported source")


@dataclass(frozen=True, slots=True)
class OrchestratorEvent:
    event_type: OrchestratorEventType
    timestamp_ms: int
    source: SignalSource
    payload: dict
    orchestrator_health: OrchestratorHealth


@dataclass(frozen=True, slots=True)
class OrchestratorSnapshot:
    timestamp_ms: int
    pending_unwind_count: int
    active_position_count: int
    ofi_ledger: OfiLedgerSnapshot
    observability: ObservabilitySnapshot
    health: OrchestratorHealth


class MultiSignalOrchestrator:
    def __init__(
        self,
        *,
        guard: DispatchGuard,
        dispatcher: PriorityDispatcher,
        config: OrchestratorConfig,
        load_shedder: OrchestratorLoadShedder | None = None,
        wallet_balance_provider: LiveWalletBalanceProvider | None = None,
        ofi_exit_router: OfiExitRouter | None = None,
        ofi_exit_trackers: Mapping[str, OrderbookTracker] | None = None,
        active_position_count_provider: Callable[[], int] | None = None,
    ) -> None:
        self._guard = guard
        self._dispatcher = dispatcher
        self._config = config
        self._load_shedder = load_shedder
        self._wallet_balance_provider = wallet_balance_provider
        self._ofi_exit_router = ofi_exit_router
        self._ofi_exit_trackers = dict(ofi_exit_trackers or {})
        self._active_position_count_provider = active_position_count_provider or (lambda: 0)
        self._ofi_exit_monitors: dict[str, OfiLocalExitMonitor] = {}
        self._ofi_ledger = OfiPaperLedger()
        self._observability_panel = GuardObservabilityPanel({"DISPATCHER": guard})
        self._reward_poster_adapter = RewardPosterAdapter(dispatcher)
        bind_gate = getattr(self._dispatcher, "bind_pre_dispatch_gate", None)
        if callable(bind_gate):
            bind_gate(self)

    @property
    def guard(self) -> DispatchGuard:
        return self._guard

    @property
    def dispatcher(self) -> PriorityDispatcher:
        return self._dispatcher

    @property
    def load_shedder(self) -> OrchestratorLoadShedder | None:
        return self._load_shedder

    @property
    def wallet_balance_provider(self) -> LiveWalletBalanceProvider | None:
        return self._wallet_balance_provider

    @property
    def ofi_exit_router(self) -> OfiExitRouter | None:
        return self._ofi_exit_router

    @property
    def reward_poster_adapter(self) -> RewardPosterAdapter:
        return self._reward_poster_adapter

    def on_best_yes_ask_update(
        self,
        market_id: str,
        ask_price: Decimal,
        ask_size: Decimal,
        timestamp_ms: int,
    ) -> bool:
        _ = (market_id, ask_price, ask_size, timestamp_ms)
        return False

    def on_ofi_signal(
        self,
        signal: OfiEntrySignal,
        max_capital: Decimal,
        timestamp_ms: int,
    ) -> OrchestratorEvent:
        event_timestamp_ms = int(timestamp_ms)
        allowed_capital = min(max_capital, self._ofi_max_capital_per_signal())
        if allowed_capital <= Decimal("0"):
            self._ofi_ledger.record_signal(
                outcome="GUARD_REJECTED",
                conviction_scalar=signal.conviction_scalar,
                timestamp_ms=event_timestamp_ms,
            )
            return self._event(
                event_type="OFI_REJECTED",
                timestamp_ms=event_timestamp_ms,
                source="OFI",
                payload={
                    "reason": "CAPITAL_ZERO",
                    "market_id": signal.market_id,
                    "side": signal.side,
                },
            )

        context = ofi_to_context(
            market_id=signal.market_id,
            side=signal.side,
            target_price=signal.target_price,
            anchor_volume=signal.anchor_volume,
            max_capital=allowed_capital,
            conviction_scalar=signal.conviction_scalar,
        )
        receipt = self._dispatcher.dispatch(
            context,
            event_timestamp_ms,
            enforce_guard=True,
        )
        if receipt.executed:
            self._ofi_ledger.record_signal(
                outcome="DISPATCHED",
                conviction_scalar=signal.conviction_scalar,
                timestamp_ms=event_timestamp_ms,
                deployed_capital=self._deployed_capital_from_receipt(receipt),
            )
            return self._event(
                event_type="OFI_DISPATCHED",
                timestamp_ms=event_timestamp_ms,
                source="OFI",
                payload={
                    "market_id": signal.market_id,
                    "side": signal.side,
                    "max_capital": str(allowed_capital),
                    "dispatch_executed": receipt.executed,
                    "fill_status": receipt.fill_status,
                },
            )

        rejection_reason = receipt.guard_reason or "GUARD_REJECTED"
        self._ofi_ledger.record_signal(
            outcome=self._ofi_ledger_outcome(rejection_reason),
            conviction_scalar=signal.conviction_scalar,
            timestamp_ms=event_timestamp_ms,
        )
        return self._event(
            event_type="OFI_REJECTED",
            timestamp_ms=event_timestamp_ms,
            source="OFI",
            payload={
                "market_id": signal.market_id,
                "side": signal.side,
                "reason": rejection_reason,
            },
        )

    def on_tick(
        self,
        timestamp_ms: int,
    ) -> list[OrchestratorEvent]:
        return [
            self._event(
                event_type="TICK_PROCESSED",
                timestamp_ms=int(timestamp_ms),
                source="SYSTEM",
                payload={},
            )
        ]

    def on_reward_intent(
        self,
        intent: "RewardPosterIntent",
        timestamp_ms: int,
    ) -> tuple[RewardQuoteState, OrchestratorEvent]:
        event_timestamp_ms = int(timestamp_ms)
        quote_state = self._reward_poster_adapter.submit_intent(intent, event_timestamp_ms)
        event_type: OrchestratorEventType = "REWARD_DISPATCHED"
        payload = {
            "market_id": intent.market_id,
            "asset_id": intent.asset_id,
            "side": intent.side,
            "quote_id": intent.quote_id,
            "status": quote_state.status,
            "guard_reason": quote_state.guard_reason,
        }
        if quote_state.status == "REJECTED":
            event_type = "REWARD_REJECTED"
        return quote_state, self._event(
            event_type=event_type,
            timestamp_ms=event_timestamp_ms,
            source="REWARD",
            payload=payload,
        )

    def evaluate_ofi_exit(
        self,
        *,
        asset_id: str,
        position_state: dict,
        current_timestamp_ms: int,
    ) -> OfiExitDecision:
        monitor = self._ofi_exit_monitor(asset_id)
        if monitor is None:
            return self._fallback_ofi_exit_decision(position_state, current_timestamp_ms)
        return monitor.evaluate_exit(position_state, current_timestamp_ms)

    def route_ofi_exit(
        self,
        position_state: dict,
        decision: OfiExitDecision,
    ):
        if self._ofi_exit_router is None:
            return None
        return self._ofi_exit_router.route_exit(position_state, decision)

    def evaluate_ofi_exit_promotion(
        self,
        *,
        position_id: str,
        current_timestamp_ms: int,
        current_bbo: dict,
    ):
        if self._ofi_exit_router is None:
            return None
        return self._ofi_exit_router.evaluate_passive_promotion(
            position_id,
            current_timestamp_ms,
            current_bbo,
        )

    def clear_ofi_exit(self, position_id: str) -> None:
        if self._ofi_exit_router is None:
            return
        self._ofi_exit_router.clear_exit(position_id)

    def orchestrator_snapshot(
        self,
        timestamp_ms: int,
    ) -> OrchestratorSnapshot:
        snapshot_timestamp_ms = int(timestamp_ms)
        observability = self._observability_panel.full_snapshot(snapshot_timestamp_ms)
        return OrchestratorSnapshot(
            timestamp_ms=snapshot_timestamp_ms,
            pending_unwind_count=0,
            active_position_count=self._active_position_count(),
            ofi_ledger=self._ofi_ledger.snapshot(),
            observability=observability,
            health=observability.system_health,
        )

    def dispatch_guard_reason(
        self,
        context: PriorityOrderContext,
        dispatch_timestamp_ms: int,
    ) -> str | None:
        _ = dispatch_timestamp_ms
        if not self._source_enabled(context.signal_source):
            return "SOURCE_DISABLED"
        if not self._market_allowed(context.market_id):
            return "MARKET_NOT_ALLOWED"
        return None

    def _event(
        self,
        *,
        event_type: OrchestratorEventType,
        timestamp_ms: int,
        source: SignalSource,
        payload: dict,
    ) -> OrchestratorEvent:
        snapshot = self.orchestrator_snapshot(timestamp_ms)
        return OrchestratorEvent(
            event_type=event_type,
            timestamp_ms=int(timestamp_ms),
            source=source,
            payload=self._json_safe_dict(payload),
            orchestrator_health=snapshot.health,
        )

    def _json_safe_dict(self, payload: dict) -> dict:
        return {str(key): self._json_safe_value(value) for key, value in payload.items()}

    def _active_position_count(self) -> int:
        try:
            active_count = int(self._active_position_count_provider())
        except Exception:
            return 0
        return max(active_count, 0)

    def _json_safe_value(self, value):
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): self._json_safe_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._json_safe_value(item) for item in value]
        if hasattr(value, "__dataclass_fields__"):
            return self._json_safe_value(asdict(value))
        return str(value)

    def _market_allowed(self, market_id: str) -> bool:
        if self._load_shedder is None:
            return True
        return self._load_shedder.is_market_allowed(market_id)

    def _source_enabled(self, signal_source: str) -> bool:
        return str(signal_source or "").strip() in self._config.signal_sources_enabled

    @staticmethod
    def _ofi_max_capital_per_signal() -> Decimal:
        return Decimal("1000000000")

    @staticmethod
    def _ofi_ledger_outcome(rejection_reason: str) -> str:
        if rejection_reason == "SOURCE_DISABLED":
            return "SOURCE_DISABLED"
        return "GUARD_REJECTED"

    @staticmethod
    def _deployed_capital_from_receipt(receipt) -> Decimal:
        if receipt.fill_price is not None and receipt.fill_size is not None:
            return receipt.fill_price * receipt.fill_size
        if receipt.partial_fill_price is not None and receipt.partial_fill_size is not None:
            return receipt.partial_fill_price * receipt.partial_fill_size
        return Decimal("0")

    def _ofi_exit_monitor(self, asset_id: str) -> OfiLocalExitMonitor | None:
        asset_key = str(asset_id or "").strip()
        if not asset_key:
            return None
        monitor = self._ofi_exit_monitors.get(asset_key)
        if monitor is not None:
            return monitor
        tracker = self._ofi_exit_trackers.get(asset_key)
        if tracker is None:
            return None
        monitor = OfiLocalExitMonitor(OrderbookBestBidProvider(tracker))
        self._ofi_exit_monitors[asset_key] = monitor
        return monitor

    @staticmethod
    def _fallback_ofi_exit_decision(position_state: dict, current_timestamp_ms: int) -> OfiExitDecision:
        current_best_bid = MultiSignalOrchestrator._decimal_field(position_state, "current_best_bid")
        drawn_tp = MultiSignalOrchestrator._decimal_field(position_state, "drawn_tp")
        if drawn_tp > Decimal("0") and current_best_bid >= drawn_tp:
            return OfiExitDecision(action="TARGET_HIT", trigger_price=drawn_tp)
        drawn_stop = MultiSignalOrchestrator._decimal_field(position_state, "drawn_stop")
        if drawn_stop > Decimal("0") and current_best_bid > Decimal("0") and current_best_bid <= drawn_stop:
            return OfiExitDecision(action="STOP_HIT", trigger_price=drawn_stop)
        drawn_time_ms = int(position_state.get("drawn_time_ms", 0) or 0)
        if drawn_time_ms > 0 and int(current_timestamp_ms) > drawn_time_ms:
            return OfiExitDecision(action="TIME_STOP_TRIGGERED", trigger_price=current_best_bid)
        return OfiExitDecision(action="HOLD", trigger_price=current_best_bid)

    @staticmethod
    def _decimal_field(position_state: dict, field_name: str) -> Decimal:
        value = position_state.get(field_name, Decimal("0"))
        if isinstance(value, Decimal) and value.is_finite() and value >= Decimal("0"):
            return value
        return Decimal("0")
