from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.entry_signals import OfiEntrySignal
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.orchestrator_factory import build_paper_orchestrator
from src.rewards.models import RewardPosterIntent


def _guard_config() -> DispatchGuardConfig:
    return DispatchGuardConfig(
        dedup_window_ms=100,
        max_dispatches_per_source_per_window=10,
        rate_window_ms=200,
        circuit_breaker_threshold=2,
        circuit_breaker_reset_ms=300,
        max_open_positions_per_market=10,
    )


def _orchestrator_config(enabled_sources: frozenset[str] | None = None) -> OrchestratorConfig:
    return OrchestratorConfig(
        tick_interval_ms=50,
        max_pending_unwinds=0,
        max_concurrent_clusters=1,
        signal_sources_enabled=enabled_sources or frozenset({"OFI", "CONTAGION", "REWARD"}),
    )


def _build_orchestrator(
    ask_proxy: dict[str, Decimal] | None = None,
    enabled_sources: frozenset[str] | None = None,
) -> MultiSignalOrchestrator:
    return build_paper_orchestrator(
        guard_config=_guard_config(),
        orchestrator_config=_orchestrator_config(enabled_sources),
        ask_proxy={} if ask_proxy is None else dict(ask_proxy),
    )


def _ofi_signal(market_id: str = "mkt-a") -> OfiEntrySignal:
    return OfiEntrySignal(
        market_id=market_id,
        side="NO",
        target_price=Decimal("0.640000"),
        anchor_volume=Decimal("50.000000"),
        conviction_scalar=Decimal("0.850000"),
        signal_timestamp_ms=1000,
        tvi_kappa=Decimal("1.000000"),
        ofi_window_ms=200,
    )


def test_factory_builds_runtime_orchestrator() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.19")})

    assert isinstance(orchestrator, MultiSignalOrchestrator)
    assert orchestrator.dispatcher is not None


def test_ofi_signal_dispatches_and_updates_snapshot() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.19")})

    event = orchestrator.on_ofi_signal(_ofi_signal(), Decimal("12.000000"), 1000)
    snapshot = orchestrator.orchestrator_snapshot(1001)

    assert event.event_type == "OFI_DISPATCHED"
    assert event.payload["market_id"] == "mkt-a"
    assert snapshot.ofi_ledger.total_dispatched == 1


def test_ofi_source_disabled_rejects() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.19")}, frozenset({"CONTAGION"}))

    event = orchestrator.on_ofi_signal(_ofi_signal(), Decimal("12.000000"), 1000)

    assert event.event_type == "OFI_REJECTED"
    assert event.payload["reason"] == "SOURCE_DISABLED"


def test_market_not_allowed_rejects_ofi_signal() -> None:
    orchestrator = _build_orchestrator({"mkt-b": Decimal("0.19")})

    event = orchestrator.on_ofi_signal(_ofi_signal(), Decimal("12.000000"), 1000)

    assert event.event_type == "OFI_REJECTED"
    assert event.payload["reason"] == "MARKET_NOT_ALLOWED"


def test_reward_source_disabled_rejects_reward_intent() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.19")}, frozenset({"OFI"}))
    intent = RewardPosterIntent(
        market_id="mkt-a",
        asset_id="asset-a",
        side="YES",
        reference_mid_price=Decimal("0.48"),
        target_price=Decimal("0.48"),
        target_size=Decimal("5"),
        max_capital=Decimal("2.4000"),
        quote_id="reward-1",
        reward_program="mid_tier_reward_v1",
        reward_daily_rate_usd=Decimal("25"),
        reward_to_competition=Decimal("8"),
        competition_score=Decimal("3"),
        reward_max_spread_cents=Decimal("3"),
        cancel_on_stale_ms=15_000,
        replace_only_if_price_moves_ticks=1,
    )

    quote_state, event = orchestrator.on_reward_intent(intent, 1000)

    assert event.event_type == "REWARD_REJECTED"
    assert quote_state.status == "REJECTED"
    assert quote_state.guard_reason == "SOURCE_DISABLED"


def test_event_and_snapshot_contracts_are_frozen() -> None:
    orchestrator = _build_orchestrator({"mkt-a": Decimal("0.19")})
    event = orchestrator.on_ofi_signal(_ofi_signal(), Decimal("12.000000"), 1000)
    snapshot = orchestrator.orchestrator_snapshot(1001)

    with pytest.raises(FrozenInstanceError):
        event.event_type = "OFI_REJECTED"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        snapshot.health = "RED"  # type: ignore[misc]
