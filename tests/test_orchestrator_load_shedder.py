from __future__ import annotations

import builtins
from decimal import Decimal

import pytest

from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.ofi_signal_bridge import OfiEntrySignal
from src.execution.orchestrator_factory import build_paper_orchestrator
from src.execution.orchestrator_load_shedder import OrchestratorLoadShedder


class _OpenMarketPositionManager:
    def __init__(self, open_market_ids: set[str] | None = None, open_positions: list[object] | None = None) -> None:
        self._open_market_ids = set(open_market_ids or set())
        self._open_positions = list(open_positions or [])

    def get_open_market_ids(self) -> set[str]:
        return set(self._open_market_ids)

    def get_open_positions(self) -> list[object]:
        return list(self._open_positions)


class _PositionRecord:
    def __init__(self, market_id: str) -> None:
        self.market_id = market_id


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


def _build_orchestrator() -> MultiSignalOrchestrator:
    return build_paper_orchestrator(
        guard_config=_guard_config(),
        orchestrator_config=_orchestrator_config(),
        ask_proxy={"mkt-a": Decimal("0.19"), "mkt-b": Decimal("0.19"), "mkt-c": Decimal("0.19")},
    )


def _ofi_signal(market_id: str = "mkt-a") -> OfiEntrySignal:
    return OfiEntrySignal(
        market_id=market_id,
        side="YES",
        target_price=Decimal("0.41"),
        anchor_volume=Decimal("4"),
        conviction_scalar=Decimal("0.8"),
        signal_timestamp_ms=1000,
        tvi_kappa=Decimal("0.2"),
        ofi_window_ms=250,
    )


def test_constructor_requires_positive_max_active_l2_markets() -> None:
    with pytest.raises(ValueError, match="max_active_l2_markets"):
        OrchestratorLoadShedder(0, ["mkt-a"])


def test_live_mode_caps_max_active_l2_markets_at_25() -> None:
    shedder = OrchestratorLoadShedder(40, [f"mkt-{index}" for index in range(40)], deployment_phase="LIVE")

    assert shedder.max_active_l2_markets == 25
    assert len(shedder.allowed_market_ids) == 25


def test_is_market_allowed_strictly_enforces_top_tier_cutoff() -> None:
    shedder = OrchestratorLoadShedder(3, ["mkt-a", "mkt-b", "mkt-c", "mkt-d"], deployment_phase="PAPER")

    assert shedder.is_market_allowed("mkt-a") is True
    assert shedder.is_market_allowed("mkt-c") is True
    assert shedder.is_market_allowed("mkt-d") is False


def test_is_market_allowed_is_o1_lookup_without_list_allocation() -> None:
    shedder = OrchestratorLoadShedder(3, ["mkt-a", "mkt-b", "mkt-c"], deployment_phase="PAPER")
    original_list = builtins.list

    def _forbidden_list(*args, **kwargs):
        raise AssertionError("list allocation is not allowed in is_market_allowed hot path")

    try:
        builtins.list = _forbidden_list
        allowed = shedder.is_market_allowed("mkt-b")
    finally:
        builtins.list = original_list

    assert allowed is True


def test_update_target_map_drops_markets_that_fall_out_of_top_tier_without_exposure() -> None:
    shedder = OrchestratorLoadShedder(2, ["mkt-a", "mkt-b"], deployment_phase="PAPER")

    shedder.update_target_map(["mkt-c", "mkt-d"])

    assert shedder.is_market_allowed("mkt-a") is False
    assert shedder.is_market_allowed("mkt-c") is True


def test_update_target_map_retains_market_with_open_market_exposure() -> None:
    shedder = OrchestratorLoadShedder(
        2,
        ["mkt-a", "mkt-b"],
        position_manager=_OpenMarketPositionManager(open_market_ids={"mkt-a"}),
        deployment_phase="PAPER",
    )

    shedder.update_target_map(["mkt-c", "mkt-d"])

    assert shedder.is_market_allowed("mkt-a") is True
    assert shedder.is_market_allowed("mkt-c") is True


def test_update_target_map_retains_market_via_open_positions_fallback() -> None:
    shedder = OrchestratorLoadShedder(
        2,
        ["mkt-a", "mkt-b"],
        position_manager=_OpenMarketPositionManager(open_positions=[_PositionRecord("mkt-b")]),
        deployment_phase="PAPER",
    )

    shedder.update_target_map(["mkt-c", "mkt-d"])

    assert shedder.is_market_allowed("mkt-b") is True


def test_update_target_map_drops_market_after_exposure_clears() -> None:
    position_manager = _OpenMarketPositionManager(open_market_ids={"mkt-a"})
    shedder = OrchestratorLoadShedder(2, ["mkt-a", "mkt-b"], position_manager=position_manager, deployment_phase="PAPER")

    shedder.update_target_map(["mkt-c", "mkt-d"])
    position_manager._open_market_ids.clear()
    shedder.update_target_map(["mkt-c", "mkt-d"])

    assert shedder.is_market_allowed("mkt-a") is False


def test_orchestrator_rejects_ofi_signal_for_shed_market() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator.load_shedder is not None
    orchestrator.load_shedder.update_target_map(["mkt-a"])

    event = orchestrator.on_ofi_signal(_ofi_signal("mkt-z"), Decimal("10"), 1000)

    assert event.event_type == "OFI_REJECTED"
    assert event.payload["reason"] == "MARKET_NOT_ALLOWED"


def test_orchestrator_drops_best_yes_ask_updates_for_lean_kernel() -> None:
    orchestrator = _build_orchestrator()

    accepted = orchestrator.on_best_yes_ask_update("mkt-a", Decimal("0.19"), Decimal("2.5"), 1000)

    assert accepted is False
