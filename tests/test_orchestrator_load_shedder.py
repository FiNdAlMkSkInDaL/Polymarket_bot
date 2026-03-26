from __future__ import annotations

import builtins
from decimal import Decimal

import pytest

from src.detectors.ctf_peg_config import CtfPegConfig
from src.events.mev_events import CtfMergeSignal
from src.execution.ctf_paper_adapter import CtfPaperAdapterConfig
from src.execution.dispatch_guard_config import DispatchGuardConfig
from src.execution.multi_signal_orchestrator import MultiSignalOrchestrator, OrchestratorConfig
from src.execution.ofi_signal_bridge import OfiEntrySignal, OfiSignalBridgeConfig
from src.execution.orchestrator_factory import build_paper_orchestrator
from src.execution.orchestrator_load_shedder import OrchestratorLoadShedder
from src.execution.si9_paper_adapter import Si9PaperAdapterConfig
from src.execution.si9_unwind_manifest import Si9UnwindConfig
from src.execution.signal_coordination_bus import CoordinationBusConfig
from src.signals.si9_matrix_detector import Si9MatrixSignal


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


def _ctf_config() -> CtfPegConfig:
    return CtfPegConfig(
        min_yield=Decimal("0.050000"),
        taker_fee_yes=Decimal("0.010000"),
        taker_fee_no=Decimal("0.010000"),
        slippage_budget=Decimal("0.005000"),
        gas_ewma_alpha=Decimal("0.500000"),
        max_desync_ms=400,
    )


def _ctf_adapter_config() -> CtfPaperAdapterConfig:
    return CtfPaperAdapterConfig(
        max_expected_net_edge=Decimal("0.250000"),
        max_capital_per_signal=Decimal("25.000000"),
        default_anchor_volume=Decimal("10.000000"),
        taker_fee_yes=Decimal("0.010000"),
        taker_fee_no=Decimal("0.010000"),
        cancel_on_stale_ms=250,
        max_size_per_leg=Decimal("8.000000"),
        mode="paper",
        bus=None,
    )


def _si9_adapter_config() -> Si9PaperAdapterConfig:
    return Si9PaperAdapterConfig(
        max_expected_net_edge=Decimal("0.050000"),
        max_capital_per_cluster=Decimal("20.000000"),
        max_leg_fill_wait_ms=100,
        cancel_on_stale_ms=50,
        mode="paper",
        unwind_config=Si9UnwindConfig(
            market_sell_threshold=Decimal("0.040000"),
            passive_unwind_threshold=Decimal("0.010000"),
            max_hold_recovery_ms=100,
            min_best_bid=Decimal("0.010000"),
        ),
        bus=None,
    )


def _ofi_bridge_config() -> OfiSignalBridgeConfig:
    return OfiSignalBridgeConfig(
        max_capital_per_signal=Decimal("15.000000"),
        mode="paper",
        slot_side_lock=True,
        source_enabled=True,
    )


def _bus_config() -> CoordinationBusConfig:
    return CoordinationBusConfig(
        slot_lease_ms=500,
        max_slots_per_source=10,
        max_total_slots=10,
        allow_same_source_reentry=False,
    )


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
        max_pending_unwinds=4,
        max_concurrent_clusters=4,
        signal_sources_enabled=enabled_sources or frozenset({"CTF", "SI9", "OFI"}),
    )


def _build_orchestrator() -> MultiSignalOrchestrator:
    return build_paper_orchestrator(
        ctf_config=_ctf_config(),
        ctf_adapter_config=_ctf_adapter_config(),
        si9_cluster_configs=[("cluster-1", ["mkt-a", "mkt-b", "mkt-c"])],
        si9_adapter_config=_si9_adapter_config(),
        ofi_bridge_config=_ofi_bridge_config(),
        bus_config=_bus_config(),
        guard_config=_guard_config(),
        orchestrator_config=_orchestrator_config(),
        ask_proxy={"mkt-a": Decimal("0.19"), "mkt-b": Decimal("0.19"), "mkt-c": Decimal("0.19")},
    )


def _ctf_signal(market_id: str = "mkt-a") -> CtfMergeSignal:
    return CtfMergeSignal(
        market_id=market_id,
        yes_ask=Decimal("0.380000"),
        no_ask=Decimal("0.400000"),
        gas_estimate=Decimal("0.010000"),
        net_edge=Decimal("0.185000"),
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


def _si9_signal() -> Si9MatrixSignal:
    return Si9MatrixSignal(
        cluster_id="cluster-1",
        market_ids=("mkt-a", "mkt-b", "mkt-c"),
        best_yes_asks={"mkt-a": Decimal("0.18"), "mkt-b": Decimal("0.19"), "mkt-c": Decimal("0.20")},
        ask_sizes={"mkt-a": Decimal("2.5"), "mkt-b": Decimal("2.5"), "mkt-c": Decimal("2.5")},
        total_yes_ask=Decimal("0.57"),
        gross_edge=Decimal("0.43"),
        net_edge=Decimal("0.03"),
        target_yield=Decimal("0.02"),
        bottleneck_market_id="mkt-a",
        required_share_counts=Decimal("2.5"),
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


def test_orchestrator_rejects_ctf_signal_for_shed_market() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator.load_shedder is not None
    orchestrator.load_shedder.update_target_map(["mkt-a"])

    event = orchestrator.on_ctf_signal(_ctf_signal("mkt-z"), 1000)

    assert event.event_type == "CTF_REJECTED"
    assert event.payload["reason"] == "MARKET_NOT_ALLOWED"


def test_orchestrator_rejects_ofi_signal_for_shed_market() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator.load_shedder is not None
    orchestrator.load_shedder.update_target_map(["mkt-a"])

    event = orchestrator.on_ofi_signal(_ofi_signal("mkt-z"), Decimal("10"), 1000)

    assert event.event_type == "OFI_REJECTED"
    assert event.payload["reason"] == "MARKET_NOT_ALLOWED"


def test_orchestrator_rejects_si9_signal_for_shed_cluster_market() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator.load_shedder is not None
    orchestrator.load_shedder.update_target_map(["mkt-a", "mkt-b"])

    event = orchestrator.on_si9_signal(_si9_signal(), 1000)

    assert event.event_type == "SI9_REJECTED"
    assert event.payload["reason"] == "MARKET_NOT_ALLOWED"
    assert event.payload["blocked_market_ids"] == ["mkt-c"]


def test_orchestrator_drops_best_yes_ask_updates_for_shed_market() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator.si9_detector is not None
    assert orchestrator.load_shedder is not None
    orchestrator.load_shedder.update_target_map(["mkt-a"])

    accepted = orchestrator.on_best_yes_ask_update("mkt-b", Decimal("0.19"), Decimal("2.5"), 1000)

    assert accepted is False
    assert "mkt-b" not in orchestrator.si9_detector.top_of_book_by_market


def test_orchestrator_accepts_best_yes_ask_updates_for_allowed_market() -> None:
    orchestrator = _build_orchestrator()
    assert orchestrator.si9_detector is not None

    accepted = orchestrator.on_best_yes_ask_update("mkt-a", Decimal("0.19"), Decimal("2.5"), 1000)

    assert accepted is True
    assert orchestrator.si9_detector.top_of_book_by_market["mkt-a"].ask_price == Decimal("0.19")