from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.events.mev_events import (
    DisputeArbitrageSignal,
    MMPredationSignal,
    ShadowSweepSignal,
)
from src.execution.mev_router import MevMarketSnapshot

ScenarioKind = Literal["shadow_sweep", "mm_trap", "d3_panic_absorption"]

DEFAULT_SHADOW_MAX_CAPITAL = 500_000.0
DEFAULT_SHADOW_PREMIUM_PCT = 0.02
DEFAULT_ATTACK_VOLUME = 25_000.0
DEFAULT_D3_MAX_CAPITAL = 250_000.0
DEFAULT_D3_PANIC_LIMIT = 0.88


@dataclass(frozen=True, slots=True)
class MevScenarioFixture:
    name: str
    title: str
    description: str
    kind: ScenarioKind
    snapshots: dict[str, MevMarketSnapshot]


REGIME_THIN_BOOK = MevScenarioFixture(
    name="REGIME_THIN_BOOK",
    title="Shadow Sweep Audit",
    description="High spread, thin top-of-book liquidity to stress shadow sweep slippage.",
    kind="shadow_sweep",
    snapshots={
        "MKT_MEMPOOL_WHALE": MevMarketSnapshot(
            yes_bid=0.38,
            yes_ask=0.45,
            no_bid=0.53,
            no_ask=0.61,
        ),
    },
)

REGIME_TRAPPED_MM = MevScenarioFixture(
    name="REGIME_TRAPPED_MM",
    title="MM Trap Audit",
    description="Deep target book with a volatile correlated book for maker-first predation.",
    kind="mm_trap",
    snapshots={
        "MKT_MM_TARGET": MevMarketSnapshot(
            yes_bid=0.44,
            yes_ask=0.45,
            no_bid=0.54,
            no_ask=0.55,
        ),
        "MKT_MM_CORRELATED": MevMarketSnapshot(
            yes_bid=0.36,
            yes_ask=0.44,
            no_bid=0.50,
            no_ask=0.58,
        ),
    },
)

REGIME_FLASH_CRASH = MevScenarioFixture(
    name="REGIME_FLASH_CRASH",
    title="D3 Panic Absorption Audit",
    description="Flash-crash regime with a panic limit materially below the current midpoint.",
    kind="d3_panic_absorption",
    snapshots={
        "MKT_UMA_PANIC": MevMarketSnapshot(
            yes_bid=0.84,
            yes_ask=0.99,
            no_bid=0.01,
            no_ask=0.16,
        ),
    },
)

SCENARIO_FIXTURES: tuple[MevScenarioFixture, ...] = (
    REGIME_THIN_BOOK,
    REGIME_TRAPPED_MM,
    REGIME_FLASH_CRASH,
)


def build_shadow_signal(*, max_capital: float, premium_pct: float) -> ShadowSweepSignal:
    return ShadowSweepSignal(
        target_market_id="MKT_MEMPOOL_WHALE",
        direction="YES",
        max_capital=max_capital,
        premium_pct=premium_pct,
    )


def build_mm_signal(*, attack_volume: float) -> MMPredationSignal:
    return MMPredationSignal(
        target_market_id="MKT_MM_TARGET",
        correlated_market_id="MKT_MM_CORRELATED",
        v_attack=attack_volume,
        trap_direction="YES",
    )


def build_dispute_signal(*, max_capital: float, panic_limit: float) -> DisputeArbitrageSignal:
    return DisputeArbitrageSignal(
        market_id="MKT_UMA_PANIC",
        panic_direction="YES",
        limit_price=panic_limit,
        max_capital=max_capital,
    )