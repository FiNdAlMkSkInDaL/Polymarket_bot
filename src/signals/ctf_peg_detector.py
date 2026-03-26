"""Isolated CTF peg detector for cross-domain merge-edge opportunities."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.detectors.ctf_peg_config import CtfPegConfig
from src.events.mev_events import CtfMergeSignal


@dataclass(frozen=True, slots=True)
class CtfPegState:
    market_id: str
    yes_ask: Decimal = Decimal("0")
    no_ask: Decimal = Decimal("0")
    gas_estimate: Decimal = Decimal("0")
    net_edge: Decimal = Decimal("0")
    yes_timestamp_ms: int = 0
    no_timestamp_ms: int = 0


class CtfPegDetector:
    """Track binary top-of-book plus rolling L1 gas and emit merge signals.

    The detector maintains only O(1) state:
    - latest YES ask
    - latest NO ask
    - EWMA gas estimate
    - latest computed merge edge

    Merge edge:
        E_merge = 1 - (Ask_YES + Ask_NO) - Gas_L1
    """

    __slots__ = (
        "market_id",
        "config",
        "_yes_ask",
        "_no_ask",
        "_yes_timestamp_ms",
        "_no_timestamp_ms",
        "_gas_estimate",
        "_gas_initialized",
        "_net_edge",
    )

    def __init__(
        self,
        market_id: str,
        config: CtfPegConfig,
    ) -> None:
        self.market_id = str(market_id).strip()
        self.config = config
        self._yes_ask = Decimal("0")
        self._no_ask = Decimal("0")
        self._yes_timestamp_ms = 0
        self._no_timestamp_ms = 0
        self._gas_estimate = Decimal("0")
        self._gas_initialized = False
        self._net_edge = Decimal("0")

    @property
    def state(self) -> CtfPegState:
        return CtfPegState(
            market_id=self.market_id,
            yes_ask=self._yes_ask,
            no_ask=self._no_ask,
            gas_estimate=self._gas_estimate,
            net_edge=self._net_edge,
            yes_timestamp_ms=self._yes_timestamp_ms,
            no_timestamp_ms=self._no_timestamp_ms,
        )

    def record_base_fee(self, base_fee: Decimal) -> Decimal:
        fee = max(Decimal("0"), Decimal(base_fee))
        if not self._gas_initialized:
            self._gas_estimate = fee
            self._gas_initialized = True
        else:
            alpha = self.config.gas_ewma_alpha
            self._gas_estimate = (alpha * fee) + ((Decimal("1") - alpha) * self._gas_estimate)
        self._recompute_net_edge()
        return self._gas_estimate

    def evaluate(
        self,
        *,
        yes_ask: Decimal,
        no_ask: Decimal,
        yes_timestamp_ms: int,
        no_timestamp_ms: int,
        base_fee: Decimal | None = None,
    ) -> CtfMergeSignal | None:
        candidate_yes_ask = Decimal(yes_ask)
        candidate_no_ask = Decimal(no_ask)

        if candidate_yes_ask <= Decimal("0") or candidate_no_ask <= Decimal("0"):
            return None
        if abs(int(yes_timestamp_ms) - int(no_timestamp_ms)) > self.config.max_desync_ms:
            return None

        self._yes_ask = candidate_yes_ask
        self._no_ask = candidate_no_ask
        self._yes_timestamp_ms = int(yes_timestamp_ms)
        self._no_timestamp_ms = int(no_timestamp_ms)
        if base_fee is not None:
            self.record_base_fee(base_fee)
        else:
            self._recompute_net_edge()

        if not self._gas_initialized:
            return None
        if self._net_edge <= self.config.min_yield:
            return None

        return CtfMergeSignal(
            market_id=self.market_id,
            yes_ask=self._yes_ask,
            no_ask=self._no_ask,
            gas_estimate=self._gas_estimate,
            net_edge=self._net_edge,
        )

    def _recompute_net_edge(self) -> None:
        self._net_edge = (
            Decimal("1")
            - (self._yes_ask + self._no_ask)
            - self._gas_estimate
            - self.config.taker_fee_yes
            - self.config.taker_fee_no
            - self.config.slippage_budget
        )
