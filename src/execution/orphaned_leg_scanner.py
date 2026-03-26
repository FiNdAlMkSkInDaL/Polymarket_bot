from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from src.execution.ctf_execution_manifest import CtfExecutionManifest, build_ctf_execution_manifest
from src.execution.ctf_unwind_manifest import CtfUnwindLeg, CtfUnwindManifest
from src.execution.si9_execution_manifest import Si9ExecutionManifest, Si9LegManifest
from src.execution.si9_unwind_manifest import Si9UnwindConfig, Si9UnwindLeg, Si9UnwindManifest
from src.execution.unwind_manifest import RecommendedAction, UnwindManifest


_DECIMAL_TICK = Decimal("0.000001")


@dataclass(frozen=True, slots=True)
class RecoveryOpenPosition:
    coordination_id: str
    strategy_source: Literal["SI9", "CTF"]
    market_id: str
    side: Literal["YES", "NO"]
    filled_size: Decimal
    filled_price: Decimal
    venue_timestamp_ms: int
    expected_leg_count: int
    leg_index: int = 0

    def __post_init__(self) -> None:
        if not str(self.coordination_id or "").strip():
            raise ValueError("coordination_id must be a non-empty string")
        if self.strategy_source not in {"SI9", "CTF"}:
            raise ValueError("strategy_source must be 'SI9' or 'CTF'")
        if not str(self.market_id or "").strip():
            raise ValueError("market_id must be a non-empty string")
        if self.side not in {"YES", "NO"}:
            raise ValueError("side must be 'YES' or 'NO'")
        for field_name in ("filled_size", "filled_price"):
            value = getattr(self, field_name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= Decimal("0"):
                raise ValueError(f"{field_name} must be a strictly positive Decimal")
        if not isinstance(self.venue_timestamp_ms, int) or self.venue_timestamp_ms <= 0:
            raise ValueError("venue_timestamp_ms must be a strictly positive int")
        if not isinstance(self.expected_leg_count, int) or self.expected_leg_count <= 0:
            raise ValueError("expected_leg_count must be a strictly positive int")
        if not isinstance(self.leg_index, int) or self.leg_index < 0:
            raise ValueError("leg_index must be a non-negative int")


@dataclass(frozen=True, slots=True)
class _AggregatedLeg:
    market_id: str
    side: Literal["YES", "NO"]
    filled_size: Decimal
    filled_price: Decimal
    venue_timestamp_ms: int
    leg_index: int


class OrphanedLegRecoveryScanner:
    def __init__(
        self,
        unwind_config: Si9UnwindConfig,
        *,
        si9_max_leg_fill_wait_ms: int = 1,
        si9_cancel_on_stale_ms: int = 1,
        ctf_cancel_on_stale_ms: int = 1,
    ) -> None:
        if not isinstance(si9_max_leg_fill_wait_ms, int) or si9_max_leg_fill_wait_ms <= 0:
            raise ValueError("si9_max_leg_fill_wait_ms must be a strictly positive int")
        if not isinstance(si9_cancel_on_stale_ms, int) or si9_cancel_on_stale_ms <= 0:
            raise ValueError("si9_cancel_on_stale_ms must be a strictly positive int")
        if not isinstance(ctf_cancel_on_stale_ms, int) or ctf_cancel_on_stale_ms <= 0:
            raise ValueError("ctf_cancel_on_stale_ms must be a strictly positive int")
        self._unwind_config = unwind_config
        self._si9_max_leg_fill_wait_ms = si9_max_leg_fill_wait_ms
        self._si9_cancel_on_stale_ms = si9_cancel_on_stale_ms
        self._ctf_cancel_on_stale_ms = ctf_cancel_on_stale_ms

    def scan(
        self,
        open_positions: list[RecoveryOpenPosition],
        current_timestamp_ms: int,
    ) -> tuple[UnwindManifest, ...]:
        timestamp_ms = int(current_timestamp_ms)
        grouped: dict[tuple[str, str], list[RecoveryOpenPosition]] = {}
        for position in open_positions:
            grouped.setdefault((position.strategy_source, position.coordination_id), []).append(position)

        manifests: list[UnwindManifest] = []
        for (strategy_source, coordination_id), positions in grouped.items():
            if strategy_source == "CTF":
                manifest = self._scan_ctf_group(coordination_id, positions, timestamp_ms)
            else:
                manifest = self._scan_si9_group(coordination_id, positions, timestamp_ms)
            if manifest is not None:
                manifests.append(manifest)
        return tuple(manifests)

    def _scan_ctf_group(
        self,
        coordination_id: str,
        positions: list[RecoveryOpenPosition],
        current_timestamp_ms: int,
    ) -> CtfUnwindManifest | None:
        aggregated = self._aggregate_ctf_positions(positions)
        yes_leg = aggregated.get("YES")
        no_leg = aggregated.get("NO")
        yes_size = yes_leg.filled_size if yes_leg is not None else Decimal("0")
        no_size = no_leg.filled_size if no_leg is not None else Decimal("0")
        if yes_leg is not None and no_leg is not None and yes_size == no_size:
            return None

        matched_size = min(yes_size, no_size) if yes_leg is not None and no_leg is not None else Decimal("0")
        hanging_legs: list[CtfUnwindLeg] = []
        orphan_timestamps: list[int] = []
        for leg in (yes_leg, no_leg):
            if leg is None:
                continue
            residual_size = (leg.filled_size - matched_size).quantize(_DECIMAL_TICK)
            if residual_size <= Decimal("0"):
                continue
            hanging_legs.append(
                CtfUnwindLeg(
                    market_id=leg.market_id,
                    side=leg.side,
                    filled_size=residual_size,
                    filled_price=leg.filled_price,
                    current_best_bid=leg.filled_price,
                    estimated_unwind_cost=Decimal("0"),
                    leg_index=leg.leg_index,
                )
            )
            orphan_timestamps.append(leg.venue_timestamp_ms)
        if not hanging_legs:
            return None

        unwind_timestamp_ms = min(orphan_timestamps)
        recommended_action = self._seed_recommendation(unwind_timestamp_ms, current_timestamp_ms)
        original_manifest = self._build_ctf_manifest(coordination_id, yes_leg, no_leg)
        return CtfUnwindManifest(
            cluster_id=coordination_id,
            hanging_legs=tuple(sorted(hanging_legs, key=lambda leg: leg.leg_index)),
            unwind_reason="SECOND_LEG_REJECTED",
            original_manifest=original_manifest,
            unwind_timestamp_ms=unwind_timestamp_ms,
            total_estimated_unwind_cost=Decimal("0"),
            recommended_action=recommended_action,
        )

    def _scan_si9_group(
        self,
        coordination_id: str,
        positions: list[RecoveryOpenPosition],
        current_timestamp_ms: int,
    ) -> Si9UnwindManifest | None:
        aggregated = self._aggregate_si9_positions(positions)
        expected_leg_count = max(position.expected_leg_count for position in positions)
        if len(aggregated) == expected_leg_count:
            filled_sizes = {leg.filled_size for leg in aggregated.values()}
            if len(filled_sizes) == 1:
                return None
            matched_size = min(filled_sizes)
        else:
            matched_size = Decimal("0")

        hanging_legs: list[Si9UnwindLeg] = []
        orphan_timestamps: list[int] = []
        for leg in sorted(aggregated.values(), key=lambda aggregated_leg: aggregated_leg.leg_index):
            residual_size = (leg.filled_size - matched_size).quantize(_DECIMAL_TICK)
            if residual_size <= Decimal("0"):
                continue
            hanging_legs.append(
                Si9UnwindLeg(
                    market_id=leg.market_id,
                    side="YES",
                    filled_size=residual_size,
                    filled_price=leg.filled_price,
                    current_best_bid=leg.filled_price,
                    estimated_unwind_cost=Decimal("0"),
                    leg_index=leg.leg_index,
                )
            )
            orphan_timestamps.append(leg.venue_timestamp_ms)
        if not hanging_legs:
            return None

        unwind_timestamp_ms = min(orphan_timestamps)
        recommended_action = self._seed_recommendation(unwind_timestamp_ms, current_timestamp_ms)
        original_manifest = self._build_si9_manifest(coordination_id, aggregated)
        return Si9UnwindManifest(
            cluster_id=coordination_id,
            hanging_legs=tuple(hanging_legs),
            unwind_reason="MANUAL_ABORT",
            original_manifest=original_manifest,
            unwind_timestamp_ms=unwind_timestamp_ms,
            total_estimated_unwind_cost=Decimal("0"),
            recommended_action=recommended_action,
        )

    @staticmethod
    def _aggregate_si9_positions(positions: list[RecoveryOpenPosition]) -> dict[str, _AggregatedLeg]:
        weighted_notional: dict[str, Decimal] = {}
        totals: dict[str, _AggregatedLeg] = {}
        for position in positions:
            existing = totals.get(position.market_id)
            position_notional = position.filled_size * position.filled_price
            if existing is None:
                totals[position.market_id] = _AggregatedLeg(
                    market_id=position.market_id,
                    side="YES",
                    filled_size=position.filled_size.quantize(_DECIMAL_TICK),
                    filled_price=position.filled_price.quantize(_DECIMAL_TICK),
                    venue_timestamp_ms=position.venue_timestamp_ms,
                    leg_index=position.leg_index,
                )
                weighted_notional[position.market_id] = position_notional
                continue
            total_size = (existing.filled_size + position.filled_size).quantize(_DECIMAL_TICK)
            weighted_notional[position.market_id] += position_notional
            totals[position.market_id] = _AggregatedLeg(
                market_id=position.market_id,
                side="YES",
                filled_size=total_size,
                filled_price=(weighted_notional[position.market_id] / total_size).quantize(_DECIMAL_TICK),
                venue_timestamp_ms=min(existing.venue_timestamp_ms, position.venue_timestamp_ms),
                leg_index=min(existing.leg_index, position.leg_index),
            )
        return totals

    @staticmethod
    def _aggregate_ctf_positions(positions: list[RecoveryOpenPosition]) -> dict[str, _AggregatedLeg]:
        weighted_notional: dict[str, Decimal] = {}
        totals: dict[str, _AggregatedLeg] = {}
        for position in positions:
            side_key = position.side
            existing = totals.get(side_key)
            position_notional = position.filled_size * position.filled_price
            if existing is None:
                totals[side_key] = _AggregatedLeg(
                    market_id=position.market_id,
                    side=position.side,
                    filled_size=position.filled_size.quantize(_DECIMAL_TICK),
                    filled_price=position.filled_price.quantize(_DECIMAL_TICK),
                    venue_timestamp_ms=position.venue_timestamp_ms,
                    leg_index=position.leg_index,
                )
                weighted_notional[side_key] = position_notional
                continue
            total_size = (existing.filled_size + position.filled_size).quantize(_DECIMAL_TICK)
            weighted_notional[side_key] += position_notional
            totals[side_key] = _AggregatedLeg(
                market_id=position.market_id,
                side=position.side,
                filled_size=total_size,
                filled_price=(weighted_notional[side_key] / total_size).quantize(_DECIMAL_TICK),
                venue_timestamp_ms=min(existing.venue_timestamp_ms, position.venue_timestamp_ms),
                leg_index=min(existing.leg_index, position.leg_index),
            )
        return totals

    def _build_si9_manifest(
        self,
        coordination_id: str,
        aggregated: dict[str, _AggregatedLeg],
    ) -> Si9ExecutionManifest:
        ordered_legs = tuple(sorted(aggregated.values(), key=lambda leg: leg.leg_index))
        required_share_counts = max(leg.filled_size for leg in ordered_legs)
        legs = tuple(
            Si9LegManifest(
                market_id=leg.market_id,
                side="YES",
                target_price=leg.filled_price,
                target_size=required_share_counts,
                is_bottleneck=(index == 0),
                leg_index=index,
            )
            for index, leg in enumerate(ordered_legs)
        )
        manifest_timestamp_ms = min(leg.venue_timestamp_ms for leg in ordered_legs)
        return Si9ExecutionManifest(
            cluster_id=coordination_id,
            legs=legs,
            net_edge=Decimal("0"),
            required_share_counts=required_share_counts,
            bottleneck_market_id=legs[0].market_id,
            manifest_timestamp_ms=manifest_timestamp_ms,
            max_leg_fill_wait_ms=self._si9_max_leg_fill_wait_ms,
            cancel_on_stale_ms=self._si9_cancel_on_stale_ms,
        )

    def _build_ctf_manifest(
        self,
        coordination_id: str,
        yes_leg: _AggregatedLeg | None,
        no_leg: _AggregatedLeg | None,
    ) -> CtfExecutionManifest:
        yes_price = self._ctf_leg_price(yes_leg, no_leg, preferred_side="YES")
        no_price = self._ctf_leg_price(no_leg, yes_leg, preferred_side="NO")
        required_size = max(
            yes_leg.filled_size if yes_leg is not None else Decimal("0"),
            no_leg.filled_size if no_leg is not None else Decimal("0"),
        )
        manifest_timestamp_ms = min(
            [leg.venue_timestamp_ms for leg in (yes_leg, no_leg) if leg is not None]
        )
        return build_ctf_execution_manifest(
            market_id=coordination_id,
            yes_price=yes_price,
            no_price=no_price,
            net_edge=Decimal("0"),
            gas_estimate=Decimal("0"),
            default_anchor_volume=required_size,
            max_capital_per_signal=(required_size * (yes_price + no_price)).quantize(_DECIMAL_TICK),
            max_size_per_leg=required_size,
            taker_fee_yes=Decimal("0"),
            taker_fee_no=Decimal("0"),
            manifest_timestamp_ms=manifest_timestamp_ms,
            cancel_on_stale_ms=self._ctf_cancel_on_stale_ms,
        )

    @staticmethod
    def _ctf_leg_price(
        present_leg: _AggregatedLeg | None,
        other_leg: _AggregatedLeg | None,
        *,
        preferred_side: Literal["YES", "NO"],
    ) -> Decimal:
        if present_leg is not None:
            return present_leg.filled_price.quantize(_DECIMAL_TICK)
        assert other_leg is not None
        if preferred_side == "YES":
            return min(Decimal("0.999999"), (other_leg.filled_price + Decimal("0.010000"))).quantize(_DECIMAL_TICK)
        return min(Decimal("0.999999"), (other_leg.filled_price + Decimal("0.010000"))).quantize(_DECIMAL_TICK)

    def _seed_recommendation(
        self,
        venue_timestamp_ms: int,
        current_timestamp_ms: int,
    ) -> RecommendedAction:
        elapsed_ms = int(current_timestamp_ms) - int(venue_timestamp_ms)
        if elapsed_ms > self._unwind_config.max_hold_recovery_ms:
            return "MARKET_SELL"
        return "HOLD_FOR_RECOVERY"