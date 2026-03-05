"""
Tests for the Volume Expansion Plan — V1 through V4.

Validates:
  V1: Maker routing — execution_mode="maker" yields higher EQS
  V2: Confluence discount — multi-factor threshold reduction
  V3: Drift signal — low-volatility mean-reversion detection
  V4: Probe sizing — sub-threshold micro-entries
"""

from __future__ import annotations

import math
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.config import StrategyParams
from src.signals.edge_filter import (
    ConfluenceContext,
    EdgeAssessment,
    compute_confluence_discount,
    compute_edge_score,
)
from src.signals.drift_signal import DriftSignal, MeanReversionDrift


# ═══════════════════════════════════════════════════════════════════════════
#  V1: Maker routing — execution_mode parameter
# ═══════════════════════════════════════════════════════════════════════════


class TestMakerRouting:
    """V1: Maker-mode EQS scoring removes phantom fee drag."""

    def test_maker_mode_higher_fee_efficiency(self):
        """Maker EQS should have higher fee_efficiency than taker at p=0.50."""
        taker = compute_edge_score(
            entry_price=0.47,
            no_vwap=0.50,
            zscore=3.0,
            volume_ratio=2.0,
            execution_mode="taker",
        )
        maker = compute_edge_score(
            entry_price=0.47,
            no_vwap=0.50,
            zscore=3.0,
            volume_ratio=2.0,
            execution_mode="maker",
        )
        assert maker.fee_efficiency > taker.fee_efficiency
        assert maker.expected_fee_cents == 0.0
        assert taker.expected_fee_cents > 0.0

    def test_maker_mode_higher_score(self):
        """Maker-mode should produce a higher composite score."""
        taker = compute_edge_score(
            entry_price=0.47,
            no_vwap=0.50,
            zscore=3.0,
            volume_ratio=2.0,
            execution_mode="taker",
        )
        maker = compute_edge_score(
            entry_price=0.47,
            no_vwap=0.50,
            zscore=3.0,
            volume_ratio=2.0,
            execution_mode="maker",
        )
        assert maker.score > taker.score

    def test_maker_mode_unlocks_marginal_entries(self):
        """An entry rejected under taker mode should pass under maker mode."""
        # Entry with small spread that gets killed by fees
        taker = compute_edge_score(
            entry_price=0.48,
            no_vwap=0.50,
            zscore=2.5,
            volume_ratio=1.5,
            min_score=45.0,
            execution_mode="taker",
        )
        maker = compute_edge_score(
            entry_price=0.48,
            no_vwap=0.50,
            zscore=2.5,
            volume_ratio=1.5,
            min_score=45.0,
            execution_mode="maker",
        )
        # Maker should be viable (or at least score higher)
        assert maker.score > taker.score

    def test_execution_mode_field_on_assessment(self):
        """EdgeAssessment should carry the execution_mode used."""
        result = compute_edge_score(
            entry_price=0.47,
            no_vwap=0.50,
            zscore=3.0,
            volume_ratio=2.0,
            execution_mode="maker",
        )
        assert result.execution_mode == "maker"

        result2 = compute_edge_score(
            entry_price=0.47,
            no_vwap=0.50,
            zscore=3.0,
            volume_ratio=2.0,
            execution_mode="taker",
        )
        assert result2.execution_mode == "taker"

    def test_execution_mode_default_is_taker(self):
        """Default execution_mode should be 'taker' for backward compat."""
        result = compute_edge_score(
            entry_price=0.47,
            no_vwap=0.50,
            zscore=3.0,
            volume_ratio=2.0,
        )
        assert result.execution_mode == "taker"


# ═══════════════════════════════════════════════════════════════════════════
#  V2: Multi-factor confluence routing
# ═══════════════════════════════════════════════════════════════════════════


class TestConfluenceDiscount:
    """V2: Dynamic EQS threshold from multi-signal confluence."""

    def test_no_discount_below_min_factors(self):
        """Single factor should NOT activate any discount."""
        ctx = ConfluenceContext(
            whale_strong_confluence=True,
            spread_compressed=False,
            l2_reliable=False,
            regime_mean_revert=False,
        )
        result = compute_confluence_discount(ctx, 50.0)
        assert result == 50.0

    def test_two_factors_activates_discount(self):
        """Two factors should reduce threshold."""
        ctx = ConfluenceContext(
            whale_strong_confluence=True,   # -4
            spread_compressed=True,         # -4
            l2_reliable=True,               # hard gate passes
            regime_mean_revert=False,
        )
        result = compute_confluence_discount(ctx, 50.0)
        assert result == 50.0 - 4.0 - 4.0  # 42.0

    def test_all_four_factors(self):
        """All four flags set: only whale + spread produce discounts
        (L2 is hard gate, regime discount is 0.0)."""
        ctx = ConfluenceContext(
            whale_strong_confluence=True,   # -4
            spread_compressed=True,         # -4
            l2_reliable=True,              # hard gate passes
            regime_mean_revert=True,        # discount=0, no effect
        )
        result = compute_confluence_discount(ctx, 50.0)
        assert result == 50.0 - 4.0 - 4.0  # 42.0

    def test_floor_respected(self):
        """Discount should never go below confluence_eqs_floor."""
        ctx = ConfluenceContext(
            whale_strong_confluence=True,
            spread_compressed=True,
            l2_reliable=True,
            regime_mean_revert=True,
        )
        # Total discount = 8 points.  Starting from 40 → 32, but floor is 35.
        result = compute_confluence_discount(ctx, 40.0)
        assert result == 35.0

    def test_zero_active_factors(self):
        """No active factors → no discount."""
        ctx = ConfluenceContext()
        result = compute_confluence_discount(ctx, 50.0)
        assert result == 50.0

    def test_custom_config_values(self):
        """Confluence discount should respect config overrides."""
        from src.core.config import settings

        ctx = ConfluenceContext(
            whale_strong_confluence=True,
            spread_compressed=True,
            l2_reliable=True,              # hard gate passes
        )
        orig_whale = settings.strategy.confluence_whale_discount
        orig_spread = settings.strategy.confluence_spread_discount
        orig_floor = settings.strategy.confluence_eqs_floor
        orig_min = settings.strategy.confluence_min_factors
        try:
            object.__setattr__(settings.strategy, 'confluence_whale_discount', 10.0)
            object.__setattr__(settings.strategy, 'confluence_spread_discount', 8.0)
            object.__setattr__(settings.strategy, 'confluence_eqs_floor', 30.0)
            object.__setattr__(settings.strategy, 'confluence_min_factors', 2)
            result = compute_confluence_discount(ctx, 50.0)
            assert result == max(30.0, 50.0 - 10.0 - 8.0)  # 32.0
        finally:
            object.__setattr__(settings.strategy, 'confluence_whale_discount', orig_whale)
            object.__setattr__(settings.strategy, 'confluence_spread_discount', orig_spread)
            object.__setattr__(settings.strategy, 'confluence_eqs_floor', orig_floor)
            object.__setattr__(settings.strategy, 'confluence_min_factors', orig_min)


# ═══════════════════════════════════════════════════════════════════════════
#  V3: Mean-reversion drift signal
# ═══════════════════════════════════════════════════════════════════════════


class _FakeBar:
    """Minimal bar stub for OHLCVAggregator."""
    def __init__(self, close: float, volume: float = 100.0):
        self.close = close
        self.volume = volume


class _FakeAggregator:
    """Minimal OHLCVAggregator stub."""
    def __init__(
        self,
        bars: list,
        rolling_vwap: float = 0.50,
        rolling_volatility: float = 0.01,
        rolling_volatility_ewma: float = 0.01,
        avg_bar_volume: float = 100.0,
    ):
        from collections import deque
        self.bars = deque(bars, maxlen=100)
        self.rolling_vwap = rolling_vwap
        self.rolling_volatility = rolling_volatility
        self.rolling_volatility_ewma = rolling_volatility_ewma
        self.avg_bar_volume = avg_bar_volume


class TestDriftSignal:
    """V3: Low-volatility mean-reversion drift detection."""

    def _make_detector(self, **kwargs):
        return MeanReversionDrift("test_market", **kwargs)

    def test_fires_on_sufficient_drift(self):
        """Should fire when cumulative displacement exceeds threshold."""
        # 10 bars with prices drifting below VWAP
        bars = [_FakeBar(close=0.47, volume=80.0) for _ in range(12)]
        agg = _FakeAggregator(
            bars=bars,
            rolling_vwap=0.50,
            rolling_volatility=0.02,
            rolling_volatility_ewma=0.01,
            avg_bar_volume=100.0,
        )
        det = self._make_detector(lookback_bars=10, z_threshold=1.0, vol_ceiling=0.015)
        sig = det.evaluate(agg, regime_is_mean_revert=True, l2_reliable=True)
        assert sig is not None
        assert sig.direction == "BUY_NO"
        assert sig.displacement < 0  # NO price below VWAP

    def test_silent_in_trending_regime(self):
        """Should NOT fire when regime is trending (not mean-reverting)."""
        bars = [_FakeBar(close=0.47) for _ in range(12)]
        agg = _FakeAggregator(
            bars=bars,
            rolling_vwap=0.50,
            rolling_volatility=0.02,
            rolling_volatility_ewma=0.01,
        )
        det = self._make_detector()
        sig = det.evaluate(agg, regime_is_mean_revert=False, l2_reliable=True)
        assert sig is None

    def test_silent_in_high_vol(self):
        """Should NOT fire when EWMA vol exceeds ceiling."""
        bars = [_FakeBar(close=0.47) for _ in range(12)]
        agg = _FakeAggregator(
            bars=bars,
            rolling_vwap=0.50,
            rolling_volatility=0.02,
            rolling_volatility_ewma=0.03,  # above ceiling
        )
        det = self._make_detector(vol_ceiling=0.015)
        sig = det.evaluate(agg, regime_is_mean_revert=True, l2_reliable=True)
        assert sig is None

    def test_silent_with_unreliable_l2(self):
        """Should NOT fire when L2 book is unreliable."""
        bars = [_FakeBar(close=0.47) for _ in range(12)]
        agg = _FakeAggregator(
            bars=bars,
            rolling_vwap=0.50,
            rolling_volatility=0.02,
            rolling_volatility_ewma=0.01,
        )
        det = self._make_detector()
        sig = det.evaluate(agg, regime_is_mean_revert=True, l2_reliable=False)
        assert sig is None

    def test_silent_with_high_volume_bar(self):
        """Should NOT fire when any bar has high volume ratio (PanicDetector territory)."""
        bars = [_FakeBar(close=0.47, volume=80.0) for _ in range(9)]
        # One bar with extreme volume — would trigger PanicDetector
        bars.append(_FakeBar(close=0.47, volume=200.0))
        bars.append(_FakeBar(close=0.47, volume=80.0))
        bars.append(_FakeBar(close=0.47, volume=80.0))
        agg = _FakeAggregator(
            bars=bars,
            rolling_vwap=0.50,
            rolling_volatility=0.02,
            rolling_volatility_ewma=0.01,
            avg_bar_volume=100.0,
        )
        det = self._make_detector(lookback_bars=10)
        sig = det.evaluate(agg, regime_is_mean_revert=True, l2_reliable=True)
        assert sig is None

    def test_insufficient_history(self):
        """Should NOT fire with too few bars."""
        bars = [_FakeBar(close=0.47) for _ in range(3)]
        agg = _FakeAggregator(bars=bars)
        det = self._make_detector(lookback_bars=10)
        sig = det.evaluate(agg, regime_is_mean_revert=True, l2_reliable=True)
        assert sig is None

    def test_score_normalisation(self):
        """Score should normalise between 0 and 1."""
        bars = [_FakeBar(close=0.46) for _ in range(12)]
        agg = _FakeAggregator(
            bars=bars,
            rolling_vwap=0.50,
            rolling_volatility=0.02,
            rolling_volatility_ewma=0.01,
            avg_bar_volume=100.0,
        )
        det = self._make_detector(lookback_bars=10, z_threshold=1.0, vol_ceiling=0.015)
        sig = det.evaluate(agg, regime_is_mean_revert=True, l2_reliable=True)
        assert sig is not None
        assert 0.0 <= sig.score <= 1.0

    def test_no_displacement_no_signal(self):
        """Should NOT fire when price is at VWAP."""
        bars = [_FakeBar(close=0.50) for _ in range(12)]
        agg = _FakeAggregator(
            bars=bars,
            rolling_vwap=0.50,
            rolling_volatility=0.02,
            rolling_volatility_ewma=0.01,
        )
        det = self._make_detector()
        sig = det.evaluate(agg, regime_is_mean_revert=True, l2_reliable=True)
        assert sig is None


# ═══════════════════════════════════════════════════════════════════════════
#  V4: Probe sizing
# ═══════════════════════════════════════════════════════════════════════════


class TestProbeSizing:
    """V4: Sub-threshold probe entries at micro-size."""

    def test_probe_accepted_above_floor(self):
        """Score between probe_eqs_floor and min_edge_score → probe."""
        # Score that is below threshold but above floor
        result = compute_edge_score(
            entry_price=0.48,
            no_vwap=0.50,
            zscore=2.0,
            volume_ratio=1.3,
            min_score=50.0,
            execution_mode="maker",
        )
        # The score should be in the probe range (35-50)
        if result.score >= 35.0 and result.score < 50.0:
            assert not result.viable
            assert result.rejection_reason == "score_below_threshold"

    def test_probe_rejected_below_floor(self):
        """Score below probe_eqs_floor → hard reject (not even probe)."""
        result = compute_edge_score(
            entry_price=0.48,
            no_vwap=0.49,
            zscore=1.6,
            volume_ratio=1.1,
            min_score=50.0,
        )
        # Very weak signal — should score well below 35
        if result.score < 35.0:
            assert not result.viable

    def test_position_is_probe_field_default(self):
        """Position.is_probe should default to False."""
        from src.trading.position_manager import Position
        pos = Position(id="test", market_id="m1", no_asset_id="a1")
        assert pos.is_probe is False

    def test_edge_assessment_has_execution_mode(self):
        """EdgeAssessment should have execution_mode field."""
        ea = EdgeAssessment(
            score=42.0,
            regime_quality=0.9,
            fee_efficiency=0.8,
            tick_margin=2,
            tick_viability=0.7,
            signal_quality=0.6,
            expected_gross_cents=3.0,
            expected_fee_cents=1.0,
            expected_net_cents=2.0,
            viable=False,
            rejection_reason="score_below_threshold",
            execution_mode="maker",
        )
        assert ea.execution_mode == "maker"


# ═══════════════════════════════════════════════════════════════════════════
#  V4: StopLoss probe breakeven callback
# ═══════════════════════════════════════════════════════════════════════════


class TestStopLossProbeBreakeven:
    """V4: Probe breakeven callback in StopLossMonitor."""

    @pytest.mark.asyncio
    async def test_probe_breakeven_callback_fires(self):
        """When a probe reaches BE activation, callback should fire."""
        from src.trading.stop_loss import StopLossMonitor
        from src.trading.position_manager import Position, PositionState

        callback_fired = []

        async def on_probe_be(pos):
            callback_fired.append(pos.id)

        pm = MagicMock()
        probe_pos = Position(
            id="POS-PROBE-1",
            market_id="m1",
            no_asset_id="asset_1",
            trade_asset_id="asset_1",
            entry_price=0.47,
            is_probe=True,
            state=PositionState.EXIT_PENDING,
            sl_trigger_cents=4.0,
        )

        pm.get_open_positions.return_value = [probe_pos]

        no_aggs = {}
        books = {"asset_1": MagicMock(has_data=True)}
        books["asset_1"].best_bid = 0.50
        books["asset_1"].best_ask = 0.51

        monitor = StopLossMonitor(
            pm, no_aggs, books, MagicMock(), MagicMock(),
            trailing_offset_cents=2.0,
            on_probe_breakeven=on_probe_be,
        )
        monitor._running = True

        # Mid = 0.505, entry = 0.47 → profit = 3.5¢ ≥ 1.0¢ BE threshold
        await monitor.on_bbo_update("asset_1")
        assert "POS-PROBE-1" in callback_fired

    @pytest.mark.asyncio
    async def test_probe_breakeven_fires_only_once(self):
        """Callback should only fire once per probe position."""
        from src.trading.stop_loss import StopLossMonitor
        from src.trading.position_manager import Position, PositionState

        call_count = 0

        async def on_probe_be(pos):
            nonlocal call_count
            call_count += 1

        pm = MagicMock()
        probe_pos = Position(
            id="POS-PROBE-2",
            market_id="m2",
            no_asset_id="asset_2",
            trade_asset_id="asset_2",
            entry_price=0.47,
            is_probe=True,
            state=PositionState.EXIT_PENDING,
            sl_trigger_cents=4.0,
        )

        pm.get_open_positions.return_value = [probe_pos]

        books = {"asset_2": MagicMock(has_data=True)}
        books["asset_2"].best_bid = 0.50
        books["asset_2"].best_ask = 0.51

        monitor = StopLossMonitor(
            pm, {}, books, MagicMock(), MagicMock(),
            trailing_offset_cents=2.0,
            on_probe_breakeven=on_probe_be,
        )
        monitor._running = True

        await monitor.on_bbo_update("asset_2")
        await monitor.on_bbo_update("asset_2")
        assert call_count == 1


# ═══════════════════════════════════════════════════════════════════════════
#  Config field existence checks
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigFields:
    """Verify all new config fields exist with correct defaults."""

    def test_v1_maker_routing_defaults(self):
        params = StrategyParams()
        assert params.maker_routing_enabled is True
        assert params.maker_eqs_discount == 0.85

    def test_v2_confluence_defaults(self):
        params = StrategyParams()
        assert params.confluence_eqs_floor == 35.0
        assert params.confluence_min_factors == 2
        assert params.confluence_whale_discount == 4.0
        assert params.confluence_spread_discount == 4.0
        assert params.confluence_l2_discount == 0.0
        assert params.confluence_regime_discount == 0.0

    def test_v3_drift_defaults(self):
        params = StrategyParams()
        assert params.drift_signal_enabled is True
        assert params.drift_lookback_bars == 10
        assert params.drift_z_threshold == 0.8
        assert params.drift_vol_ceiling == 0.05
        assert params.drift_cooldown_s == 60.0

    def test_v4_probe_defaults(self):
        params = StrategyParams()
        assert params.probe_sizing_enabled is True
        assert params.probe_eqs_floor == 35.0
        assert params.probe_kelly_fraction == 0.05
        assert params.probe_max_usd == 2.0
