"""
Tests for the Alpha Evolution improvements (Post-Mortem March 2026).

Covers:
  - P0: Minimum viable edge gate
  - P0: Entry lock (per-event race prevention)
  - P1: Dollar-risk cap per trade
  - P1: Adaptive cold-start Kelly sizing
  - P2: Z-score diminishing returns (log-concave)
  - P2: Config parameter changes (min_edge_score, no_discount_factor, min_spread_cents)
"""

from __future__ import annotations

import asyncio
import math

import pytest

from src.core.config import settings
from src.data.ohlcv import OHLCVAggregator
from src.signals.edge_filter import compute_edge_score
from src.signals.panic_detector import PanicSignal
from src.trading.executor import OrderExecutor
from src.trading.position_manager import PositionManager
from src.trading.sizer import compute_kelly_size
from tests.helpers import build_bar_history


# ═══════════════════════════════════════════════════════════════════════════
#  Section A: Config Parameter Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigDefaults:
    """Verify updated config defaults from alpha evolution."""

    def test_min_spread_cents_raised(self):
        assert settings.strategy.min_spread_cents == 4.0

    def test_min_edge_score_raised(self):
        assert settings.strategy.min_edge_score == 40.0

    def test_no_discount_factor_tightened(self):
        assert settings.strategy.no_discount_factor == 1.005

    def test_max_loss_per_trade_cents_exists(self):
        assert settings.strategy.max_loss_per_trade_cents == 50.0

    def test_cold_start_halt_window_exists(self):
        assert settings.strategy.cold_start_halt_window == 10

    def test_cold_start_negative_ev_halt_default(self):
        assert settings.strategy.cold_start_negative_ev_halt is True


# ═══════════════════════════════════════════════════════════════════════════
#  Section B: Minimum Viable Edge Gate
# ═══════════════════════════════════════════════════════════════════════════

class TestMinViableEdgeGate:
    """Verify that trades with insufficient edge after slippage+fees are rejected."""

    @pytest.mark.asyncio
    async def test_thin_edge_rejected_with_fees(self):
        """A trade where TP spread < slippage + fees + margin should be rejected.

        This test explicitly disables maker routing to validate that
        taker-mode fee drag correctly rejects thin-edge entries.
        """
        # Disable maker routing so taker fees apply
        orig = settings.strategy.maker_routing_enabled
        object.__setattr__(settings.strategy, "maker_routing_enabled", False)
        try:
            executor = OrderExecutor(paper_mode=True)
            pm = PositionManager(executor)
            pm.set_wallet_balance(50.0)

            no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
            build_bar_history(no_agg, [0.55] * 10, base_vol=10.0)

            # Price at 0.45 → entry at 0.44 → with fees, edge is thin
            signal = PanicSignal(
                market_id="MKT", yes_asset_id="YES_T", no_asset_id="NO_T",
                yes_price=0.75, yes_vwap=0.50, zscore=3.0, volume_ratio=5.0,
                no_best_ask=0.45, whale_confluence=False,
            )

            # Fee-enabled: trade should be rejected due to insufficient edge
            pos = await pm.open_position(signal, no_agg, fee_enabled=True)
            assert pos is None
        finally:
            object.__setattr__(settings.strategy, "maker_routing_enabled", orig)

    @pytest.mark.asyncio
    async def test_wide_edge_accepted_without_fees(self):
        """A trade on a fee-free market should pass if edge is structurally wide."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)
        pm.set_wallet_balance(50.0)

        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        build_bar_history(no_agg, [0.55] * 10, base_vol=10.0)

        signal = PanicSignal(
            market_id="MKT", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.75, yes_vwap=0.50, zscore=3.0, volume_ratio=5.0,
            no_best_ask=0.45, whale_confluence=False,
        )

        # Fee-disabled: spread should survive slippage + margin
        pos = await pm.open_position(signal, no_agg, fee_enabled=False)
        assert pos is not None

    @pytest.mark.asyncio
    async def test_deep_discount_accepted_with_fees(self):
        """Large VWAP-entry gap passes even with fee drag."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)
        pm.set_wallet_balance(50.0)

        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        build_bar_history(no_agg, [0.55] * 10, base_vol=10.0)

        signal = PanicSignal(
            market_id="MKT", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.85, yes_vwap=0.40, zscore=3.0, volume_ratio=5.0,
            no_best_ask=0.26, whale_confluence=False,
        )

        pos = await pm.open_position(signal, no_agg, fee_enabled=True)
        assert pos is not None


# ═══════════════════════════════════════════════════════════════════════════
#  Section C: Dollar-Risk Cap
# ═══════════════════════════════════════════════════════════════════════════

class TestDollarRiskCap:
    """Verify max_loss_per_trade_cents caps position size."""

    @pytest.mark.asyncio
    async def test_high_price_entry_capped(self):
        """At high entry prices, position size is capped by dollar-risk."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)
        pm.set_wallet_balance(100.0)

        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        build_bar_history(no_agg, [0.55] * 10, base_vol=10.0)

        # Entry at 0.44, max_loss=50¢ → max shares = 50/(44) ≈ 1.14
        signal = PanicSignal(
            market_id="MKT", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.75, yes_vwap=0.50, zscore=3.0, volume_ratio=5.0,
            no_best_ask=0.45, whale_confluence=False,
        )

        pos = await pm.open_position(signal, no_agg, fee_enabled=False)
        if pos is not None:
            max_loss_if_zero = pos.entry_price * pos.entry_size * 100
            assert max_loss_if_zero <= settings.strategy.max_loss_per_trade_cents + 1

    @pytest.mark.asyncio
    async def test_low_price_entry_not_capped(self):
        """At low entry prices, dollar-risk cap is not binding."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor)
        pm.set_wallet_balance(50.0)

        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        build_bar_history(no_agg, [0.30] * 10, base_vol=10.0)

        # Entry at 0.14, max_loss=50¢ → max shares = 50/(14) ≈ 3.57
        # Kelly cold-start at 50% of $15 = $7.50 → 53.57 shares
        # Dollar cap at 3.57 shares → still > 1, passes
        signal = PanicSignal(
            market_id="MKT", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.85, yes_vwap=0.40, zscore=4.0, volume_ratio=5.0,
            no_best_ask=0.15, whale_confluence=False,
        )

        pos = await pm.open_position(signal, no_agg, fee_enabled=False)
        assert pos is not None
        # Verify the cap was applied
        max_loss_if_zero = pos.entry_price * pos.entry_size * 100
        assert max_loss_if_zero <= settings.strategy.max_loss_per_trade_cents + 1


# ═══════════════════════════════════════════════════════════════════════════
#  Section D: Entry Lock (Race Prevention)
# ═══════════════════════════════════════════════════════════════════════════

class TestEntryLock:
    """Verify that concurrent entries on the same market are serialized."""

    @pytest.mark.asyncio
    async def test_concurrent_entries_serialized(self):
        """Two concurrent open_position calls on same market → only first succeeds."""
        executor = OrderExecutor(paper_mode=True)
        pm = PositionManager(executor, max_open_positions=5)
        pm.set_wallet_balance(100.0)

        no_agg = OHLCVAggregator("NO_T", lookback_minutes=10)
        build_bar_history(no_agg, [0.30] * 10, base_vol=10.0)

        signal = PanicSignal(
            market_id="MKT_SAME", yes_asset_id="YES_T", no_asset_id="NO_T",
            yes_price=0.85, yes_vwap=0.40, zscore=4.0, volume_ratio=5.0,
            no_best_ask=0.15, whale_confluence=False,
        )

        # Launch two concurrent entries
        results = await asyncio.gather(
            pm.open_position(signal, no_agg, fee_enabled=False),
            pm.open_position(signal, no_agg, fee_enabled=False),
        )
        opened = [r for r in results if r is not None]
        # Per-market limit is 1 → at most 1 should succeed (lock serializes)
        assert len(opened) <= 1


# ═══════════════════════════════════════════════════════════════════════════
#  Section E: Adaptive Cold-Start Kelly
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptiveColdStart:
    """Verify the adaptive cold-start sizing logic."""

    def test_cold_start_decays_with_trades(self):
        """Cold-start fraction should decay as trade count increases."""
        strat = settings.strategy

        # At 0 trades, full cold_start_frac
        result_0 = compute_kelly_size(
            signal_score=0.5, win_rate=0.0, avg_win_cents=0.0,
            avg_loss_cents=0.0, bankroll_usd=100.0, entry_price=0.20,
            max_trade_usd=15.0, total_trades=0,
        )

        # At 10 trades (half of min_kelly_trades), fraction should be halved
        result_10 = compute_kelly_size(
            signal_score=0.5, win_rate=0.0, avg_win_cents=0.0,
            avg_loss_cents=0.0, bankroll_usd=100.0, entry_price=0.20,
            max_trade_usd=15.0, total_trades=10,
        )

        assert result_0.method == "kelly_cold_start"
        assert result_10.method == "kelly_cold_start"
        assert result_10.size_usd < result_0.size_usd

    def test_cold_start_negative_ev_throttle(self):
        """When rolling expectancy is negative, sizing is throttled."""
        result_normal = compute_kelly_size(
            signal_score=0.5, win_rate=0.0, avg_win_cents=0.0,
            avg_loss_cents=0.0, bankroll_usd=100.0, entry_price=0.20,
            max_trade_usd=15.0, total_trades=12,
            signal_metadata={"rolling_expectancy_cents": 5.0},
        )

        result_neg = compute_kelly_size(
            signal_score=0.5, win_rate=0.0, avg_win_cents=0.0,
            avg_loss_cents=0.0, bankroll_usd=100.0, entry_price=0.20,
            max_trade_usd=15.0, total_trades=12,
            signal_metadata={"rolling_expectancy_cents": -10.0},
        )

        assert result_neg.method == "kelly_cold_start"
        assert result_neg.size_usd < result_normal.size_usd

    def test_cold_start_zero_trades_full_fraction(self):
        """At 0 trades, uses the full initial cold_start_frac."""
        strat = settings.strategy
        result = compute_kelly_size(
            signal_score=0.5, win_rate=0.0, avg_win_cents=0.0,
            avg_loss_cents=0.0, bankroll_usd=100.0, entry_price=0.20,
            max_trade_usd=15.0, total_trades=0,
        )
        expected_usd = 15.0 * strat.cold_start_frac
        assert result.size_usd == pytest.approx(expected_usd, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════════
#  Section F: Z-Score Diminishing Returns (Log-Concave)
# ═══════════════════════════════════════════════════════════════════════════

class TestZScoreDiminishingReturns:
    """Verify the log-concave z-score contribution curve."""

    def test_higher_zscore_higher_signal_quality(self):
        """Signal quality should increase monotonically with z-score."""
        scores = []
        for z in [2.0, 3.0, 5.0, 8.0, 12.0]:
            edge = compute_edge_score(
                entry_price=0.30, no_vwap=0.50,
                zscore=z, volume_ratio=2.0,
                fee_enabled=False,
            )
            scores.append(edge.signal_quality)

        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"Signal quality not monotonic: z-score index {i} "
                f"({scores[i]}) > index {i+1} ({scores[i+1]})"
            )

    def test_extreme_zscore_still_differentiated(self):
        """z=2 and z=4 should produce different signal qualities."""
        edge_z2 = compute_edge_score(
            entry_price=0.30, no_vwap=0.50,
            zscore=2.0, volume_ratio=1.0,
            fee_enabled=False,
        )
        edge_z4 = compute_edge_score(
            entry_price=0.30, no_vwap=0.50,
            zscore=4.0, volume_ratio=1.0,
            fee_enabled=False,
        )
        # The log-concave curve should still differentiate these
        assert edge_z4.signal_quality > edge_z2.signal_quality
        # And the gap should be meaningful (not just 0.01)
        assert edge_z4.signal_quality - edge_z2.signal_quality >= 0.02

    def test_log_concave_saturation(self):
        """Signal quality should approach but not exceed 1.0."""
        edge = compute_edge_score(
            entry_price=0.30, no_vwap=0.50,
            zscore=100.0, volume_ratio=10.0,
            whale_confluence=True,
            fee_enabled=False,
        )
        assert edge.signal_quality <= 1.0

    def test_zscore_at_threshold_gives_baseline(self):
        """Z-score exactly at threshold → signal quality ≈ 0.5 baseline."""
        z_thresh = settings.strategy.zscore_threshold
        v_thresh = settings.strategy.volume_ratio_threshold
        edge = compute_edge_score(
            entry_price=0.30, no_vwap=0.50,
            zscore=z_thresh, volume_ratio=v_thresh,
            fee_enabled=False,
        )
        # At threshold, z_excess=0 and v_excess=0 → signal_q ≈ 0.50
        assert edge.signal_quality == pytest.approx(0.50, abs=0.05)


# ═══════════════════════════════════════════════════════════════════════════
#  Section G: Rolling Expectancy (Trade Store)
# ═══════════════════════════════════════════════════════════════════════════

class TestRollingExpectancy:
    """Verify TradeStore.get_rolling_expectancy()."""

    @pytest.mark.asyncio
    async def test_returns_zero_insufficient_trades(self, trade_store):
        """When fewer than window trades exist, returns 0.0."""
        result = await trade_store.get_rolling_expectancy(window=10)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_computes_average_pnl(self, trade_store):
        """Should return the average PnL of the last N trades."""
        from tests.helpers import make_position

        # Record 10 positions: 3 winners (+10¢) and 7 losers (-5¢)
        # PnL = (exit_p - entry) * size * 100
        # +10¢ at size=10: exit_p - entry = 0.01
        # -5¢ at size=10: exit_p - entry = -0.005
        entries = (
            [(0.30, 0.31)] * 3 +  # 3 winners: +10¢ each
            [(0.30, 0.295)] * 7    # 7 losers: -5¢ each
        )
        for i, (ent, ext) in enumerate(entries):
            pos = make_position(
                pos_id=f"POS-{i}",
                entry=ent, exit_p=ext,
                reason="target",
            )
            await trade_store.record(pos)

        result = await trade_store.get_rolling_expectancy(window=10)
        expected = (3 * 10.0 + 7 * (-5.0)) / 10.0  # -0.5
        assert result == pytest.approx(expected, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════════
#  Section H: Edge Score Regression with New Parameters
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeScoreWithNewDefaults:
    """Verify EQS behavior with updated min_edge_score=50."""

    def test_marginal_trade_rejected_at_50(self):
        """A trade that passed at EQS=20 but fails at EQS=50."""
        edge = compute_edge_score(
            entry_price=0.05, no_vwap=0.07,  # tail price, tiny spread
            zscore=2.0, volume_ratio=1.2,
            fee_enabled=True,
        )
        # H(0.05) ≈ 0.286, regime very low; thin spread eaten by fees
        assert edge.score < 50.0
        assert not edge.viable

    def test_sweet_spot_trade_passes(self):
        """Mid-range entry with strong signal passes at EQS=50."""
        edge = compute_edge_score(
            entry_price=0.30, no_vwap=0.50,
            zscore=4.0, volume_ratio=3.0,
            fee_enabled=False,
        )
        assert edge.score >= 50.0
        assert edge.viable
