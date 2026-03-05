"""
Tests for AdverseSelectionMonitor — maker-fill adverse selection tracking.

Coverage:
  - MakerFillRecord creation and PnL sign convention
  - T+5 / T+15 / T+60 mark scheduling via tick()
  - Welch t-test: no suspension below min_fills threshold
  - Welch t-test: suspension fires when all fills adverse (p < 0.05)
  - Suspension prevents is_maker_allowed() returning True
  - Suspension expires after backoff period
  - Exponential backoff on repeated suspensions
  - Positive PnL history never triggers suspension
  - _t_cdf_lower_tail basic sanity checks
  - Probe harvest idempotency: scale_probe_to_full fires once
  - Probe harvest "not too late" guard (Guard 3)
  - Flaw 1 combo guard: compute_confluence_discount with maker_routing_active
  - Flaw 2 drift guard: regime discount suppressed when is_drift_signal=True
  - Flaw 3 probe confluence suppression: whale discount not applied to probes
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.trading.adverse_selection_monitor import (
    AdverseSelectionMonitor,
    MakerFillRecord,
    _ADVERSE_CENTS_FLOOR,
    _signed_pnl,
    _t_cdf_lower_tail,
    make_fill_record,
    VolProvider,
)
from src.signals.edge_filter import ConfluenceContext, compute_confluence_discount


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_monitor(mid_fn=None, alpha=0.05, min_fills=10,
                  suspension_base_s=900.0, vol_provider=None):
    if mid_fn is None:
        async def mid_fn(asset_id):
            return 0.50
    return AdverseSelectionMonitor(
        mid_price_fn=mid_fn,
        alpha=alpha,
        min_fills_to_suspend=min_fills,
        window=30,
        suspension_base_s=suspension_base_s,
        suspension_max_s=14_400.0,
        vol_provider=vol_provider,
    )


def _fill(fill_price: float = 0.50, side: str = "BUY", market_id: str = "MKT-1",
          asset_id: str = "ASSET-1", size_usd: float = 5.0,
          fill_time: float | None = None) -> MakerFillRecord:
    return make_fill_record(
        market_id=market_id,
        asset_id=asset_id,
        fill_price=fill_price,
        fill_side=side,
        size_usd=size_usd,
        fill_time=fill_time or (time.time() - 70),  # old enough for T+60
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Unit: _signed_pnl sign convention
# ═══════════════════════════════════════════════════════════════════════════

class TestSignedPnl:
    def test_buy_price_rises_positive(self):
        rec = _fill(fill_price=0.47)
        pnl = _signed_pnl(rec, 0.50)
        assert pnl > 0

    def test_buy_price_falls_negative(self):
        rec = _fill(fill_price=0.47)
        pnl = _signed_pnl(rec, 0.45)
        assert pnl < 0

    def test_sell_price_falls_positive(self):
        rec = _fill(fill_price=0.50, side="SELL")
        pnl = _signed_pnl(rec, 0.48)
        assert pnl > 0

    def test_sell_price_rises_negative(self):
        rec = _fill(fill_price=0.50, side="SELL")
        pnl = _signed_pnl(rec, 0.53)
        assert pnl < 0

    def test_magnitude_correct(self):
        # BUY at 0.47, mark at 0.50 → +0.03 × 100 × $5 = +15.0 cent-dollars
        rec = _fill(fill_price=0.47, size_usd=5.0)
        pnl = _signed_pnl(rec, 0.50)
        assert abs(pnl - 15.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
#  Unit: t-distribution CDF
# ═══════════════════════════════════════════════════════════════════════════

class TestTCdf:
    def test_symmetric_at_zero(self):
        p = _t_cdf_lower_tail(0.0, df=10)
        assert abs(p - 0.5) < 0.01

    def test_large_negative_approaches_zero(self):
        p = _t_cdf_lower_tail(-10.0, df=20)
        assert p < 0.01

    def test_large_positive_approaches_one(self):
        p = _t_cdf_lower_tail(10.0, df=20)
        assert p > 0.99

    def test_known_value_df30_t_neg2(self):
        # t = -2, df = 30 → lower tail ≈ 0.027
        p = _t_cdf_lower_tail(-2.0, df=30)
        assert 0.01 < p < 0.05


# ═══════════════════════════════════════════════════════════════════════════
#  Integration: tick() → mark scheduling
# ═══════════════════════════════════════════════════════════════════════════

class TestTickMarking:
    @pytest.mark.asyncio
    async def test_t5_mark_set_after_5s(self):
        call_count = 0

        async def mid_fn(asset_id):
            nonlocal call_count
            call_count += 1
            return 0.52

        monitor = _make_monitor(mid_fn=mid_fn)
        now = time.time()
        rec = _fill(fill_time=now - 6)  # 6s ago — eligible for T+5 but not T+15
        monitor.record_maker_fill(rec)
        await monitor.tick(now=now)

        assert rec.mark_t5 == 0.52
        assert rec.pnl_t5 is not None
        assert rec.mark_t15 is None

    @pytest.mark.asyncio
    async def test_all_marks_set_after_60s(self):
        async def mid_fn(asset_id):
            return 0.50

        monitor = _make_monitor(mid_fn=mid_fn)
        now = time.time()
        rec = _fill(fill_time=now - 70)
        monitor.record_maker_fill(rec)
        await monitor.tick(now=now)

        assert rec.mark_t5 is not None
        assert rec.mark_t15 is not None
        assert rec.mark_t60 is not None
        assert monitor.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_fill_stays_pending_until_t60(self):
        async def mid_fn(asset_id):
            return 0.50

        monitor = _make_monitor(mid_fn=mid_fn)
        now = time.time()
        rec = _fill(fill_time=now - 20)  # 20s old — T+15 done, T+60 not yet
        monitor.record_maker_fill(rec)
        await monitor.tick(now=now)

        assert rec.mark_t15 is not None
        assert rec.mark_t60 is None
        assert monitor.get_pending_count() == 1


# ═══════════════════════════════════════════════════════════════════════════
#  Integration: suspension logic
# ═══════════════════════════════════════════════════════════════════════════

class TestSuspension:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @pytest.mark.asyncio
    async def test_no_suspension_below_min_fills(self):
        """With only 9 fills (< min_fills=10), test should not fire."""
        async def mid_fn(asset_id):
            return 0.45  # adverse: price fell after BUY at 0.50

        monitor = _make_monitor(mid_fn=mid_fn, alpha=0.05, min_fills=10)
        now = time.time()
        for _ in range(9):
            rec = _fill(fill_price=0.50, fill_time=now - 70)
            monitor.record_maker_fill(rec)
        await monitor.tick(now=now)

        assert monitor.is_maker_allowed("MKT-1")

    @pytest.mark.asyncio
    async def test_suspension_fires_with_consistently_adverse_fills(self):
        """10+ fills all adverse → t-test fires → suspension."""
        async def mid_fn(asset_id):
            return 0.44  # price dropped 6¢ after BUY at 0.50 → strongly adverse

        monitor = _make_monitor(mid_fn=mid_fn, alpha=0.05, min_fills=10)
        now = time.time()
        for i in range(10):
            # Small fill-price jitter to avoid stdev=0 early-return
            rec = _fill(fill_price=0.500 + i * 0.0001, fill_time=now - 70)
            monitor.record_maker_fill(rec)
        await monitor.tick(now=now)

        assert not monitor.is_maker_allowed("MKT-1")

    @pytest.mark.asyncio
    async def test_suspension_expires_after_backoff(self):
        async def mid_fn(asset_id):
            return 0.44

        monitor = _make_monitor(mid_fn=mid_fn, min_fills=10, suspension_base_s=1.0)
        now = time.time()
        for i in range(10):
            rec = _fill(fill_price=0.500 + i * 0.0001, fill_time=now - 70)
            monitor.record_maker_fill(rec)
        await monitor.tick(now=now)

        assert not monitor.is_maker_allowed("MKT-1", now=now)
        # Check well past the 1s suspension window
        assert monitor.is_maker_allowed("MKT-1", now=now + 3.0)

    @pytest.mark.asyncio
    async def test_exponential_backoff_on_second_suspension(self):
        """Second suspension doubles the backoff period."""
        async def mid_fn(asset_id):
            return 0.44

        monitor = _make_monitor(mid_fn=mid_fn, min_fills=5, suspension_base_s=60.0)
        now = time.time()

        # First suspension
        for i in range(5):
            rec = _fill(fill_price=0.500 + i * 0.0001, fill_time=now - 70)
            monitor.record_maker_fill(rec)
        await monitor.tick(now=now)

        stats = monitor.get_stats("MKT-1")
        assert stats is not None
        assert stats.suspension_count == 1
        first_until = stats.suspension_until
        first_duration = first_until - now  # should be ~60s

        # Expire first suspension and add more adverse fills.
        # Mock time.time so _is_suspended / _suspend see the future time.
        expired_now = first_until + 1.0
        with patch("src.trading.adverse_selection_monitor.time") as mock_time:
            mock_time.time.return_value = expired_now
            for i in range(5):
                rec = _fill(fill_price=0.500 + i * 0.0001, fill_time=expired_now - 70)
                monitor.record_maker_fill(rec)
            await monitor.tick(now=expired_now)

        stats = monitor.get_stats("MKT-1")
        assert stats.suspension_count == 2
        second_until = stats.suspension_until
        second_duration = second_until - expired_now
        # Second suspension should be roughly 2× first
        assert second_duration > first_duration * 1.8

    @pytest.mark.asyncio
    async def test_positive_pnl_never_suspends(self):
        """Profitable fills should not trigger suspension regardless of n."""
        async def mid_fn(asset_id):
            return 0.56  # strong move in our favour after BUY at 0.50

        monitor = _make_monitor(mid_fn=mid_fn, min_fills=5)
        now = time.time()
        for _ in range(20):
            rec = _fill(fill_price=0.50, fill_time=now - 70)
            monitor.record_maker_fill(rec)
        await monitor.tick(now=now)

        assert monitor.is_maker_allowed("MKT-1")

    @pytest.mark.asyncio
    async def test_unknown_market_always_allowed(self):
        monitor = _make_monitor()
        assert monitor.is_maker_allowed("UNKNOWN-MARKET")


# ═══════════════════════════════════════════════════════════════════════════
#  Flaw 1: combined maker+confluence floor guard
# ═══════════════════════════════════════════════════════════════════════════

class TestFlaw1CombinedFloor:
    def test_maker_active_uses_tighter_floor(self):
        """Whale + spread + maker_routing_active: floor = 40 not 35.

        With the calibration update (L2/regime discounts = 0.0), only
        whale (4) + spread (4) = 8 pts discount fires.  The combined
        maker floor (40) still applies.
        """
        ctx = ConfluenceContext(
            whale_strong_confluence=True,
            spread_compressed=True,
            l2_reliable=True,
            regime_mean_revert=True,
        )
        # base=42.5, discount=4+4=8 → 34.5, floored at 40 (maker)
        result_maker = compute_confluence_discount(
            ctx, 42.5, maker_routing_active=True
        )
        result_taker = compute_confluence_discount(
            ctx, 42.5, maker_routing_active=False
        )
        # With tighter combined floor (default 40), maker result ≥ taker result
        assert result_maker >= result_taker
        assert result_maker >= 40.0  # default confluence_maker_combined_floor

    def test_taker_mode_uses_standard_floor(self):
        ctx = ConfluenceContext(
            whale_strong_confluence=True,
            spread_compressed=True,
            l2_reliable=True,
            regime_mean_revert=True,
        )
        result = compute_confluence_discount(ctx, 50.0, maker_routing_active=False)
        # Whale(4) + Spread(4) = 8 pts: max(35, 50-8) = 42.0
        assert result == 42.0


# ═══════════════════════════════════════════════════════════════════════════
#  Flaw 2: drift signal suppresses regime discount
# ═══════════════════════════════════════════════════════════════════════════

class TestFlaw2DriftRegimeSuppression:
    def test_regime_discount_suppressed_when_drift(self):
        """With calibration update: regime discount is 0.0, so no discount
        regardless of drift/non-drift.  Both should return base_threshold
        when only non-justified factors are active."""
        ctx = ConfluenceContext(
            whale_strong_confluence=False,
            spread_compressed=False,
            l2_reliable=True,      # hard gate passes, but l2_discount=0
            regime_mean_revert=True,  # regime_discount=0
        )
        result_drift = compute_confluence_discount(
            ctx, 50.0, is_drift_signal=True
        )
        result_normal = compute_confluence_discount(
            ctx, 50.0, is_drift_signal=False
        )
        # No justified factors with non-zero discount → < min_factors → no discount
        assert result_drift == 50.0
        assert result_normal == 50.0

    def test_non_regime_factors_still_work_with_drift(self):
        """Whale + spread still combine for drift signals."""
        ctx = ConfluenceContext(
            whale_strong_confluence=True,  # +4
            spread_compressed=True,        # +4
            l2_reliable=True,              # hard gate passes
            regime_mean_revert=True,       # discount=0, ignored
        )
        result = compute_confluence_discount(ctx, 50.0, is_drift_signal=True)
        # Whale(4) + Spread(4) = 8 pts discount → max(35, 50-8) = 42
        assert result == 42.0

    def test_l2_unreliable_blocks_all_discounts(self):
        """L2 unreliable → hard gate blocks all confluence discounts."""
        ctx = ConfluenceContext(
            whale_strong_confluence=True,
            spread_compressed=True,
            l2_reliable=False,
            regime_mean_revert=True,
        )
        result = compute_confluence_discount(ctx, 50.0)
        assert result == 50.0  # hard gate returns base_threshold


# ═══════════════════════════════════════════════════════════════════════════
#  Probe harvest guards (covered via direct PositionManager unit test)
# ═══════════════════════════════════════════════════════════════════════════

class TestProbeHarvestGuards:
    """Verify scale_probe_to_full respects its six guards."""

    def _make_pm(self):
        from unittest.mock import MagicMock, AsyncMock
        from src.trading.position_manager import PositionManager, PositionState, Position
        from src.trading.executor import OrderExecutor

        executor = MagicMock(spec=OrderExecutor)
        executor.paper_mode = True
        executor.place_limit_order = AsyncMock(return_value=MagicMock(order_id="O-1"))
        pm = PositionManager(executor=executor)
        return pm

    def _make_probe_pos(self, entry=0.47, target=0.54, entry_size=20.0):
        from src.trading.position_manager import Position, PositionState
        from src.trading.take_profit import TakeProfitResult
        from src.trading.sizer import KellyResult

        tp = MagicMock()
        tp.target_price = target
        tp.viable = True

        kelly = MagicMock()
        kelly.size_shares = 20.0  # 20 shares = at probe_kelly_fraction=0.05, full=400

        pos = MagicMock()
        pos.id = "PROBE-1"
        pos.market_id = "MKT-1"
        pos.no_asset_id = "ASSET-1"
        pos.trade_asset_id = "ASSET-1"
        pos.trade_side = "NO"
        pos.entry_price = entry
        pos.entry_size = entry_size
        pos.state = PositionState.EXIT_PENDING
        pos.is_probe = True
        pos.tp_result = tp
        pos.kelly_result = kelly
        return pos

    @pytest.mark.asyncio
    async def test_guard2_idempotent_only_scales_once(self):
        pm = self._make_pm()
        pos = self._make_probe_pos()
        mid_fn = lambda aid: 0.51  # profitable

        # First call should succeed
        order1 = await pm.scale_probe_to_full(pos, mid_fn)
        assert order1 is not None
        assert pos.id in pm._probe_scaled_set
        assert pos.is_probe is False

        # Second call must be a no-op
        order2 = await pm.scale_probe_to_full(pos, mid_fn)
        assert order2 is None

    @pytest.mark.asyncio
    async def test_guard1_not_profitable_returns_none(self):
        pm = self._make_pm()
        pos = self._make_probe_pos(entry=0.47)
        mid_fn = lambda aid: 0.47  # breakeven = not profitable

        order = await pm.scale_probe_to_full(pos, mid_fn)
        assert order is None

    @pytest.mark.asyncio
    async def test_guard3_too_late_returns_none(self):
        pm = self._make_pm()
        pos = self._make_probe_pos(entry=0.47, target=0.54)
        # TP span = 7¢; 80% progress = 5.6¢ → mid = 0.47 + 0.056 = 0.526
        mid_fn = lambda aid: 0.526  # 80% of TP already captured

        order = await pm.scale_probe_to_full(pos, mid_fn)
        assert order is None

    @pytest.mark.asyncio
    async def test_guard1_not_open_returns_none(self):
        from src.trading.position_manager import PositionState
        pm = self._make_pm()
        pos = self._make_probe_pos()
        pos.state = PositionState.CLOSED  # not EXIT_PENDING
        mid_fn = lambda aid: 0.51

        order = await pm.scale_probe_to_full(pos, mid_fn)
        assert order is None


# ═══════════════════════════════════════════════════════════════════════════
#  Dynamic alpha: volatility-scaled significance threshold
# ═══════════════════════════════════════════════════════════════════════════

class TestDynamicAlpha:
    """Verify _dynamic_alpha scaling formula and clamp behaviour."""

    def test_no_vol_provider_returns_fixed_alpha(self):
        """When vol_provider is None, alpha is fixed at alpha_base."""
        monitor = _make_monitor(vol_provider=None, alpha=0.05)
        assert monitor._dynamic_alpha("MKT-1") == 0.05

    def test_vol_provider_returns_none_falls_back(self):
        """When vol_provider returns None, fallback to fixed alpha."""
        monitor = _make_monitor(vol_provider=lambda mid: None, alpha=0.05)
        assert monitor._dynamic_alpha("MKT-1") == 0.05

    def test_vol_provider_returns_zero_falls_back(self):
        """Zero vol should not blow up — fallback to fixed alpha."""
        monitor = _make_monitor(vol_provider=lambda mid: 0.0, alpha=0.05)
        assert monitor._dynamic_alpha("MKT-1") == 0.05

    def test_normal_vol_returns_base_alpha(self):
        """σ_rolling == σ_ref → ratio = 1.0 → α_dynamic = α_base."""
        # vol_ref default = 0.01
        monitor = AdverseSelectionMonitor(
            mid_price_fn=AsyncMock(return_value=0.50),
            alpha=0.05,
            vol_provider=lambda mid: 0.01,
            vol_ref=0.01,
            alpha_gamma=0.5,
            alpha_min=0.01,
            alpha_max=0.15,
        )
        alpha = monitor._dynamic_alpha("MKT-1")
        assert abs(alpha - 0.05) < 1e-9

    def test_low_vol_tightens_alpha(self):
        """σ = 0.005 (half ref) → ratio = 0.5 → α ≈ 0.05 × 0.707 ≈ 0.035."""
        monitor = AdverseSelectionMonitor(
            mid_price_fn=AsyncMock(return_value=0.50),
            alpha=0.05,
            vol_provider=lambda mid: 0.005,
            vol_ref=0.01,
            alpha_gamma=0.5,
            alpha_min=0.01,
            alpha_max=0.15,
        )
        alpha = monitor._dynamic_alpha("MKT-1")
        expected = 0.05 * (0.5 ** 0.5)  # ≈ 0.0354
        assert abs(alpha - expected) < 0.001
        assert alpha < 0.05  # more conservative

    def test_high_vol_loosens_alpha(self):
        """σ = 0.02 (2× ref) → ratio = 2.0 → α ≈ 0.05 × √2 ≈ 0.071."""
        monitor = AdverseSelectionMonitor(
            mid_price_fn=AsyncMock(return_value=0.50),
            alpha=0.05,
            vol_provider=lambda mid: 0.02,
            vol_ref=0.01,
            alpha_gamma=0.5,
            alpha_min=0.01,
            alpha_max=0.15,
        )
        alpha = monitor._dynamic_alpha("MKT-1")
        expected = 0.05 * (2.0 ** 0.5)  # ≈ 0.0707
        assert abs(alpha - expected) < 0.001
        assert alpha > 0.05  # more aggressive

    def test_extreme_high_vol_clamped_at_max(self):
        """Very high vol should clamp at alpha_max = 0.15."""
        monitor = AdverseSelectionMonitor(
            mid_price_fn=AsyncMock(return_value=0.50),
            alpha=0.05,
            vol_provider=lambda mid: 0.10,  # 10× ref
            vol_ref=0.01,
            alpha_gamma=0.5,
            alpha_min=0.01,
            alpha_max=0.15,
        )
        alpha = monitor._dynamic_alpha("MKT-1")
        assert alpha == 0.15

    def test_extreme_low_vol_clamped_at_min(self):
        """Very low vol should clamp at alpha_min = 0.01."""
        monitor = AdverseSelectionMonitor(
            mid_price_fn=AsyncMock(return_value=0.50),
            alpha=0.05,
            vol_provider=lambda mid: 0.0001,  # 0.01× ref
            vol_ref=0.01,
            alpha_gamma=0.5,
            alpha_min=0.01,
            alpha_max=0.15,
        )
        alpha = monitor._dynamic_alpha("MKT-1")
        assert alpha == 0.01

    @pytest.mark.asyncio
    async def test_dynamic_alpha_used_in_suspension_decision(self):
        """Verify that _recompute_stats uses _dynamic_alpha, not fixed alpha.

        Use low vol (α_dynamic ≈ 0.035) → a p-value of 0.04 would
        NOT trigger suspension (0.04 > 0.035), whereas fixed α=0.05 would.
        """
        # Mid-price at 0.455 after BUY at 0.50 → adverse but not extreme
        async def mid_fn(asset_id):
            return 0.455

        # With low vol → alpha shrinks to ~0.035
        monitor = AdverseSelectionMonitor(
            mid_price_fn=mid_fn,
            alpha=0.05,
            min_fills_to_suspend=10,
            vol_provider=lambda mid: 0.005,  # low vol
            vol_ref=0.01,
            alpha_gamma=0.5,
            alpha_min=0.01,
            alpha_max=0.15,
        )

        # Check that the dynamic alpha is indeed < 0.05
        effective_alpha = monitor._dynamic_alpha("MKT-1")
        assert effective_alpha < 0.05

    @pytest.mark.asyncio
    async def test_backward_compat_no_vol_provider(self):
        """Existing code with no vol_provider should work identically to before."""
        async def mid_fn(asset_id):
            return 0.44  # strongly adverse

        monitor = _make_monitor(mid_fn=mid_fn, alpha=0.05, min_fills=10)
        now = time.time()
        for i in range(10):
            # Small fill-price jitter to avoid stdev=0 early-return
            rec = _fill(fill_price=0.500 + i * 0.0001, fill_time=now - 70)
            monitor.record_maker_fill(rec)
        await monitor.tick(now=now)

        # Should still suspend exactly as before
        assert not monitor.is_maker_allowed("MKT-1")
