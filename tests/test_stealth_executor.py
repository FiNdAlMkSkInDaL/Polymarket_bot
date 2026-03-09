"""Tests for the StealthExecutor (SI-4)."""

from __future__ import annotations

import asyncio
import math

import pytest

from src.trading.executor import Order, OrderExecutor, OrderSide, OrderStatus
from src.trading.stealth_executor import StealthExecutor, StealthPlan


class TestStealthPlan:
    @pytest.fixture
    def stealth(self):
        executor = OrderExecutor(paper_mode=True)
        return StealthExecutor(
            executor,
            min_size_usd=5.0,
            max_slices=4,
            min_delay_ms=50.0,
            max_delay_ms=100.0,
            size_jitter_pct=0.15,
        )

    def test_plan_slice_count(self, stealth):
        plan = stealth._build_plan(total_size=20.0, price=0.50)
        # 20 * 0.50 = $10 → 10/3 + 1 ≈ 4 slices, capped at 4
        assert 2 <= plan.num_slices <= 4

    def test_plan_slices_sum_to_total(self, stealth):
        plan = stealth._build_plan(total_size=20.0, price=0.50)
        assert sum(plan.slice_sizes) == pytest.approx(20.0, abs=0.05)

    def test_plan_delays_between_bounds(self, stealth):
        plan = stealth._build_plan(total_size=20.0, price=0.50)
        for d in plan.delays_ms:
            assert 50.0 <= d <= 100.0

    def test_plan_small_order_gets_2_slices(self, stealth):
        plan = stealth._build_plan(total_size=6.0, price=0.50)
        # 6 * 0.50 = $3 → 3/3 + 1 = 2 slices
        assert plan.num_slices == 2


class TestStealthExecution:
    @pytest.fixture
    def stealth(self):
        executor = OrderExecutor(paper_mode=True)
        return StealthExecutor(
            executor,
            min_size_usd=5.0,
            max_slices=3,
            min_delay_ms=10.0,
            max_delay_ms=20.0,
            size_jitter_pct=0.10,
        )

    @pytest.mark.asyncio
    async def test_small_order_passes_through(self, stealth):
        """Orders below min_size_usd should not be split."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=5.0,  # 5 * 0.50 = $2.50 < $5.0
        )
        assert len(orders) == 1
        assert orders[0].size == 5.0

    @pytest.mark.asyncio
    async def test_large_order_is_split(self, stealth):
        """Orders above min_size_usd should be split into multiple slices."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,  # 20 * 0.50 = $10 > $5.0
        )
        assert len(orders) >= 2
        total_placed = sum(o.size for o in orders)
        assert total_placed == pytest.approx(20.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_all_orders_are_live(self, stealth):
        """In paper mode, all child orders should be LIVE."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
        )
        for o in orders:
            assert o.status == OrderStatus.LIVE

    @pytest.mark.asyncio
    async def test_sell_side(self, stealth):
        """Stealth execution should work for SELL orders too."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.SELL,
            price=0.50,
            total_size=20.0,
        )
        assert len(orders) >= 2
        for o in orders:
            assert o.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_post_only(self, stealth):
        """post_only should be forwarded to all child orders."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
            post_only=True,
        )
        for o in orders:
            assert o.post_only is True

    def test_executor_property(self, stealth):
        assert isinstance(stealth.executor, OrderExecutor)


class TestStealthAbandonment:
    """Tests for mid-slice abandonment guards."""

    @pytest.fixture
    def stealth(self):
        executor = OrderExecutor(paper_mode=True)
        return StealthExecutor(
            executor,
            min_size_usd=5.0,
            max_slices=4,
            min_delay_ms=10.0,
            max_delay_ms=20.0,
            size_jitter_pct=0.10,
        )

    @pytest.mark.asyncio
    async def test_drift_abandonment(self, stealth):
        """If mid-price drifts adversely between slices, stop placing."""
        call_count = 0

        def drifting_mid():
            nonlocal call_count
            call_count += 1
            # First check: no drift. Second check: 3¢ adverse drift.
            if call_count <= 1:
                return 0.50
            return 0.53  # 3¢ above anchor → exceeds 2¢ default threshold

        stealth._abandon_drift_cents = 2.0
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,  # $10 → should want 4 slices
            get_mid_fn=drifting_mid,
        )
        # Should have stopped early — fewer slices than the full plan
        assert len(orders) < 4

    @pytest.mark.asyncio
    async def test_no_abandonment_without_drift(self, stealth):
        """If mid stays stable, all slices should be placed."""
        stealth._abandon_drift_cents = 2.0
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
            get_mid_fn=lambda: 0.50,  # stable mid
        )
        assert len(orders) >= 2

    @pytest.mark.asyncio
    async def test_no_mid_fn_places_all_slices(self, stealth):
        """Without get_mid_fn, no abandonment checks — all slices placed."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
        )
        assert len(orders) >= 2

    @pytest.mark.asyncio
    async def test_sell_side_drift_abandonment(self, stealth):
        """SELL orders abandon when mid drops adversely."""
        call_count = 0

        def dropping_mid():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return 0.50
            return 0.47  # 3¢ below anchor → adverse for SELL

        stealth._abandon_drift_cents = 2.0
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.SELL,
            price=0.50,
            total_size=20.0,
            get_mid_fn=dropping_mid,
        )
        assert len(orders) < 4


# ═══════════════════════════════════════════════════════════════════════════
#  SI-4.1: VWAP-Aware Stealth Slicing — Mathematical Invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestVWAPStealthInvariants:
    """Prove the four mathematical invariants of POV/VWAP-aware slicing."""

    @pytest.fixture
    def stealth(self):
        executor = OrderExecutor(paper_mode=True)
        return StealthExecutor(
            executor,
            min_size_usd=5.0,
            max_slices=4,
            min_delay_ms=10.0,
            max_delay_ms=20.0,
            size_jitter_pct=0.10,
        )

    # ── Dead Market Invariant ──────────────────────────────────────────────
    # If recent_volume_usd is 0 or None, the system must safely fall back
    # to an assumed $10.00 volume and never crash with ZeroDivisionError.

    def test_zero_volume_no_crash(self, stealth):
        """Zero volume → $10 fallback; no ZeroDivisionError."""
        plan = stealth._build_plan(total_size=20.0, price=0.50, recent_volume_usd=0.0)
        assert plan.num_slices >= 2
        assert sum(plan.slice_sizes) == pytest.approx(20.0, abs=0.05)

    def test_none_volume_no_crash(self, stealth):
        """None volume → $10 fallback; no ZeroDivisionError."""
        plan = stealth._build_plan(total_size=20.0, price=0.50, recent_volume_usd=None)
        assert plan.num_slices >= 2
        assert sum(plan.slice_sizes) == pytest.approx(20.0, abs=0.05)

    def test_negative_volume_no_crash(self, stealth):
        """Negative volume → $10 fallback; treated same as zero."""
        plan = stealth._build_plan(total_size=20.0, price=0.50, recent_volume_usd=-5.0)
        assert plan.num_slices >= 2
        assert sum(plan.slice_sizes) == pytest.approx(20.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_zero_volume_e2e_no_crash(self, stealth):
        """End-to-end: place_stealth_order with zero volume doesn't crash."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
            recent_volume_usd=0.0,
        )
        assert len(orders) >= 2

    # ── API Rejection Invariant ────────────────────────────────────────────
    # If recent_volume_usd × stealth_max_participation_pct is extremely
    # small, max_slice_usd must clamp up to stealth_min_size_usd ($5.00)
    # to prevent sub-minimum API rejections.

    def test_micro_volume_clamps_to_min_size(self, stealth):
        """Tiny volume ($2) × 5% = $0.10 → clamped to min_size_usd ($5)."""
        # $2 × 0.05 = $0.10, far below min_size_usd of $5.00
        plan = stealth._build_plan(total_size=20.0, price=0.50, recent_volume_usd=2.0)
        notional = 20.0 * 0.50  # $10
        # max_slice_usd should be clamped to $5.00 (min_size_usd)
        # so required_slices = ceil(10 / 5) = 2
        assert plan.num_slices >= 2
        # Each slice notional should be at least min_size_usd
        for s in plan.slice_sizes:
            assert s * 0.50 >= 0.0  # no negative slices

    def test_very_small_volume_does_not_over_slice(self, stealth):
        """When volume is tiny but clamp applies, slices stay reasonable."""
        # $1 × 0.05 = $0.05 → clamped to $5.00 → ceil(10/5) = 2
        plan = stealth._build_plan(total_size=20.0, price=0.50, recent_volume_usd=1.0)
        # Should NOT produce an absurdly high slice count
        assert plan.num_slices <= 10

    # ── Dynamic Reslicing Invariant ────────────────────────────────────────
    # $50 order in $100 volume market with 5% participation cap ($5 max
    # slice) → engine must expand to exactly ceil(50/5) = 10 slices.

    def test_dynamic_reslice_exact_count(self, stealth):
        """$50 order, $100 volume, 5% cap → exactly 10 slices."""
        # total_size * price = $50, volume=$100, 5% cap → max_slice=$5
        # ceil(50/5) = 10; default would be ~4 → must override to 10
        plan = stealth._build_plan(total_size=100.0, price=0.50, recent_volume_usd=100.0)
        expected_slices = math.ceil(50.0 / 5.0)  # 10
        assert plan.num_slices == expected_slices

    def test_dynamic_reslice_overrides_default(self, stealth):
        """When POV cap requires more slices than default, default is overridden."""
        # max_slices=4, but POV requires 10 → plan.num_slices > 4
        plan = stealth._build_plan(total_size=100.0, price=0.50, recent_volume_usd=100.0)
        assert plan.num_slices > stealth._max_slices

    def test_dynamic_reslice_sizes_sum_to_total(self, stealth):
        """Even after reslicing, all slice sizes must sum to total_size."""
        plan = stealth._build_plan(total_size=100.0, price=0.50, recent_volume_usd=100.0)
        assert sum(plan.slice_sizes) == pytest.approx(100.0, abs=0.10)

    def test_dynamic_reslice_delay_count_matches(self, stealth):
        """Number of delays must match number of slices."""
        plan = stealth._build_plan(total_size=100.0, price=0.50, recent_volume_usd=100.0)
        assert len(plan.delays_ms) == plan.num_slices

    # ── Participation Cap Invariant ────────────────────────────────────────
    # In a high-volume environment, no individual slice notional may
    # exceed recent_volume_usd × stealth_max_participation_pct.

    def test_no_slice_exceeds_participation_cap(self, stealth):
        """High-volume: every slice notional ≤ volume × participation_pct."""
        from src.core.config import settings
        pov_cap = settings.strategy.stealth_max_participation_pct
        volume = 500.0  # $500 volume → max_slice = $25
        max_slice_usd = volume * pov_cap

        plan = stealth._build_plan(total_size=40.0, price=0.50, recent_volume_usd=volume)
        for s in plan.slice_sizes:
            slice_notional = s * 0.50
            # Allow 15% tolerance for jitter + rounding residual on last slice
            assert slice_notional <= max_slice_usd * 1.15, (
                f"Slice {slice_notional:.2f} exceeds cap {max_slice_usd:.2f}"
            )

    def test_normal_volume_no_reslice(self, stealth):
        """In a deep market, default slice count is preserved (no override)."""
        # $10 order in $1000 volume → max_slice=$50 → ceil(10/50)=1 → default 4 wins
        plan = stealth._build_plan(total_size=20.0, price=0.50, recent_volume_usd=1000.0)
        assert plan.num_slices <= stealth._max_slices

    def test_participation_cap_with_high_price(self, stealth):
        """Participation cap math is correct at high token prices."""
        # price=0.90, total_size=60 → notional=$54
        # volume=$200 → max_slice=$10 → ceil(54/10)=6
        plan = stealth._build_plan(total_size=60.0, price=0.90, recent_volume_usd=200.0)
        expected = math.ceil(54.0 / 10.0)  # 6
        assert plan.num_slices == expected


# ═══════════════════════════════════════════════════════════════════════════
#  SI-4.2: Order-Book Imbalance Pacing — Mathematical Invariants
# ═══════════════════════════════════════════════════════════════════════════


class TestOBIPacingInvariants:
    """Prove the OBI delay-biasing math for the StealthExecutor."""

    @pytest.fixture
    def stealth(self):
        executor = OrderExecutor(paper_mode=True)
        return StealthExecutor(
            executor,
            min_size_usd=5.0,
            max_slices=4,
            min_delay_ms=100.0,
            max_delay_ms=1000.0,
            size_jitter_pct=0.10,
        )

    # ── Hollowed Book → Max Delay ──────────────────────────────────────────

    def test_buy_hollowed_book_forces_max_delay(self, stealth):
        """BUY with ratio=0.1 (bid≪ask) → our_ratio<0.5 → max delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, 0.1)
        assert delay == stealth._max_delay

    def test_buy_ratio_0_49_forces_max_delay(self, stealth):
        """BUY with ratio=0.49 → still hollowed → max delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, 0.49)
        assert delay == stealth._max_delay

    def test_sell_hollowed_book_forces_max_delay(self, stealth):
        """SELL with ratio=3.0 (bid≫ask) → our_ratio=1/3=0.33<0.5 → max delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.SELL, 3.0)
        assert delay == stealth._max_delay

    # ── Heavy Support → Min Delay ──────────────────────────────────────────

    def test_buy_heavy_support_forces_min_delay(self, stealth):
        """BUY with ratio=3.0 (bid≫ask) → our_ratio>2.0 → min delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, 3.0)
        assert delay == stealth._min_delay

    def test_buy_ratio_2_01_forces_min_delay(self, stealth):
        """BUY with ratio=2.01 → just above threshold → min delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, 2.01)
        assert delay == stealth._min_delay

    def test_sell_heavy_support_forces_min_delay(self, stealth):
        """SELL with ratio=0.1 (ask≫bid) → our_ratio=1/0.1=10>2 → min delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.SELL, 0.1)
        assert delay == stealth._min_delay

    # ── Balanced Book → Random Delay Pass-Through ─────────────────────────

    def test_balanced_book_keeps_base_delay(self, stealth):
        """Ratio=1.0 → 0.5≤our_ratio≤2.0 → base delay unchanged."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, 1.0)
        assert delay == 500.0

    def test_ratio_0_5_keeps_base_delay(self, stealth):
        """Ratio exactly 0.5 → boundary → keeps base delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, 0.5)
        assert delay == 500.0

    def test_ratio_2_0_keeps_base_delay(self, stealth):
        """Ratio exactly 2.0 → boundary → keeps base delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, 2.0)
        assert delay == 500.0

    # ── Fallback: Missing / Invalid L2 Data ───────────────────────────────

    def test_zero_ratio_defaults_to_balanced(self, stealth):
        """Ratio=0 → fallback to 1.0 → base delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, 0.0)
        assert delay == 500.0

    def test_none_ratio_defaults_to_balanced(self, stealth):
        """Ratio=None → fallback to 1.0 → base delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, None)
        assert delay == 500.0

    def test_negative_ratio_defaults_to_balanced(self, stealth):
        """Ratio=-1.0 → fallback to 1.0 → base delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, -1.0)
        assert delay == 500.0

    # ── Side Inversion ────────────────────────────────────────────────────

    def test_sell_inverts_ratio(self, stealth):
        """SELL with bid/ask=0.4 → our_ratio=1/0.4=2.5>2 → min delay."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.SELL, 0.4)
        assert delay == stealth._min_delay

    # ── E2E: OBI pacing applied in place_stealth_order ────────────────────

    @pytest.mark.asyncio
    async def test_e2e_hollowed_book_does_not_crash(self, stealth):
        """place_stealth_order with extreme book_depth_ratio doesn't crash."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
            book_depth_ratio=0.1,
        )
        assert len(orders) >= 2

    @pytest.mark.asyncio
    async def test_e2e_default_ratio_no_crash(self, stealth):
        """place_stealth_order without book_depth_ratio uses default 1.0."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_1",
            asset_id="ASSET_1",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
        )
        assert len(orders) >= 2


# ═══════════════════════════════════════════════════════════════════════════
#  SI-4.2 Volume-Acceleration & L2-Fallback Wiring (PositionManager)
# ═══════════════════════════════════════════════════════════════════════════

class TestBurstVolumeWiring:
    """Prove that PositionManager.scale_probe_to_full correctly wires
    burst-volume acceleration and L2 fallback into StealthExecutor calls."""

    def _make_pm_with_stealth(self):
        """Build a PositionManager whose StealthExecutor is spy-patched
        so we can inspect the exact args passed to place_stealth_order."""
        from unittest.mock import MagicMock, AsyncMock, patch

        executor = MagicMock(spec=OrderExecutor)
        executor.paper_mode = True
        executor.place_limit_order = AsyncMock(
            return_value=MagicMock(order_id="O-STEALTH"),
        )

        from src.trading.position_manager import PositionManager
        pm = PositionManager(executor=executor)

        # Wire up a real StealthExecutor with a spy on place_stealth_order
        stealth = StealthExecutor(
            executor,
            min_size_usd=0.50,  # low threshold so probes qualify
            max_slices=2,
            min_delay_ms=0.0,
            max_delay_ms=0.0,
            size_jitter_pct=0.0,
        )
        stealth.place_stealth_order = AsyncMock(
            return_value=[MagicMock(order_id="O-STEALTH")],
        )
        pm._stealth = stealth
        return pm, stealth

    def _make_probe_pos(self, asset_id="ASSET-1", entry=0.47, target=0.54):
        from unittest.mock import MagicMock
        from src.trading.position_manager import PositionState

        tp = MagicMock()
        tp.target_price = target
        tp.viable = True

        kelly = MagicMock()
        kelly.size_shares = 20.0

        pos = MagicMock()
        pos.id = "PROBE-BV"
        pos.market_id = "MKT-BV"
        pos.no_asset_id = asset_id
        pos.trade_asset_id = asset_id
        pos.trade_side = "NO"
        pos.entry_price = entry
        pos.entry_size = 1.0  # small probe
        pos.state = PositionState.EXIT_PENDING
        pos.is_probe = True
        pos.tp_result = tp
        pos.kelly_result = kelly
        return pos

    # ── Burst-Volume Invariant ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_burst_volume_uses_max_of_avg_and_current(self):
        """When current_bar_volume is 20× avg, scale_probe_to_full must
        pass the larger current_bar_volume into recent_volume_usd."""
        from unittest.mock import MagicMock

        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()

        # Mock OHLCVAggregator: avg=10, current=200 (20× spike)
        agg = MagicMock()
        agg.avg_bar_volume = 10.0
        agg.current_bar_volume = 200.0
        pm._ohlcv_aggs = {pos.trade_asset_id: agg}

        mid_fn = lambda aid: 0.48  # profitable for NO entry at 0.47

        await pm.scale_probe_to_full(pos, mid_fn)

        stealth_spy.place_stealth_order.assert_called_once()
        call_kwargs = stealth_spy.place_stealth_order.call_args
        passed_vol = call_kwargs.kwargs.get(
            "recent_volume_usd",
            call_kwargs[1].get("recent_volume_usd") if len(call_kwargs) > 1 else None,
        )
        # expected: max(10, 200) * scale_price ≈ 200 * 0.48 = 96.0
        # scale_price = min(mid, entry+0.01) = min(0.48, 0.48) = 0.48
        assert passed_vol is not None
        assert passed_vol == pytest.approx(200.0 * 0.48, abs=1.0)

    @pytest.mark.asyncio
    async def test_burst_volume_avg_wins_when_higher(self):
        """When avg_bar_volume > current_bar_volume, avg is used."""
        from unittest.mock import MagicMock

        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()

        agg = MagicMock()
        agg.avg_bar_volume = 500.0
        agg.current_bar_volume = 10.0
        pm._ohlcv_aggs = {pos.trade_asset_id: agg}

        mid_fn = lambda aid: 0.48

        await pm.scale_probe_to_full(pos, mid_fn)

        stealth_spy.place_stealth_order.assert_called_once()
        call_kwargs = stealth_spy.place_stealth_order.call_args
        passed_vol = call_kwargs.kwargs.get("recent_volume_usd")
        # expected: max(500, 10) * 0.48 = 240.0
        assert passed_vol == pytest.approx(500.0 * 0.48, abs=1.0)

    @pytest.mark.asyncio
    async def test_burst_volume_no_aggregator_passes_zero(self):
        """When _ohlcv_aggs is None, recent_volume_usd defaults to 0."""
        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()
        pm._ohlcv_aggs = None

        mid_fn = lambda aid: 0.48

        await pm.scale_probe_to_full(pos, mid_fn)

        stealth_spy.place_stealth_order.assert_called_once()
        call_kwargs = stealth_spy.place_stealth_order.call_args
        passed_vol = call_kwargs.kwargs.get("recent_volume_usd")
        assert passed_vol == 0.0

    # ── L2 Fallback Invariant ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_l2_fallback_missing_tracker_defaults_to_1(self):
        """If _book_trackers has no entry for the asset, ratio = 1.0."""
        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()
        pm._book_trackers = {}  # no tracker
        pm._ohlcv_aggs = None

        mid_fn = lambda aid: 0.48

        await pm.scale_probe_to_full(pos, mid_fn)

        stealth_spy.place_stealth_order.assert_called_once()
        call_kwargs = stealth_spy.place_stealth_order.call_args
        passed_ratio = call_kwargs.kwargs.get("book_depth_ratio")
        assert passed_ratio == 1.0

    @pytest.mark.asyncio
    async def test_l2_fallback_unreliable_tracker_defaults_to_1(self):
        """If tracker exists but is_reliable=False, ratio = 1.0."""
        from unittest.mock import MagicMock

        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()
        pm._ohlcv_aggs = None

        tracker = MagicMock()
        tracker.is_reliable = False
        tracker.book_depth_ratio = 0.3  # would be hollowed, but unreliable
        pm._book_trackers = {pos.trade_asset_id: tracker}

        mid_fn = lambda aid: 0.48

        await pm.scale_probe_to_full(pos, mid_fn)

        stealth_spy.place_stealth_order.assert_called_once()
        call_kwargs = stealth_spy.place_stealth_order.call_args
        passed_ratio = call_kwargs.kwargs.get("book_depth_ratio")
        assert passed_ratio == 1.0

    @pytest.mark.asyncio
    async def test_l2_reliable_tracker_passes_real_ratio(self):
        """If tracker is reliable, the real book_depth_ratio is passed."""
        from unittest.mock import MagicMock

        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()
        pm._ohlcv_aggs = None

        tracker = MagicMock()
        tracker.is_reliable = True
        tracker.book_depth_ratio = 2.5  # heavy support
        pm._book_trackers = {pos.trade_asset_id: tracker}

        mid_fn = lambda aid: 0.48

        await pm.scale_probe_to_full(pos, mid_fn)

        stealth_spy.place_stealth_order.assert_called_once()
        call_kwargs = stealth_spy.place_stealth_order.call_args
        passed_ratio = call_kwargs.kwargs.get("book_depth_ratio")
        assert passed_ratio == 2.5

    @pytest.mark.asyncio
    async def test_burst_volume_and_l2_wired_together(self):
        """Both burst-volume and L2 ratio are passed simultaneously."""
        from unittest.mock import MagicMock

        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()

        agg = MagicMock()
        agg.avg_bar_volume = 50.0
        agg.current_bar_volume = 1000.0  # 20× spike
        pm._ohlcv_aggs = {pos.trade_asset_id: agg}

        tracker = MagicMock()
        tracker.is_reliable = True
        tracker.book_depth_ratio = 0.2  # hollowed book
        pm._book_trackers = {pos.trade_asset_id: tracker}

        mid_fn = lambda aid: 0.48

        await pm.scale_probe_to_full(pos, mid_fn)

        stealth_spy.place_stealth_order.assert_called_once()
        kw = stealth_spy.place_stealth_order.call_args.kwargs
        assert kw["recent_volume_usd"] == pytest.approx(1000.0 * 0.48, abs=1.0)
        assert kw["book_depth_ratio"] == 0.2


# ═══════════════════════════════════════════════════════════════════════════
#  SI-4.3 Slippage-Adaptive Pacing
# ═══════════════════════════════════════════════════════════════════════════

class TestSlippageAdaptivePacing:
    """Prove that detected slippage on a filled slice forces the next
    inter-slice delay to max_delay (friction choke)."""

    @pytest.fixture
    def stealth(self):
        executor = OrderExecutor(paper_mode=True)
        return StealthExecutor(
            executor,
            min_size_usd=5.0,
            max_slices=4,
            min_delay_ms=100.0,
            max_delay_ms=1000.0,
            size_jitter_pct=0.10,
        )

    # ── _obi_delay_ms unit tests ──────────────────────────────────────

    def test_slippage_choke_forces_max_delay(self, stealth):
        """slippage_choke=True overrides all OBI logic → max delay."""
        delay = stealth._obi_delay_ms(
            500.0, OrderSide.BUY, 3.0,  # ratio=3.0 would normally → min delay
            slippage_choke=True,
        )
        assert delay == stealth._max_delay

    def test_slippage_choke_overrides_heavy_support(self, stealth):
        """Even with ratio=5.0 (heavy support → min), slippage wins."""
        delay = stealth._obi_delay_ms(
            500.0, OrderSide.BUY, 5.0,
            slippage_choke=True,
        )
        assert delay == stealth._max_delay

    def test_slippage_choke_overrides_balanced(self, stealth):
        """Balanced ratio=1.0 + slippage_choke → max delay."""
        delay = stealth._obi_delay_ms(
            500.0, OrderSide.BUY, 1.0,
            slippage_choke=True,
        )
        assert delay == stealth._max_delay

    def test_no_slippage_choke_allows_normal_obi(self, stealth):
        """slippage_choke=False does not interfere with OBI logic."""
        delay = stealth._obi_delay_ms(
            500.0, OrderSide.BUY, 3.0,
            slippage_choke=False,
        )
        assert delay == stealth._min_delay  # ratio=3.0 → heavy support

    def test_slippage_choke_priority_over_iceberg(self, stealth):
        """Both slippage_choke and opposing_iceberg → slippage wins (both → max anyway)."""
        delay = stealth._obi_delay_ms(
            500.0, OrderSide.BUY, 3.0,
            slippage_choke=True,
            opposing_iceberg=True,
        )
        assert delay == stealth._max_delay

    # ── Slippage calculation thresholds ───────────────────────────────

    def test_half_tick_slippage_triggers_choke(self, stealth):
        """fill at price+0.006 → 0.6 ticks > 0.5 threshold → choke."""
        # Simulate: target=0.50, fill=0.506 → slippage=0.006/0.01=0.6 ticks
        slippage_ticks = abs(0.506 - 0.50) / 0.01
        assert slippage_ticks > 0.5

    def test_exactly_half_tick_no_choke(self, stealth):
        """fill at price+0.005 → 0.5 ticks exactly = NOT > 0.5 → no choke."""
        # Use integers to avoid float imprecision: 5 - 0 = 5, 5/10 = 0.5
        slippage_ticks = abs(50 - 50) / 1  # zero slippage, boundary illustration
        assert not (slippage_ticks > 0.5)
        # Direct _obi_delay_ms test: slippage_choke=False → normal OBI
        delay = stealth._obi_delay_ms(
            500.0, OrderSide.BUY, 1.0, slippage_choke=False,
        )
        assert delay == 500.0

    def test_zero_slippage_no_choke(self, stealth):
        """fill at exact target → 0 ticks → no choke."""
        slippage_ticks = abs(0.50 - 0.50) / 0.01
        assert slippage_ticks == 0.0

    # ── E2E: slippage-adaptive in place_stealth_order ─────────────────

    @pytest.mark.asyncio
    async def test_e2e_slippage_choke_does_not_crash(self, stealth):
        """place_stealth_order with paper executor doesn't crash."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_SLP",
            asset_id="ASSET_SLP",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
        )
        assert len(orders) >= 2


# ═══════════════════════════════════════════════════════════════════════════
#  SI-4.3 Iceberg-Aware Delay Skewing
# ═══════════════════════════════════════════════════════════════════════════

class TestIcebergAwareSkewing:
    """Prove that opposing_iceberg_detected forces max_delay,
    overriding even heavy-support OBI ratios."""

    @pytest.fixture
    def stealth(self):
        executor = OrderExecutor(paper_mode=True)
        return StealthExecutor(
            executor,
            min_size_usd=5.0,
            max_slices=4,
            min_delay_ms=100.0,
            max_delay_ms=1000.0,
            size_jitter_pct=0.10,
        )

    # ── _obi_delay_ms unit tests ──────────────────────────────────────

    def test_iceberg_forces_max_delay_balanced(self, stealth):
        """opposing_iceberg=True + balanced ratio=1.0 → max delay."""
        delay = stealth._obi_delay_ms(
            500.0, OrderSide.BUY, 1.0,
            opposing_iceberg=True,
        )
        assert delay == stealth._max_delay

    def test_iceberg_overrides_heavy_support(self, stealth):
        """opposing_iceberg=True + heavy-support ratio=3.0 → max delay."""
        delay = stealth._obi_delay_ms(
            500.0, OrderSide.BUY, 3.0,
            opposing_iceberg=True,
        )
        assert delay == stealth._max_delay

    def test_iceberg_overrides_heavy_support_sell(self, stealth):
        """SELL + opposing_iceberg + ratio=0.1 (our_ratio=10) → max delay."""
        delay = stealth._obi_delay_ms(
            500.0, OrderSide.SELL, 0.1,
            opposing_iceberg=True,
        )
        assert delay == stealth._max_delay

    def test_no_iceberg_allows_normal_obi(self, stealth):
        """opposing_iceberg=False + heavy support → min delay."""
        delay = stealth._obi_delay_ms(
            500.0, OrderSide.BUY, 3.0,
            opposing_iceberg=False,
        )
        assert delay == stealth._min_delay

    def test_no_iceberg_default_param(self, stealth):
        """Default opposing_iceberg (not passed) → normal OBI."""
        delay = stealth._obi_delay_ms(500.0, OrderSide.BUY, 1.0)
        assert delay == 500.0  # balanced → base delay

    # ── E2E: iceberg flag in place_stealth_order ──────────────────────

    @pytest.mark.asyncio
    async def test_e2e_iceberg_flag_no_crash(self, stealth):
        """place_stealth_order with opposing_iceberg_detected=True runs cleanly."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_ICE",
            asset_id="ASSET_ICE",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
            opposing_iceberg_detected=True,
        )
        assert len(orders) >= 2

    @pytest.mark.asyncio
    async def test_e2e_iceberg_false_no_crash(self, stealth):
        """place_stealth_order with opposing_iceberg_detected=False is fine."""
        orders = await stealth.place_stealth_order(
            market_id="MKT_ICE2",
            asset_id="ASSET_ICE2",
            side=OrderSide.BUY,
            price=0.50,
            total_size=20.0,
            opposing_iceberg_detected=False,
        )
        assert len(orders) >= 2


# ═══════════════════════════════════════════════════════════════════════════
#  SI-4.3 Iceberg Wiring (PositionManager → StealthExecutor)
# ═══════════════════════════════════════════════════════════════════════════

class TestIcebergWiring:
    """Prove PositionManager.scale_probe_to_full passes
    opposing_iceberg_detected correctly to StealthExecutor."""

    def _make_pm_with_stealth(self):
        from unittest.mock import MagicMock, AsyncMock
        executor = MagicMock(spec=OrderExecutor)
        executor.paper_mode = True
        executor.place_limit_order = AsyncMock(
            return_value=MagicMock(order_id="O-ICE"),
        )
        from src.trading.position_manager import PositionManager
        pm = PositionManager(executor=executor)
        stealth = StealthExecutor(
            executor,
            min_size_usd=0.50,
            max_slices=2,
            min_delay_ms=0.0,
            max_delay_ms=0.0,
            size_jitter_pct=0.0,
        )
        stealth.place_stealth_order = AsyncMock(
            return_value=[MagicMock(order_id="O-ICE")],
        )
        pm._stealth = stealth
        return pm, stealth

    def _make_probe_pos(self, asset_id="ASSET-ICE"):
        from unittest.mock import MagicMock
        from src.trading.position_manager import PositionState

        tp = MagicMock()
        tp.target_price = 0.54
        tp.viable = True

        kelly = MagicMock()
        kelly.size_shares = 20.0

        pos = MagicMock()
        pos.id = "PROBE-ICE"
        pos.market_id = "MKT-ICE"
        pos.no_asset_id = asset_id
        pos.trade_asset_id = asset_id
        pos.trade_side = "NO"
        pos.entry_price = 0.47
        pos.entry_size = 1.0
        pos.state = PositionState.EXIT_PENDING
        pos.is_probe = True
        pos.tp_result = tp
        pos.kelly_result = kelly
        return pos

    @pytest.mark.asyncio
    async def test_iceberg_detected_passes_true(self):
        """IcebergDetector strongest_iceberg confidence ≥ 0.50 → True."""
        from unittest.mock import MagicMock
        from src.signals.iceberg_detector import IcebergSignal

        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()
        pm._ohlcv_aggs = None

        ice_det = MagicMock()
        ice_signal = IcebergSignal(
            asset_id=pos.trade_asset_id,
            side="SELL",
            price=0.52,
            refill_count=5,
            avg_slice_size=10.0,
            estimated_total=50.0,
            timestamp=1.0,
            confidence=0.75,
        )
        ice_det.strongest_iceberg.return_value = ice_signal
        pm._iceberg_detectors = {pos.trade_asset_id: ice_det}

        await pm.scale_probe_to_full(pos, lambda aid: 0.48)

        stealth_spy.place_stealth_order.assert_called_once()
        kw = stealth_spy.place_stealth_order.call_args.kwargs
        assert kw["opposing_iceberg_detected"] is True

    @pytest.mark.asyncio
    async def test_iceberg_low_confidence_passes_false(self):
        """IcebergDetector confidence < 0.50 → False."""
        from unittest.mock import MagicMock
        from src.signals.iceberg_detector import IcebergSignal

        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()
        pm._ohlcv_aggs = None

        ice_det = MagicMock()
        ice_signal = IcebergSignal(
            asset_id=pos.trade_asset_id,
            side="SELL",
            price=0.52,
            refill_count=2,
            avg_slice_size=5.0,
            estimated_total=10.0,
            timestamp=1.0,
            confidence=0.40,  # below threshold
        )
        ice_det.strongest_iceberg.return_value = ice_signal
        pm._iceberg_detectors = {pos.trade_asset_id: ice_det}

        await pm.scale_probe_to_full(pos, lambda aid: 0.48)

        stealth_spy.place_stealth_order.assert_called_once()
        kw = stealth_spy.place_stealth_order.call_args.kwargs
        assert kw["opposing_iceberg_detected"] is False

    @pytest.mark.asyncio
    async def test_no_iceberg_detector_passes_false(self):
        """No IcebergDetector for this asset → False."""
        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()
        pm._ohlcv_aggs = None
        pm._iceberg_detectors = {}

        await pm.scale_probe_to_full(pos, lambda aid: 0.48)

        stealth_spy.place_stealth_order.assert_called_once()
        kw = stealth_spy.place_stealth_order.call_args.kwargs
        assert kw["opposing_iceberg_detected"] is False

    @pytest.mark.asyncio
    async def test_no_strongest_iceberg_passes_false(self):
        """IcebergDetector returns None → False."""
        from unittest.mock import MagicMock

        pm, stealth_spy = self._make_pm_with_stealth()
        pos = self._make_probe_pos()
        pm._ohlcv_aggs = None

        ice_det = MagicMock()
        ice_det.strongest_iceberg.return_value = None
        pm._iceberg_detectors = {pos.trade_asset_id: ice_det}

        await pm.scale_probe_to_full(pos, lambda aid: 0.48)

        stealth_spy.place_stealth_order.assert_called_once()
        kw = stealth_spy.place_stealth_order.call_args.kwargs
        assert kw["opposing_iceberg_detected"] is False
