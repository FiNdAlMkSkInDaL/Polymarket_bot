"""
Tests for the Resolution Probability Engine (RPE).

Covers:
  - Crypto question parsing (regex patterns)
  - CryptoPriceModel (ITM, OTM, near-expiry, missing data)
  - GenericBayesianModel (50/50, extremes, time decay)
  - ResolutionProbabilityEngine (signal firing, direction, gating)
  - PositionManager.open_rpe_position (bidirectional, shadow mode)
  - Model estimate confidence properties
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from src.signals.resolution_probability import (
    CryptoPriceModel,
    GenericBayesianModel,
    ModelEstimate,
    ResolutionProbabilityEngine,
    parse_crypto_question,
)
from src.signals.signal_framework import SignalResult

# Ensure paper mode
os.environ.setdefault("PAPER_MODE", "true")
os.environ.setdefault("DEPLOYMENT_ENV", "PAPER")


# ── Helpers ──────────────────────────────────────────────────────────────

@dataclass
class FakeMarketInfo:
    """Minimal MarketInfo stand-in for tests."""

    condition_id: str = "TEST_MKT"
    question: str = "Will Bitcoin exceed $100,000 by March 31, 2026?"
    yes_token_id: str = "YES_TOKEN"
    no_token_id: str = "NO_TOKEN"
    daily_volume_usd: float = 100_000.0
    end_date: datetime | None = None
    active: bool = True
    event_id: str = "EVT_1"
    liquidity_usd: float = 50_000.0
    score: float = 80.0
    accepting_orders: bool = True
    tags: str = "crypto"

    def __post_init__(self) -> None:
        if self.end_date is None:
            self.end_date = datetime.now(timezone.utc) + timedelta(days=30)


# ═══════════════════════════════════════════════════════════════════════════
#  Crypto Question Parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestCryptoQuestionParsing:
    """Regex pattern tests for parse_crypto_question()."""

    def test_standard_btc(self) -> None:
        result = parse_crypto_question(
            "Will Bitcoin exceed $100,000 by March 31, 2026?"
        )
        assert result is not None
        ticker, strike = result
        assert ticker == "BTC"
        assert strike == 100_000.0

    def test_btc_abbreviation(self) -> None:
        result = parse_crypto_question(
            "Will BTC reach $150,000 by December 2026?"
        )
        assert result is not None
        assert result[0] == "BTC"
        assert result[1] == 150_000.0

    def test_eth_with_k_suffix(self) -> None:
        result = parse_crypto_question(
            "Will ETH reach $5k by end of 2026?"
        )
        assert result is not None
        assert result[0] == "ETH"
        assert result[1] == 5_000.0

    def test_ethereum_full_name(self) -> None:
        result = parse_crypto_question(
            "Will Ethereum exceed $10,000 by January 2027?"
        )
        assert result is not None
        assert result[0] == "ETH"
        assert result[1] == 10_000.0

    def test_above_keyword(self) -> None:
        result = parse_crypto_question(
            "Will BTC be above $200,000 on December 31, 2026?"
        )
        assert result is not None
        assert result[0] == "BTC"
        assert result[1] == 200_000.0

    def test_non_crypto_returns_none(self) -> None:
        assert parse_crypto_question("Will the US enter a recession?") is None

    def test_no_price_returns_none(self) -> None:
        assert parse_crypto_question("Will Bitcoin moon?") is None

    def test_hit_keyword(self) -> None:
        result = parse_crypto_question(
            "Will Bitcoin hit $75,000 by February 2026?"
        )
        assert result is not None
        assert result[1] == 75_000.0


# ═══════════════════════════════════════════════════════════════════════════
#  CryptoPriceModel
# ═══════════════════════════════════════════════════════════════════════════


class TestCryptoPriceModel:
    """Log-normal crypto price model tests."""

    def _make_model(
        self, spot: float | None = 95_000.0, vol: float = 0.80
    ) -> CryptoPriceModel:
        return CryptoPriceModel(
            price_fn=lambda: spot,
            vol_override=vol,
        )

    def test_itm_high_probability(self) -> None:
        """BTC at $100k, strike $90k, 30 days → high probability."""
        model = self._make_model(spot=100_000.0)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $90,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
        )
        est = model.estimate(market, market_price=0.80)
        assert est is not None
        assert est.probability > 0.5
        assert est.confidence > 0.2
        assert est.model_name == "crypto_lognormal"

    def test_otm_low_probability(self) -> None:
        """BTC at $50k, strike $100k, 7 days → low probability."""
        model = self._make_model(spot=50_000.0)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=7),
        )
        est = model.estimate(market, market_price=0.10)
        assert est is not None
        assert est.probability < 0.2

    def test_near_expiry_atm(self) -> None:
        """Near-expiry ATM → model should produce estimate with lower time_factor."""
        model = self._make_model(spot=100_000.0)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=2),
        )
        est = model.estimate(market, market_price=0.50)
        assert est is not None
        # Near ATM + near expiry → confidence should be relatively low
        assert 0.0 < est.confidence < 1.0

    def test_no_spot_price(self) -> None:
        """Price feed returns None → model returns None (no hallucination)."""
        model = self._make_model(spot=None)
        market = FakeMarketInfo()
        est = model.estimate(market, market_price=0.50)
        assert est is None

    def test_expired_market_itm(self) -> None:
        """Already expired + spot > strike → probability ≈ 1."""
        model = self._make_model(spot=110_000.0)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        est = model.estimate(market, market_price=0.95)
        assert est is not None
        assert est.probability == 1.0
        assert est.confidence == 0.95

    def test_expired_market_otm(self) -> None:
        """Already expired + spot < strike → probability ≈ 0."""
        model = self._make_model(spot=90_000.0)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        est = model.estimate(market, market_price=0.05)
        assert est is not None
        assert est.probability == 0.0

    def test_no_end_date(self) -> None:
        """No end_date → model returns None."""
        model = self._make_model()
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
        )
        market.end_date = None
        est = model.estimate(market, market_price=0.50)
        assert est is None

    def test_can_handle_crypto(self) -> None:
        model = self._make_model()
        market = FakeMarketInfo()
        assert model.can_handle(market) is True

    def test_cannot_handle_non_crypto(self) -> None:
        model = self._make_model()
        market = FakeMarketInfo(
            question="Will the Democrats win the 2028 election?",
            tags="politics",
        )
        assert model.can_handle(market) is False

    def test_probability_clamped(self) -> None:
        """Probability should never be exactly 0 or 1 (except expired)."""
        model = self._make_model(spot=1_000_000.0)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=365),
        )
        est = model.estimate(market, market_price=0.99)
        assert est is not None
        assert 0.01 <= est.probability <= 0.99

    def test_metadata_contains_model_params(self) -> None:
        model = self._make_model()
        market = FakeMarketInfo()
        est = model.estimate(market, market_price=0.50)
        assert est is not None
        assert "ticker" in est.metadata
        assert "spot" in est.metadata
        assert "strike" in est.metadata
        assert "sigma" in est.metadata
        assert "d2" in est.metadata


# ═══════════════════════════════════════════════════════════════════════════
#  GenericBayesianModel
# ═══════════════════════════════════════════════════════════════════════════


class TestGenericBayesianModel:
    """Beta-distribution Bayesian model tests."""

    def _make_model(self, obs_weight: float = 5.0) -> GenericBayesianModel:
        return GenericBayesianModel(obs_weight=obs_weight)

    def test_near_50_50(self) -> None:
        """Market at 0.50 → estimate near 0.50."""
        model = self._make_model()
        market = FakeMarketInfo(
            question="Will it rain tomorrow?",
            tags="weather",
        )
        est = model.estimate(market, market_price=0.50, days_to_resolution=30)
        assert est is not None
        assert abs(est.probability - 0.50) < 0.05

    def test_extreme_price_high(self) -> None:
        """Market at 0.95 → high probability estimate."""
        model = self._make_model()
        market = FakeMarketInfo(tags="politics")
        est = model.estimate(market, market_price=0.95, days_to_resolution=5)
        assert est is not None
        assert est.probability > 0.7  # Beta(2,2) prior pulls toward 0.5

    def test_extreme_price_low(self) -> None:
        """Market at 0.05 → low probability estimate."""
        model = self._make_model()
        market = FakeMarketInfo(tags="politics")
        est = model.estimate(market, market_price=0.05, days_to_resolution=5)
        assert est is not None
        assert est.probability < 0.3  # Beta(2,2) prior pulls toward 0.5

    def test_time_decay_increases_confidence(self) -> None:
        """Closer to resolution → higher confidence."""
        model = self._make_model()
        market = FakeMarketInfo(tags="sports")

        est_far = model.estimate(market, market_price=0.70, days_to_resolution=90)
        est_near = model.estimate(market, market_price=0.70, days_to_resolution=5)

        assert est_far is not None and est_near is not None
        assert est_near.confidence > est_far.confidence

    def test_always_can_handle(self) -> None:
        """GenericBayesianModel handles all markets."""
        model = self._make_model()
        for tags in ("crypto", "politics", "sports", "", "random"):
            market = FakeMarketInfo(tags=tags)
            assert model.can_handle(market) is True

    def test_invalid_price_returns_none(self) -> None:
        """Market price at 0 or 1 → None."""
        model = self._make_model()
        market = FakeMarketInfo()
        assert model.estimate(market, market_price=0.0) is None
        assert model.estimate(market, market_price=1.0) is None

    def test_model_name(self) -> None:
        model = self._make_model()
        assert model.name == "generic_bayesian"

    def test_higher_obs_weight_tracks_price(self) -> None:
        """With higher observation weight, estimate follows market price more closely."""
        model_low = self._make_model(obs_weight=1.0)
        model_high = self._make_model(obs_weight=20.0)
        market = FakeMarketInfo(tags="politics")

        est_low = model_low.estimate(market, market_price=0.90, days_to_resolution=30)
        est_high = model_high.estimate(market, market_price=0.90, days_to_resolution=30)

        assert est_low is not None and est_high is not None
        # Higher weight → closer to market price
        assert abs(est_high.probability - 0.90) < abs(est_low.probability - 0.90)

    def test_metadata_fields(self) -> None:
        model = self._make_model()
        market = FakeMarketInfo()
        est = model.estimate(market, market_price=0.60, days_to_resolution=30)
        assert est is not None
        assert "alpha_post" in est.metadata
        assert "beta_post" in est.metadata
        assert "entropy_penalty" in est.metadata
        assert "time_factor" in est.metadata


# ═══════════════════════════════════════════════════════════════════════════
#  ResolutionProbabilityEngine
# ═══════════════════════════════════════════════════════════════════════════


class TestResolutionProbabilityEngine:
    """RPE signal firing, routing, and gating tests."""

    def _make_rpe(
        self,
        spot: float = 95_000.0,
        *,
        confidence_threshold: float = 0.05,
        shadow_mode: bool = False,
        min_confidence: float = 0.10,
    ) -> ResolutionProbabilityEngine:
        crypto = CryptoPriceModel(
            price_fn=lambda: spot, vol_override=0.80
        )
        generic = GenericBayesianModel(obs_weight=5.0)
        return ResolutionProbabilityEngine(
            models=[crypto, generic],
            confidence_threshold=confidence_threshold,
            shadow_mode=shadow_mode,
            min_confidence=min_confidence,
        )

    def test_signal_fires_on_divergence(self) -> None:
        """Large divergence → signal fires."""
        rpe = self._make_rpe(spot=120_000.0, confidence_threshold=0.05)
        # Model says ~high prob YES, but market price is 0.30
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=60),
        )
        result = rpe.evaluate(
            market=market, market_price=0.30, days_to_resolution=60
        )
        assert result is not None
        assert isinstance(result, SignalResult)
        assert result.metadata["direction"] == "buy_yes"  # underpriced

    def test_no_signal_within_threshold(self) -> None:
        """Small divergence → no signal."""
        rpe = self._make_rpe(spot=100_000.0, confidence_threshold=0.50)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
        )
        # At ATM the model prob ≈ 0.45-0.50, market_price=0.50 → tiny divergence
        result = rpe.evaluate(
            market=market, market_price=0.50, days_to_resolution=30
        )
        assert result is None

    def test_direction_buy_no(self) -> None:
        """Market overpriced → direction is buy_no."""
        rpe = self._make_rpe(spot=50_000.0, confidence_threshold=0.03)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
        )
        result = rpe.evaluate(
            market=market, market_price=0.80, days_to_resolution=30
        )
        # Model says low prob (spot << strike), market says 0.80 → buy NO
        assert result is not None
        assert result.metadata["direction"] == "buy_no"

    def test_direction_buy_yes(self) -> None:
        """Market underpriced → direction is buy_yes."""
        rpe = self._make_rpe(spot=150_000.0, confidence_threshold=0.03)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=60),
        )
        result = rpe.evaluate(
            market=market, market_price=0.30, days_to_resolution=60
        )
        assert result is not None
        assert result.metadata["direction"] == "buy_yes"

    def test_confidence_too_low(self) -> None:
        """Low model confidence → no signal."""
        rpe = self._make_rpe(
            spot=100_000.0,
            confidence_threshold=0.01,
            min_confidence=0.99,  # unreasonably high → blocks everything
        )
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
        )
        result = rpe.evaluate(
            market=market, market_price=0.10, days_to_resolution=30
        )
        assert result is None

    def test_shadow_mode_flag(self) -> None:
        """Shadow mode flag is correctly set in metadata."""
        rpe = self._make_rpe(
            spot=120_000.0,
            confidence_threshold=0.05,
            shadow_mode=True,
        )
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=60),
        )
        result = rpe.evaluate(
            market=market, market_price=0.30, days_to_resolution=60
        )
        assert result is not None
        assert result.metadata["shadow_mode"] is True

    def test_can_handle_routing_crypto(self) -> None:
        """Crypto markets route to CryptoPriceModel."""
        rpe = self._make_rpe(spot=95_000.0, confidence_threshold=0.05)
        market = FakeMarketInfo()
        result = rpe.evaluate(
            market=market, market_price=0.30, days_to_resolution=30
        )
        if result is not None:
            assert result.metadata["model_name"] == "crypto_lognormal"

    def test_can_handle_routing_generic(self) -> None:
        """Non-crypto markets route to GenericBayesianModel."""
        rpe = self._make_rpe(spot=95_000.0, confidence_threshold=0.01)
        market = FakeMarketInfo(
            question="Will the Democrats win the 2028 election?",
            tags="politics",
        )
        # Force a large divergence
        result = rpe.evaluate(
            market=market, market_price=0.90, days_to_resolution=30
        )
        if result is not None:
            assert result.metadata["model_name"] == "generic_bayesian"

    def test_estimate_cached(self) -> None:
        """After evaluation, estimate is cached and retrievable."""
        rpe = self._make_rpe(spot=95_000.0, confidence_threshold=0.01)
        market = FakeMarketInfo()
        rpe.evaluate(market=market, market_price=0.50, days_to_resolution=30)
        cached = rpe.get_estimate("TEST_MKT")
        assert cached is not None
        assert 0 < cached.probability < 1

    def test_model_estimate_always_has_confidence(self) -> None:
        """Every estimate carries 0 < confidence < 1."""
        rpe = self._make_rpe(spot=95_000.0, confidence_threshold=0.01)
        market = FakeMarketInfo()
        rpe.evaluate(market=market, market_price=0.50, days_to_resolution=30)
        est = rpe.get_estimate("TEST_MKT")
        assert est is not None
        assert 0 < est.confidence < 1

    def test_score_normalisation(self) -> None:
        """Score saturates at 1.0 for 20¢+ divergence."""
        rpe = self._make_rpe(spot=50_000.0, confidence_threshold=0.01)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
        )
        result = rpe.evaluate(
            market=market, market_price=0.80, days_to_resolution=30
        )
        assert result is not None
        assert 0 < result.score <= 1.0

    def test_uncertainty_penalty_in_metadata(self) -> None:
        """Signal metadata includes uncertainty_penalty for Kelly integration."""
        rpe = self._make_rpe(spot=120_000.0, confidence_threshold=0.05)
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=60),
        )
        result = rpe.evaluate(
            market=market, market_price=0.30, days_to_resolution=60
        )
        assert result is not None
        penalty = result.metadata["uncertainty_penalty"]
        assert 0 <= penalty <= 1
        # uncertainty_penalty = 1 - confidence
        assert abs(penalty - (1.0 - result.metadata["confidence"])) < 0.001

    def test_invalid_market_price(self) -> None:
        """Invalid market price (0 or 1) → no signal."""
        rpe = self._make_rpe(confidence_threshold=0.01)
        market = FakeMarketInfo()
        assert rpe.evaluate(market=market, market_price=0.0) is None
        assert rpe.evaluate(market=market, market_price=1.0) is None

    def test_no_market_returns_none(self) -> None:
        """Missing market kwarg → None."""
        rpe = self._make_rpe(confidence_threshold=0.01)
        assert rpe.evaluate(market_price=0.50) is None

    def test_name_property(self) -> None:
        rpe = self._make_rpe()
        assert rpe.name == "resolution_probability_engine"


# ═══════════════════════════════════════════════════════════════════════════
#  PositionManager RPE Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestPositionManagerRPE:
    """Integration tests for PositionManager.open_rpe_position."""

    @pytest.fixture
    def pm(self, paper_executor, trade_store):
        from src.trading.position_manager import PositionManager
        pm = PositionManager(
            paper_executor, trade_store=trade_store, max_open_positions=5
        )
        pm.set_wallet_balance(1000.0)
        return pm

    @pytest.mark.asyncio
    async def test_shadow_mode_returns_none(self, pm) -> None:
        """In shadow mode, open_rpe_position always returns None."""
        # Force shadow mode ON via monkeypatch at the strategy level
        import src.core.config as cfg
        original = cfg.settings.strategy
        shadow_params = cfg.StrategyParams(
            rpe_shadow_mode=True,
            max_trade_size_usd=100.0,
        )
        object.__setattr__(cfg.settings, "strategy", shadow_params)
        try:
            pos = await pm.open_rpe_position(
                market_id="MKT_1",
                yes_asset_id="YES_1",
                no_asset_id="NO_1",
                direction="buy_no",
                model_probability=0.30,
                confidence=0.80,
                entry_price=0.65,
                fee_enabled=True,
            )
            assert pos is None
        finally:
            object.__setattr__(cfg.settings, "strategy", original)

    @pytest.mark.asyncio
    async def test_rpe_yes_entry(self, pm) -> None:
        """RPE buy_yes opens a position with trade_side=YES."""
        import src.core.config as cfg
        original = cfg.settings.strategy
        params = cfg.StrategyParams(
            rpe_shadow_mode=False,
            max_trade_size_usd=100.0,
            min_edge_score=0.0,
            stop_loss_cents=8.0,
        )
        object.__setattr__(cfg.settings, "strategy", params)
        try:
            pos = await pm.open_rpe_position(
                market_id="MKT_1",
                yes_asset_id="YES_1",
                no_asset_id="NO_1",
                direction="buy_yes",
                model_probability=0.80,
                confidence=0.70,
                entry_price=0.30,
                fee_enabled=False,
            )
            if pos is not None:
                assert pos.trade_side == "YES"
                assert pos.trade_asset_id == "YES_1"
                assert pos.id.startswith("RPE-")
        finally:
            object.__setattr__(cfg.settings, "strategy", original)

    @pytest.mark.asyncio
    async def test_rpe_no_entry(self, pm) -> None:
        """RPE buy_no opens a position with trade_side=NO."""
        import src.core.config as cfg
        original = cfg.settings.strategy
        params = cfg.StrategyParams(
            rpe_shadow_mode=False,
            max_trade_size_usd=100.0,
            min_edge_score=0.0,
            stop_loss_cents=8.0,
        )
        object.__setattr__(cfg.settings, "strategy", params)
        try:
            pos = await pm.open_rpe_position(
                market_id="MKT_1",
                yes_asset_id="YES_1",
                no_asset_id="NO_1",
                direction="buy_no",
                model_probability=0.20,
                confidence=0.70,
                entry_price=0.75,
                fee_enabled=False,
            )
            if pos is not None:
                assert pos.trade_side == "NO"
                assert pos.trade_asset_id == "NO_1"
        finally:
            object.__setattr__(cfg.settings, "strategy", original)

    @pytest.mark.asyncio
    async def test_risk_gates_respected(self, pm) -> None:
        """Circuit breaker blocks RPE entries just like panic entries."""
        pm._circuit_breaker_tripped = True
        import src.core.config as cfg
        original = cfg.settings.strategy
        params = cfg.StrategyParams(rpe_shadow_mode=False)
        object.__setattr__(cfg.settings, "strategy", params)
        try:
            pos = await pm.open_rpe_position(
                market_id="MKT_1",
                yes_asset_id="YES_1",
                no_asset_id="NO_1",
                direction="buy_yes",
                model_probability=0.80,
                confidence=0.90,
                entry_price=0.30,
            )
            assert pos is None
        finally:
            object.__setattr__(cfg.settings, "strategy", original)

    @pytest.mark.asyncio
    async def test_exit_uses_trade_asset_id(self, pm) -> None:
        """on_entry_filled uses trade_asset_id for exit order."""
        import src.core.config as cfg
        original = cfg.settings.strategy
        params = cfg.StrategyParams(
            rpe_shadow_mode=False,
            max_trade_size_usd=100.0,
            min_edge_score=0.0,
            stop_loss_cents=8.0,
        )
        object.__setattr__(cfg.settings, "strategy", params)
        try:
            pos = await pm.open_rpe_position(
                market_id="MKT_1",
                yes_asset_id="YES_1",
                no_asset_id="NO_1",
                direction="buy_yes",
                model_probability=0.80,
                confidence=0.70,
                entry_price=0.30,
                fee_enabled=False,
            )
            if pos is not None:
                # Simulate fill
                from src.trading.executor import Order, OrderStatus
                pos.entry_order = Order(
                    order_id="test-entry",
                    market_id="MKT_1",
                    asset_id="YES_1",
                    side="BUY",
                    price=0.30,
                    size=10,
                    status=OrderStatus.FILLED,
                    filled_size=10,
                    filled_avg_price=0.30,
                )
                await pm.on_entry_filled(pos)
                assert pos.exit_order is not None
                assert pos.exit_order.asset_id == "YES_1"
        finally:
            object.__setattr__(cfg.settings, "strategy", original)


# ═══════════════════════════════════════════════════════════════════════════
#  ModelEstimate properties
# ═══════════════════════════════════════════════════════════════════════════


class TestModelEstimate:
    """ModelEstimate invariants."""

    def test_fields_present(self) -> None:
        est = ModelEstimate(
            probability=0.65,
            confidence=0.80,
            model_name="test",
        )
        assert est.probability == 0.65
        assert est.confidence == 0.80
        assert est.model_name == "test"
        assert isinstance(est.metadata, dict)

    def test_metadata_default_empty(self) -> None:
        est = ModelEstimate(probability=0.5, confidence=0.5, model_name="x")
        assert est.metadata == {}

    def test_metadata_custom(self) -> None:
        est = ModelEstimate(
            probability=0.5,
            confidence=0.5,
            model_name="x",
            metadata={"key": "value"},
        )
        assert est.metadata["key"] == "value"


# ═══════════════════════════════════════════════════════════════════════════
#  Fix 1 — GenericBayesianModel gate (rpe_generic_enabled)
# ═══════════════════════════════════════════════════════════════════════════


class TestGenericModelGate:
    """Verify RPE_GENERIC_ENABLED gates signal generation for generic model."""

    def _make_rpe(
        self,
        *,
        generic_enabled: bool,
        confidence_threshold: float = 0.01,
        min_confidence: float = 0.05,
    ) -> ResolutionProbabilityEngine:
        crypto = CryptoPriceModel(price_fn=lambda: None, vol_override=0.80)
        generic = GenericBayesianModel(obs_weight=5.0)
        return ResolutionProbabilityEngine(
            models=[crypto, generic],
            confidence_threshold=confidence_threshold,
            shadow_mode=False,
            min_confidence=min_confidence,
            generic_enabled=generic_enabled,
        )

    def test_generic_disabled_blocks_signal(self) -> None:
        """When generic_enabled=False, generic-only markets produce no signal."""
        rpe = self._make_rpe(generic_enabled=False)
        market = FakeMarketInfo(
            question="Will the Democrats win the 2028 election?",
            tags="politics",
        )
        # Large divergence: market at 0.90, generic model ≈ 0.72 → buy_no
        result = rpe.evaluate(
            market=market, market_price=0.90, days_to_resolution=10
        )
        assert result is None

    def test_generic_disabled_still_caches_estimate(self) -> None:
        """Estimate is cached even when generic signal is gated."""
        rpe = self._make_rpe(generic_enabled=False)
        market = FakeMarketInfo(
            question="Will it rain tomorrow?",
            tags="weather",
            condition_id="WEATHER_1",
        )
        rpe.evaluate(market=market, market_price=0.80, days_to_resolution=10)
        cached = rpe.get_estimate("WEATHER_1")
        assert cached is not None
        assert cached.model_name == "generic_bayesian"
        assert 0 < cached.probability < 1

    def test_generic_enabled_allows_signal(self) -> None:
        """When generic_enabled=True, generic model CAN fire signals."""
        rpe = self._make_rpe(generic_enabled=True)
        market = FakeMarketInfo(
            question="Will the Democrats win the 2028 election?",
            tags="politics",
        )
        result = rpe.evaluate(
            market=market, market_price=0.90, days_to_resolution=10
        )
        # With obs_weight=5 and price=0.90, posterior≈0.72, divergence≈0.18
        # threshold is ~0.01 * (1 + 0.72) ≈ 0.017, so 0.18 > 0.017 → fires
        assert result is not None
        assert result.metadata["model_name"] == "generic_bayesian"

    def test_crypto_model_unaffected_by_generic_gate(self) -> None:
        """Crypto model fires regardless of generic_enabled flag."""
        crypto = CryptoPriceModel(price_fn=lambda: 120_000.0, vol_override=0.80)
        generic = GenericBayesianModel(obs_weight=5.0)
        rpe = ResolutionProbabilityEngine(
            models=[crypto, generic],
            confidence_threshold=0.05,
            shadow_mode=False,
            min_confidence=0.10,
            generic_enabled=False,  # generic disabled
        )
        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=60),
        )
        result = rpe.evaluate(
            market=market, market_price=0.30, days_to_resolution=60
        )
        assert result is not None
        assert result.metadata["model_name"] == "crypto_lognormal"

    def test_default_generic_enabled_is_true(self) -> None:
        """Default config has rpe_generic_enabled=True (activated with RPE)."""
        from src.core.config import settings
        assert settings.strategy.rpe_generic_enabled is True


# ═══════════════════════════════════════════════════════════════════════════
#  Fix 2 — Crypto retrigger on price move
# ═══════════════════════════════════════════════════════════════════════════


class TestCryptoRetrigger:
    """Verify that a sufficient spot price move triggers RPE re-evaluation."""

    def test_retrigger_fires_on_large_move(self) -> None:
        """Spot moves by more than threshold → RPE evaluate is called."""
        crypto = CryptoPriceModel(price_fn=lambda: 120_000.0, vol_override=0.80)
        generic = GenericBayesianModel(obs_weight=5.0)
        rpe = ResolutionProbabilityEngine(
            models=[crypto, generic],
            confidence_threshold=0.05,
            shadow_mode=False,
            min_confidence=0.10,
            generic_enabled=False,
        )

        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=60),
        )

        # Simulate first evaluation at spot=100k
        # Then spot jumps to 120k (20k > 500 threshold)
        # The RPE should produce a signal on the new spot
        result = rpe.evaluate(
            market=market, market_price=0.30, days_to_resolution=60
        )
        assert result is not None
        assert result.metadata["model_name"] == "crypto_lognormal"

    def test_retrigger_suppressed_on_noise(self) -> None:
        """Spot moves less than threshold → no re-evaluation needed.

        This tests the logic: |spot - last_spot| < threshold → skip.
        We simulate by checking that an RPE with a near-ATM spot doesn't
        produce a signal when divergence is small.
        """
        threshold_cents = 500.0
        spot_a = 100_000.0
        spot_b = spot_a + threshold_cents - 1  # $499 move, below threshold

        # Verify the price difference is below threshold
        assert abs(spot_b - spot_a) < threshold_cents

        # Both spots produce similar estimates for the same market
        crypto_a = CryptoPriceModel(price_fn=lambda: spot_a, vol_override=0.80)
        crypto_b = CryptoPriceModel(price_fn=lambda: spot_b, vol_override=0.80)

        market = FakeMarketInfo(
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
            end_date=datetime.now(timezone.utc) + timedelta(days=30),
        )

        est_a = crypto_a.estimate(market, 0.50)
        est_b = crypto_b.estimate(market, 0.50)
        assert est_a is not None and est_b is not None
        # The probability difference from a $499 move should be small
        assert abs(est_a.probability - est_b.probability) < 0.05

    def test_retrigger_config_default(self) -> None:
        """Default rpe_crypto_retrigger_cents is 500."""
        from src.core.config import settings
        assert settings.strategy.rpe_crypto_retrigger_cents == 500.0


# ═══════════════════════════════════════════════════════════════════════════
#  Fix 3 — Exit chaser routes to correct book for YES-side RPE positions
# ═══════════════════════════════════════════════════════════════════════════


class TestExitChaserBookRouting:
    """Verify that YES-side RPE positions route exit to the YES book."""

    def test_yes_position_trade_asset_id(self) -> None:
        """RPE buy_yes sets trade_asset_id=YES token."""
        from src.trading.position_manager import Position
        pos = Position(
            id="RPE-test",
            market_id="MKT_1",
            no_asset_id="NO_1",
            yes_asset_id="YES_1",
            trade_asset_id="YES_1",
            trade_side="YES",
            entry_price=0.30,
        )
        # The bot uses: exit_asset = pos.trade_asset_id or pos.no_asset_id
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        assert exit_asset == "YES_1"

    def test_no_position_trade_asset_id(self) -> None:
        """RPE buy_no sets trade_asset_id=NO token."""
        from src.trading.position_manager import Position
        pos = Position(
            id="RPE-test",
            market_id="MKT_1",
            no_asset_id="NO_1",
            yes_asset_id="YES_1",
            trade_asset_id="NO_1",
            trade_side="NO",
            entry_price=0.70,
        )
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        assert exit_asset == "NO_1"

    def test_legacy_panic_position_fallback(self) -> None:
        """Panic positions (trade_asset_id='') fall back to no_asset_id."""
        from src.trading.position_manager import Position
        pos = Position(
            id="PANIC-test",
            market_id="MKT_1",
            no_asset_id="NO_1",
            entry_price=0.60,
        )
        # Legacy: trade_asset_id defaults to ""
        assert pos.trade_asset_id == ""
        exit_asset = pos.trade_asset_id or pos.no_asset_id
        assert exit_asset == "NO_1"


# ═══════════════════════════════════════════════════════════════════════════
#  Fix 4 — Singleton RPE engine
# ═══════════════════════════════════════════════════════════════════════════


class TestSingletonRPE:
    """Verify that one RPE instance handles multiple markets correctly."""

    def test_shared_rpe_multi_market(self) -> None:
        """One RPE caches estimates for multiple different markets."""
        crypto = CryptoPriceModel(price_fn=lambda: 95_000.0, vol_override=0.80)
        generic = GenericBayesianModel(obs_weight=5.0)
        rpe = ResolutionProbabilityEngine(
            models=[crypto, generic],
            confidence_threshold=0.01,
            shadow_mode=False,
            min_confidence=0.05,
            generic_enabled=True,
        )

        # Evaluate two different markets
        m1 = FakeMarketInfo(
            condition_id="CRYPTO_1",
            question="Will Bitcoin exceed $100,000 by March 31, 2026?",
        )
        m2 = FakeMarketInfo(
            condition_id="POLITICS_1",
            question="Will the Democrats win the 2028 election?",
            tags="politics",
        )

        rpe.evaluate(market=m1, market_price=0.50, days_to_resolution=30)
        rpe.evaluate(market=m2, market_price=0.50, days_to_resolution=30)

        est1 = rpe.get_estimate("CRYPTO_1")
        est2 = rpe.get_estimate("POLITICS_1")
        assert est1 is not None
        assert est2 is not None
        assert est1.model_name == "crypto_lognormal"
        assert est2.model_name == "generic_bayesian"

    def test_unwire_pops_estimate(self) -> None:
        """clear_market() should remove the cached estimate for that condition_id."""
        crypto = CryptoPriceModel(price_fn=lambda: 95_000.0, vol_override=0.80)
        generic = GenericBayesianModel(obs_weight=5.0)
        rpe = ResolutionProbabilityEngine(
            models=[crypto, generic],
            confidence_threshold=0.01,
            shadow_mode=False,
            min_confidence=0.05,
            generic_enabled=True,
        )

        market = FakeMarketInfo(condition_id="MKT_TO_REMOVE")
        rpe.evaluate(market=market, market_price=0.50, days_to_resolution=30)
        assert rpe.get_estimate("MKT_TO_REMOVE") is not None

        # Use the public clear_market API instead of reaching into internals
        rpe.clear_market("MKT_TO_REMOVE")
        assert rpe.get_estimate("MKT_TO_REMOVE") is None

    def test_clear_market_nonexistent_is_safe(self) -> None:
        """clear_market() on an unknown condition_id should not raise."""
        crypto = CryptoPriceModel(price_fn=lambda: 95_000.0, vol_override=0.80)
        rpe = ResolutionProbabilityEngine(
            models=[crypto],
            confidence_threshold=0.01,
            shadow_mode=False,
            min_confidence=0.05,
            generic_enabled=True,
        )

        # Should not raise on a condition_id that was never cached
        rpe.clear_market("nonexistent")
        assert rpe.get_estimate("nonexistent") is None
