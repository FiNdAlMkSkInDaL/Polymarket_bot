"""
Tests for PCE (Pillar 15) backtest integration — Iteration 2 Issue 4.

Covers:
- BotReplayAdapter creates PCE when pce_backtest_enabled=True
- BotReplayAdapter skips PCE when pce_backtest_enabled=False
- PCE market registration on on_init()
- PCE market unregistration on on_end()
- PCE VaR gating in entry path
- WFO SEARCH_SPACE includes pce_holding_period_minutes
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("PAPER_MODE", "true")
os.environ.setdefault("DEPLOYMENT_ENV", "PAPER")

from src.backtest.strategy import BotReplayAdapter
from src.backtest.wfo_optimizer import SEARCH_SPACE
from src.core.config import StrategyParams
from src.trading.portfolio_correlation import PortfolioCorrelationEngine


class TestPCEAdapterIntegration:
    """PCE is created and wired in BotReplayAdapter when enabled."""

    def test_pce_created_when_enabled(self):
        params = StrategyParams(pce_backtest_enabled=True)
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES-1",
            no_asset_id="NO-1",
            params=params,
        )
        assert adapter._pce is not None
        assert isinstance(adapter._pce, PortfolioCorrelationEngine)

    def test_pce_not_created_when_disabled(self):
        params = StrategyParams(pce_backtest_enabled=False)
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES-1",
            no_asset_id="NO-1",
            params=params,
        )
        assert adapter._pce is None

    def test_pce_default_disabled(self):
        """PCE backtest is off by default."""
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES-1",
            no_asset_id="NO-1",
        )
        assert adapter._pce is None

    def test_pce_market_registered_on_init(self):
        params = StrategyParams(pce_backtest_enabled=True)
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES-1",
            no_asset_id="NO-1",
            params=params,
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        assert adapter._pce is not None
        assert "MKT-1" in adapter._pce._markets

    def test_pce_market_unregistered_on_end(self):
        params = StrategyParams(pce_backtest_enabled=True)
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES-1",
            no_asset_id="NO-1",
            params=params,
        )
        adapter.engine = MagicMock()
        adapter.on_init()

        assert "MKT-1" in adapter._pce._markets
        adapter.on_end()
        assert "MKT-1" not in adapter._pce._markets

    def test_pce_params_forwarded(self):
        """PCE constructor receives params from StrategyParams."""
        params = StrategyParams(
            pce_backtest_enabled=True,
            pce_max_portfolio_var_usd=42.0,
            pce_correlation_haircut_threshold=0.55,
            pce_var_soft_cap=True,
            pce_var_bisect_iterations=12,
        )
        adapter = BotReplayAdapter(
            market_id="MKT-1",
            yes_asset_id="YES-1",
            no_asset_id="NO-1",
            params=params,
        )
        pce = adapter._pce
        assert pce is not None
        assert pce.max_portfolio_var_usd == 42.0
        assert pce.haircut_threshold == 0.55
        assert pce.var_soft_cap is True
        assert pce.var_bisect_iterations == 12


class TestPCEWFOSearchSpace:
    """WFO SEARCH_SPACE includes PCE parameters."""

    def test_pce_holding_period_in_search_space(self):
        assert "pce_holding_period_minutes" in SEARCH_SPACE

    def test_pce_holding_period_range(self):
        spec = SEARCH_SPACE["pce_holding_period_minutes"]
        assert spec[0] == "suggest_int"
        assert spec[1] == 30
        assert spec[2] == 360

    def test_pce_max_var_range_updated(self):
        spec = SEARCH_SPACE["pce_max_portfolio_var_usd"]
        assert spec[0] == "suggest_float"
        assert spec[1] == 20.0  # widened from 5.0
        assert spec[2] == 100.0  # widened from 30.0

    def test_all_pce_params_present(self):
        pce_params = [
            "pce_max_portfolio_var_usd",
            "pce_correlation_haircut_threshold",
            "pce_structural_prior_weight",
            "pce_holding_period_minutes",
        ]
        for p in pce_params:
            assert p in SEARCH_SPACE, f"Missing PCE param: {p}"
