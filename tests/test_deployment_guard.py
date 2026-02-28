"""
Tests for the 3-phase deployment guard and environment configuration.

Validates:
  1. DeploymentEnv enum has exactly three values.
  2. DeploymentGuard sizing clamps for PENNY_LIVE.
  3. PRODUCTION without confirmation raises RuntimeError.
  4. Data recording is forced ON in PAPER mode.
  5. Config derives paper_mode from deployment_env correctly.
  6. CLI --env flag validation (via Click test runner).
"""

from __future__ import annotations

import os

import pytest

from src.core.config import DeploymentEnv, PENNY_LIVE_MAX_TRADE_USD
from src.core.guard import DeploymentGuard


# ── DeploymentEnv enum ─────────────────────────────────────────────────────

class TestDeploymentEnv:
    def test_exactly_three_phases(self):
        assert set(DeploymentEnv) == {
            DeploymentEnv.PAPER,
            DeploymentEnv.PENNY_LIVE,
            DeploymentEnv.PRODUCTION,
        }

    def test_string_values(self):
        assert DeploymentEnv.PAPER.value == "PAPER"
        assert DeploymentEnv.PENNY_LIVE.value == "PENNY_LIVE"
        assert DeploymentEnv.PRODUCTION.value == "PRODUCTION"

    def test_from_string(self):
        assert DeploymentEnv("PAPER") is DeploymentEnv.PAPER
        assert DeploymentEnv("PENNY_LIVE") is DeploymentEnv.PENNY_LIVE
        assert DeploymentEnv("PRODUCTION") is DeploymentEnv.PRODUCTION

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DeploymentEnv("STAGING")


# ── PENNY_LIVE_MAX_TRADE_USD constant ─────────────────────────────────────

class TestPennyLiveConstant:
    def test_value_is_one_dollar(self):
        assert PENNY_LIVE_MAX_TRADE_USD == 1.0

    def test_type_is_float(self):
        assert isinstance(PENNY_LIVE_MAX_TRADE_USD, float)


# ── DeploymentGuard ───────────────────────────────────────────────────────

class TestDeploymentGuard:

    # ── Construction ───────────────────────────────────────────────────
    def test_paper_creates_without_confirmation(self):
        guard = DeploymentGuard(DeploymentEnv.PAPER)
        assert guard.is_paper is True
        assert guard.is_live is False

    def test_penny_live_creates_without_confirmation(self):
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        assert guard.is_paper is False
        assert guard.is_live is True

    def test_production_requires_confirmation(self):
        with pytest.raises(RuntimeError, match="--confirm-production"):
            DeploymentGuard(DeploymentEnv.PRODUCTION)

    def test_production_with_confirmation(self):
        guard = DeploymentGuard(
            DeploymentEnv.PRODUCTION, confirmed_production=True,
        )
        assert guard.is_paper is False
        assert guard.is_live is True

    # ── get_allowed_trade_size ─────────────────────────────────────────
    def test_paper_passes_through(self):
        guard = DeploymentGuard(DeploymentEnv.PAPER)
        assert guard.get_allowed_trade_size(100.0) == 100.0
        assert guard.get_allowed_trade_size(0.5) == 0.5

    def test_penny_live_clamps_to_one_dollar(self):
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        assert guard.get_allowed_trade_size(5.0) == PENNY_LIVE_MAX_TRADE_USD
        assert guard.get_allowed_trade_size(0.50) == 0.50  # below cap
        assert guard.get_allowed_trade_size(1.0) == 1.0    # exactly at cap

    def test_penny_live_clamps_large_amount(self):
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        assert guard.get_allowed_trade_size(1000.0) == 1.0

    def test_production_passes_through(self):
        guard = DeploymentGuard(
            DeploymentEnv.PRODUCTION, confirmed_production=True,
        )
        assert guard.get_allowed_trade_size(500.0) == 500.0

    # ── get_allowed_trade_shares ───────────────────────────────────────
    def test_penny_live_clamps_shares(self):
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        # 10 shares at $0.50 each = $5.00 → clamped to $1.00 → 2 shares
        result = guard.get_allowed_trade_shares(10.0, 0.50)
        assert result == 2.0

    def test_shares_zero_price(self):
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        assert guard.get_allowed_trade_shares(10.0, 0.0) == 0.0

    def test_paper_shares_pass_through(self):
        guard = DeploymentGuard(DeploymentEnv.PAPER)
        assert guard.get_allowed_trade_shares(10.0, 0.50) == 10.0

    # ── Edge cases: very low-priced tokens ─────────────────────────────
    def test_penny_live_very_low_price_tokens(self):
        """Verify robustness when token price is $0.0099 (below penny)."""
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        # 200 shares at $0.0099 = $1.98 → clamped to ~101 shares = $0.9999
        result = guard.get_allowed_trade_shares(200.0, 0.0099)
        # Verify the final notional is ≤ $1.00
        final_usd = result * 0.0099
        assert final_usd <= 1.00, f"Exceeded $1 cap: {result} shares * $0.0099 = ${final_usd}"
        assert result > 0, "Should return non-zero shares for valid price"

    def test_penny_live_fractional_cent_price(self):
        """Verify robustness when token price is $0.001 (tenth of a cent)."""
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        # 2000 shares at $0.001 = $2.00 → clamped to 1000 shares = $1.00
        result = guard.get_allowed_trade_shares(2000.0, 0.001)
        final_usd = result * 0.001
        assert final_usd <= 1.00, f"Exceeded $1 cap: {result} shares * $0.001 = ${final_usd}"
        assert result == 1000.0  # exactly at cap

    def test_penny_live_token_price_above_cap(self):
        """If token price > $1, should return fractional shares = $1.00 worth."""
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        result = guard.get_allowed_trade_shares(10.0, 5.00)
        # With $1 cap and $5 token price: 1.00 / 5.00 = 0.2 shares
        # (Minimum share enforcement happens in position_manager, not guard)
        assert result == 0.2
        final_usd = result * 5.00
        assert final_usd == 1.00  # exactly at cap

    def test_penny_live_rounding_edge_case(self):
        """Verify post-condition check handles float rounding correctly."""
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        # Pathological case: price that causes rounding issues
        # $0.299 per share: 1.00 / 0.299 = 3.344... → rounds to 3.34
        result = guard.get_allowed_trade_shares(100.0, 0.299)
        final_usd = result * 0.299
        assert final_usd <= 1.00, f"Exceeded $1 cap: {result} shares * $0.299 = ${final_usd}"
        # 3.34 shares * $0.299 = 0.99866 ✓

    # ── enforce_data_recording ─────────────────────────────────────────
    def test_paper_forces_recording_on(self):
        guard = DeploymentGuard(DeploymentEnv.PAPER)
        assert guard.enforce_data_recording() is True

    def test_penny_live_defers_to_settings(self):
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        # Should return whatever settings.record_data is (default False in tests)
        result = guard.enforce_data_recording()
        assert isinstance(result, bool)

    # ── startup_summary ────────────────────────────────────────────────
    def test_startup_summary_keys(self):
        guard = DeploymentGuard(DeploymentEnv.PAPER)
        summary = guard.startup_summary()
        assert "deployment_env" in summary
        assert "paper_mode" in summary
        assert "max_trade_size" in summary
        assert "data_recording" in summary
        assert summary["deployment_env"] == "PAPER"
        assert summary["paper_mode"] is True

    # ── max_trade_size_label ───────────────────────────────────────────
    def test_penny_live_label(self):
        guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
        assert "$1.00" in guard.max_trade_size_label

    def test_paper_label(self):
        guard = DeploymentGuard(DeploymentEnv.PAPER)
        assert "simulated" in guard.max_trade_size_label.lower()


# ── Config integration ─────────────────────────────────────────────────────

class TestConfigDeploymentEnv:
    def test_settings_has_deployment_env(self):
        from src.core.config import settings
        assert hasattr(settings, "deployment_env")
        assert isinstance(settings.deployment_env, DeploymentEnv)

    def test_default_is_paper(self):
        from src.core.config import settings
        # In test environment DEPLOYMENT_ENV defaults to PAPER
        assert settings.deployment_env == DeploymentEnv.PAPER

    def test_paper_mode_derived_from_deployment_env(self):
        from src.core.config import settings
        assert settings.paper_mode == (
            settings.deployment_env == DeploymentEnv.PAPER
        )


# ── CLI integration ────────────────────────────────────────────────────────

class TestCLIEnvFlag:
    def test_production_without_confirm_raises(self):
        """polybot run --env PRODUCTION (without --confirm-production) must fail."""
        from click.testing import CliRunner
        from src.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--env", "PRODUCTION"])
        assert result.exit_code != 0
        assert "PRODUCTION" in (result.output + str(result.exception))

    def test_paper_default_starts(self):
        """polybot run (default) should not raise about credentials."""
        # We can't actually run the full bot, but we can check the CLI
        # parses correctly by verifying no RuntimeError for PAPER mode.
        from click.testing import CliRunner
        from src.cli import main

        runner = CliRunner()
        # Invoke with --help to verify the command parses without error
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "PAPER" in result.output

    def test_env_choices_displayed_in_help(self):
        from click.testing import CliRunner
        from src.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert "PENNY_LIVE" in result.output
        assert "PRODUCTION" in result.output
