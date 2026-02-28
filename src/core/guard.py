"""
Deployment guard — phase-aware middleware that enforces environment-specific
constraints without mutating the frozen configuration dataclasses.

Usage:
    from src.core.guard import DeploymentGuard
    guard = DeploymentGuard(DeploymentEnv.PENNY_LIVE)
    clamped = guard.get_allowed_trade_size(raw_usd)
"""

from __future__ import annotations

from src.core.config import (
    DeploymentEnv,
    PENNY_LIVE_MAX_TRADE_USD,
    settings,
)
from src.core.logger import get_logger

log = get_logger(__name__)


class DeploymentGuard:
    """Authoritative enforcement layer for the 3-phase deployment pipeline.

    Sits between the sizing engine output and the order executor to ensure
    that phase-specific capital constraints are respected regardless of
    what the Kelly sizer or depth-aware sizer recommends.

    Parameters
    ----------
    deployment_env:
        The current deployment phase (PAPER, PENNY_LIVE, or PRODUCTION).
    confirmed_production:
        Must be ``True`` when ``deployment_env`` is PRODUCTION.
        Raises ``RuntimeError`` otherwise.
    """

    def __init__(
        self,
        deployment_env: DeploymentEnv,
        *,
        confirmed_production: bool = False,
    ) -> None:
        self._env = deployment_env

        if self._env == DeploymentEnv.PRODUCTION and not confirmed_production:
            raise RuntimeError(
                "PRODUCTION mode requires --confirm-production flag. "
                "This is not a drill."
            )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def deployment_env(self) -> DeploymentEnv:
        return self._env

    @property
    def is_paper(self) -> bool:
        return self._env == DeploymentEnv.PAPER

    @property
    def is_live(self) -> bool:
        return self._env in (DeploymentEnv.PENNY_LIVE, DeploymentEnv.PRODUCTION)

    @property
    def max_trade_size_label(self) -> str:
        """Human-readable max trade size for the current phase."""
        if self._env == DeploymentEnv.PAPER:
            return "unlimited (simulated)"
        if self._env == DeploymentEnv.PENNY_LIVE:
            return f"${PENNY_LIVE_MAX_TRADE_USD:.2f} USDC"
        return f"${settings.strategy.max_trade_size_usd:.2f} USDC (Kelly-derived)"

    # ── Sizing enforcement ────────────────────────────────────────────────

    def get_allowed_trade_size(self, raw_size_usd: float) -> float:
        """Clamp ``raw_size_usd`` to the phase-appropriate maximum.

        PAPER       → pass through unchanged (simulated capital).
        PENNY_LIVE  → min(raw, $1.00)  — hardcoded, un-overrideable.
        PRODUCTION  → pass through unchanged (Kelly sizer governs).
        """
        if self._env == DeploymentEnv.PENNY_LIVE:
            clamped = min(raw_size_usd, PENNY_LIVE_MAX_TRADE_USD)
            if clamped < raw_size_usd:
                log.warning(
                    "penny_live_size_clamped",
                    original_usd=round(raw_size_usd, 4),
                    clamped_usd=round(clamped, 4),
                    cap=PENNY_LIVE_MAX_TRADE_USD,
                )
            return clamped
        return raw_size_usd

    def get_allowed_trade_shares(
        self,
        raw_shares: float,
        entry_price: float,
    ) -> float:
        """Clamp a *share count* to the phase-appropriate USD maximum.

        Converts shares → USD, clamps, then converts back.

        Robust against floating-point rounding errors: verifies the final
        notional value and iteratively decreases the share count if needed
        to strictly enforce the USD cap (critical for PENNY_LIVE).
        """
        if entry_price <= 0:
            return 0.0

        raw_usd = raw_shares * entry_price
        allowed_usd = self.get_allowed_trade_size(raw_usd)

        # Convert USD → shares with rounding
        clamped_shares = round(allowed_usd / entry_price, 2)

        # ── Post-condition check: verify notional value ───────────────────
        # Guard against floating-point rounding causing the final USD value
        # to exceed the cap (e.g., for very low-priced tokens like $0.0099).
        if self._env == DeploymentEnv.PENNY_LIVE:
            final_usd = clamped_shares * entry_price
            if final_usd > PENNY_LIVE_MAX_TRADE_USD:
                # Iteratively reduce by 0.01 shares until under cap
                # (max 100 iterations = safety net for $0.01+ tokens)
                for _ in range(100):
                    clamped_shares = round(clamped_shares - 0.01, 2)
                    if clamped_shares <= 0:
                        return 0.0
                    final_usd = clamped_shares * entry_price
                    if final_usd <= PENNY_LIVE_MAX_TRADE_USD:
                        break
                else:
                    # Fallback: force to 1 share if price is sane
                    if entry_price <= PENNY_LIVE_MAX_TRADE_USD:
                        clamped_shares = 1.0
                    else:
                        clamped_shares = 0.0

                log.warning(
                    "penny_live_shares_post_clamped",
                    entry_price=round(entry_price, 6),
                    original_shares=round(raw_shares, 4),
                    final_shares=round(clamped_shares, 4),
                    final_usd=round(clamped_shares * entry_price, 6),
                )

        return clamped_shares

    # ── Data recording policy ─────────────────────────────────────────────

    def enforce_data_recording(self) -> bool:
        """Return ``True`` if the data recorder must be forced ON.

        PAPER → always ON (harvesting data is the primary mission).
        Other → defer to the existing ``settings.record_data`` flag.
        """
        if self._env == DeploymentEnv.PAPER:
            return True
        return settings.record_data

    # ── Wallet balance policy ─────────────────────────────────────────────

    def get_wallet_balance(self) -> float:
        """Return the appropriate wallet balance for the current phase.

        PAPER       → mocked $50.
        PENNY_LIVE  → real balance is used, but capped by sizing guard.
        PRODUCTION  → real balance is used.
        """
        if self._env == DeploymentEnv.PAPER:
            return 50.0
        # For PENNY_LIVE and PRODUCTION, the caller should query the real
        # wallet.  Provide a conservative default if that's not available.
        return settings.strategy.max_trade_size_usd * 5

    # ── Display helpers ───────────────────────────────────────────────────

    def startup_summary(self) -> dict:
        """Structured summary for startup logging."""
        return {
            "deployment_env": self._env.value,
            "paper_mode": self.is_paper,
            "max_trade_size": self.max_trade_size_label,
            "data_recording": self.enforce_data_recording(),
        }
