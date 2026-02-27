"""
Centralised configuration loaded from environment variables.

Usage:
    from src.core.config import settings
    print(settings.polygon_rpc_url)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate .env — prefer tmpfs (VPS decrypted secrets), fall back to project root
# ---------------------------------------------------------------------------
_TMPFS_ENV = Path("/dev/shm/secrets/.env")
_LOCAL_ENV = Path(__file__).resolve().parents[2] / ".env"

if _TMPFS_ENV.exists():
    load_dotenv(_TMPFS_ENV)
else:
    load_dotenv(_LOCAL_ENV)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float = 0.0) -> float:
    raw = os.getenv(key, "")
    return float(raw) if raw else default


def _env_bool(key: str, default: bool = True) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# ── Strategy defaults ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class StrategyParams:
    """Tunable knobs for the mean-reversion strategy."""

    # Panic spike detector
    zscore_threshold: float = 2.0
    volume_ratio_threshold: float = 3.0
    lookback_minutes: int = 60

    # Take-profit
    alpha_default: float = 0.50
    alpha_min: float = 0.30
    alpha_max: float = 0.70
    min_spread_cents: float = 4.0  # minimum profit target in cents

    # Risk
    max_trade_size_usd: float = _env_float("MAX_TRADE_SIZE_USD", 5.0)
    max_wallet_risk_pct: float = _env_float("MAX_WALLET_RISK_PCT", 20.0)

    # Time limits
    entry_timeout_seconds: int = 300   # 5 min to fill entry
    exit_timeout_seconds: int = 1800   # 30 min to fill exit

    # Market selection filters
    min_daily_volume_usd: float = 50_000.0
    min_days_to_resolution: int = 7


@dataclass(frozen=True)
class Settings:
    """Application-wide settings derived from the environment."""

    # Polymarket CLOB
    polymarket_api_key: str = field(default_factory=lambda: _env("POLYMARKET_API_KEY"))
    polymarket_secret: str = field(default_factory=lambda: _env("POLYMARKET_SECRET"))
    polymarket_passphrase: str = field(default_factory=lambda: _env("POLYMARKET_PASSPHRASE"))

    # EOA
    eoa_private_key: str = field(default_factory=lambda: _env("EOA_PRIVATE_KEY"))

    # Polygon RPC
    polygon_rpc_url: str = field(default_factory=lambda: _env("POLYGON_RPC_URL"))

    # Polygonscan
    polygonscan_api_key: str = field(default_factory=lambda: _env("POLYGONSCAN_API_KEY"))

    # Telegram
    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    telegram_chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))

    # Mode
    paper_mode: bool = field(default_factory=lambda: _env_bool("PAPER_MODE", True))

    # Strategy
    strategy: StrategyParams = field(default_factory=StrategyParams)

    # Polymarket CLOB endpoints
    clob_http_url: str = "https://clob.polymarket.com"
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Whale monitoring
    whale_poll_interval_seconds: int = 30
    whale_lookback_seconds: int = 600  # 10 min window for confluence

    # Logging
    log_dir: str = field(default_factory=lambda: _env("LOG_DIR", "logs"))


# Singleton
settings = Settings()
