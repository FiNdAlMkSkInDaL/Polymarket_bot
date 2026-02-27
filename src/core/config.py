"""
Centralised configuration loaded from environment variables.

Usage:
    from src.core.config import settings
    print(settings.polygon_rpc_url)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
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


def _env_int(key: str, default: int = 0) -> int:
    raw = os.getenv(key, "")
    return int(raw) if raw else default


def _env_bool(key: str, default: bool = True) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# ── Strategy defaults ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class StrategyParams:
    """Tunable knobs for the mean-reversion strategy.

    Every field can be overridden via its corresponding env-var
    (e.g.  ZSCORE_THRESHOLD=2.5  in .env).
    """

    # Panic spike detector
    zscore_threshold: float = _env_float("ZSCORE_THRESHOLD", 2.0)
    volume_ratio_threshold: float = _env_float("VOLUME_RATIO_THRESHOLD", 3.0)
    lookback_minutes: int = _env_int("LOOKBACK_MINUTES", 60)

    # Take-profit
    alpha_default: float = _env_float("ALPHA_DEFAULT", 0.50)
    alpha_min: float = _env_float("ALPHA_MIN", 0.30)
    alpha_max: float = _env_float("ALPHA_MAX", 0.70)
    min_spread_cents: float = _env_float("MIN_SPREAD_CENTS", 4.0)

    # Risk
    max_trade_size_usd: float = _env_float("MAX_TRADE_SIZE_USD", 5.0)
    max_wallet_risk_pct: float = _env_float("MAX_WALLET_RISK_PCT", 20.0)

    # Time limits
    entry_timeout_seconds: int = _env_int("ENTRY_TIMEOUT_SECONDS", 300)
    exit_timeout_seconds: int = _env_int("EXIT_TIMEOUT_SECONDS", 1800)

    # Market selection filters
    min_daily_volume_usd: float = _env_float("MIN_DAILY_VOLUME_USD", 50_000.0)
    min_days_to_resolution: int = _env_int("MIN_DAYS_TO_RESOLUTION", 7)
    min_liquidity_usd: float = _env_float("MIN_LIQUIDITY_USD", 0.0)

    # Discovery behaviour
    discovery_tags: str = _env("DISCOVERY_TAGS", "")  # empty = all categories
    reject_neg_risk: bool = _env_bool("REJECT_NEG_RISK", True)
    one_market_per_event: bool = _env_bool("ONE_MARKET_PER_EVENT", True)
    market_refresh_minutes: int = _env_int("MARKET_REFRESH_MINUTES", 30)

    # Market scoring
    min_market_score: float = _env_float("MIN_MARKET_SCORE", 40.0)
    observation_period_minutes: int = _env_int("OBSERVATION_PERIOD_MINUTES", 5)
    demotion_cycles_before_evict: int = _env_int("DEMOTION_CYCLES_BEFORE_EVICT", 3)
    api_rate_limit_per_sec: int = _env_int("API_RATE_LIMIT_PER_SEC", 5)

    # Risk controls
    max_open_positions: int = _env_int("MAX_OPEN_POSITIONS", 5)
    max_positions_per_market: int = _env_int("MAX_POSITIONS_PER_MARKET", 1)
    max_positions_per_event: int = _env_int("MAX_POSITIONS_PER_EVENT", 2)
    daily_loss_limit_usd: float = _env_float("DAILY_LOSS_LIMIT_USD", 25.0)
    max_drawdown_cents: float = _env_float("MAX_DRAWDOWN_CENTS", 2500.0)
    stop_loss_cents: float = _env_float("STOP_LOSS_CENTS", 8.0)
    signal_cooldown_minutes: int = _env_int("SIGNAL_COOLDOWN_MINUTES", 15)
    max_total_exposure_pct: float = _env_float("MAX_TOTAL_EXPOSURE_PCT", 60.0)

    # ── Pillar 1: Passive-Aggressive Chasing ───────────────────────────────
    chase_interval_ms: int = _env_int("CHASE_INTERVAL_MS", 250)
    max_chase_depth_cents: float = _env_float("MAX_CHASE_DEPTH_CENTS", 3.0)
    post_only_enabled: bool = _env_bool("POST_ONLY_ENABLED", True)

    # ── Pillar 2: Liquidity-Sensing Sizing ─────────────────────────────────
    max_impact_pct: float = _env_float("MAX_IMPACT_PCT", 15.0)
    impact_depth_cents: float = _env_float("IMPACT_DEPTH_CENTS", 5.0)

    # ── Pillar 3: Adaptive TP Rescaling ────────────────────────────────────
    tp_rescale_interval_s: int = _env_int("TP_RESCALE_INTERVAL_S", 15)
    tp_vol_sensitivity: float = _env_float("TP_VOL_SENSITIVITY", 1.5)
    tp_spread_min_mult: float = _env_float("TP_SPREAD_MIN_MULT", 0.5)
    tp_spread_max_mult: float = _env_float("TP_SPREAD_MAX_MULT", 3.0)

    # ── Pillar 4: Stale-Data Kill-Switch ───────────────────────────────────
    latency_block_ms: int = _env_int("LATENCY_BLOCK_MS", 1500)
    latency_warn_ms: int = _env_int("LATENCY_WARN_MS", 800)
    latency_recovery_count: int = _env_int("LATENCY_RECOVERY_COUNT", 3)

    # ── Pillar 5: Anti-Adverse-Selection ("Fast-Kill") ─────────────────────
    adverse_sel_enabled: bool = _env_bool("ADVERSE_SEL_ENABLED", True)
    adverse_sel_tick_threshold: int = _env_int("ADVERSE_SEL_TICK_THRESHOLD", 5000)
    adverse_sel_book_stale_ms: int = _env_int("ADVERSE_SEL_BOOK_STALE_MS", 30000)
    adverse_sel_cooldown_s: float = _env_float("ADVERSE_SEL_COOLDOWN_S", 2.0)
    adverse_sel_poll_ms: int = _env_int("ADVERSE_SEL_POLL_MS", 50)
    adverse_sel_polygon_head_lag_ms: int = _env_int("ADVERSE_SEL_POLYGON_HEAD_LAG_MS", 3000)
    binance_ws_url: str = _env(
        "BINANCE_WS_URL", "wss://stream.binance.com:9443/ws/btcusdc@trade"
    )

    # ── Pillar 6: Dynamic Fee-Curve Integration ────────────────────────────
    fee_cache_ttl_s: int = _env_int("FEE_CACHE_TTL_S", 300)
    fee_default_bps: int = _env_int("FEE_DEFAULT_BPS", 200)
    desired_margin_cents: float = _env_float("DESIRED_MARGIN_CENTS", 1.0)

    # ── Pillar 7: Hybrid-Aggressive Chaser Escalation ──────────────────────
    chaser_max_rejections: int = _env_int("CHASER_MAX_REJECTIONS", 3)
    chaser_escalation_ticks: int = _env_int("CHASER_ESCALATION_TICKS", 1)

    # ── Pillar 8: Clock-Skew & Stale Book Safety ──────────────────────────
    heartbeat_check_ms: int = _env_int("HEARTBEAT_CHECK_MS", 500)
    heartbeat_stale_ms: int = _env_int("HEARTBEAT_STALE_MS", 1500)
    ws_silence_timeout_s: float = _env_float("WS_SILENCE_TIMEOUT_S", 5.0)

    # ── Pillar 9: Toxic Flow Avoidance (2026 Dynamic Fee Regime) ───────────
    # MTI — Maker/Taker Imbalance penalty
    mti_threshold: float = _env_float("MTI_THRESHOLD", 0.80)
    mti_penalty_points: float = _env_float("MTI_PENALTY_POINTS", 40.0)

    # Fee-adaptive stop-loss
    fee_max_pct: float = _env_float("FEE_MAX_PCT", 1.56)  # peak fee %
    fee_enabled_categories: str = _env("FEE_ENABLED_CATEGORIES", "crypto,sports")

    # Ghost Liquidity Circuit Breaker
    ghost_depth_drop_threshold: float = _env_float("GHOST_DEPTH_DROP_THRESHOLD", 0.50)
    ghost_window_s: float = _env_float("GHOST_WINDOW_S", 2.0)
    ghost_recovery_s: float = _env_float("GHOST_RECOVERY_S", 30.0)
    ghost_check_interval_ms: int = _env_int("GHOST_CHECK_INTERVAL_MS", 500)

    # Whale cluster detection
    whale_cluster_lookback_blocks: int = _env_int("WHALE_CLUSTER_LOOKBACK_BLOCKS", 10000)
    whale_cluster_refresh_hours: float = _env_float("WHALE_CLUSTER_REFRESH_HOURS", 6.0)


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

    # ── Security: mask secrets in repr / str / logs ─────────────────────
    _SENSITIVE_PATTERN: re.Pattern = field(
        default=re.compile(r"(SECRET|KEY|PASSPHRASE|TOKEN)", re.IGNORECASE),
        repr=False,
        compare=False,
    )

    def __repr__(self) -> str:
        parts: list[str] = []
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if self._SENSITIVE_PATTERN.search(f.name):
                masked = (str(value)[:4] + "***") if value else "<empty>"
                parts.append(f"{f.name}={masked!r}")
            else:
                parts.append(f"{f.name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    __str__ = __repr__


# Singleton
settings = Settings()
