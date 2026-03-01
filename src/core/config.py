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
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv


# ── Deployment phases ──────────────────────────────────────────────────────
class DeploymentEnv(str, Enum):
    """Strict 3-phase deployment pipeline.

    PAPER        — simulated fills, mocked wallet, data recorder forced ON.
    PENNY_LIVE   — real CLOB + real wallet, hardcoded $1 max trade size.
    PRODUCTION   — all guardrails lifted; defers to Kelly sizer.
    """

    PAPER = "PAPER"
    PENNY_LIVE = "PENNY_LIVE"
    PRODUCTION = "PRODUCTION"


# Hardcoded — intentionally NOT configurable via env var.
PENNY_LIVE_MAX_TRADE_USD: float = 1.0

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
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"Invalid float for env var {key!r}: {raw!r}.  "
            f"Expected a numeric value (e.g. '2.5')."
        )


def _env_int(key: str, default: int = 0) -> int:
    raw = os.getenv(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Invalid integer for env var {key!r}: {raw!r}.  "
            f"Expected an integer value (e.g. '10')."
        )


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

    # Edge quality filter: minimum EQS (0-100) for entry.  Uses binary
    # entropy, fee efficiency, tick viability, and signal strength.
    min_edge_score: float = _env_float("MIN_EDGE_SCORE", 40.0)

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
    #
    # Intrinsic detection engine — four intra-Polymarket signals detect
    # toxic flow using data the bot already collects.  A kill fires when
    # ANY two of the four signals trigger simultaneously (2-of-4 rule),
    # reducing false positives from idiosyncratic market noise.
    #
    # Core lifecycle knobs (preserved from v1):
    adverse_sel_enabled: bool = _env_bool("ADVERSE_SEL_ENABLED", True)
    adverse_sel_cooldown_s: float = _env_float("ADVERSE_SEL_COOLDOWN_S", 2.0)
    adverse_sel_poll_ms: int = _env_int("ADVERSE_SEL_POLL_MS", 50)

    # Polygon head-lag threshold (used by PolygonHeadLagChecker in heartbeat.py)
    adverse_sel_polygon_head_lag_ms: int = _env_int("ADVERSE_SEL_POLYGON_HEAD_LAG_MS", 3000)

    # Signal 1 — Cross-market flow coherence
    # When taker-initiated trades dominate 3+ independent markets within
    # a 5-second window, it indicates a platform-wide information event
    # (news drop, API leak).  Idiosyncratic noise cannot produce
    # simultaneous directional flow across unrelated markets.
    adverse_sel_mti_threshold: float = _env_float("ADVERSE_SEL_MTI_THRESHOLD", 0.85)
    adverse_sel_mti_min_markets: int = _env_int("ADVERSE_SEL_MTI_MIN_MARKETS", 3)
    adverse_sel_mti_window_s: float = _env_float("ADVERSE_SEL_MTI_WINDOW_S", 5.0)

    # Signal 2 — Book depth evaporation
    # When informed traders arrive, market makers pull quotes before the
    # price moves.  A 60% depth drop within 2 seconds within 5¢ of mid
    # is the classic signature.  Differs from ghost liquidity (which
    # detects fake depth without corresponding trades) — this detects
    # genuine quote withdrawal by real market makers.
    adverse_sel_depth_drop_pct: float = _env_float("ADVERSE_SEL_DEPTH_DROP_PCT", 0.60)
    adverse_sel_depth_window_s: float = _env_float("ADVERSE_SEL_DEPTH_WINDOW_S", 2.0)
    adverse_sel_depth_near_mid_cents: float = _env_float("ADVERSE_SEL_DEPTH_NEAR_MID_CENTS", 5.0)

    # Signal 3 — Spread blow-out
    # When the bid-ask spread on a positioned market widens to 3× its
    # 5-minute rolling average, market makers are widening defensively
    # in response to perceived information asymmetry.  The 5-minute
    # window normalizes for time-of-day spread variation.
    adverse_sel_spread_blowout_mult: float = _env_float("ADVERSE_SEL_SPREAD_BLOWOUT_MULT", 3.0)
    adverse_sel_spread_avg_window_s: float = _env_float("ADVERSE_SEL_SPREAD_AVG_WINDOW_S", 300.0)

    # Signal 4 — Velocity anomaly on positioned assets
    # A 5× spike in trade arrival rate over a 10-minute baseline on a
    # market where the bot holds a position indicates a burst of
    # informed activity.  Under Poisson arrival, a 5× spike has
    # p < 0.001 unless event-driven.
    adverse_sel_velocity_mult: float = _env_float("ADVERSE_SEL_VELOCITY_MULT", 5.0)
    adverse_sel_velocity_window_s: float = _env_float("ADVERSE_SEL_VELOCITY_WINDOW_S", 600.0)
    # Adaptive multiplier boost for high-frequency markets.
    # Markets with a long-term baseline above this rate (trades/min)
    # use velocity_mult * high_freq_mult_boost instead of velocity_mult,
    # preventing false positives on naturally active markets.
    adverse_sel_high_freq_baseline: float = _env_float("ADVERSE_SEL_HIGH_FREQ_BASELINE", 20.0)
    adverse_sel_high_freq_mult_boost: float = _env_float("ADVERSE_SEL_HIGH_FREQ_MULT_BOOST", 1.5)

    # Kill outcome retrospective analysis.
    # After each fast-kill, wait outcome_delay_s then re-read mid-prices
    # and classify the kill as TP (price moved adversely ≥ threshold) or
    # FP (it didn’t).  Results are logged and persisted to JSONL.
    adverse_sel_outcome_delay_s: float = _env_float("ADVERSE_SEL_OUTCOME_DELAY_S", 60.0)
    adverse_sel_tp_threshold_cents: float = _env_float("ADVERSE_SEL_TP_THRESHOLD_CENTS", 3.0)
    # ── Pillar 6: Dynamic Fee-Curve Integration ────────────────────────────
    fee_cache_ttl_s: int = _env_int("FEE_CACHE_TTL_S", 300)
    fee_default_bps: int = _env_int("FEE_DEFAULT_BPS", 200)
    desired_margin_cents: float = _env_float("DESIRED_MARGIN_CENTS", 1.0)

    # ── Pillar 7: Hybrid-Aggressive Chaser Escalation ──────────────────────
    chaser_max_rejections: int = _env_int("CHASER_MAX_REJECTIONS", 3)
    chaser_escalation_ticks: int = _env_int("CHASER_ESCALATION_TICKS", 1)

    # ── Pillar 8: Clock-Skew & Stale Book Safety ──────────────────────────
    heartbeat_check_ms: int = _env_int("HEARTBEAT_CHECK_MS", 500)
    heartbeat_stale_ms: int = _env_int("HEARTBEAT_STALE_MS", 2500)
    heartbeat_stale_count: int = _env_int("HEARTBEAT_STALE_COUNT", 2)
    ws_silence_timeout_s: float = _env_float("WS_SILENCE_TIMEOUT_S", 5.0)

    # ── Pillar 9: Toxic Flow Avoidance (2026 Dynamic Fee Regime) ───────────
    # MTI — Maker/Taker Imbalance penalty
    mti_threshold: float = _env_float("MTI_THRESHOLD", 0.80)
    mti_penalty_points: float = _env_float("MTI_PENALTY_POINTS", 40.0)

    # Fee-adaptive stop-loss
    fee_max_pct: float = _env_float("FEE_MAX_PCT", 1.56)  # peak fee %
    fee_enabled_categories: str = _env("FEE_ENABLED_CATEGORIES", "crypto,sports")

    # ── Pillar 10: Order Status Polling ─────────────────────────────────
    order_status_poll_s: float = _env_float("ORDER_STATUS_POLL_S", 2.0)
    order_status_max_retries: int = _env_int("ORDER_STATUS_MAX_RETRIES", 3)
    # ── Pillar 11: Active Stop-Loss Engine ────────────────────────────────
    stop_loss_poll_ms: int = _env_int("STOP_LOSS_POLL_MS", 500)
    trailing_stop_offset_cents: float = _env_float("TRAILING_STOP_OFFSET_CENTS", 0.0)

    # ── Pillar 12: Multi-Signal Framework ─────────────────────────────────
    imbalance_threshold: float = _env_float("IMBALANCE_THRESHOLD", 2.0)
    spread_compression_pct: float = _env_float("SPREAD_COMPRESSION_PCT", 0.5)
    min_composite_signal_score: float = _env_float("MIN_COMPOSITE_SIGNAL_SCORE", 0.3)

    # ── Pillar 13: Kelly Sizing ───────────────────────────────────────────
    kelly_fraction: float = _env_float("KELLY_FRACTION", 0.25)
    kelly_max_pct: float = _env_float("KELLY_MAX_PCT", 10.0)
    kelly_p_cap: float = _env_float("KELLY_P_CAP", 0.85)                      # max win probability estimate
    kelly_default_uncertainty: float = _env_float("KELLY_DEFAULT_UNCERTAINTY", 0.5)  # fallback when signal metadata missing

    # ── Uncertainty penalty weights for edge discounting ──────────────────
    uncertainty_spread_weight: float = _env_float("UNCERTAINTY_SPREAD_WEIGHT", 0.6)
    uncertainty_conf_weight: float = _env_float("UNCERTAINTY_CONF_WEIGHT", 0.4)

    # Ghost Liquidity Circuit Breaker
    ghost_depth_drop_threshold: float = _env_float("GHOST_DEPTH_DROP_THRESHOLD", 0.50)
    ghost_window_s: float = _env_float("GHOST_WINDOW_S", 2.0)
    ghost_recovery_s: float = _env_float("GHOST_RECOVERY_S", 30.0)
    ghost_check_interval_ms: int = _env_int("GHOST_CHECK_INTERVAL_MS", 500)

    # Whale cluster detection
    whale_cluster_lookback_blocks: int = _env_int("WHALE_CLUSTER_LOOKBACK_BLOCKS", 10000)
    whale_cluster_refresh_hours: float = _env_float("WHALE_CLUSTER_REFRESH_HOURS", 6.0)

    # ── Pillar 11: Real-Time L2 Order Book ─────────────────────────────────
    l2_enabled: bool = _env_bool("L2_ENABLED", True)
    l2_max_levels: int = _env_int("L2_MAX_LEVELS", 50)
    l2_snapshot_timeout_s: float = _env_float("L2_SNAPSHOT_TIMEOUT_S", 10.0)
    l2_delta_buffer_size: int = _env_int("L2_DELTA_BUFFER_SIZE", 500)
    l2_seq_gap_max_retries: int = _env_int("L2_SEQ_GAP_MAX_RETRIES", 3)
    l2_spread_score_top_n: int = _env_int("L2_SPREAD_SCORE_TOP_N", 3)

    # ── Pillar 14: Resolution Probability Engine (RPE) ─────────────────────
    rpe_shadow_mode: bool = _env_bool("RPE_SHADOW_MODE", True)
    rpe_confidence_threshold: float = _env_float("RPE_CONFIDENCE_THRESHOLD", 0.08)
    rpe_weight: float = _env_float("RPE_WEIGHT", 0.5)
    rpe_crypto_vol_default: float = _env_float("RPE_CRYPTO_VOL_DEFAULT", 0.80)  # 80% annualised
    rpe_bayesian_obs_weight: float = _env_float("RPE_BAYESIAN_OBS_WEIGHT", 5.0)
    rpe_min_confidence: float = _env_float("RPE_MIN_CONFIDENCE", 0.15)
    rpe_generic_enabled: bool = _env_bool("RPE_GENERIC_ENABLED", False)
    rpe_crypto_retrigger_cents: float = _env_float("RPE_CRYPTO_RETRIGGER_CENTS", 500.0)

    # ── Pillar 15: Portfolio Correlation Engine (PCE) ───────────────────────
    pce_shadow_mode: bool = _env_bool("PCE_SHADOW_MODE", True)
    pce_max_portfolio_var_usd: float = _env_float("PCE_MAX_PORTFOLIO_VAR_USD", 50.0)
    pce_correlation_haircut_threshold: float = _env_float("PCE_CORRELATION_HAIRCUT_THRESHOLD", 0.50)
    pce_structural_same_event_corr: float = _env_float("PCE_STRUCTURAL_SAME_EVENT_CORR", 0.85)
    pce_structural_same_tag_corr: float = _env_float("PCE_STRUCTURAL_SAME_TAG_CORR", 0.30)
    pce_structural_baseline_corr: float = _env_float("PCE_STRUCTURAL_BASELINE_CORR", 0.05)
    pce_structural_prior_weight: int = _env_int("PCE_STRUCTURAL_PRIOR_WEIGHT", 10)
    pce_min_overlap_bars: int = _env_int("PCE_MIN_OVERLAP_BARS", 30)
    pce_staleness_halflife_hours: float = _env_float("PCE_STALENESS_HALFLIFE_HOURS", 24.0)
    pce_var_confidence_z: float = _env_float("PCE_VAR_CONFIDENCE_Z", 1.645)
    pce_correlation_refresh_minutes: float = _env_float("PCE_CORRELATION_REFRESH_MINUTES", 30.0)
    pce_holding_period_minutes: int = _env_int("PCE_HOLDING_PERIOD_MINUTES", 120)
    pce_var_soft_cap: bool = _env_bool("PCE_VAR_SOFT_CAP", True)
    pce_var_bisect_iterations: int = _env_int("PCE_VAR_BISECT_ITERATIONS", 10)
    pce_near_extreme_threshold: float = _env_float("PCE_NEAR_EXTREME_THRESHOLD", 0.85)
    pce_near_extreme_overlap_multiplier: int = _env_int("PCE_NEAR_EXTREME_OVERLAP_MULTIPLIER", 3)
    pce_backtest_enabled: bool = _env_bool("PCE_BACKTEST_ENABLED", False)


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

    # ── Deployment phase (replaces simple paper_mode boolean) ──────────
    deployment_env: DeploymentEnv = field(
        default_factory=lambda: DeploymentEnv(_env("DEPLOYMENT_ENV", "PAPER"))
    )
    # Derived — kept for backward compatibility with executor / poller
    paper_mode: bool = field(init=False, default=True)

    # Data recording
    record_data: bool = field(default_factory=lambda: _env_bool("RECORD_DATA", False))
    record_data_dir: str = field(default_factory=lambda: _env("RECORD_DATA_DIR", "data"))

    def __post_init__(self) -> None:
        # Derive paper_mode from the canonical deployment_env.
        # frozen=True prevents normal assignment; use object.__setattr__.
        object.__setattr__(
            self, "paper_mode", self.deployment_env == DeploymentEnv.PAPER
        )
        # Legacy PAPER_MODE env-var override: if someone explicitly sets
        # PAPER_MODE=true *without* setting DEPLOYMENT_ENV, honour it.
        raw_dep = os.getenv("DEPLOYMENT_ENV", "")
        if not raw_dep and _env_bool("PAPER_MODE", True):
            object.__setattr__(self, "deployment_env", DeploymentEnv.PAPER)
            object.__setattr__(self, "paper_mode", True)

    # Strategy
    strategy: StrategyParams = field(default_factory=StrategyParams)

    # Polymarket CLOB endpoints
    clob_http_url: str = field(
        default_factory=lambda: _env("CLOB_HTTP_URL", "https://clob.polymarket.com")
    )
    clob_ws_url: str = field(
        default_factory=lambda: _env(
            "CLOB_WS_URL",
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        )
    )
    clob_l2_ws_url: str = field(
        default_factory=lambda: _env(
            "CLOB_L2_WS_URL",
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        )
    )
    clob_book_url: str = field(
        default_factory=lambda: _env(
            "CLOB_BOOK_URL",
            "https://clob.polymarket.com/book",
        )
    )

    # Whale monitoring
    whale_poll_interval_seconds: int = field(
        default_factory=lambda: _env_int("WHALE_POLL_INTERVAL_S", 30)
    )
    whale_lookback_seconds: int = field(
        default_factory=lambda: _env_int("WHALE_LOOKBACK_S", 600)
    )  # 10 min window for confluence

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

    def validate_credentials(self) -> list[str]:
        """Return a list of missing credential fields (empty = all OK).

        Checks credentials required for any non-PAPER deployment phase
        (PENNY_LIVE and PRODUCTION both need real CLOB credentials).
        """
        errors: list[str] = []
        if self.deployment_env == DeploymentEnv.PAPER:
            return errors  # no real credentials needed
        required_live = [
            ("polymarket_api_key", "POLYMARKET_API_KEY"),
            ("polymarket_secret", "POLYMARKET_SECRET"),
            ("polymarket_passphrase", "POLYMARKET_PASSPHRASE"),
            ("eoa_private_key", "EOA_PRIVATE_KEY"),
        ]
        for attr, env_name in required_live:
            if not getattr(self, attr, ""):
                errors.append(f"{env_name} is required for {self.deployment_env.value} mode")
        return errors


# Singleton
settings = Settings()
