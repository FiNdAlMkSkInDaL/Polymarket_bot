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
PENNY_LIVE_MAX_TRADE_USD: float = 5.0

# Polymarket exchange minimums — orders below these are rejected.
EXCHANGE_MIN_USD: float = 1.0
EXCHANGE_MIN_SHARES: float = 5.0

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
    """Tunable knobs for the live strategy stack.

    Every field can be overridden via its corresponding env-var
    when a runtime override is required.
    """

    # Tradeable price band — markets outside this range are too close
    # to resolution for mean-reversion alpha to exist.
    min_tradeable_price: float = _env_float("MIN_TRADEABLE_PRICE", 0.05)
    max_tradeable_price: float = _env_float("MAX_TRADEABLE_PRICE", 0.95)

    # Whale monitor adaptive polling threshold.
    whale_zscore_threshold: float = _env_float("WHALE_ZSCORE_THRESHOLD", 0.20)

    # Take-profit
    alpha_default: float = _env_float("ALPHA_DEFAULT", 0.50)
    alpha_min: float = _env_float("ALPHA_MIN", 0.40)
    alpha_max: float = _env_float("ALPHA_MAX", 0.55)
    min_spread_cents: float = _env_float("MIN_SPREAD_CENTS", 4.0)

    # Edge quality filter: minimum EQS (0-100) for entry.  Uses binary
    # entropy, fee efficiency, tick viability, and signal strength.
    # Set to 40 based on 3-day tick data analysis (March 2026): marginal
    # signals at EQS 40-50 are +EV (avg PnL 56.82c vs 57.43c at 50+);
    # the 3.2% additional entries justify the 1% PnL dilution.
    # The geometric-mean formula already hard-rejects trades where any
    # factor is zero.
    min_edge_score: float = _env_float("MIN_EDGE_SCORE", 40.0)

    # ── V1: Maker routing ─────────────────────────────────────────────────
    # When True, entries priced as maker (best_ask - 1¢) use 0-fee EQS
    # scoring and a relaxed viability gate (no slippage/fee drag).
    maker_routing_enabled: bool = _env_bool("MAKER_ROUTING_ENABLED", True)
    # Multiplier on min_edge_score when executing as maker (0.85 = 15% lower).
    maker_eqs_discount: float = _env_float("MAKER_EQS_DISCOUNT", 0.85)

    # ── V2: Multi-factor confluence routing ───────────────────────────────
    # Dynamic EQS threshold discount when multiple independent signals
    # confirm simultaneously.  Requires ≥ confluence_min_factors active.
    confluence_eqs_floor: float = _env_float("CONFLUENCE_EQS_FLOOR", 35.0)
    confluence_min_factors: int = _env_int("CONFLUENCE_MIN_FACTORS", 2)
    # Per-factor EQS threshold discounts (points subtracted).
    # Whale: 5→4 to tighten Sobol S1 variance (dominant first-order index).
    # L2: 3→0 — reclassified as hard gate; failed binomial H₀: w_on ≤ w_off.
    # Regime: 3→0 — failed binomial test; already required by drift Gate 1.
    confluence_whale_discount: float = _env_float("CONFLUENCE_WHALE_DISCOUNT", 4.0)
    confluence_spread_discount: float = _env_float("CONFLUENCE_SPREAD_DISCOUNT", 4.0)
    confluence_l2_discount: float = _env_float("CONFLUENCE_L2_DISCOUNT", 0.0)
    confluence_regime_discount: float = _env_float("CONFLUENCE_REGIME_DISCOUNT", 0.0)

    # ── V4: Probe sizing ──────────────────────────────────────────────────
    # Allow sub-threshold entries (EQS in [probe_eqs_floor, min_edge_score))
    # at micro-size for market exploration and scale-in upon confirmation.
    probe_sizing_enabled: bool = _env_bool("PROBE_SIZING_ENABLED", True)
    probe_eqs_floor: float = _env_float("PROBE_EQS_FLOOR", 35.0)
    probe_kelly_fraction: float = _env_float("PROBE_KELLY_FRACTION", 0.05)
    probe_max_usd: float = _env_float("PROBE_MAX_USD", 2.0)

    # ── V1-V4 Risk Guard: combined maker+confluence floor (Flaw 1) ────────
    # When both maker_routing AND all-4-confluence fire simultaneously,
    # the score is inflated (0 fees) while the threshold is compressed.
    # This hard floor prevents the adjusted threshold from falling below
    # this value when maker routing is active.
    confluence_maker_combined_floor: float = _env_float("CONFLUENCE_MAKER_COMBINED_FLOOR", 40.0)

    # ── V3/V2 Drift-regime double-count guard (Flaw 2) ────────────────────
    # DriftSignal gate 1 requires regime_mean_revert; confluence also awards
    # confluence_regime_discount when regime is mean-reverting.  These are
    # NOT independent bits.  When the primary signal is a drift signal,
    # the regime discount is suppressed to avoid double-counting.
    drift_suppress_regime_discount: bool = _env_bool("DRIFT_SUPPRESS_REGIME_DISCOUNT", True)

    # ── V4 Probe risk controls (Flaw 3) ───────────────────────────────────
    # Probes are sub-threshold micro-entries designed for data gathering, not
    # strong-signal confirmation trades.  Confluence discounts (especially
    # the 5-pt whale discount) should NOT lower the probe threshold: probes
    # are admitted via the confluence-independent probe_eqs_floor, so
    # applying confluence discounts to them double-counts unconfirmed signals.
    probe_suppress_confluence: bool = _env_bool("PROBE_SUPPRESS_CONFLUENCE", True)

    # ── Probe harvesting ──────────────────────────────────────────────────
    # When a probe reaches breakeven (StopLossMonitor fires on_probe_breakeven),
    # scale_probe_to_full() pyramids the position up to full Kelly.
    # harvest_max_progress: skip scale-in if > this fraction of TP achieved.
    # harvest_min_profit_cents: position must still be net-positive.
    # harvest_min_increment_usd: don't bother with tiny scale-ins.
    harvest_max_progress: float = _env_float("HARVEST_MAX_PROGRESS", 0.60)
    harvest_min_profit_cents: float = _env_float("HARVEST_MIN_PROFIT_CENTS", 0.50)
    harvest_min_increment_usd: float = _env_float("HARVEST_MIN_INCREMENT_USD", 0.50)

    # ── Adverse Selection Monitor: dynamic alpha ──────────────────────────
    # The fixed α=0.05 over-suspends in low-vol and under-suspends in
    # high-vol.  Scale α with rolling EWMA volatility:
    #   α_dynamic = α_base × (σ_rolling / σ_ref)^γ, clamped to [α_min, α_max]
    adverse_monitor_alpha_base: float = _env_float("ADVERSE_MONITOR_ALPHA_BASE", 0.05)
    adverse_monitor_vol_ref: float = _env_float("ADVERSE_MONITOR_VOL_REF", 0.01)
    adverse_monitor_alpha_gamma: float = _env_float("ADVERSE_MONITOR_ALPHA_GAMMA", 0.5)
    adverse_monitor_alpha_min: float = _env_float("ADVERSE_MONITOR_ALPHA_MIN", 0.01)
    adverse_monitor_alpha_max: float = _env_float("ADVERSE_MONITOR_ALPHA_MAX", 0.15)

    # Risk
    max_trade_size_usd: float = _env_float("MAX_TRADE_SIZE_USD", 15.0)
    max_wallet_risk_pct: float = _env_float("MAX_WALLET_RISK_PCT", 20.0)

    # Time limits
    entry_timeout_seconds: int = _env_int("ENTRY_TIMEOUT_SECONDS", 300)
    exit_timeout_seconds: int = _env_int("EXIT_TIMEOUT_SECONDS", 1800)

    # Market selection filters
    min_daily_volume_usd: float = _env_float("MIN_DAILY_VOLUME_USD", 1500.0)
    min_days_to_resolution: int = _env_int("MIN_DAYS_TO_RESOLUTION", 3)
    min_liquidity_usd: float = _env_float("MIN_LIQUIDITY_USD", 0.0)

    # Discovery behaviour
    discovery_tags: str = _env("DISCOVERY_TAGS", "")  # empty = all categories
    reject_neg_risk: bool = _env_bool("REJECT_NEG_RISK", False)
    one_market_per_event: bool = _env_bool("ONE_MARKET_PER_EVENT", True)
    market_refresh_minutes: int = _env_int("MARKET_REFRESH_MINUTES", 10)

    # Load shedding: maximum markets with full L2 book reconstruction.
    # Remaining discovered markets use lightweight price/trade-only tracking.
    max_active_l2_markets: int = _env_int("MAX_ACTIVE_L2_MARKETS", 50)

    # Pure maker quoting universe and sizing.
    pure_mm_enabled: bool = _env_bool("PURE_MM_ENABLED", True)
    pure_mm_max_markets: int = _env_int("PURE_MM_MAX_MARKETS", 25)
    pure_mm_quote_size_usd: float = _env_float("PURE_MM_QUOTE_SIZE_USD", 5.0)
    pure_mm_inventory_cap_usd: float = _env_float("PURE_MM_INVENTORY_CAP_USD", 15.0)
    pure_mm_loop_ms: int = _env_int("PURE_MM_LOOP_MS", 250)
    pure_mm_toxic_ofi_ratio: float = _env_float("PURE_MM_TOXIC_OFI_RATIO", 0.80)
    pure_mm_depth_window_s: float = _env_float("PURE_MM_DEPTH_WINDOW_S", 2.0)
    pure_mm_depth_evaporation_pct: float = _env_float("PURE_MM_DEPTH_EVAPORATION_PCT", 0.75)

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
    stop_loss_cents: float = _env_float("STOP_LOSS_CENTS", 4.0)
    stop_loss_cooldown_s: float = _env_float("STOP_LOSS_COOLDOWN_S", 300.0)
    sl_vol_adaptive: bool = _env_bool("SL_VOL_ADAPTIVE", True)
    sl_vol_ref: float = _env_float("SL_VOL_REF", 0.70)
    sl_vol_multiplier_max: float = _env_float("SL_VOL_MULTIPLIER_MAX", 1.5)

    # ── Pillar 11.3: Preemptive Liquidity & Time-Decay Stop ────────────
    # Preemptive liquidity drain: trigger stop when support-side depth
    # is < threshold fraction of resistance-side depth AND position
    # is underwater.  Prevents slippage bleed on a "hollow book".
    sl_preemptive_obi_threshold: float = _env_float("SL_PREEMPTIVE_OBI_THRESHOLD", 0.10)
    # Time-decay: after sl_decay_start_minutes the vol multiplier
    # decays exponentially back toward 1.0 (tightening the stop)
    # with a half-life of sl_decay_half_life_minutes.
    sl_decay_start_minutes: float = _env_float("SL_DECAY_START_MINUTES", 5.0)
    sl_decay_half_life_minutes: float = _env_float("SL_DECAY_HALF_LIFE_MINUTES", 15.0)

    signal_cooldown_minutes: float = _env_float("SIGNAL_COOLDOWN_MINUTES", 5.0)
    max_total_exposure_pct: float = _env_float("MAX_TOTAL_EXPOSURE_PCT", 60.0)

    # Maximum dollar-risk per trade.  Caps position size so that a
    # gap-to-zero event cannot lose more than this many cents.
    # max_loss = entry_price × size × 100 ≤ max_loss_per_trade_cents.
    max_loss_per_trade_cents: float = _env_float("MAX_LOSS_PER_TRADE_CENTS", 1500.0)

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
    # price moves.  A 75% depth drop within 2 seconds within 5¢ of mid
    # is the classic signature.  Raised from 60% to 75% to reduce false
    # positives on thin Polymarket books where natural noise can produce
    # transient depth drops.
    adverse_sel_depth_drop_pct: float = _env_float("ADVERSE_SEL_DEPTH_DROP_PCT", 0.75)
    adverse_sel_depth_window_s: float = _env_float("ADVERSE_SEL_DEPTH_WINDOW_S", 2.0)
    adverse_sel_depth_near_mid_cents: float = _env_float("ADVERSE_SEL_DEPTH_NEAR_MID_CENTS", 5.0)

    # Signal 3 — Spread blow-out
    # When the bid-ask spread on a positioned market widens to 4× its
    # 5-minute rolling average, market makers are widening defensively
    # in response to perceived information asymmetry.  Raised from 3×
    # to 4× to reduce false positives on naturally noisy spreads.
    adverse_sel_spread_blowout_mult: float = _env_float("ADVERSE_SEL_SPREAD_BLOWOUT_MULT", 4.0)
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

    # Confirmation persistence: require the 2-of-4 condition to hold
    # for N consecutive poll cycles before firing.  At 50ms cadence,
    # 4 cycles = 200ms delay.  Filters single-cycle transient spikes.
    adverse_sel_confirmation_cycles: int = _env_int("ADVERSE_SEL_CONFIRMATION_CYCLES", 4)
    # ── Pillar 6: Dynamic Fee-Curve Integration ────────────────────────────
    fee_cache_ttl_s: int = _env_int("FEE_CACHE_TTL_S", 300)
    fee_default_bps: int = _env_int("FEE_DEFAULT_BPS", 200)
    desired_margin_cents: float = _env_float("DESIRED_MARGIN_CENTS", 2.5)

    # ── Pillar 7: Hybrid-Aggressive Chaser Escalation ──────────────────────
    chaser_max_rejections: int = _env_int("CHASER_MAX_REJECTIONS", 3)
    chaser_escalation_ticks: int = _env_int("CHASER_ESCALATION_TICKS", 1)
    # Anti-toxicity guard: if the adverse-selection p-value drops below
    # this ceiling during a chase, cancel rather than crossing the spread.
    chaser_toxicity_p_value_ceil: float = _env_float("CHASER_TOXICITY_P_VALUE_CEIL", 0.10)

    # ── Pillar 8: Clock-Skew & Stale Book Safety ──────────────────────────
    heartbeat_check_ms: int = _env_int("HEARTBEAT_CHECK_MS", 500)
    heartbeat_stale_ms: int = _env_int("HEARTBEAT_STALE_MS", 5000)
    heartbeat_stale_count: int = _env_int("HEARTBEAT_STALE_COUNT", 3)
    ws_silence_timeout_s: float = _env_float("WS_SILENCE_TIMEOUT_S", 10.0)
    # L2 uses a separate, longer timeout because low-volume markets
    # can go minutes without book changes.  Ping/pong keeps the TCP
    # socket alive; application-level silence is expected.
    l2_silence_timeout_s: float = _env_float("L2_SILENCE_TIMEOUT_S", 120.0)

    # ── Pillar 9: Toxic Flow Avoidance (2026 Dynamic Fee Regime) ───────────
    # MTI — Maker/Taker Imbalance penalty
    mti_threshold: float = _env_float("MTI_THRESHOLD", 0.80)
    mti_penalty_points: float = _env_float("MTI_PENALTY_POINTS", 40.0)

    # Fee-adaptive stop-loss
    fee_max_pct: float = _env_float("FEE_MAX_PCT", 2.00)  # peak fee %
    fee_enabled_categories: str = _env("FEE_ENABLED_CATEGORIES", "crypto,sports")

    # Fee-efficiency floor for EQS.  Prevents fees from zeroing the
    # entire geometric-mean EQS score.  At 0.30 trades losing >70% of
    # gross spread to fees are hard-rejected as mathematically -EV.
    eqs_fee_efficiency_floor: float = _env_float("EQS_FEE_EFFICIENCY_FLOOR", 0.30)

    # ── Pillar 10: Order Status Polling ─────────────────────────────────
    order_status_poll_s: float = _env_float("ORDER_STATUS_POLL_S", 2.0)
    order_status_max_retries: int = _env_int("ORDER_STATUS_MAX_RETRIES", 3)
    # ── Pillar 11: Active Stop-Loss Engine ────────────────────────────────
    stop_loss_poll_ms: int = _env_int("STOP_LOSS_POLL_MS", 500)
    trailing_stop_offset_cents: float = _env_float("TRAILING_STOP_OFFSET_CENTS", 0.0)

    # ── Pillar 12: Multi-Signal Framework ─────────────────────────────────
    imbalance_threshold: float = _env_float("IMBALANCE_THRESHOLD", 1.5)
    spread_compression_pct: float = _env_float("SPREAD_COMPRESSION_PCT", 0.70)
    min_composite_signal_score: float = _env_float("MIN_COMPOSITE_SIGNAL_SCORE", 0.20)

    # ── Kelly cold-start ──────────────────────────────────────────────────
    min_kelly_trades: int = _env_int("MIN_KELLY_TRADES", 20)
    cold_start_frac: float = _env_float("COLD_START_FRAC", 0.50)
    # Adaptive cold-start: halt new entries when rolling N-trade
    # expectancy is negative, preventing compounding losses.
    cold_start_halt_window: int = _env_int("COLD_START_HALT_WINDOW", 10)
    cold_start_negative_ev_halt: bool = _env_bool("COLD_START_NEGATIVE_EV_HALT", True)

    # ── Spread-based signal source ────────────────────────────────────────
    spread_signal_enabled: bool = _env_bool("SPREAD_SIGNAL_ENABLED", True)
    spread_signal_cooldown_s: float = _env_float("SPREAD_SIGNAL_COOLDOWN_S", 30.0)

    # ── EWMA volatility (RiskMetrics) ──────────────────────────────────────────
    # Exponentially-weighted moving average of 1-min log-return variance.
    # Reacts to regime changes in ~6 bars vs ~30 for equal-weight std.
    # λ = 0.94 is the RiskMetrics standard for daily data scaled to 1-min.
    volatility_ewma_lambda: float = _env_float("VOLATILITY_EWMA_LAMBDA", 0.94)

    # ── Regime-adaptive EQS threshold (OE-6) ─────────────────────────────────
    # Scale min_edge_score by EWMA σ.  Low-vol → raise threshold
    # (avoid noise), high-vol → lower threshold (mean-reversion α).
    eqs_vol_adaptive: bool = _env_bool("EQS_VOL_ADAPTIVE", False)
    # Recalibrated from 0.02 to 0.70 based on 3-day tick data analysis:
    # actual market EWMA vols have median 0.72 (P10=0.21, P75=1.22).
    # At 0.02, the ratio was always 10-96x, jamming the ±25% adjuster
    # permanently at -25%.  At 0.70, the feature is actually adaptive:
    # quiet bars (σ<0.41) → threshold +25% (more selective),
    # normal bars (σ≈0.72) → threshold ≈ baseline,
    # active bars (σ>1.22) → threshold -25% (exploit mean-reversion).
    eqs_vol_ref: float = _env_float("EQS_VOL_REF", 0.70)         # median 1-min σ from data
    eqs_vol_scale_range: float = _env_float("EQS_VOL_SCALE_RANGE", 0.25)  # ±25% max adjustment
    # ── Pillar 13: Kelly Sizing ───────────────────────────────────────────
    kelly_fraction: float = _env_float("KELLY_FRACTION", 0.25)
    kelly_max_pct: float = _env_float("KELLY_MAX_PCT", 10.0)
    kelly_p_cap: float = _env_float("KELLY_P_CAP", 0.85)                      # max win probability estimate
    kelly_default_uncertainty: float = _env_float("KELLY_DEFAULT_UNCERTAINTY", 0.5)  # fallback when signal metadata missing

    # Prior win rate used during cold-start (< min_kelly_trades).
    # Set this from WFO OOS results to avoid the 0.55 guess.
    kelly_prior_win_rate: float = _env_float("KELLY_PRIOR_WIN_RATE", 0.55)

    # Decay factor for exponentially-weighted win rate.
    # ŵ_t = α · 1[win_t] + (1-α) · ŵ_{t-1}.  α=0.10 → ~10-trade half-life.
    kelly_wr_decay_alpha: float = _env_float("KELLY_WR_DECAY_ALPHA", 0.10)

    # ── Uncertainty penalty weights for edge discounting ──────────────────
    uncertainty_spread_weight: float = _env_float("UNCERTAINTY_SPREAD_WEIGHT", 0.6)
    uncertainty_conf_weight: float = _env_float("UNCERTAINTY_CONF_WEIGHT", 0.4)

    # Ghost Liquidity Circuit Breaker
    ghost_depth_drop_threshold: float = _env_float("GHOST_DEPTH_DROP_THRESHOLD", 0.50)
    ghost_window_s: float = _env_float("GHOST_WINDOW_S", 2.0)
    ghost_recovery_s: float = _env_float("GHOST_RECOVERY_S", 60.0)
    ghost_check_interval_ms: int = _env_int("GHOST_CHECK_INTERVAL_MS", 500)

    # ── Vacuum / Stink Bid Strategy ────────────────────────────────────────
    # When ghost liquidity is detected, place POST_ONLY limit orders deeply
    # out-of-the-money on both sides of the book to catch flash-crash wicks.
    vacuum_stink_bid_enabled: bool = _env_bool("VACUUM_STINK_BID_ENABLED", True)
    vacuum_stink_bid_offset_cents: float = _env_float("VACUUM_STINK_BID_OFFSET_CENTS", 8.0)
    vacuum_stink_bid_max_risk_usd: float = _env_float("VACUUM_STINK_BID_MAX_RISK_USD", 2.0)

    # Paper-mode fill slippage.  In paper mode, fills are simulated by
    # crossing the limit price.  This adds a configurable adverse
    # slippage (in cents) to avoid overstating performance.  Set to 0
    # for the legacy behaviour (fill at exact limit price).
    paper_slippage_cents: float = _env_float("PAPER_SLIPPAGE_CENTS", 0.5)

    # Minimum ask-side depth (USD) required when discovering markets.
    # Markets with less than this on the ask side are too illiquid to
    # enter without excessive impact.
    min_ask_depth_usd: float = _env_float("MIN_ASK_DEPTH_USD", 25.0)

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
    # Maximum acceptable seq-gap rate for an L2 book.  Books that exceed
    # this rate (after ≥50 deltas) are flagged unreliable and excluded
    # from signal evaluation until they stabilise.
    l2_max_seq_gap_rate: float = _env_float("L2_MAX_SEQ_GAP_RATE", 0.02)

    # ── Pillar 14: Resolution Probability Engine (RPE) ─────────────────────
    #
    # RPE_SHADOW_MODE (default False → live mode):
    #   When True, RPE signals are evaluated, logged, recorded in the
    #   calibration tracker, and sent to Telegram (prefixed "SHADOW"),
    #   but NO positions are opened.  Use this to collect calibration
    #   data (Brier score, log-loss, direction accuracy) before going
    #   live.  Set to False to enable real/paper position opening.
    rpe_shadow_mode: bool = _env_bool("RPE_SHADOW_MODE", False)
    rpe_confidence_threshold: float = _env_float("RPE_CONFIDENCE_THRESHOLD", 0.08)
    rpe_weight: float = _env_float("RPE_WEIGHT", 0.5)
    rpe_crypto_vol_default: float = _env_float("RPE_CRYPTO_VOL_DEFAULT", 0.80)  # 80% annualised
    rpe_bayesian_obs_weight: float = _env_float("RPE_BAYESIAN_OBS_WEIGHT", 5.0)
    rpe_min_confidence: float = _env_float("RPE_MIN_CONFIDENCE", 0.15)
    # GenericBayesianModel uses a Beta(2,2)-derived prior that shrinks
    # every market toward 0.50.  This creates false divergence on any
    # market not priced at 0.50.  Disabled by default; enable ONLY after
    # RPECalibrationTracker shows direction_accuracy > 55% on ≥30
    # resolved signals.
    rpe_generic_enabled: bool = _env_bool("RPE_GENERIC_ENABLED", False)
    rpe_crypto_retrigger_cents: float = _env_float("RPE_CRYPTO_RETRIGGER_CENTS", 500.0)

    # RPE cooldown / freshness / calibration (Pillar 14 overhaul)
    rpe_cooldown_seconds: int = _env_int("RPE_COOLDOWN_SECONDS", 120)
    rpe_max_data_age_seconds: int = _env_int("RPE_MAX_DATA_AGE_SECONDS", 300)
    # Dedicated stale-market eviction threshold (seconds without any
    # trade).  Markets exceeding this are evicted or drained.  Separate
    # from rpe_max_data_age_seconds so the two concerns can be tuned
    # independently.
    stale_market_eviction_s: float = _env_float("STALE_MARKET_EVICTION_S", 1800.0)

    rpe_prior_k: float = _env_float("RPE_PRIOR_K", 4.0)
    rpe_min_eqs: float = _env_float("RPE_MIN_EQS", 25.0)
    rpe_tail_veto_threshold: float = _env_float("RPE_TAIL_VETO_THRESHOLD", 0.10)

    # ── Dynamic Prior Generation Engine (RPE upgrade) ──────────────────────
    # L2 order-book imbalance sensitivity — scales ln(book_depth_ratio)
    # into Beta pseudo-counts.  Higher = more responsive to book skew.
    rpe_l2_kappa: float = _env_float("RPE_L2_KAPPA", 1.5)
    # Time-decay sigmoid steepness (higher = sharper transition).
    rpe_theta_gamma: float = _env_float("RPE_THETA_GAMMA", 8.0)
    # Time-decay half-life as fraction of market duration remaining.
    # At this fraction the prior / observation weights are equal.
    rpe_theta_half: float = _env_float("RPE_THETA_HALF", 0.15)
    # Minimum divergence (model vs market) to trigger a model-only
    # probe entry when PanicDetector is silent.
    rpe_probe_divergence_min: float = _env_float("RPE_PROBE_DIVERGENCE_MIN", 0.12)
    # Enable tag-based dynamic priors (vs fixed/market-anchored).
    rpe_dynamic_prior_enabled: bool = _env_bool("RPE_DYNAMIC_PRIOR_ENABLED", True)

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

    # ── SI-1: Regime Detector ──────────────────────────────────────────────
    # Per-market HMM-lite regime classifier.  Score ∈ [0,1] where
    # 1 = mean-reversion favourable.  Entries are suppressed when the
    # score drops below regime_threshold.
    regime_enabled: bool = _env_bool("REGIME_ENABLED", True)
    regime_ewma_lambda: float = _env_float("REGIME_EWMA_LAMBDA", 0.90)
    regime_autocorr_window: int = _env_int("REGIME_AUTOCORR_WINDOW", 20)
    regime_persistence_window: int = _env_int("REGIME_PERSISTENCE_WINDOW", 15)
    regime_threshold: float = _env_float("REGIME_THRESHOLD", 0.40)

    # ── SI-2: Iceberg Detector ─────────────────────────────────────────────
    # Detects hidden reserve orders via repeated size replenishment at
    # persistent price levels in the L2 book.
    iceberg_enabled: bool = _env_bool("ICEBERG_ENABLED", True)
    iceberg_refill_window_s: float = _env_float("ICEBERG_REFILL_WINDOW_S", 5.0)
    iceberg_min_refills: int = _env_int("ICEBERG_MIN_REFILLS", 3)
    iceberg_size_tolerance_pct: float = _env_float("ICEBERG_SIZE_TOLERANCE_PCT", 0.30)
    # Minimum iceberg confidence to activate peg routing in the chaser.
    iceberg_peg_min_confidence: float = _env_float("ICEBERG_PEG_MIN_CONFIDENCE", 0.50)
    # EQS bonus applied when iceberg detection confirms hidden liquidity.
    iceberg_eqs_bonus: float = _env_float("ICEBERG_EQS_BONUS", 0.15)
    # Take-profit alpha adjustment when iceberg is detected.
    iceberg_tp_alpha: float = _env_float("ICEBERG_TP_ALPHA", 0.05)

    # ── SI-3: Cross-Market Signal Generator ────────────────────────────────
    # Offensive pairs-style alpha from correlated market lag divergences.
    cross_mkt_enabled: bool = _env_bool("CROSS_MKT_ENABLED", True)
    cross_mkt_shadow: bool = _env_bool("CROSS_MKT_SHADOW", True)  # log-only until calibrated
    cross_mkt_min_correlation: float = _env_float("CROSS_MKT_MIN_CORRELATION", 0.50)
    cross_mkt_z_entry: float = _env_float("CROSS_MKT_Z_ENTRY", 2.0)
    cross_mkt_spread_ewma_lambda: float = _env_float("CROSS_MKT_SPREAD_EWMA_LAMBDA", 0.94)

    # ── SI-4: Stealth Execution ────────────────────────────────────────────
    # Time-sliced order splitting to reduce market footprint.
    stealth_enabled: bool = _env_bool("STEALTH_ENABLED", True)
    stealth_min_size_usd: float = _env_float("STEALTH_MIN_SIZE_USD", 5.0)
    stealth_max_slices: int = _env_int("STEALTH_MAX_SLICES", 4)
    stealth_min_delay_ms: float = _env_float("STEALTH_MIN_DELAY_MS", 200.0)
    stealth_max_delay_ms: float = _env_float("STEALTH_MAX_DELAY_MS", 1500.0)
    stealth_size_jitter_pct: float = _env_float("STEALTH_SIZE_JITTER_PCT", 0.15)
    # POV cap: max fraction of recent volume a single slice may represent.
    stealth_max_participation_pct: float = _env_float("STEALTH_MAX_PARTICIPATION_PCT", 0.05)
    # Abandonment: abort remaining slices if mid drifts adversely.
    stealth_abandon_drift_cents: float = _env_float("STEALTH_ABANDON_DRIFT_CENTS", 2.0)
    # Abandonment: skip remaining slices if this fraction already filled.
    stealth_abandon_fill_pct: float = _env_float("STEALTH_ABANDON_FILL_PCT", 0.75)

    # ── SI-8: Oracle Latency Arbitrage ─────────────────────────────────────
    # Generalised off-chain oracle system.  Polls real-world APIs (election
    # feeds, live sports) and triggers the SI-7 fast-strike taker path when
    # the CLOB price diverges from the oracle-derived probability.
    oracle_arb_enabled: bool = _env_bool("ORACLE_ARB_ENABLED", False)
    # Default polling interval (ms) for REST-based oracle adapters.
    oracle_default_poll_ms: int = _env_int("ORACLE_DEFAULT_POLL_MS", 1000)
    # Critical-window polling interval (ms) — throttled during decisive moments.
    oracle_critical_poll_ms: int = _env_int("ORACLE_CRITICAL_POLL_MS", 200)
    # Idle-period polling interval (ms) — relaxed during halftime / pre-event.
    oracle_idle_poll_ms: int = _env_int("ORACLE_IDLE_POLL_MS", 30000)
    # Base divergence threshold (tighter than RPE's 0.08 — oracle signals
    # are high-conviction by nature).
    oracle_confidence_threshold: float = _env_float("ORACLE_CONFIDENCE_THRESHOLD", 0.06)
    # Minimum oracle confidence to fire a signal.  Higher bar than RPE's
    # 0.15 because we only want to trade on highly verified API states.
    oracle_min_confidence: float = _env_float("ORACLE_MIN_CONFIDENCE", 0.80)
    # Per-market fire cooldown (seconds).  Longer than RPE's 120 s because
    # oracle events are slower-moving.
    oracle_cooldown_seconds: int = _env_int("ORACLE_COOLDOWN_SECONDS", 300)
    # Maximum CLOB spread (cents) before suppressing the signal.  Wide
    # spreads indicate the market is already pricing in the event or has
    # low liquidity.
    oracle_max_spread_cents: float = _env_float("ORACLE_MAX_SPREAD_CENTS", 15.0)
    # JSON-encoded list of oracle market bindings.  Each entry:
    #   {"market_id": "...", "oracle_type": "ap_election"|"sports"
    #    |"odds_api_ws"|"tree_news_ws",
    #    "oracle_params": {...}, "yes_asset_id": "...", "no_asset_id": "...",
    #    "event_id": "..."}
    oracle_market_configs: str = _env("ORACLE_MARKET_CONFIGS", "[]")
    # AP Election API credentials
    oracle_ap_api_key: str = _env("ORACLE_AP_API_KEY")
    oracle_ap_api_url: str = _env("ORACLE_AP_API_URL")
    # Sports API credentials
    oracle_sports_api_key: str = _env("ORACLE_SPORTS_API_KEY")
    oracle_sports_api_url: str = _env("ORACLE_SPORTS_API_URL")
    # Odds API WebSocket credentials
    oracle_odds_api_ws_url: str = _env("ORACLE_ODDS_API_WS_URL")
    oracle_odds_api_ws_key: str = _env("ORACLE_ODDS_API_WS_KEY")
    # Tree News WebSocket credentials
    oracle_tree_news_ws_url: str = _env("ORACLE_TREE_NEWS_WS_URL")
    oracle_tree_news_ws_key: str = _env("ORACLE_TREE_NEWS_WS_KEY")
    # Shadow mode — evaluate and log oracle signals without placing orders.
    # Enabled by default for safe rollout; disable to go live.
    oracle_shadow_mode: bool = _env_bool("ORACLE_SHADOW_MODE", True)

    # ── SI-9: Mutually Exclusive Combinatorial Arbitrage ───────────────────
    # Passive maker-only arbitrage across negRisk event clusters where
    # sum(YES best_bids) < 1.0 - margin  ⇒  risk-free if all legs fill.
    si9_arb_enabled: bool = _env_bool("SI9_ARB_ENABLED", False)
    si9_min_margin_cents: float = _env_float("SI9_MIN_MARGIN_CENTS", 2.0)
    si9_margin_buffer_cents: float = _env_float("SI9_MARGIN_BUFFER_CENTS", 1.0)
    si9_max_legs: int = _env_int("SI9_MAX_LEGS", 6)
    si9_max_concurrent_combos: int = _env_int("SI9_MAX_CONCURRENT_COMBOS", 2)
    si9_max_total_exposure_usd: float = _env_float("SI9_MAX_TOTAL_EXPOSURE_USD", 50.0)
    si9_max_per_combo_usd: float = _env_float("SI9_MAX_PER_COMBO_USD", 25.0)
    si9_target_size_usd: float = _env_float("SI9_TARGET_SIZE_USD", 10.0)
    si9_max_leg_delay_ms: int = _env_int("SI9_MAX_LEG_DELAY_MS", 30000)
    si9_max_leg_chase_cents: float = _env_float("SI9_MAX_LEG_CHASE_CENTS", 2.0)
    si9_chase_interval_ms: int = _env_int("SI9_CHASE_INTERVAL_MS", 500)
    si9_scan_interval_ms: int = _env_int("SI9_SCAN_INTERVAL_MS", 500)
    si9_min_leg_depth_usd: float = _env_float("SI9_MIN_LEG_DEPTH_USD", 50.0)
    # Max taker fee (cents) tolerated when emergency-hedging a hanging leg.
    # If the taker cost exceeds this, the bot dumps filled legs instead.
    si9_emergency_taker_max_cents: float = _env_float("SI9_EMERGENCY_TAKER_MAX_CENTS", 4.0)
    # Max spread (cents) on best_ask for a lagging leg before the bot
    # gives up crossing and dumps filled legs instead.
    si9_emergency_max_spread_cents: float = _env_float("SI9_EMERGENCY_MAX_SPREAD_CENTS", 5.0)


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
    polygon_rpc_wss_url: str = field(
        default_factory=lambda: _env("POLYGON_RPC_WSS_URL")
    )

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
    whale_threshold_shares: float = field(
        default_factory=lambda: _env_float("WHALE_THRESHOLD_SHARES", 50_000.0)
    )
    whale_ws_heartbeat_s: float = field(
        default_factory=lambda: _env_float("WHALE_WS_HEARTBEAT_S", 30.0)
    )

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
