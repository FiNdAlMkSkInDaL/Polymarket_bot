# Pillar 16 — Alpha-Source Attribution

## Overview

Pillar 16 adds **alpha-source attribution** to the trading system, enabling
performance analysis broken down by the originating signal type. Every
position is tagged at creation with a `signal_type` and the SI-6
`meta_weight` that was active at entry time, and these fields are persisted
to `trades.db` for post-hoc analytics.

## Signal Types

| Signal Type  | Origin                                      | Position ID Prefix |
|-------------|---------------------------------------------|--------------------|
| `panic`     | PanicDetector fires on z-score spike         | `POS-`             |
| `drift`     | DriftDetector fires on cumulative displacement| `POS-`             |
| `rpe`       | Resolution Probability Engine model divergence| `RPE-`             |
| `stink_bid` | Cascade harvester places gap bids during panic| `STINK-`           |

## Database Schema

Two columns added to the `trades` table:

```sql
signal_type   TEXT DEFAULT ''    -- "panic", "drift", "rpe", "stink_bid"
meta_weight   REAL DEFAULT 1.0  -- SI-6 MetaStrategyController weight at entry
```

**Migration:** On first `init()` after upgrade, the `TradeStore` runs
`ALTER TABLE ... ADD COLUMN` for each missing column. Existing rows default
to `signal_type=''` and `meta_weight=1.0`. The tearsheet script infers
legacy signal types from position ID prefixes.

## Tagging Logic

### Panic & Drift (`_open_position_inner`)

The `signal_metadata` dict passed from `bot.py` carries `signal_source`
(set to `"drift"` for drift signals) and `meta_weight` from SI-6.  The
`_open_position_inner` method reads these to populate `Position.signal_type`
and `Position.meta_weight`.

```python
_signal_type = "drift" if meta.get("signal_source") == "drift" else "panic"
_meta_weight = float(meta.get("meta_weight", 1.0))
```

### RPE (`_open_rpe_position_inner`)

All RPE positions are unconditionally tagged `signal_type="rpe"`. The
`meta_weight` is read from `rpe_signal_meta["meta_weight"]`.

### Stink-Bids (`harvest_cascades`)

All positions created by the cascade harvester use the prefix `STINK-` and
are tagged `signal_type="stink_bid"`. Meta-weight defaults to 1.0 (cascade
bids are not routed through SI-6).

## SI-6 Meta-Weight Influence

The `MetaStrategyController` evaluates each signal against the current
regime score and returns a weight multiplier:

| Regime    | Panic/Drift | RPE  |
|-----------|-------------|------|
| Deep MR   | 1.5×        | 0.5× |
| Neutral   | 1.0×        | 1.0× |
| Trending  | VETOED      | 1.0× |

This weight is stored on the Position and persisted to `trades.db` as
`meta_weight`. The tearsheet reports the average meta-weight per signal
type, which reveals how often each strategy fires in its preferred regime.

**Expected Value calculation:** The meta-weight does not directly scale
PnL — it influences *sizing* at entry time. When the Kelly sizer runs,
the meta-weight from SI-6 scales the maximum trade allocation, effectively
expressing the controller's conviction. A trade entered at weight=1.5×
deploys more capital (and thus higher absolute PnL on wins/losses) than
one entered at weight=0.5×.

## Analytics: Strategy Tearsheet

The `scripts/strategy_tearsheet.py` script generates a performance
breakdown per alpha source.

### Metrics Computed

| Metric               | Description                                       |
|---------------------|---------------------------------------------------|
| Total PnL            | Sum of `pnl_cents` for all closed trades           |
| Win Rate             | Fraction of trades with `pnl_cents > 0`           |
| Profit Factor        | Gross profit / gross loss                         |
| Avg Hold Time        | Mean `hold_seconds` across closed trades          |
| Max Drawdown         | Largest peak-to-trough cumulative PnL decline     |
| Capital Used         | Sum of `entry_price × entry_size` (USD deployed)  |
| Strategy Efficiency  | `PnL / Capital Used` — return per dollar of VaR   |
| Avg Meta Weight      | Mean SI-6 weight — indicates regime alignment     |

### Usage

```bash
# Default: reads logs/trades.db, writes strategy_performance.json
python scripts/strategy_tearsheet.py

# Custom paths
python scripts/strategy_tearsheet.py --db data/trades.db --json-out reports/perf.json
```

### Output

1. **Console:** A Markdown table with per-signal-type and portfolio-aggregate rows.
2. **JSON:** `strategy_performance.json` with the same metrics in machine-readable format.

### Legacy Data Handling

Trades recorded before Pillar 16 have no `signal_type`. The script infers
the type from position ID prefixes:

- `RPE-*` → `rpe`
- `STINK-*` → `stink_bid`
- All others → `panic` (conservative default; drift signals also use `POS-`
  prefixes, so pre-Pillar-16 drift trades are attributed to panic)

---

## Section 2: Self-Healing Mechanisms and Throttling Thresholds (Pillar 16.2)

### Motivation

Not all alpha sources remain profitable at all times. Market regime shifts,
structural changes in liquidity, or model degradation can cause a formerly
profitable signal type to bleed capital. Pillar 16.2 automatically detects
this and reduces position sizing for the underperforming source, allowing
the portfolio to self-heal without manual intervention.

### Architecture

```
TradeStore.get_strategy_expectancy(signal_type, window=50)
    ↓
PositionManager._compute_strategy_multiplier(signal_type)
    ↓  returns strategy_mult ∈ {0.1, 0.5, 1.0}
compute_kelly_size(..., strategy_multiplier=strategy_mult)
    ↓  f_final = f_adj × strategy_mult
Position opened at reduced size (or full size if healthy)
```

### TradeStore Query

`get_strategy_expectancy(signal_type, window=50)` returns `(avg_pnl_cents,
n_trades)` for the last 50 closed non-probe trades of the given signal
type. Probe trades (`is_probe=1`) are excluded so micro-sized exploratory
entries do not pollute the expectancy estimator.

### Penalty Scale

The self-healing gate activates only after **≥ 20 trades** of the given
signal type, ensuring sufficient sample size:

| Condition                              | `strategy_mult` | Effect             |
|----------------------------------------|-----------------|--------------------|
| `rolling_ev < -2.0¢` (significant)    | **0.1**         | 90% Kelly reduction |
| `rolling_ev < 0¢` (mild negative)     | **0.5**         | 50% Kelly reduction |
| `rolling_ev ≥ 0¢`                     | **1.0**         | No penalty          |
| `n_trades < 20`                        | **1.0**         | Insufficient data   |

### Kelly Integration

The `strategy_multiplier` is applied inside `compute_kelly_size()` after
the base Kelly fraction (`f* × kelly_fraction`) but before the MCPV
covariance penalty:

```
f_final = (f* × kelly_fraction) × strategy_multiplier × mcpv_penalty
```

This ordering ensures the self-healing throttle compounds with (rather
than overrides) the existing PCE diversification penalty.

### Logging

Every throttle event emits a structured log at WARNING level:

```json
{
  "event": "alpha_source_throttled",
  "signal_type": "drift",
  "rolling_ev": -1.23,
  "n_trades": 34,
  "strategy_mult": 0.5
}
```

### Recovery

The throttle is **automatic and stateless** — it re-evaluates on every
entry attempt. As soon as the rolling 50-trade window for the affected
signal type returns to non-negative expectancy, `strategy_mult` resets to
1.0 and full sizing resumes. No manual reset or configuration change is
needed.
