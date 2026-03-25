# Architecture

This document describes the checked-in trading architecture as it exists in the
repository today. It is written for quantitative engineers who need exact
runtime and replay behavior, not a simplified product summary.

## System Shape

The repository is a multi-strategy event-driven trading system built around a
shared runtime substrate:

- src/bot.py orchestrates books, trades, detectors, and background control
  loops.
- src/trading/position_manager.py owns position lifecycle, local exits, and
  shared execution-time risk checks.
- src/trading/stop_loss.py converts BBO changes into immediate exit evaluation.
- src/data/orderbook.py maintains live top-of-book state and O(1) depth
  baselines.
- src/backtest/strategy.py mirrors live behavior in deterministic replay
  adapters.
- src/backtest/wfo_optimizer.py drives Optuna walk-forward studies and artifact
  export.
- src/core/live_hyperparameters.py and scripts/inject_wfo_champions.py bridge
  optimization outputs into production configuration.

The codebase contains multiple alpha families. The runtime does not assume a
single flagship strategy.

## Alpha Portfolio

### OFI Momentum

Primary modules:

- src/signals/ofi_momentum.py
- src/execution/momentum_taker.py
- src/trading/position_manager.py
- src/trading/stop_loss.py
- src/backtest/strategy.py

OFI Momentum is a taker-style microstructure strategy built around rolling
top-of-book imbalance and TVI-sensitive confirmation. The signal is short-hold,
latency-sensitive, and explicitly designed to monetize queue-pressure rather
than passive spread capture.

### Contagion Arb

Primary modules:

- src/signals/contagion_arb.py
- src/signals/microstructure_utils.py
- src/backtest/strategy.py
- src/backtest/wfo_optimizer.py

Contagion Arb monitors thematic groups of markets, detects toxicity spikes on a
leader book, and propagates that shock into lagging books subject to
correlation, residual, spread, and synchronization gates.

### SI-9 Combinatorial Arb

Primary modules:

- src/signals/combinatorial_arb.py
- src/data/arb_clusters.py
- src/trading/position_manager.py
- src/bot.py

SI-9 looks for mutually exclusive clusters whose YES best bids imply a
sub-$1.00$ Dutch-book. It is structurally multi-leg and therefore uses a
different execution model from the directional strategies.

### SI-10 Bayesian Joint-Probability Arb

Primary modules:

- src/signals/bayesian_arb.py
- src/backtest/strategy.py
- src/backtest/wfo_optimizer.py
- src/bot.py

SI-10 evaluates configured base/base/joint relationships directly from live
YES-book snapshots and fee-aware edge math. It uses the same synchronization
gate as the other coordinated strategies.

### Legacy Paths

PURE_MM, panic, drift, RPE, oracle, and related support code remain in the
repository. They are relevant for replay, regression, and selective deployment,
but they are not the correct umbrella description of the architecture.

## Execution Layer

### Event-Driven Runtime

The live runtime is built around market-data events rather than polling loops.

- OrderbookTracker updates top-of-book state from price_change and book events.
- BBO changes invoke stop-loss and local-exit evaluation.
- PositionManager owns entry state, exit state, fill reconciliation, and exit
  routing.
- The timeout loop exists as a backstop, not as the primary execution trigger.

This matters because the high-conviction paths are latency-sensitive and should
not wait for coarse periodic polling to react to market state.

### Stochastic Hazard-Rate Exits For OFI

The OFI execution layer no longer posts a deterministic visible take-profit
order after entry fill.

Instead:

1. src/execution/momentum_taker.py draws a private DrawnMomentumBracket via
   draw_stochastic_momentum_bracket().
2. The draw uses bounded exponential hazard-style sampling around the configured
   means for take-profit percentage, stop-loss percentage, and max hold time.
3. src/trading/position_manager.py stores the sampled target, stop, and hold
   horizon on the position as drawn_tp, drawn_stop, drawn_time,
   drawn_tp_pct, and drawn_stop_pct.
4. on_entry_filled() arms the position for local monitoring and deliberately
   leaves pos.exit_order unset.

Why this design exists:

- a posted deterministic TP advertises the strategy footprint to the market
- the signal is short-lived and does not benefit from revealing its full exit
  geometry
- replay parity is still preserved because the same bracket sampler is reused
  there with deterministic seeding

This is not a cosmetic randomization layer. It is a change in execution model:
the exit is now hidden and strategy-owned.

### Local Exit Monitoring

The local OFI exit path is centered in PositionManager.evaluate_ofi_local_exit().
That function is triggered from two places:

- src/trading/stop_loss.py when a BBO update arrives for the traded asset
- PositionManager.check_timeouts() as a periodic backstop

Exit logic is evaluated in this order:

1. if a local target exit is already working, avoid duplicate submissions
2. if smart-passive time-stop mode is active, either continue waiting or
   promote to taker
3. if current best bid reaches the hidden target, call force_target_exit()
4. if current best bid breaches the hidden stop, call force_stop_loss()
5. if the stochastic hold horizon expires, begin time-stop handling

The important architectural point is that OFI exits are no longer generic sell
orders sitting in the book. They are stateful local decisions tied to the live
book.

### Liquidity-Vacuum Suppression

Time-stop exits are not fired blindly. The runtime suppresses OFI time-stop
exits during transient order-book vacuums.

Mechanics:

- OrderbookTracker maintains O(1) EWMA baselines for top bid and ask depth.
- PositionManager._should_suppress_ofi_time_stop_exit() compares current depth
  against those EWMAs.
- The exit is suppressed when depth collapses below the configured vacuum ratio
  and spread simultaneously blows out beyond the configured multiple of the
  baseline spread.
- The exit becomes eligible again only after recovery relative to the EWMA
  baselines.

This prevents the bot from converting a stale momentum exit into a worst-quote
liquidity donation during a temporary microstructure vacuum.

### Smart-Passive Time-Stop Fallback

When the hold horizon expires without target or stop being hit, the OFI path
can enter smart-passive exit mode before promoting to taker. This preserves the
existing maker-fallback logic while keeping the exit locally owned.

### Replay Parity

Replay mirrors the same OFI model.

- src/backtest/strategy.py uses draw_stochastic_momentum_bracket() when opening
  OFI positions.
- Replay positions carry the same drawn fields as live positions.
- Replay exit checks mirror local target, local stop, and time-stop handling.
- WFO injects stochastic_seed values so each Optuna trial is deterministic even
  though brackets are stochastic.

The result is parity without reintroducing visible deterministic brackets.

## Risk Layer

### EnsembleRiskManager

src/trading/ensemble_risk.py implements an O(1) directional exposure gate
across strategies.

The core problem is that scalar net exposure is insufficient once multiple
strategies can hold YES and NO inventory on the same market. The manager
therefore maintains:

- one hash map keyed by market_id for directional ownership counts
- one position index keyed by position_id for exact release bookkeeping

Per market, exposure is split into:

- yes_by_strategy
- no_by_strategy

The operational rule is simple: a strategy may not open new exposure in a
direction if another strategy already owns that same directional slot on the
market.

This prevents gross risk stacking such as:

- OFI Momentum opening NO
- RPE opening more NO on the same market
- SI-10 opening another NO expression on the same book

All of can_enter(), register_position(), and release_position() are O(1) in the
size of the tracked state.

### CrossBookSyncGate

src/signals/microstructure_utils.py implements CrossBookSyncGate, another O(1)
primitive.

It performs a max-minus-min timestamp divergence check across a small set of
related book snapshots and returns a CrossBookSyncAssessment containing:

- is_synchronized
- latest_timestamp
- delta_ms
- book_count

If any snapshot is missing a valid timestamp, synchronization fails
immediately. Otherwise:

$$
\Delta_{ms} = (\max t_i - \min t_i) \times 1000
$$

Signals are suppressed when:

$$
\Delta_{ms} > \text{MAX\_CROSS\_BOOK\_DESYNC\_MS}
$$

This gate is instantiated in:

- ComboArbDetector for SI-9 cluster evaluation
- ContagionArbDetector for leader/lagger synchronization
- BayesianArbDetector for base/base/joint triplets

The purpose is not cosmetic data hygiene. It is a hard defense against trading
one leg of a relationship on stale or asynchronously updated books.

### Additional Shared Risk Controls

Other live risk components remain layered around the two primitives above:

- DeploymentGuard caps size by deployment phase
- PortfolioCorrelationEngine applies concentration haircuts and VaR-based size
  control
- LatencyGuard and heartbeat infrastructure mark stale books and throttle
  execution
- hanging-leg logic in SI-9 flattens partial fills that can no longer be
  completed safely

Those are important, but EnsembleRiskManager and CrossBookSyncGate are the key
new architectural primitives that changed how gross risk is controlled.

## Replay And Research Layer

### O(1) Aggregation

The replay plane relies on incremental OHLCV and depth state rather than full
window recomputation.

- rolling VWAP, volatility, and related moments are updated incrementally
- replay order books keep OFI windows and top-depth EWMAs in O(1) state
- BotReplayAdapter advertises self_aggregates_trades = True so the engine does
  not duplicate bar work

This optimization is operationally significant because WFO trial throughput
depends on it.

### WFO Model

src/backtest/wfo_optimizer.py runs walk-forward studies with these properties:

- rolling or anchored fold generation
- Optuna study storage in SQLite
- per-trial child-process isolation
- hard wall-clock timeout enforcement
- champion report export with per-fold and aggregate metrics

Champion parameters are exported by _export_champion_params() as:

```json
{
  "params": {"param_name": 1.23},
  "meta": {
    "champion_fold": 3,
    "oos_sharpe": 1.11,
    "generated_at": "..."
  }
}
```

OFI studies additionally thread stochastic_seed through replay so the sampled
hidden brackets remain deterministic within each trial and champion replay.

## Deployment Parameter Plane

### Artifact Handoff

The research-to-production bridge is explicit.

1. A WFO run emits champion_params.json.
2. scripts/inject_wfo_champions.py loads one or more champion artifacts.
3. The injector validates all overrides against StrategyParams.
4. The injector merges the params in input order and writes
   live_hyperparameters.json atomically.
5. src/core/config.py applies those overrides during bootstrap.

### Validation Model

src/core/live_hyperparameters.py is the sole validation authority for this
handoff.

It enforces at minimum:

- payload must be a JSON object
- params must resolve to known StrategyParams fields
- None, NaN, and infinite values are invalid
- strictly positive edge thresholds remain strictly positive
- percentile and correlation-style fields remain in [0, 1]
- max_cross_book_desync_ms remains strictly positive

If live_hyperparameters.json exists but is invalid, config bootstrap raises a
hard error. The architecture chooses deterministic failure over silent drift.

### File Locations

- default live override path: repository-root live_hyperparameters.json
- override env var: LIVE_HYPERPARAMETERS_PATH

## Operational Distinctions That Must Stay Explicit

When documenting or extending the system, keep these categories separate:

- code present in source
- code wired into the runtime
- code enabled by default
- code approved for a given desk deployment

This repository has enough optionality that collapsing those categories will
produce incorrect operational guidance.

## Bottom Line

The current architecture is best understood as four cooperating layers:

1. multiple alpha generators, including OFI, contagion, SI-9, and SI-10
2. an execution layer that now hides OFI exit geometry with stochastic local
   brackets
3. a risk layer built around O(1) exposure and synchronization gates
4. a research-to-production parameter plane driven by WFO artifacts and strict
   boot-time validation

Describing the repository as a generic market maker would now be materially
wrong.