# Polymarket Bot

Institutional trading stack for Polymarket microstructure, combinatorial, and
joint-probability execution.

This repository is not a single-strategy market maker. The checked-in runtime,
replay, and optimization layers support multiple alpha streams, a shared risk
and execution substrate, and an Optuna-to-production parameter deployment path.

Use this file as the operator-facing overview. Use ARCHITECTURE.md for the
technical source of truth.

## Current State

- Deployment mode defaults to PAPER via DEPLOYMENT_ENV.
- The runtime wires multiple alpha streams into one bot and one shared risk
  surface.
- Research and deployment now share a formal parameter handoff via
  champion_params.json and live_hyperparameters.json.
- PURE_MM remains in the repository for legacy replay and regression, but it is
  no longer the right mental model for the system.

## Feature Set

### Alpha Streams

- OFI Momentum with TVI-sensitive confirmation and toxicity-aware sizing.
- Contagion Arb, a Domino-style cross-market toxicity propagation strategy.
- SI-9 combinatorial arbitrage for mutually exclusive negRisk clusters.
- SI-10 Bayesian joint-probability arbitrage across configured base/base/joint
  triplets.
- Legacy panic, drift, RPE, oracle, and passive-maker paths that remain in the
  codebase for compatibility, research, and selective deployment.

### Shared Infrastructure

- Event-driven order-book tracking with O(1) top-of-book metrics and top-depth
  EWMA baselines.
- Hidden stochastic OFI exit management monitored locally instead of posting a
  deterministic take-profit order to the book.
- O(1) ensemble exposure gating to prevent multi-strategy directional stacking
  on the same market.
- Cross-book synchronization gating for multi-leg and multi-market signals.
- Replay adapters for OFI, contagion, Bayesian, and legacy strategies.
- Optuna walk-forward optimization with child-process timeouts and SQLite study
  storage.
- Boot-time live hyperparameter override loading with strict validation.

## Strategy Matrix

| Strategy | Runtime Status | Notes |
| --- | --- | --- |
| OFI Momentum | Implemented in live/replay | Uses aggressive entry and hidden local exits |
| Contagion Arb | Implemented in live/replay | Enabled by config, shadow mode on by default |
| SI-9 Combinatorial Arb | Implemented in live/replay | Runtime-wired, explicitly gated by config |
| SI-10 Bayesian Arb | Implemented in live/replay | Runtime-wired, explicitly gated by config |
| PURE_MM | Legacy | Kept for replay, comparative study, and regression |

"Implemented" means the code path is merged and wired into the bot or replay
stack. It does not imply that every strategy is enabled in the same deployment.

## Alpha Summaries

### OFI Momentum With TVI

OFI Momentum trades fast queue-pressure dislocations using aggressive entry.
The detector measures rolling order-flow imbalance and uses TVI sensitivity via
the ofi_tvi_kappa parameter. Once a position is filled, the strategy does not
leave a visible deterministic take-profit order resting on the book. Instead,
the runtime draws a private bracket and monitors hidden target, stop, and hold
time locally.

Operational consequences:

- entry is taker-style and latency-sensitive
- exits are monitored from BBO changes and timeout backstops
- replay and WFO reuse the same bracket draw logic for parity

### Contagion Arb

Contagion Arb groups markets by thematic tags, watches leader-book toxicity,
and projects that impulse into lagging books through correlation and residual
filters. The signal is suppressed when the relevant books are not synchronized
within the configured cross-book desync budget.

### SI-9 Combinatorial Arb

SI-9 scans mutually exclusive event clusters and looks for the Dutch-book case
where the sum of YES best bids drops sufficiently below $1.00$. It keeps share
counts equal across legs, works the bottleneck leg passively first, and uses a
hanging-leg unwind path to avoid naked partial exposure.

### SI-10 Bayesian Joint-Probability Arb

SI-10 evaluates configured base/base/joint triplets in O(1) time using live
YES-book snapshots, fee-aware edge math, depth constraints, and the same
cross-book synchronization gate used by the other coordinated strategies.

## MLOps Pipeline

The repository now has a formal research-to-runtime handoff.

### Research Side

- WFO runs through src/backtest/wfo_optimizer.py using Optuna studies backed by
  SQLite.
- Each study exports a report and a champion_params.json artifact.
- champion_params.json contains a params object plus metadata such as fold,
  degradation, trade counts, and generation time.

### Deployment Side

- scripts/inject_wfo_champions.py accepts one or more WFO output directories or
  champion_params.json files.
- The injector resolves each artifact, validates parameter names and values,
  merges them in input order, and writes live_hyperparameters.json atomically.
- Existing live_hyperparameters.json values are loaded first, then overlaid by
  the supplied champion artifacts.

### Boot-Time Loading

- src/core/config.py loads live_hyperparameters.json during settings bootstrap.
- src/core/live_hyperparameters.py validates overrides before they touch the
  frozen StrategyParams dataclass.
- Invalid live hyperparameters fail the boot with a hard error instead of
  silently degrading into stale defaults.
- LIVE_HYPERPARAMETERS_PATH can override the default file location.

### Validation Rules

- unknown strategy parameter names are rejected
- None, NaN, and infinite values are rejected
- edge thresholds ending in _usd or _cents must stay strictly positive
- percentile and correlation-style fields must remain inside [0, 1]
- max_cross_book_desync_ms must be strictly positive

## WFO Usage Notes

Before heavy optimization runs, disable file logging:

```powershell
$env:BOT_DISABLE_FILE_LOGGING='1'
```

Why:

- Optuna launches many child-process backtests
- rotating file logging adds avoidable disk contention
- the optimizer already has checkpoint and artifact outputs

Representative entry points:

```powershell
python wfo_ofi_momentum.py
python wfo_contagion_arb.py
python wfo_bayesian_arb.py
```

## Repository Layout

```text
src/
├── backtest/      Replay engine, matching engine, WFO optimizer, telemetry
├── core/          Config, guards, process management, live hyperparameter loader
├── data/          Market discovery, OHLCV, L2 books, adapters
├── execution/     Execution helpers including OFI bracket drawing
├── monitoring/    Trade store, health reporting, Telegram notifications
├── signals/       OFI, contagion, SI-9, SI-10, and legacy detectors
├── strategies/    Legacy live strategy implementations such as PURE_MM
├── trading/       Position manager, stop-loss engine, combo lifecycle, risk
├── bot.py         Runtime orchestrator
└── cli.py         Operator CLI
```

## Documentation Rules

Keep these distinctions explicit when updating docs:

- code present in the repository
- code wired into the runtime
- code enabled by default
- code approved for a specific deployment

This repo supports more than one thing at once. The docs should reflect the
actual wiring without collapsing everything into a single live posture claim.

- PURE_MM defaults that still exist in config
- legacy directional code paths still present for compatibility
- SI-9 and OFI components that are wired in source but used differently across
   live paper, replay, and research workflows

## Further Reading

- See `ARCHITECTURE.md` for the full system architecture and rationale
- See `src/backtest/wfo_optimizer.py` for timeout and study orchestration
- See `src/data/ohlcv.py` for the incremental rolling-state implementation
