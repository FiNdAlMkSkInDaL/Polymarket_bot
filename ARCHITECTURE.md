# Architecture

This document explains the current checked-in architecture as of March 24,
2026, with an explicit separation between live deployment posture and research
or replay-only capabilities.

## Architectural Position

The repository is a hybrid trading system, but it is no longer correct to
describe it as a passive market-making repo with a few optional side modules.

The current strategic center is:

1. SI-9 combinatorial arbitrage for live paper deployment
2. OFI momentum taker architecture for directional microstructure replay and
   optimization
3. An optimized O(1) OHLCV engine that makes the replay path operationally
   viable

The PURE_MM passive maker stack still exists in source, but it is now a legacy
research and replay path rather than the approved live strategy.

## Architecture Separation

### Live Ops

Live Ops should be read as the approved deployed posture.

Current posture:

- Deployment environment is PAPER
- SI-9 is the only active live strategy
- PURE_MM remains disabled on the high-volume target map

### Backtest Ops

Backtest Ops should be read as the research and optimization plane.

Current posture:

- OFI momentum is fully represented in replay
- PURE_MM is still available for replay and comparative study
- WFO uses child-process isolation and hard trial timeouts
- Logging must be throttled operationally to avoid I/O choking

This separation is intentional and should survive future refactors.

## The Strategic Pivot

### PURE_MM Is Retired Operationally

PURE_MM failed in two different ways:

1. Long-tail markets produced too few passive fills
2. Top 25 markets produced fills in the wrong regime: toxic, informed, or
   adverse-selection-dominated flow

The high-volume universe failure is reflected in the current repo narrative and
recent OOS WFO conclusions. The long-tail failure is an execution-reality
problem: passive quoting without enough organic taker flow creates operational
motion with little or no actual business throughput.

The result is a clear desk-level conclusion:

- PURE_MM remains in source
- PURE_MM may still be replayed
- PURE_MM should not be treated as the approved live strategy

### Why The New Direction Looks Different

The replacement architecture does not try to force one execution style onto all
books.

- SI-9 targets explicit structural mispricings in mutually exclusive clusters
- OFI momentum accepts that some edges are only monetizable with taker speed

That is the deeper reason the repo now contains both a combinatorial arb engine
and an aggressive momentum path.

## Strategy Portfolio

### PURE_MM Passive Maker

Primary code locations:

- `src/strategies/pure_market_maker.py`
- `src/backtest/strategy.py` via `PureMarketMakerReplayAdapter`
- `src/core/config.py` pure maker parameters

Status:

- Present in code
- Configurable
- Still instantiated by the bot when enabled
- Not approved as the current live production path

Operational interpretation:

- Keep it for research and historical regression
- Do not write new documentation that frames it as the flagship live strategy

### SI-9 Combinatorial Arbitrage

Primary code locations:

- `src/signals/combinatorial_arb.py`
- `src/trading/position_manager.py`
- `src/bot.py`
- `src/data/arb_clusters.py`

Status:

- Approved live strategy
- Deployed in PAPER mode
- Structured around explicit multi-leg safety constraints

### OFI Momentum Taker

Primary code locations:

- `src/signals/ofi_momentum.py`
- `src/execution/momentum_taker.py`
- `src/trading/position_manager.py`
- `src/backtest/strategy.py`

Status:

- Implemented and wired
- Part of the replay and optimization stack
- Important for forward strategy development
- Not documented here as the sole live strategy

## SI-9 Architecture

### Signal Detection

SI-9 scans mutually exclusive event clusters and evaluates whether the sum of
YES best bids implies a structural arbitrage.

Detection logic:

- Read the YES-side best bid and ask across all legs of a cluster
- Compute $\sum bids$
- Require the cluster to clear the configured margin threshold
- Size all legs to the same share count to preserve the hedge invariant

This is not Kelly sizing. It is share-count pegging. That distinction is
critical. Unequal leg sizes would destroy the Dutch-book structure.

### Guardrails

SI-9 contains explicit guardrails that should remain documented because they
encode hard-won market structure lessons.

#### Ghost-Town Filter

The cluster is rejected when:

- $\sum bids < 0.85$
- edge exceeds $\$0.15$

This prevents the detector from confusing dead books with genuine liquid arb.

#### Liquidity Gate

Each leg must clear a minimum bid-depth threshold before the cluster is even
considered viable.

#### Exposure Limits

SI-9 enforces:

- max concurrent combos
- max total combo exposure
- max collateral per combo
- wallet-risk budget checks

These checks live in the execution path, not just the detector, which is the
correct architecture. Signal validity and executable safety are not the same
thing.

### Maker-First Execution

SI-9 is deliberately maker-first.

Execution sequence:

1. Rank legs by bottleneck characteristics
2. Work the hardest leg passively first
3. Hold the remaining legs as pending taker legs
4. Sweep the takers only after the maker leg fills

The point is to avoid crossing the full combo before the hard leg proves the
arb is actually executable.

### Hanging-Leg State Machine

This is the most important SI-9 safety component.

If the combo partially fills:

1. Re-evaluate missing legs using current best ask and spread
2. If emergency taker completion is still affordable, cross and finish the arb
3. Otherwise dump the already filled legs and flatten the book

The invariant is simple:

- never leave partial combo exposure sitting in the portfolio as naked
  directional risk

That guardrail exists because multi-leg risk is qualitatively different from a
single-position timeout.

## OFI Momentum Architecture

### Detector

The OFI detector tracks rolling top-of-book volume imbalance:

$$
VI = \frac{Q_{bid} - Q_{ask}}{Q_{bid} + Q_{ask}}
$$

It maintains a rolling millisecond window and triggers when the rolling VI
crosses a configurable threshold.

Core properties:

- window-based, not bar-close-based
- built from top-of-book queue pressure
- directional output: BUY or SELL
- intended for fast microstructure momentum, not slow mean reversion

### Execution Philosophy

OFI momentum uses aggressive taker entry.

That is deliberate. If the edge comes from short-lived order-flow imbalance,
waiting passively is usually equivalent to declining the trade.

The execution helper places a spread-crossing buy at the current best ask and,
in PAPER mode, simulates the fill immediately for deterministic testing.

### Hard Brackets

OFI momentum positions use fixed post-entry brackets:

- take profit: 3.0%
- stop-loss: 1.5%
- time-stop: 300 seconds

The 1.5% stop-loss is not a soft suggestion. It is a hard bailout bracket.
The 300-second time-stop exists because stale momentum is inventory, not edge.

### Timeout Integration

The position manager treats OFI momentum differently from generic exits.

- When an OFI momentum position exceeds its hold window, the exit reason is
  `time_stop`
- Non-OFI positions use the generic timeout path

This distinction should remain in the docs because the desk needs to know that
momentum decays are handled as strategy-specific bracket exits, not generic
housekeeping.

## OHLCV And Replay Engine Optimization

### Why The Refactor Was Necessary

Momentum WFO was previously bottlenecked by OHLCV recomputation cost.

The fix was architectural, not cosmetic:

- stop rebuilding rolling arrays on every tick
- maintain rolling statistics incrementally
- avoid duplicate aggregation inside the replay loop

### O(1) Incremental State

The current OHLCV aggregator maintains rolling state explicitly:

- rolling volume sum
- rolling VWAP volume sum
- rolling VWAP sum
- rolling return sum
- rolling squared-return sum
- short-window return moments
- EWMA volatility state
- downside EWMA volatility state

When a bar enters or expires, the aggregator updates those moments directly.
That turns the hot path from repeated window recomputation into O(1) state
maintenance.

### BotReplayAdapter Owns Its Aggregation

BotReplayAdapter now declares:

- `self_aggregates_trades = True`

That declaration matters because the backtest engine checks it and skips its own
duplicate OHLCV pass when the strategy already manages aggregation internally.

Architectural consequence:

- live-like strategy replay remains faithful
- the backtest engine avoids paying for the same bar work twice

### Why This Must Stay Documented

Three months from now, this will otherwise look like an arbitrary optimization.
It is not. It is the reason momentum WFO can finish within the operational
timeout budget.

## Backtest Engine

The replay engine is synchronous and event-driven.

Core behavior:

1. replay events in timestamp order
2. activate orders after simulated latency
3. process real L2 book events when available
4. synthesize a BBO in trade-only mode when necessary
5. pass trade and book events into the strategy
6. record fills and equity telemetry

This design is simpler than an async backtest loop and is easier to reason
about deterministically.

## WFO Architecture

### Study Model

WFO uses:

- rolling or anchored windows
- Optuna-based parameter search
- child-process backtest execution
- hard timeout enforcement per trial

### Timeout Wrapper

Every trial is executed in a spawned child process. The parent waits up to
60 seconds. If the child is still alive, the process is terminated and the
trial is marked as a timeout.

This is critical operational hygiene. A single pathological trial must not
freeze a large study.

### Logging Rule

Before any serious WFO run, set:

```powershell
$env:BOT_DISABLE_FILE_LOGGING='1'
```

Reason:

- rotating JSONL file output is unnecessary during dense optimization runs
- child-process studies amplify I/O contention
- the logger already supports stdout-only operation when this flag is set

### Generic WFO Workflow

```powershell
$env:BOT_DISABLE_FILE_LOGGING='1'
python -m src.cli wfo `
  --data-dir data/vps_march2026 `
  --train-days 35 `
  --test-days 7 `
  --step-days 7 `
  --anchored `
  --embargo-days 1 `
  --n-trials 500 `
  --max-workers -1 `
  --strategy-adapter bot_replay
```

### OFI Momentum Top 25 Workflow

```powershell
$env:BOT_DISABLE_FILE_LOGGING='1'
python wfo_ofi_momentum.py `
  --train-days 35 `
  --test-days 7 `
  --step-days 7 `
  --embargo-days 1 `
  --n-trials 500 `
  --max-workers -1 `
  --trial-timeout-s 60
```

## Deployment State

### What Is Checked In

The checked-in service configuration launches the bot in PAPER mode.

### What Should Be Documented Carefully

This repository has already produced runtime-versus-source mismatches in prior
analysis. For that reason, architecture docs should not casually merge:

- code that exists
- code that is enabled by default
- code that is approved for live use
- code that was observed in a historical runtime artifact

The approved documentation rule is:

- state what the code supports
- state what the live desk is actually running
- do not assume those are identical unless there is direct evidence

## Guardrails Worth Keeping In The Docs

These are precisely the kinds of details future readers will otherwise rip out
without understanding why they exist.

### PURE_MM Retirement Guardrail

- PURE_MM remains in source for research only until the quoting model proves it
  can survive both long-tail starvation and Top 25 toxic flow

### SI-9 Ghost-Town Guardrail

- reject low-$\sum bids$ and absurd-edge clusters because dead books masquerade
  as fake arb

### OFI Bracket Guardrail

- 1.5% stop-loss and 300-second time-stop are structural constraints, not mere
  optimization parameters

### WFO Throughput Guardrail

- disable file logging during WFO and keep the 60-second timeout wrapper, or
  optimization throughput collapses under avoidable overhead

### Replay Fidelity Guardrail

- preserve `self_aggregates_trades = True` behavior for BotReplayAdapter so the
  engine does not double-charge OHLCV work

## Bottom Line

The repository is now built around a clear division of labor:

- SI-9 for live paper deployment
- OFI momentum for aggressive directional microstructure execution and replay
- O(1) OHLCV state as the enabling infrastructure
- WFO as a controlled, timeout-bounded research process

If future documentation collapses this back into "a Polymarket market maker,"
it will be wrong.