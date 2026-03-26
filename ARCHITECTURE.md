# Architecture

This document describes the current checked-in architecture at HEAD. It is a
runtime and control-flow document, not a product brief. Where the codebase
contains both legacy and current paths, this file distinguishes between code
that exists, code that is wired, and code that actively governs live trading.

## Current Runtime Shape

The live system is centered on three layers:

1. `src/bot.py` owns process startup, websocket consumption, order-book
   updates, legacy strategy loops, and background health tasks.
2. `src/execution/live_execution_boundary.py` is the hard boundary between the
   Python runtime and the Polymarket venue. In LIVE mode it builds the
   transport, signer, nonce manager, wallet balance provider, and OFI exit
   router. In PAPER mode it intentionally returns a no-op boundary.
3. `src/execution/multi_signal_orchestrator.py` coordinates the shared live
   execution substrate: dispatch guard, priority dispatcher, OFI entry bridge,
   CTF adapter, SI-9 adapter, unwind escalation, position lifecycle, load
   shedding, observability, and wallet balance cache.

The architecture is therefore not a single-strategy bot. It is a shared
execution surface with multiple alpha lanes attached to it.

## Live Graph

The current live graph is:

1. Market data enters through the bot websocket and L2 websocket tasks.
2. `OrderbookTracker` instances update best bid/ask, depth, toxicity, and
   snapshot timestamps in O(1) time.
3. Alpha lanes react to those updates in parallel:
   - OFI Momentum emits live entry intents into `MultiSignalOrchestrator`.
   - Contagion arb evaluates on BBO updates and routes accepted signals into
     the RPE fast-strike position path.
   - Legacy and optional loops such as `PureMarketMaker` can still run, but
     they are not part of the orchestrator control plane.
4. The orchestrator sends live orders through `PriorityDispatcher`, which is
   the single bottleneck for venue dispatch, margin checks, MEV routing, and
   normalized receipts.
5. `LiveWalletBalanceProvider` polls the venue in the background, maintains an
   in-memory USDC cache, and feeds margin state back into both the orchestrator
   and `PositionManager`.
6. `OrchestratorHealthMonitor` snapshots the orchestrator and exposes a
   fail-closed `is_safe_to_trade(current_timestamp_ms)` gate used by the live
   event loops.

Two important consequences follow from this design:

- The runtime is event-driven. Health, exits, and signal conversion are driven
  by market-data updates and orchestrator ticks rather than slow periodic
  polling.
- The venue-facing path is centralized. Even when the bot evaluates multiple
  alpha families, network submission is serialized through the same dispatcher
  and balance cache.

## MultiSignalOrchestrator

`MultiSignalOrchestrator` is the live control graph, not just a convenience
wrapper.

It owns these shared components:

- `SignalCoordinationBus` for cross-strategy slot management and coordination.
- `DispatchGuard` for centralized suppression accounting and dispatch checks.
- `PriorityDispatcher` for routing a `PriorityOrderContext` into paper, dry
  run, or live execution.
- `CtfPaperAdapter`, `OfiSignalBridge`, and `Si9PaperAdapter` as the strategy
  adapters that convert higher-level signals into dispatch intents.
- `UnwindExecutor` and escalation policy for hanging-leg and recovery flows.
- `PositionLifecycleInterface` for reservation, confirmation, and release.
- `OrchestratorLoadShedder` for ranked-market admission control.
- `LiveWalletBalanceProvider` and `OfiExitRouter` when the deployment phase is
  LIVE.

The orchestrator currently exposes four operational surfaces:

1. `on_ctf_signal(...)`
2. `on_ofi_signal(...)`
3. `on_si9_signal(...)`
4. `on_tick(...)`

`on_tick(...)` handles unwind escalation, surrender decisions, and lifecycle
release. Snapshot generation is also centralized there, which is what lets the
health monitor reason over one coherent state object instead of a loose set of
side channels.

## OFI Momentum Lane

### Entry Path

The current OFI live path is the most deeply integrated orchestrator lane.

On each eligible NO-side OFI momentum signal in `src/bot.py`, the bot:

1. checks `is_safe_to_trade(...)` when the live OFI runtime is active,
2. constructs an `OfiEntrySignal` using `Decimal` prices, sizes, conviction,
   and timestamps,
3. sends that signal into `MultiSignalOrchestrator.on_ofi_signal(...)`,
4. lets `OfiSignalBridge` allocate side locks and convert the signal into a
   `PriorityOrderContext`, and
5. dispatches through `PriorityDispatcher`.

This is the current live OFI entry contract. In PAPER mode the bot still has a
legacy `_on_panic_signal(...)` path, but that is not the live venue path.

### Exit Path

The OFI exit model is intentionally severed from the old visible generic-exit
flow.

Current live behavior is:

1. Entry fills leave the position with locally owned exit state rather than a
   generic resting take-profit order.
2. `MultiSignalOrchestrator.evaluate_ofi_exit(...)` uses
   `OfiLocalExitMonitor` when book trackers are available, or falls back to a
   local target/stop/time-stop decision derived from the stored `Decimal`
   fields.
3. `MultiSignalOrchestrator.route_ofi_exit(...)` hands the decision to
   `OfiExitRouter`.
4. `OfiExitRouter` submits one of three exit forms:
   - immediate taker exit for target or stop hits,
   - passive limit exit for time-stop handling,
   - promotion from passive to taker when the passive wait budget or slippage
     budget is breached.

### Hazard-Style Brackets And Liquidity Vacuums

The execution model remains a hidden, locally monitored OFI bracket system.
The stored `drawn_tp`, `drawn_stop`, and `drawn_time_ms` fields are the control
surface for that bracket. Time-stop exits are not fired blindly:

- `OfiLocalExitMonitor` can suppress time-stop action during a liquidity
  vacuum.
- The replay and tracing utilities use the same vacuum concept through the
  replay order book EWMA baselines.
- `scripts/trace_ofi_fold.py` explicitly measures the OFI drop funnel,
  including TVI penalty suppressions, depth-vacuum suppressions, cooldowns,
  and size-floor vetoes.

Operationally, OFI exits are now strategy-owned microstructure decisions, not
just generic sell orders.

## Contagion Lane

The contagion architecture has two distinct pieces at HEAD: a live signal lane
and an archive qualification lane.

### Live Contagion Evaluation

Live contagion is evaluated on every relevant BBO update in `src/bot.py`.

The bot:

1. checks `is_safe_to_trade(...)` before evaluating contagion in live mode,
2. computes current YES mid-price and toxicity from live YES and NO books,
3. calls `ContagionArbDetector.evaluate_market(...)`, and
4. routes accepted non-shadow signals into `_on_contagion_signal(...)`, which
   then opens an RPE fast-strike position when spread, freshness, cooldown,
   stop-loss cooldown, and ensemble-risk gates all pass.

This is parallel to OFI Momentum, but it is not currently dispatched through a
dedicated orchestrator adapter. The orchestrator protects the lane via shared
health gating; the actual contagion execution still enters through the
position-manager fast-strike path.

### Archive Qualification: UniverseBuilder + ContagionValidator

The current contagion research and curation stack lives in moved modules:

- `src/data/universe_builder.py`
- `src/tools/contagion_validator.py`
- `scripts/cli_universe_builder.py`

`UniverseBuilder` builds candidate leader-lagger clusters from archived YES
series. It enforces:

- minimum empirical correlation,
- minimum observed events per day,
- minimum archive days,
- maximum lagger freshness,
- optional causal ordering.

`ContagionValidator` then replays archived data with an instrumented
`ContagionReplayAdapter` and records:

- cross-market pairs evaluated,
- causal gate pass rate,
- legacy sync pass rate,
- signals fired,
- fills executed,
- dominant suppressor,
- lagger-age percentiles,
- optional per-event telemetry.

The accepted archive baseline used by `scripts/cli_universe_builder.py` is:

- `max_lagger_age_ms = 600000`
- `max_causal_lag_ms = 600000`
- `max_leader_age_ms = 5000`

That 600,000 ms freshness contract is not a doc convention. It is encoded in
the CLI defaults and reinforced by the validator sweep artifacts.

## SI-9 And CTF Adapters

The orchestrator also carries CTF and SI-9 execution contracts.

### CTF

`CtfPaperAdapter` consumes fee-aware merge signals and emits receipts backed by
strict `Decimal` manifests. `CtfExecutionManifest`, `CtfLegManifest`, and
`CtfExecutionReceipt` reject non-`Decimal` monetary fields and enforce a
well-formed two-leg execution contract.

### SI-9

`MultiSignalOrchestrator.on_si9_signal(...)` supports cluster reservation,
execution via `Si9PaperAdapter`, hanging-leg unwind manifests, and escalation
through `on_tick(...)`.

That contract is present and tested in HEAD. The bot also still maintains its
legacy combo-arb loop, so documentation must not claim that every SI-9 action
already routes through the orchestrator in production.

## LiveExecutionBoundary

`build_live_execution_boundary(...)` is the hard runtime split between paper
and live.

Outside LIVE deployment it returns:

- `venue_adapter = None`
- `wallet_balance_provider = None`
- `ofi_exit_router = None`

In LIVE deployment it constructs:

1. an `AiohttpClobTransport` running on a dedicated async transport loop,
2. a `PolymarketClobAdapter`,
3. a `ClobNonceManager`,
4. a `ClobSigner` bound to the Polymarket CTF exchange EIP-712 domain,
5. a `LiveWalletBalanceProvider` tracking `USDC`, and
6. an `OfiExitRouter` with OFI-scoped client-order IDs.

That is the current live boundary. No live order path should bypass it.

## PriorityDispatcher And The Network Bottleneck

`PriorityDispatcher` is where all venue-bound traffic converges.

It performs the following in order:

1. optional `DispatchGuard` check,
2. MEV routing via `MevExecutionRouter.plan_priority_sequence(...)`,
3. envelope serialization,
4. mode-specific execution,
5. receipt normalization into a `DispatchReceipt`.

In LIVE mode it also performs an O(1) margin gate before the order reaches the
venue:

$$
\text{required margin} = \text{order price} \times \text{effective size}
$$

The dispatcher compares that against:

$$
\text{available margin} = \text{LiveWalletBalanceProvider.get\_available\_margin("USDC")}
$$

If available margin is insufficient, the order is rejected locally with
`guard_reason = "INSUFFICIENT_MARGIN"` and never reaches the exchange.

This is the current network bottleneck by design. Multiple alpha lanes can
race to produce intent, but only one dispatcher contract is allowed to convert
intent into venue traffic.

## O(1) LiveWalletBalanceProvider Margin Gate

`LiveWalletBalanceProvider` maintains a cached `Decimal` balance per tracked
asset and updates it from a background poll loop.

Important properties:

- `get_available_margin(...)` is O(1) dictionary access.
- The provider rejects negative or non-finite balances.
- The bot registers a balance update callback so USDC changes are reflected in
  `PositionManager` immediately.
- Poll failures caused by rate limits, timeouts, or transport circuit opens are
  logged but do not silently fabricate balances.

This is what makes the dispatcher-side margin check fast enough to sit directly
on the live path.

## Strict Decimal-Only Execution Boundary

HEAD now enforces a strict `Decimal` execution contract across the live venue
path.

Examples:

- `ClobSigner` requires decimal strings for venue payload prices and sizes,
  quantizes them to micro units, and rejects non-string or non-finite numeric
  inputs.
- execution manifests such as `CtfExecutionManifest` and related receipts
  reject non-`Decimal` values for prices, sizes, fees, and PnL fields.
- live wallet balance polling and dispatch receipts store balances, prices,
  sizes, and remaining size as `Decimal` values.

The rule is:

- strategy code may still originate from float-heavy market data,
- but once a signal is translated into an execution intent, the venue-facing
  path is `Decimal` only until the final payload boundary.

This is a correctness constraint, not stylistic preference.

## Fail-Closed Health Layer

`src/execution/orchestrator_health_monitor.py` wraps the orchestrator in a
conservative health gate.

It marks the runtime unsafe when any of the following occurs:

- orchestrator snapshot health is RED,
- snapshot age exceeds the stale-snapshot threshold,
- heartbeat gap breaches the configured interval budget,
- consecutive release failures reach the configured halt threshold.

The live bot checks `is_safe_to_trade(...)` before:

- converting OFI momentum signals into live orchestrator entries,
- evaluating contagion arb on live BBO updates,
- running the combo-arb loop in live mode.

This is the current fail-closed posture. Health does not merely annotate the
runtime; it actively suppresses signal conversion.

## Deployment And Observability

The production-facing PAPER deployment is run under `systemd` using:

- `scripts/polymarket-bot.service`
- `scripts/install_paper_service.sh`

The service model is:

- tmpfs secret material prepared in `ExecStartPre`,
- stale shared-memory segments removed before startup,
- bot started with `python -m src.cli run --env PAPER`,
- stdout and stderr shipped to `journald`,
- automatic restart enabled.

Offline latency profiling is performed outside the running process by exporting
`journalctl` output and feeding it into `scripts/profile_latency_logs.py`.

That split is intentional: the bot emits journal-backed telemetry online, and
profiling happens offline against exported logs.

## Module Location Corrections

Several architectural modules moved relative to older documentation. The
current locations are:

- `src/execution/live_wallet_balance.py`
- `src/tools/contagion_validator.py`
- `src/data/universe_builder.py`

Any document that still refers to `src/core/live_wallet_balance.py`,
`src/trading/contagion_validator.py`, or `src/trading/universe_builder.py` is
describing a pre-HEAD layout.