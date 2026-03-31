# Architecture

This document describes the current checked-in runtime shape that actively
governs trading. It is not a product brief. It is a control-surface document.
If a component is not mentioned here, assume it is either frozen, legacy, or
not on the hot path.

## Level 0 Reset

The repository is no longer organized around a sprawling multi-adapter
execution graph.

The current live kernel is deliberately smaller:

1. `src/bot.py` owns process startup, websocket ingestion, BBO-driven event
  handling, lane wiring, and the background loops.
2. `src/execution/live_execution_boundary.py` is the only venue-facing runtime
  boundary. In LIVE mode it builds the transport, signer, nonce manager, venue
  adapter, wallet-balance provider, and OFI exit router. In PAPER mode it is
  intentionally hollow.
3. `src/execution/multi_signal_orchestrator.py` is now a thin shared control
  plane around `PriorityDispatcher`, guard observability, OFI accounting, and
  the reward poster adapter.
4. `src/execution/priority_dispatcher.py` is the unified pre-dispatch choke
  point. Admission logic is collapsed here through bound pre-dispatch gates
  plus the shared `DispatchGuard`.
5. `src/execution/orchestrator_health_monitor.py` is the fail-closed health
  gate that feeds the dispatcher and higher-level bot loops.
6. `src/monitoring/trade_store.py` is the durable persistence surface for both
  closed trades and shadow trades.

This is the new Level 0 kernel. The old story about the orchestrator owning
slot buses, side locks, and strategy-specific paper adapters is obsolete.

## Deleted Mental Model

Do not model the current runtime as an orchestrator that owns:

- `SignalCoordinationBus`
- `OfiSignalBridge`
- `CtfPaperAdapter`
- `Si9PaperAdapter`
- generic cross-strategy slot management
- OFI side-lock routing

That is not the architecture that actively governs the bot now.

Current orchestrator behavior is narrower:

- validate source enablement,
- validate market admission through the load shedder when present,
- expose a dispatcher-backed OFI path,
- expose a dispatcher-backed reward adapter,
- emit a coherent snapshot for health monitoring.

## Execution Kernel

### Bot

`src/bot.py` is still the runtime owner.

It starts:

- websocket and L2 consumers,
- trade and book processors,
- stop-loss, timeout, stats, cleanup, health, and summary loops,
- the orchestrator tick loop,
- the wallet-balance poll loop when a live balance provider exists,
- the reward sidecar when the reward lane is active,
- optional frozen loops only when their lane states are explicitly enabled.

The bot also owns the high-frequency BBO callbacks that drive:

- shadow tracking,
- OFI exit evaluation,
- stop-loss monitoring,
- reward-sidecar quote maintenance,
- contagion evaluation.

### LiveExecutionBoundary

`src/execution/live_execution_boundary.py` remains the hard split between local
strategy logic and venue side effects.

In PAPER mode it returns no live transport:

- `venue_adapter = None`
- `wallet_balance_provider = None`
- `ofi_exit_router = None`

In LIVE mode it builds the only objects that are allowed to touch the venue:

1. transport,
2. adapter,
3. nonce manager,
4. signer,
5. cached wallet-balance provider,
6. OFI exit router.

No lane should invent its own venue-facing stack outside this boundary.

### MultiSignalOrchestrator

`src/execution/multi_signal_orchestrator.py` is now intentionally thin.

It owns:

- `DispatchGuard`
- `PriorityDispatcher`
- `GuardObservabilityPanel`
- `OfiPaperLedger`
- `RewardPosterAdapter`
- optional load shedding and wallet-balance plumbing
- OFI exit router wiring when LIVE

It does not own strategy-specific adapter trees for CTF or SI-9 anymore, and
it does not act as a generic strategy switchboard.

Its active operational surfaces are:

1. `on_ofi_signal(...)`
2. `on_reward_intent(...)`
3. `on_tick(...)`
4. `orchestrator_snapshot(...)`
5. `dispatch_guard_reason(...)`

The orchestrator also binds itself as a dispatcher pre-gate. That means source
enablement and market admission are enforced before the dispatcher ever tries
to serialize or submit an order.

### PriorityDispatcher

`src/execution/priority_dispatcher.py` is the single admission and dispatch
bottleneck.

The important reset is architectural, not cosmetic: pre-dispatch logic is no
longer fragmented across strategy adapters. It is unified here.

Current dispatch order is:

1. run all bound pre-dispatch gates,
2. run the shared `DispatchGuard` if enabled,
3. build and serialize the MEV envelope,
4. execute in `paper`, `dry_run`, or `live` mode,
5. normalize the result into a `DispatchReceipt`.

This is where live-entry admission now collapses.

The dispatcher is also the shared gate used by:

- OFI live entries,
- contagion dispatcher-backed admission checks,
- reward-sidecar quote submission.

### OrchestratorHealthMonitor

`src/execution/orchestrator_health_monitor.py` is the hard live-trading gate.

Its health states are:

- `GREEN`: normal operation.
- `YELLOW`: degraded-risk mode.
- `RED`: hard stop.

The important current behavior is:

- `GREEN` allows normal entry and position management.
- `YELLOW` allows position management, exits, and flattening, but it blocks new
  live entry intents.
- `RED` is a hard stop.

That degraded-risk posture is enforced two ways:

1. the monitor reports `allows_position_management=True` and
  `allows_new_panic_entries=False` outside GREEN,
2. the monitor binds itself into `PriorityDispatcher` and rejects new
  dispatcher-backed entries for `OFI`, `CONTAGION`, and `REWARD` while
  YELLOW.

This matters operationally. YELLOW is not a cosmetic warning state. It is the
mode where the runtime is allowed to clean up risk but not allowed to open new
risk.

## Persistence Contract

`src/monitoring/trade_store.py` is the current persistence authority.

The contract is journal first, ledger second.

For both `trades` and `shadow_trades` it now does the following:

1. write a `trade_persistence_journal` row first,
2. open an immediate transaction for the ledger write,
3. write the ledger row,
4. mark the journal row `RECORDED` on success,
5. mark the journal row `FAILED` on error and log loudly.

The journal is not decorative. It records:

- `journal_key`
- `ledger_kind`
- `trade_id`
- `signal_source`
- timing fields
- `payload_json`
- `ledger_state`
- `last_error`

The only acceptable steady-state ledger outcome is `RECORDED`.

The runtime now explicitly distinguishes:

- `PENDING`
- `RECORDED`
- `FAILED`

This replaced the old silent-drop failure model. Persistence failures are now
deliberately loud.

### WAL-Safe Snapshotting

The measurement path is also stricter.

`create_wal_safe_remeasurement_snapshot(...)` does not rely on a naive file
copy. It performs:

1. a passive WAL checkpoint,
2. an SQLite backup into a frozen snapshot database,
3. optional JSONL export of `trade_persistence_journal`,
4. a JSON manifest recording snapshot paths and accounting status.

That snapshot manifest is the measurement contract for offline cohort work.

### Shadow Extra Payload

`record_shadow_trade(...)` now accepts `extra_payload`.

That payload is merged into `payload_json` only, under the reserved top-level
key `extra_payload`. The SQL schema for `shadow_trades` is unchanged.

This is intentional. Reward-sidecar attribution data belongs in the journaled
payload, not in schema sprawl.

## Lane States

The default lane posture in `src/core/config.py` is now explicit.

Current defaults:

- Panic: `LIVE`
- Contagion: `SHADOW`
- OFI Momentum: `OFF`
- RPE: `OFF`
- Oracle: `OFF`
- PureMarketMaker: `OFF`
- SI-9 combo: `OFF`
- Cross-market: `OFF`
- SI-10 Bayesian: `OFF`
- Reward sidecar: `OFF` by default, present in code, activated only when the
  reward lane is enabled

Operationally, that means:

- directional Panic is the sole live candidate,
- Contagion is the active shadow incubator,
- Pure MM, SI-9 combo, standard OFI, Oracle, and the rest are frozen,
- reward posting exists as infrastructure but is not a default always-on lane.

## Panic Lane

Panic is the only current live directional candidate.

Its entry path is still bot-managed through `PositionManager`, not through a
dedicated dispatcher venue path. But it is governed by the tightened Level 0
controls:

- lane-state gating,
- tradeability checks,
- cooldowns,
- book-quality gates,
- ensemble-risk vetoes,
- orchestrator health gating at the bot layer.

In short: if you are debugging live directional behavior, start with Panic.

## Contagion Lane

Contagion is now the shadow incubator.

The key current truth is:

- the detector still evaluates on BBO updates inside `src/bot.py`,
- shadow contagion uses dispatcher-backed admission in PAPER mode before the
  shadow tracker opens a counterfactual position,
- live contagion no longer gets to bypass the unified pre-dispatch gate.

The live branch still finishes through the existing fast-strike
`PositionManager` path after admission, but the old fast-strike bypass mental
model is wrong. The lane now pays the same dispatcher tax at the front door.

That means contagion is no longer a privileged legacy side path. It is a
shadow-first incubator that must clear the unified gate.

## Reward Poster Sidecar

The reward poster is a new major subsystem.

### Runtime Role

`src/rewards/reward_poster_sidecar.py` is not a separate execution stack. It is
a shared-kernel client.

It:

- selects admitted reward markets,
- maintains working quotes,
- reprices or cancels stale quotes,
- records one-sided fills into local inventory,
- persists reward-shadow rows through `trade_store`.

It does not own its own venue adapter.

### Shared Dispatcher Contract

The sidecar uses `RewardPosterIntent` and `RewardExecutionHints` to build a
`PriorityOrderContext` with `signal_source="REWARD"`.

Those reward execution hints are strict:

- `post_only=True`
- `time_in_force="GTC"`
- `liquidity_intent="MAKER_REWARD"`
- `allow_taker_escalation=False`

That is the contract. Reward posting is maker-only and dispatcher-mediated.

The actual path is:

1. sidecar creates `RewardPosterIntent`,
2. `RewardPosterAdapter` converts it into a shared `PriorityOrderContext`,
3. `PriorityDispatcher` runs pre-gates and guard checks,
4. the returned `DispatchReceipt` is mapped into `RewardQuoteState`.

There is no separate reward venue stack hiding off to the side.

### Reward Shadow Measurement

The sidecar also persists shadow measurement rows via
`bot._persist_reward_shadow_trade(...)` and `trade_store.record_shadow_trade(...)`.

Reward attribution data is packed into `payload_json.extra_payload`, including
fields such as:

- `reward_to_competition`
- `reward_daily_usd`
- `reward_max_spread_cents`
- `queue_depth_ahead_usd`
- `queue_residency_seconds`
- `fill_occurred`
- `estimated_reward_capture_usd`
- `estimated_net_edge_usd`
- `quote_id`
- `quote_reason`

This is deliberate. The sidecar extends measurement via journal payloads
without bloating the SQL schema.

### Reporting Surface

`scripts/report_reward_shadow.py` reads persisted reward-shadow rows and groups
them by:

- market,
- signal source,
- reward-to-competition bucket,
- fill occurred,
- emergency flatten.

It emits JSON and Markdown from existing shadow rows. It is a measurement tool,
not an execution tool.

## What Actively Governs Live Trading

If you need the ruthless current truth, it is this:

1. Panic is the only live directional candidate.
2. Contagion is being incubated in shadow and must clear the unified
  pre-dispatch gate.
3. Reward posting exists as shared-kernel infrastructure, not as a special
  venue stack.
4. The dispatcher is the choke point.
5. YELLOW means manage risk only. Do not add risk.
6. Trade persistence is journal-first and loud on failure.

Everything else is secondary, frozen, or legacy until explicitly reactivated.

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