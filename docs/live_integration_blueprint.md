# Live Integration Blueprint

Status: sealed planning artifact only. This document authorizes no runtime changes and modifies no production code.

Scope constraints:
- Zero bot.py mutations in this phase.
- No new runtime dependencies.
- No executable Python is introduced here.
- All references below describe the future integration session only.

## Section 1: Startup Sequence

### Existing startup anchors

The future live orchestrator insertion is anchored to the current startup layout in [src/bot.py](../src/bot.py#L336), [src/bot.py](../src/bot.py#L358), [src/bot.py](../src/bot.py#L371), [src/bot.py](../src/bot.py#L388), and [src/bot.py](../src/bot.py#L440).

Current layout:
- Runtime placeholders begin at [src/bot.py](../src/bot.py#L336) with `_heartbeat`, `_order_poller`, and `_stop_loss_monitor` declarations.
- Heartbeat is constructed at [src/bot.py](../src/bot.py#L358).
- Order polling is constructed at [src/bot.py](../src/bot.py#L371).
- Stop-loss monitor construction begins at [src/bot.py](../src/bot.py#L388).
- Background task registration begins at [src/bot.py](../src/bot.py#L440).

### live_hyperparameters.json injection map

The repo already injects live hyperparameters centrally through configuration, not through bot.py directly:
- Default path resolution is defined at [src/core/live_hyperparameters.py](../src/core/live_hyperparameters.py#L17).
- JSON loading occurs at [src/core/live_hyperparameters.py](../src/core/live_hyperparameters.py#L53).
- Strategy overrides are applied at [src/core/live_hyperparameters.py](../src/core/live_hyperparameters.py#L62).
- The singleton settings object applies those overrides at [src/core/config.py](../src/core/config.py#L904).

Integration rule:
- The future bot.py integration must treat `settings.strategy` as already hydrated from `live_hyperparameters.json` before `TradingBot.start()` runs.
- No secondary JSON read is to be added inside bot.py.
- Live orchestrator construction must consume already-resolved values from `settings.strategy` plus the runtime `session_id` injected by the deployment/session layer.

### LiveOrchestratorConfig construction plan

Planned future startup additions:
- Add two new bot fields immediately after the runtime placeholders at [src/bot.py](../src/bot.py#L336):
  - `self._live_orchestrator`
  - `self._orchestrator_health_monitor`
- Add one deployment-scoped `session_id` acquisition step before orchestrator construction. bot.py currently has no session identifier of its own, so this value must be injected from the deployment launcher or runtime session manager, not fabricated ad hoc inside signal code.

Construction contract for `LiveOrchestratorConfig`:
- `orchestrator_config`: derived from current orchestrator/runtime knobs already reflected in strategy settings.
- `bus_config`: single shared bus configuration for all live alpha paths.
- `guard_config`: single shared dispatch guard configuration for all live alpha paths.
- `ctf_adapter_config`: reused paper adapter config until the dedicated live adapter boundary is finalized.
- `si9_adapter_config`: preserved in current paper-mode semantics for SI-9 while live dispatcher wiring lands beside it.
- `ofi_bridge_config`: reused bridge config with deployment-phase-derived dispatcher mode.
- `ctf_peg_config`: current detector configuration.
- `si9_cluster_configs`: derived from the existing cluster manager output currently initialized around [src/bot.py](../src/bot.py#L472).
- `unwind_config`: the future generalized unwind executor’s config payload.
- `deployment_phase`: mapped from deployment runtime.
- `session_id`: injected deployment/session identifier.
- `max_position_release_failures`: supplied from deployment policy.
- `heartbeat_interval_ms`: must match the cadence of the orchestrator tick task introduced beside the current task list at [src/bot.py](../src/bot.py#L440).

### Exact future invocation point for build_live_orchestrator()

Planned insertion point:
- Insert the `build_live_orchestrator()` call immediately after the `OrderStatusPoller` block at [src/bot.py](../src/bot.py#L371) and before the `StopLossMonitor` setup at [src/bot.py](../src/bot.py#L388).
- Operationally, this is the new block at planned anchor `bot.py:376`.

Reason for this placement:
- `self._book_trackers` and `self.positions` already exist by this point from construction in [src/bot.py](../src/bot.py#L167).
- `self._on_clob_fill` is already established as the live fill ingress callback at [src/bot.py](../src/bot.py#L371) and [src/bot.py](../src/bot.py#L1017).
- The orchestrator must exist before downstream signal consumers start running, but after core runtime dependencies are already initialized.

### Shared-instance handoff discipline

Non-negotiable ownership model:
- bot.py must not construct independent `SignalCoordinationBus`, `DispatchGuard`, or `PriorityDispatcher` instances for each alpha path.
- bot.py must invoke `build_live_orchestrator()` exactly once during startup.
- The factory will internally create one shared bus, one shared guard, and one shared dispatcher.
- Those shared instances are then injected by the factory into:
  - `CtfPaperAdapter`
  - `Si9PaperAdapter`
  - `OfiSignalBridge`
- bot.py may keep references only through `self._live_orchestrator.bus`, `self._live_orchestrator.guard`, and `self._live_orchestrator.dispatcher` for observability and diagnostics. It must not construct parallel copies.

Resulting ownership graph:
- bot.py owns a single `MultiSignalOrchestrator`.
- `MultiSignalOrchestrator` owns the shared coordination objects.
- The alpha adapters receive those exact shared instances transitively from the factory and nowhere else.

## Section 2: BBO Event Routing & Signal Ingress

### Existing ingress anchors

Current top-of-book flow is already split across two runtime channels:
- Non-L2 book updates are applied in [src/bot.py](../src/bot.py#L1189) via `_process_book_updates()`.
- Multi-core BBO notifications are consumed in [src/bot.py](../src/bot.py#L1237) via `_consume_bbo_events()`.
- L2 callback fan-out currently enters at [src/bot.py](../src/bot.py#L1345).
- Lightweight `OrderbookTracker` callback fan-out currently enters at [src/bot.py](../src/bot.py#L1363).
- Market tracker creation occurs in `_wire_market()` at [src/bot.py](../src/bot.py#L587) and specifically the tracker wiring block at [src/bot.py](../src/bot.py#L624).

### OrderbookTracker to OrderbookBestBidProvider flow

Future routing plan:
- The source of truth for top-of-book remains the existing per-asset tracker map in `self._book_trackers` from [src/bot.py](../src/bot.py#L167).
- The live orchestrator’s `OrderbookBestBidProvider` will wrap the injected `OrderbookTracker` instance passed into `build_live_orchestrator()`.
- No additional market data cache is to be introduced.
- bot.py will continue to update trackers through existing methods in [src/bot.py](../src/bot.py#L1189) and [src/bot.py](../src/bot.py#L1237); the provider will read the same tracker in O(1).

Practical consequence:
- Market data ownership remains with the current tracker map.
- The orchestrator reads existing tracker state; it does not subscribe to a separate feed or maintain duplicate depth history.

### OFI and contagion routing plan

Current signal entry points:
- OFI evaluation starts at [src/bot.py](../src/bot.py#L1381).
- Contagion evaluation starts at [src/bot.py](../src/bot.py#L1454).
- OFI currently routes toward `_on_panic_signal()` at [src/bot.py](../src/bot.py#L1774), which later reaches `PositionManager.open_rpe_position()` at [src/bot.py](../src/bot.py#L2133).

Future routing split:
- `_on_l2_bbo_change_inner()` at [src/bot.py](../src/bot.py#L1345) remains the primary high-frequency BBO fan-out site.
- `_on_orderbook_bbo_change()` at [src/bot.py](../src/bot.py#L1363) remains the fallback/non-L2 fan-out site.
- OFI path:
  - Detector generation remains in `_evaluate_ofi_momentum()`.
  - Once a BUY signal passes existing market/tradeability gates, dispatch must be handed to `self._live_orchestrator.on_ofi_signal(...)` instead of falling directly into the legacy RPE open path.
  - The orchestrator’s `OfiSignalBridge` then routes that directional taker-style signal through the shared dispatcher.
- Contagion path:
  - Detection remains in `_evaluate_contagion_arb()`.
  - The resulting signal objects will be normalized into the future venue-adapter boundary and routed through the same shared dispatcher/guard/bus graph, not through ad hoc order placement.

### O(1) fan-out guarantee

This integration must preserve bounded runtime behavior on each BBO event:
- No new per-tick append-only lists may be introduced.
- No new per-tick historical queues may be introduced in bot.py.
- `OrderbookBestBidProvider` must remain a read-only wrapper over the existing tracker object.
- `OrchestratorHealthMonitor` keeps only fixed-size counters and timestamps.
- Existing bounded structures remain acceptable:
  - `_recent_contagion_matrix` is already a bounded deque at [src/bot.py](../src/bot.py#L211).
- Signal fan-out remains a constant-count operation:
  - stop-loss callback
  - maker monitor tick
  - OFI evaluation
  - contagion evaluation
  - future orchestrator gate check

Operational rule:
- One incoming BBO event may schedule or execute a fixed number of downstream handlers, but may not allocate an unbounded event buffer in bot.py.

## Section 3: Health Monitor & Lifecycle Routing

### Health gate placement

Primary gate sites:
- `_evaluate_ofi_momentum()` at [src/bot.py](../src/bot.py#L1381)
- `_evaluate_contagion_arb()` at [src/bot.py](../src/bot.py#L1454)
- SI-9 combo loop bootstrap at [src/bot.py](../src/bot.py#L472) and execution loop at [src/bot.py](../src/bot.py#L4199)

Integration rule:
- `OrchestratorHealthMonitor.is_safe_to_trade(current_timestamp_ms)` must be checked before any new alpha signal is converted into execution intent.
- If the health monitor returns `False`, signal generation must short-circuit immediately.
- Short-circuit means:
  - do not call the live dispatcher
  - do not call `PositionManager.open_rpe_position()`
  - do not reserve new SI-9 positions
  - do not enqueue new execution work

Recommended future insertion sequence:
- Add a single fast health gate near the top of `_evaluate_ofi_momentum()` before detector output is acted upon.
- Add the same fast health gate near the top of `_evaluate_contagion_arb()` before signals are iterated.
- Add the same gate to the SI-9 combo execution loop before cluster execution begins.

### Orchestrator tick and heartbeat alignment

The health monitor depends on an injected timestamp, not wall clock reads inside the monitor.

Future runtime task plan:
- Add an orchestrator tick coroutine beside the current background task list at [src/bot.py](../src/bot.py#L440).
- Its cadence must match `LiveOrchestratorConfig.heartbeat_interval_ms`.
- Each iteration will:
  - compute `current_timestamp_ms`
  - call `self._live_orchestrator.on_tick(current_timestamp_ms)`
  - call `self._orchestrator_health_monitor.check(current_timestamp_ms)`
  - emit observability/logging only; no secondary state cache is required

This keeps heartbeat expectations aligned across:
- orchestrator tick cadence
- health monitor heartbeat validation
- the existing operational heartbeat reporter at [src/bot.py](../src/bot.py#L3827)

### PositionManagerLifecycle wiring

Current lifecycle/cleanup anchors:
- Live fill ingress enters at [src/bot.py](../src/bot.py#L1017).
- Position cleanup loop runs at [src/bot.py](../src/bot.py#L4159).
- Cleanup execution currently occurs at [src/bot.py](../src/bot.py#L4165).

Future lifecycle bridge model:
- `build_live_orchestrator()` already wraps the injected `PositionManager` with `PositionManagerLifecycle`.
- bot.py must continue owning the canonical `self.positions`, but new orchestrator-driven cluster reservation/release decisions must flow through the lifecycle wrapper inside `self._live_orchestrator`.
- bot.py must not create a second `PositionManager` or a parallel release ledger.

### Release-failure interception plan

The monitor counter must increment only on actual cleanup/release failures.

Future interception points:
- Orchestrator tick task:
  - Wrap `self._live_orchestrator.on_tick(current_timestamp_ms)` in a narrow `try/except`.
  - If unwind completion triggers a position release path that fails, call `self._orchestrator_health_monitor.record_position_release_failure()`.
  - After a successful release cycle, call `reset_release_failure_count()`.
- Periodic cleanup loop at [src/bot.py](../src/bot.py#L4159):
  - Wrap the `self.positions.cleanup_closed()` call at [src/bot.py](../src/bot.py#L4165).
  - On exception, increment `record_position_release_failure()` and leave trading halted if the threshold is reached.
  - On success, call `reset_release_failure_count()`.
- Fill-driven lifecycle updates at [src/bot.py](../src/bot.py#L1017):
  - If a future integration step ties order fills to orchestrator release completion, the same failure accounting rule applies there.

Reason for dual interception:
- Some release failures will occur during orchestrator unwind completion.
- Some release failures will occur during background cleanup/position reconciliation.
- The health monitor must see both classes and keep a single consecutive-failure count.

### Health reporting alignment

The existing health reporter at [src/bot.py](../src/bot.py#L3827) and per-market health log at [src/bot.py](../src/bot.py#L4086) remain the correct output surfaces.

Future extension only:
- Append orchestrator snapshot health and health monitor status into the existing health payload.
- Do not create a second competing health file.
- Keep the reporter as the single long-run telemetry sink.

## Section 4: Legacy Strategy Containment

### Pure market maker containment

Current PureMarketMaker anchors:
- Initialization happens at [src/bot.py](../src/bot.py#L423).
- Task enablement is guarded at [src/bot.py](../src/bot.py#L460).

Containment rule for the integration session:
- `PureMarketMaker` remains in the repo.
- It must be feature-flagged off for this phase.
- The deployment target map already runs Top 25 / high-volume markets with PURE_MM disabled; the blueprint preserves that posture.
- No orchestrator integration work may depend on maker quoting from this path.

### SI-9 containment

Current SI-9 anchors:
- SI-9 enablement begins at [src/bot.py](../src/bot.py#L472).
- Detector construction occurs at [src/bot.py](../src/bot.py#L479).
- Combo loop task creation occurs at [src/bot.py](../src/bot.py#L485).
- The combo loop itself begins at [src/bot.py](../src/bot.py#L4199).

Containment rule:
- SI-9 remains a maker-first state machine over mutually exclusive market clusters.
- Its current paper deployment parameters are preserved during the live wiring phase.
- The live orchestrator integration must wire alongside SI-9, not force SI-9 into live venue execution before Agent 3’s generalized unwind executor is green.
- In practice:
  - keep SI-9 detector and cluster formation behavior unchanged
  - preserve its current paper-mode adapter semantics
  - allow shared health/coordination wiring to exist without changing cluster execution policy yet

### OFI momentum containment and live dispatch bridge

Current OFI anchors:
- Evaluation begins at [src/bot.py](../src/bot.py#L1381).
- Existing panic/RPE funnel starts at [src/bot.py](../src/bot.py#L1774).
- One current taker-style open path reaches `open_rpe_position()` at [src/bot.py](../src/bot.py#L2133).

Containment and bridge rule:
- OFI momentum remains the directional microstructure alpha path built around aggressive taker entry.
- During integration, its signal generation logic stays in bot.py, but execution handoff moves to the live orchestrator.
- The handoff target is the orchestrator’s `OfiSignalBridge`, which already sits behind the shared dispatcher.
- That dispatcher then binds to the future `VenueAdapter` boundary delivered by Agent 1.

Execution contract:
- OFI signal creation remains local.
- Guarding, slot coordination, and dispatch mode selection become orchestrator responsibilities.
- bot.py stops issuing direct OFI-driven execution through the legacy RPE path once the orchestrator handoff is live.

## Planned edit matrix for the future integration session

| Anchor | Future action | Constraint |
| --- | --- | --- |
| [src/bot.py](../src/bot.py#L336) | Add orchestrator and health-monitor fields | No second runtime graph |
| [src/bot.py](../src/bot.py#L371) | Keep live fill poller construction | `_on_clob_fill` remains live fill ingress |
| `bot.py:376` planned insert | Construct `LiveOrchestratorConfig`, call `build_live_orchestrator()`, construct `OrchestratorHealthMonitor` | Use injected `session_id`; no duplicate bus/guard/dispatcher |
| [src/bot.py](../src/bot.py#L440) | Add orchestrator tick task | Cadence must equal `heartbeat_interval_ms` |
| [src/bot.py](../src/bot.py#L460) | Keep `PureMarketMaker` behind feature flag, disabled in deployment | No maker-path dependency for this phase |
| [src/bot.py](../src/bot.py#L1189) | Preserve tracker update ownership | No duplicate market-data cache |
| [src/bot.py](../src/bot.py#L1345) | Keep BBO fan-out entry | Add health-gated orchestrator handoff downstream only |
| [src/bot.py](../src/bot.py#L1363) | Keep non-L2 BBO fan-out entry | Same constant-cost routing rule |
| [src/bot.py](../src/bot.py#L1381) | Add `is_safe_to_trade()` short-circuit before OFI execution handoff | Immediate return on unsafe state |
| [src/bot.py](../src/bot.py#L1454) | Add `is_safe_to_trade()` short-circuit before contagion execution handoff | Immediate return on unsafe state |
| [src/bot.py](../src/bot.py#L1017) | Preserve fill callback ownership; optionally bridge release accounting | No duplicate fill-processing pipeline |
| [src/bot.py](../src/bot.py#L3827) | Extend health payload with orchestrator health | No second health file |
| [src/bot.py](../src/bot.py#L4159) | Wrap cleanup loop with release-failure accounting | Consecutive failure counter only |

## Final integration guardrails

The eventual bot.py integration session must satisfy all of the following:
- Exactly one live orchestrator instance per bot process.
- Exactly one shared bus, one shared guard, and one shared dispatcher for all orchestrated alpha adapters.
- No bot.py direct dependency on agent-internal adapters beyond the public orchestrator surface.
- No unbounded per-tick allocations for BBO fan-out.
- Health gate must fail closed.
- Release-failure accounting must be monotonic until a successful cleanup/reset occurs.
- Pure MM remains off for this phase.
- SI-9 stays paper-semantic while live execution wiring lands around it.
- OFI becomes the first directional path to traverse the live dispatcher boundary.
