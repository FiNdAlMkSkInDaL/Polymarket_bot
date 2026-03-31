# Polymarket Bot

This repository is no longer a broad multi-strategy playground pretending that
every checked-in subsystem is equally active.

The current runtime has been cut back to a Level 0 execution kernel with one
live directional candidate, one shadow incubator, a unified dispatcher gate,
and a journal-first persistence contract.

If you need the control-flow truth, read `ARCHITECTURE.md`. This file is the
operator summary.

## Current Reality

The bot is currently organized around:

- `src/bot.py` as the runtime owner,
- `src/execution/live_execution_boundary.py` as the only venue-facing
  boundary,
- `src/execution/multi_signal_orchestrator.py` as the shared dispatcher and
  observability control plane,
- `src/execution/priority_dispatcher.py` as the single pre-dispatch bottleneck,
- `src/execution/orchestrator_health_monitor.py` as the fail-closed health
  gate,
- `src/monitoring/trade_store.py` as the journal-first persistence authority.

The old documentation that described a large adapter graph is stale. The
current kernel is smaller and stricter.

## Lane States

Default lane states in `src/core/config.py` are:

- Panic: `LIVE`
- Contagion: `SHADOW`
- OFI Momentum: `OFF`
- RPE: `OFF`
- Oracle: `OFF`
- PureMarketMaker: `OFF`
- SI-9 combo: `OFF`
- Cross-market: `OFF`
- SI-10 Bayesian: `OFF`
- Reward sidecar: `OFF` by default

Operationally:

- Panic is the sole live candidate.
- Contagion is the shadow incubator.
- Standard OFI, Oracle, Pure MM, and SI-9 are frozen.
- Reward posting infrastructure exists, but it only runs when explicitly
  enabled.

## Unified Admission

Live entry admission is now collapsed into the shared dispatcher path.

`PriorityDispatcher` runs:

1. bound pre-dispatch gates,
2. the shared `DispatchGuard`,
3. envelope serialization,
4. mode-specific execution,
5. receipt normalization.

This is the front door for dispatcher-backed entry.

The practical consequence is that new live risk is no longer supposed to sneak
through strategy-specific routing tricks.

## Health States

`OrchestratorHealthMonitor` exposes three states:

- `GREEN`: normal operation.
- `YELLOW`: degraded-risk mode.
- `RED`: hard stop.

The important operating rule is:

- YELLOW still allows position management and flattening.
- YELLOW blocks new live entry intents.
- RED halts the runtime harder.

If you are debugging a missing entry, do not treat YELLOW as harmless.

## Persistence

`src/monitoring/trade_store.py` now uses a journal-first contract.

For both live closed trades and shadow trades it:

1. writes `trade_persistence_journal` first,
2. writes the ledger row second inside an immediate transaction,
3. marks the journal row `RECORDED` or `FAILED`.

The persistence model is intentionally loud. Silent drops are no longer an
acceptable failure mode.

The same module also provides WAL-safe measurement snapshots using:

- SQLite backup,
- optional JSONL journal capture,
- a manifest JSON that records snapshot paths and accounting status.

## Reward Poster Sidecar

The repository now contains a reward-posting subsystem:

- `src/rewards/models.py`
- `src/rewards/reward_selector.py`
- `src/rewards/reward_poster_sidecar.py`
- `src/execution/reward_poster_adapter.py`
- `src/rewards/reward_shadow_metrics.py`
- `scripts/report_reward_shadow.py`

What it is:

- a maker-only reward quote manager that uses the shared dispatcher,
- a shadow-measurement pipeline for reward attribution,
- a reporting surface over persisted reward shadow rows.

What it is not:

- a separate venue adapter,
- a separate dispatch stack,
- a reason to add schema bloat to `shadow_trades`.

Reward attribution is packed into `payload_json.extra_payload` inside the
trade-persistence journal. That is how the sidecar records fields like
`reward_to_competition` and `estimated_reward_capture_usd` without mutating the
SQL schema.

## Contagion

Contagion is not the old privileged fast-strike side path anymore.

Current truth:

- it evaluates on BBO updates in `src/bot.py`,
- it is currently run as the shadow incubator,
- it clears dispatcher-backed admission before opening shadow or live paths,
- the live branch still finishes through the existing position machinery after
  that admission check.

So the correct mental model is: contagion is shadow-first and pays the shared
gate tax.

## Deployment

The VPS-managed PAPER flow is still based on `systemd`.

Typical deploy shape on the server:

```bash
cd /home/botuser/polymarket-bot
git pull origin main
./scripts/install_paper_service.sh
```

Useful commands:

```bash
sudo systemctl status polymarket-bot.service --no-pager
sudo journalctl -u polymarket-bot.service -n 100 --no-pager
sudo journalctl -u polymarket-bot.service -f
```

## Measurement Commands

Reward shadow report:

```bash
python scripts/report_reward_shadow.py --db logs/trades.db
```

Shadow cohort study:

```bash
python scripts/shadow_cohort_study.py --db logs/trades.db
```

Trade cohort study:

```bash
python scripts/trade_cohort_study.py --db logs/trades.db
```

Those scripts are meant to consume frozen, WAL-safe measurement data rather
than optimistic live copies.

## Where To Look First

If you are trying to understand the current bot, start here in this order:

1. `src/core/config.py` for lane states.
2. `src/bot.py` for runtime wiring.
3. `src/execution/priority_dispatcher.py` for live-entry admission.
4. `src/execution/orchestrator_health_monitor.py` for degraded-risk behavior.
5. `src/monitoring/trade_store.py` for persistence and measurement.
6. `src/rewards/reward_poster_sidecar.py` if the reward lane is in scope.

Anything older than that mental model is probably lying to you.