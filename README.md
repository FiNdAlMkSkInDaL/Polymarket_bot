# Polymarket Bot

Polymarket Bot is a multi-strategy trading stack built around a shared live
execution surface, archive-backed research tools, and a paper deployment
workflow managed by `systemd`.

The important current-state distinction is:

- the bot still contains legacy strategies and compatibility code,
- the live execution architecture is now organized around
  `MultiSignalOrchestrator`, `LiveExecutionBoundary`, `PriorityDispatcher`,
  `LiveWalletBalanceProvider`, and a strict `Decimal` venue path,
- the contagion research plane is now driven by `UniverseBuilder` and
  `ContagionValidator`.

See `ARCHITECTURE.md` for the control-flow source of truth.

## Current Runtime Summary

- OFI Momentum live entries go through `MultiSignalOrchestrator`.
- Live OFI exits are locally owned and routed through `OfiExitRouter` rather
  than generic visible resting exits.
- Contagion arb runs in parallel on BBO updates and routes accepted signals
  into the fast-strike RPE path, while being protected by orchestrator health.
- `PriorityDispatcher` is the network bottleneck for live venue submission.
- `LiveWalletBalanceProvider` supplies an O(1) cached USDC margin gate before
  live dispatch.
- `OrchestratorHealthMonitor.is_safe_to_trade(...)` fail-closes the live OFI,
  contagion, and combo-arb loops when orchestrator health degrades.

## Deployment

### CI/CD Workflow

The current deployment workflow for the VPS-managed PAPER service is based on
`scripts/install_paper_service.sh`.

Typical flow on the server:

```bash
cd /home/botuser/polymarket-bot
git pull origin main
./scripts/install_paper_service.sh
```

What the installer does:

1. stops ad-hoc `python -m src.cli run --env PAPER` processes,
2. copies `scripts/polymarket-bot.service` into `/etc/systemd/system/`,
3. reloads `systemd`,
4. enables and restarts `polymarket-bot.service`,
5. runs `scripts/audit_forward_data.py` against the current UTC day, and
6. prints a service status plus the latest journal tail.

The service itself:

- runs as `botuser`,
- uses `/home/botuser/polymarket-bot/.venv/bin/python -m src.cli run --env PAPER`,
- decrypts `.env.age` into `/dev/shm/secrets/.env` at start,
- cleans `/dev/shm/pmb_*` before launch,
- writes logs to `journald`,
- restarts automatically.

Useful commands after deployment:

```bash
sudo systemctl status polymarket-bot.service --no-pager
sudo journalctl -u polymarket-bot.service -n 100 --no-pager
sudo journalctl -u polymarket-bot.service -f
```

### Offline Telemetry Profiling

The runtime logs to `journald`. Offline latency profiling is done by exporting
those logs and analyzing them with `scripts/profile_latency_logs.py`.

Example:

```bash
sudo journalctl -u polymarket-bot.service --since "2026-03-26 00:00:00" --no-pager > exported-journal.txt
python scripts/profile_latency_logs.py exported-journal.txt --output latency_hist.png
```

The profiler parses latency-like events, prints summary percentiles, and emits
a histogram with a configurable threshold marker.

## Research And Data Tools

### Universe Builder / Contagion Validator

The current universe curation tool is `scripts/cli_universe_builder.py`.

It combines:

- `src/data/universe_builder.py` for archive-backed leader-lagger cluster
  construction,
- `src/tools/contagion_validator.py` for replay validation and causal lag
  telemetry.

Default contagion freshness posture:

- `max_lagger_age_ms = 600000`
- `max_causal_lag_ms = 600000`
- `max_leader_age_ms = 5000`

Example:

```bash
python scripts/cli_universe_builder.py \
  --clusters-json data/si10_relationships_march2026.json \
  --archive-path data/vps_march2026 \
  --market-map data/market_map.json \
  --output-json artifacts/universe_builder/report.json \
  --require-causal-ordering
```

What it does:

1. loads candidate clusters,
2. builds an archive-backed recommended cluster,
3. validates that cluster with `ContagionValidator`, and
4. prints both the builder funnel and validator funnel.

Optional flags worth knowing:

- `--max-events` for faster iteration on large archives,
- `--emit-per-event-telemetry` to include pair-level causal telemetry,
- `--min-correlation`, `--min-events-per-day`, and `--min-archive-days` to
  tighten builder admission.

### OFI Fold Trace Tool

The current OFI drop-funnel tracer is `scripts/trace_ofi_fold.py`.

It replays a walk-forward fold and reports how many candidate OFI events are
lost to:

- thresholding,
- TVI penalties,
- depth-vacuum suppression,
- cooldown,
- meta vetoes,
- size-floor vetoes.

Example:

```bash
python scripts/trace_ofi_fold.py \
  --data-dir data/vps_march2026 \
  --market-configs data/market_map_top25.json \
  --fold-index 2 \
  --train-days 35 \
  --test-days 7 \
  --step-days 7 \
  --embargo-days 1 \
  --max-markets 25
```

Notes:

- `--data-dir` may point at the archive root or directly at `raw_ticks`; the
  script normalizes either form.
- If the selected fold window contains zero L2 events, the tool now fails
  loudly with a fatal error instead of silently returning an empty funnel.

## Repository Layout

```text
src/
  bot.py                     Live runtime and background loops
  core/                      Config, guards, process management
  data/                      Archive readers, market discovery, universe builder
  execution/                 Orchestrator, boundary, dispatcher, venue plumbing
  signals/                   OFI, contagion, SI-9, SI-10, and related detectors
  strategies/                Legacy standalone strategy implementations
  tools/                     Replay validators and analysis utilities
scripts/
  install_paper_service.sh   Current PAPER deployment entry point
  profile_latency_logs.py    Offline journald latency profiler
  cli_universe_builder.py    Universe builder + contagion validator CLI
  trace_ofi_fold.py          OFI walk-forward funnel tracer
```

## Notes On Legacy Paths

`PureMarketMaker` still exists and can still be started when
`settings.strategy.pure_mm_enabled` is true. That does not make it the primary
architecture.

At HEAD:

- `PureMarketMaker` is a legacy optional loop,
- live OFI entries and exits are controlled by the orchestrator and live
  execution boundary,
- contagion live execution is still a bot-managed fast-strike lane protected by
  orchestrator health rather than a dedicated orchestrator adapter.