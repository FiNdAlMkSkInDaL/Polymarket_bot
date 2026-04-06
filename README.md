# Polymarket Bot

This repository is now documented around the deployed Helsinki VPS shadow
stack, not around the legacy local backtest workflow.

The production truth surface is the VPS. The local workstation is a research,
mirror, and code-authoring environment.

## What Changed

The old operating model leaned too heavily on local CSV exports, frozen local
Parquet snapshots, and ad hoc historical inspection. That model is retired.

The live system now has two canonical moving parts:

1. A continuously written live L2 data lake on the VPS.
2. An hourly shadow pipeline that evaluates the current strategy wrappers on
   that live lake and reports results to Telegram.

Any local CSV, one-off export, or stale historical lake is now diagnostic
material only. It is not the control-plane truth surface.

## Canonical Runtime

The canonical deployment lives on the Helsinki VPS under:

- `/home/botuser/polymarket-bot/data/l2_book_live`
- `/home/botuser/polymarket-bot/shadow_mode`
- `/home/botuser/polymarket-bot/shadow_logs`

The core runtime contract is:

1. `scripts/polymarket-tick-compressor.service` keeps the live lake current.
2. `shadow_sentinel_cron.sh` runs the shadow pipeline at minute `5` of every
   hour.
3. `scripts/refresh_shadow_metadata_cache.py` refreshes the shadow metadata
   cache before any strategy lane runs.
4. The lane order is Scavenger, then Squeeze, then Mid-Tier.
5. `scripts/send_shadow_hourly_telegram.py` publishes the compact operator
   summary after the run.

## Live Data Lake

The live data lake is written by `scripts/live_tick_compressor.py` under the
systemd unit `scripts/polymarket-tick-compressor.service`.

On the VPS, it writes hourly `zstd`-compressed Parquet shards to:

- `/home/botuser/polymarket-bot/data/l2_book_live`

When scanned from the partition root, the lake is an effective ten-column
table:

- `timestamp`
- `market_id`
- `event_id`
- `token_id`
- `best_bid`
- `best_ask`
- `bid_depth`
- `ask_depth`
- `date`
- `hour`

The first eight columns are physically written into the shard. The last two are
materialized from the `date=.../hour=...` partition layout.

Important operational defaults from the deployed service:

- hourly rotation
- `zstd` compression
- `--market-limit 200`
- `--flush-rows 10000`
- `--flush-seconds 300`
- `--universe-refresh-seconds 7200`
- `--heartbeat-seconds 900`
- `--min-free-gb 10`

This live lake is the source data for shadow-mode evaluation. Local mirrors are
useful, but subordinate.

## Shadow Mode

The deployed shadow runner is `vps_shadow_mode/shadow_sentinel_cron.sh`, which
is mirrored onto the VPS as `shadow_mode/shadow_sentinel_cron.sh`.

Each hourly run does the following:

1. Build a fresh timestamped run root under
   `/home/botuser/polymarket-bot/shadow_logs/<UTC-run-stamp>/`.
2. Refresh `clob_arb_baseline_metadata.json` against the active live universe.
3. Emit a metadata coverage status line into `cron_runner.log`.
4. Run the Scavenger wrapper.
5. Run the Squeeze wrapper.
6. Run the Mid-Tier wrapper.
7. Attempt a Telegram summary send, even if an earlier lane failed.

The operator-facing scheduler log is:

- `/home/botuser/polymarket-bot/shadow_logs/cron_runner.log`

The lane wrappers still include `historical` in their filenames for backwards
compatibility, but the deployed mode is live shadow evaluation against the
current UTC day of live lake data.

## Why The Pipeline Is Sequential

The pipeline is deliberately serialized because the VPS is memory constrained.
The host budget is roughly `750 MB`, and each lane is a Polars-heavy analysis
job with its own metadata, scans, joins, and summary outputs.

Running Scavenger, Squeeze, and Mid-Tier in parallel would stack their resident
memory usage and create unnecessary OOM pressure. The design target is to keep
the active hourly step in the low-hundreds-of-megabytes range, roughly around
the `150 MB` band where possible, instead of allowing concurrent peaks.

The result is a simpler contract:

- only one heavy lane is resident at a time
- peak RSS is bounded by the most expensive single step, not the sum of all
  three
- Telegram still runs at the end so the operator always gets a status surface

## Self-Healing Metadata Preflight

`scripts/refresh_shadow_metadata_cache.py` exists to stop the shadow stack from
going blind when the active live universe moves ahead of the cached metadata.

It does four things before the hourly lanes run:

1. Scan today's live lake and discover the active market ids.
2. Fetch fresh Gamma market rows by `condition_ids`.
3. Fetch or seed matching event payloads.
4. Rewrite `clob_arb_baseline_metadata.json` and emit a machine-readable
   summary.

The active shadow cache on the VPS lives at:

- `/home/botuser/polymarket-bot/shadow_mode/artifacts/clob_arb_baseline_metadata.json`

The preflight summary for each run is written into the run root as:

- `metadata_refresh_preflight.json`

Key protections:

- coverage alarm below `95%`
- retry and backoff on Gamma failures
- fallback reuse of previously known good market rows
- fallback reuse or seeding of event payloads
- explicit `success`, `fallback`, `degraded`, `cache_only`, or `unavailable`
  status surfaces

This is the mechanism that prevents the Scavenger from silently evaluating only
an obsolete subset of the live universe.

## Strict T-72h Scavenger Gate

The Scavenger lane is explicitly time-gated.

Its default `--resolution-window-hours` is `72`, and the current strict logic
only admits units where:

- `time_to_resolution_seconds > 0`
- `time_to_resolution_seconds < 72 * 3600`

That means the window is strictly inside `T-72h`, not less-than-or-equal.
Markets exactly on the `72h` boundary are excluded.

This matters operationally because it prevents false target counts near the
window edge and keeps the hourly Scavenger report aligned with the intended
near-resolution strategy surface.

## Telegram Summary

`scripts/send_shadow_hourly_telegram.py` reads the finished run artifacts and
builds a compact operator summary.

It loads:

- `metadata_refresh_preflight.json`
- Scavenger `summary.json`
- Squeeze `batch_summary.json` and `ranking.csv`
- Mid-Tier `execution_summary.json` and `daily_panel.parquet`

It writes:

- `shadow_telegram_summary.json`

It then sends a one-message summary containing:

- metadata coverage and refresh status
- Scavenger targets, accepted orders, and fills
- Squeeze signals, FOK baskets, and completed pairs
- Mid-Tier snapshots, candidate orders, and fills

## Environment And Secrets

For the deployed shadow notifier, the required environment variables are:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

On the VPS, place them in:

- `/home/botuser/polymarket-bot/.env`

The config loader prefers `/dev/shm/secrets/.env` if it exists. If that tmpfs
file is absent, `/home/botuser/polymarket-bot/.env` is the active fallback.

`send_shadow_hourly_telegram.py` also supports optional shadow-specific
overrides:

- `SHADOW_TELEGRAM_BOT_TOKEN`
- `SHADOW_TELEGRAM_CHAT_ID`

If those are unset, it falls back to the standard Telegram variables above.

## Local Workflow

The local machine is still useful, but its role changed.

Use it to:

- sync the VPS lake for inspection
- validate lake freshness
- inspect shadow outputs
- edit and test code before VPS deployment

Do not use it as the canonical truth surface for deployment decisions.

Useful local commands:

```bash
python scripts/sync_lake_from_vps.py --local-root artifacts/l2_parquet_lake_rolling/l2_book --min-date 2026-04-04 --loop --interval-seconds 3600
python scripts/monitor_lake_health.py --sync-state artifacts/l2_parquet_lake_rolling/sync_state.json
```

## Files That Matter Most

Start here if you need to understand the deployed system quickly:

1. `scripts/live_tick_compressor.py`
2. `scripts/polymarket-tick-compressor.service`
3. `vps_shadow_mode/shadow_sentinel_cron.sh`
4. `scripts/refresh_shadow_metadata_cache.py`
5. `scripts/run_scavenger_protocol_historical_sweep.py`
6. `scripts/run_conditional_probability_squeeze_batch.py`
7. `scripts/run_mid_tier_probability_compression_historical_sweep.py`
8. `scripts/send_shadow_hourly_telegram.py`

## Runtime Boundaries

The repository tracks source, tests, docs, and static configuration.

The following are runtime surfaces and should be treated as ephemeral:

- `artifacts/`
- `data/`
- `vps_shadow_mode/` runtime mirrors copied to the VPS
- logs
- generated CSV, Parquet, and JSON summaries

## Read Next

- `ARCHITECTURE.md` for the full system design.
- `OPERATIONS.md` for the day-2 runbook.# Polymarket Bot

This repository now tracks a live-data shadow stack, not a local historical
backtest program.

The old workflow that treated large local lakes as the authoritative truth
surface was retired after the Temporal Gap investigation showed that stale or
incomplete history could drive bad deployment decisions. The canonical system
now lives on the Helsinki VPS and is built around two primitives:

- a continuous live L2 archive written to `data/l2_book_live/`
- an hourly shadow-mode sweep runner that evaluates strategies on live data
  without promoting them directly into live execution

## What Is Canonical

1. `scripts/live_tick_compressor.py` runs on the Helsinki VPS under
   `scripts/polymarket-tick-compressor.service`.
2. It writes hourly, `zstd`-compressed Parquet shards to
   `/home/botuser/polymarket-bot/data/l2_book_live/`.
3. The physical shard schema is eight quote columns:
   - `timestamp`
   - `market_id`
   - `event_id`
   - `token_id`
   - `best_bid`
   - `best_ask`
   - `bid_depth`
   - `ask_depth`
4. When scanned from the partition root, the lake is an effective ten-column
   surface because hive partition columns `date` and `hour` are materialized
   alongside the eight physical fields.
5. The local machine is only a mirror and analysis client. It is not the
   authoritative runtime.

## Shadow Mode

The deployed VPS shadow runner executes at minute `5` of every hour through
`shadow_sentinel_cron.sh`.

That cron entry first refreshes the live-universe metadata cache and checks
cache coverage against today's active live lake. If coverage drops below `95%`,
it emits a visible warning into `cron_runner.log`. It then runs three wrappers
sequentially against the live rolling lake for the current UTC date and sends a
compact Telegram summary after the run completes.

1. Agent 2: `scripts/run_scavenger_protocol_historical_sweep.py`
2. Agent 3: `scripts/run_conditional_probability_squeeze_batch.py`
3. Agent 4: `scripts/run_mid_tier_probability_compression_historical_sweep.py`

Operational rules:

- the wrappers consume live VPS lake data, not a frozen local historical export
- they produce timestamped run folders under
  `/home/botuser/polymarket-bot/shadow_logs/`
- the operator-facing cron log is
  `/home/botuser/polymarket-bot/shadow_logs/cron_runner.log`
- wrapper names still contain `historical` for compatibility, but the deployed
  operating mode is live shadow evaluation

## Local Operator Model

Local research now means mirroring the live lake, validating freshness, and
inspecting shadow outputs. It does not mean treating
`artifacts/l2_parquet_lake_full/` or other historical snapshots as the
production truth surface.

Mirror the VPS live lake into the rolling local cache with:

```bash
python scripts/sync_lake_from_vps.py --local-root artifacts/l2_parquet_lake_rolling/l2_book --min-date 2026-04-04 --loop --interval-seconds 3600
```

Check sync freshness with:

```bash
python scripts/monitor_lake_health.py --sync-state artifacts/l2_parquet_lake_rolling/sync_state.json
```

Inspect the live compressor locally if needed with:

```bash
python scripts/live_tick_compressor.py --output-dir data/l2_book_live --rotation hourly --compression zstd
```

## Production Files

The production data and control-plane contract is intentionally small:

- `scripts/live_tick_compressor.py`
- `scripts/polymarket-tick-compressor.service`
- the deployed `shadow_sentinel_cron.sh`
- `scripts/run_scavenger_protocol_historical_sweep.py`
- `scripts/run_conditional_probability_squeeze_batch.py`
- `scripts/run_mid_tier_probability_compression_historical_sweep.py`
- `scripts/sync_lake_from_vps.py`
- `scripts/monitor_lake_health.py`

Everything else in the repo is supporting code, offline research, archived
experiments, or future work.

## Repository Hygiene

Generated datasets and operator outputs do not belong in Git. The repository
now treats the following as ephemeral runtime state:

- `artifacts/`
- `data/`
- `vps_shadow_mode/`
- parquet shards, CSV exports, logs, runtime state JSON, and generated summary
  JSON

Static configuration that defines the system should stay in Git. Generated
snapshots, receipts, and runtime mirrors should not.

## Minimal VPS Checks

Inspect the live compressor:

```bash
sudo systemctl status polymarket-tick-compressor.service --no-pager
sudo journalctl -u polymarket-tick-compressor.service -n 100 --no-pager
```

Inspect the shadow scheduler log:

```bash
tail -n 200 /home/botuser/polymarket-bot/shadow_logs/cron_runner.log
```

The Telegram formatting is strategy-aware:

- Shield alerts include active target count, staged/intercepted counts,
  planned notional, top categories, and sample questions.
- Sword alerts include executable strip count, grouped event count, launch
  statuses, and a top-strip preview.
- Failure alerts include strategy, failing stage, and a plain-text error
  message.

The tick compressor does not send Telegram alerts. Its operator surface is
journald and the archive directory. Expect `tick_compressor_heartbeat` every
`900` seconds and `tick_parquet_chunk_written` whenever a new archive part is
flushed.

## OS Hardening

The checked-in log rotation policy is `config/polymarket-bot.logrotate`.

It enforces:

- daily rotation;
- 14-day retention;
- compression;
- `copytruncate` for live writers;
- a `100M` max-size threshold.

The repo also includes `config/polymarket-master.service`, which is the current
systemd hardening scaffold carrying `MemoryMax=750M`. The active paper service
templates are separate files under `scripts/`, while the memory-cap pattern is
defined in that checked-in master-service template.

## Read Next

For the full runtime description, read `ARCHITECTURE.md`.