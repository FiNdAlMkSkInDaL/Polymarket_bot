# Helsinki Shadow Architecture

This document describes the production architecture that is actually deployed.
It is written for the next engineer who needs to understand how the Helsinki
VPS ingests data, evaluates the shadow lanes, self-heals metadata, and reports
status to Telegram.

## Architectural Principles

1. The Helsinki VPS is the source of truth.
2. The live L2 lake is the canonical market-data surface.
3. The shadow pipeline evaluates live data without directly promoting trades to
   live execution.
4. The pipeline is serialized to control memory pressure on a small VPS.
5. Metadata freshness is enforced before any lane can run.
6. The operator should always get an end-of-run status surface, even when a
   lane fails.

## System Overview

The deployed system has three practical planes.

### 1. Live Data Plane

This plane captures live L2 market state and writes it into a compact Parquet
lake.

Owned by:

- `scripts/live_tick_compressor.py`
- `scripts/polymarket-tick-compressor.service`

Runtime path:

- `/home/botuser/polymarket-bot/data/l2_book_live`

### 2. Shadow Evaluation Plane

This plane runs the current research lanes on the live lake every hour.

Owned by:

- `shadow_mode/shadow_sentinel_cron.sh`
- `scripts/refresh_shadow_metadata_cache.py`
- `scripts/run_scavenger_protocol_historical_sweep.py`
- `scripts/run_conditional_probability_squeeze_batch.py`
- `scripts/run_mid_tier_probability_compression_historical_sweep.py`
- `scripts/send_shadow_hourly_telegram.py`

Runtime paths:

- `/home/botuser/polymarket-bot/shadow_mode`
- `/home/botuser/polymarket-bot/shadow_logs`

### 3. Operator And Mirror Plane

This plane exists for inspection, diagnostics, and code changes.

Owned by:

- the local workstation checkout
- `scripts/sync_lake_from_vps.py`
- `scripts/monitor_lake_health.py`

Important rule: local mirrors do not override VPS reality.

## Why Local CSVs Are No Longer Canonical

The repo used to tolerate a workflow where local CSV exports, local Parquet
snapshots, or manually assembled historical lakes were treated as the decision
surface. That model broke down once the production universe and the cached
metadata started drifting.

The modern contract is different:

- the VPS writes the live lake continuously
- the shadow lanes read the live lake directly on the VPS
- local CSVs and frozen historical snapshots are only for diagnostics

This change removes two failure modes:

1. stale local data silently driving deployment decisions
2. metadata snapshots that only cover an outdated subset of live markets

## Live Data Plane In Detail

`scripts/live_tick_compressor.py` runs under
`scripts/polymarket-tick-compressor.service`.

The unit file configures:

- `WorkingDirectory=/home/botuser/polymarket-bot`
- virtualenv Python from `/home/botuser/polymarket-bot/.venv/bin/python`
- output root pre-creation for `data/l2_book_live` and `data/l2_book_live/_state`
- `Restart=on-failure`
- `RestartSec=15`
- `Nice=10`
- best-effort IO scheduling
- `LimitNOFILE=65536`
- journald stdout and stderr

The compressor writes hourly `zstd`-compressed shards under:

- `/home/botuser/polymarket-bot/data/l2_book_live/date=YYYY-MM-DD/hour=HH/`

The effective scanned schema is ten columns:

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

The first eight fields are written directly. `date` and `hour` come from hive
partition discovery.

The service defaults matter because they define the data freshness and storage
contract:

- `--market-limit 200`
- `--flush-rows 10000`
- `--flush-seconds 300`
- `--rotation hourly`
- `--compression zstd`
- `--universe-refresh-seconds 7200`
- `--heartbeat-seconds 900`
- `--min-free-gb 10`

The `_state` directory exists for handoff and restart metadata, which makes the
writer easier to reason about across restarts.

## Shadow Pipeline In Detail

The shadow scheduler is a shell wrapper:

- `shadow_mode/shadow_sentinel_cron.sh`

The checked-in deployment mirror lives at:

- `vps_shadow_mode/shadow_sentinel_cron.sh`

It is invoked by cron at minute `5` of every hour and creates a run root:

- `/home/botuser/polymarket-bot/shadow_logs/<UTC-run-stamp>`

The pipeline order is fixed:

1. metadata preflight
2. Scavenger
3. Squeeze
4. Mid-Tier
5. Telegram summary

The wrapper uses a `run_step` helper that logs either:

- `SHADOW_STEP_OK step=<name>`
- `WARNING SHADOW_STEP_FAILED step=<name> exit_code=<n>`

This is deliberate. A lane failure marks the pipeline degraded, but does not
abort the rest of the run. Telegram still executes afterward so the operator
gets a single status surface for the hour.

## Sequential Execution And Memory Management

The shadow lanes are run sequentially because the VPS is small.

The engineering constraint is a host budget of roughly `750 MB` RAM with other
background processes already present. Each shadow lane performs Polars scans,
joins, aggregations, and artifact generation. If the lanes overlap, resident
memory stacks instead of recycling.

The scheduler therefore enforces one heavy process at a time.

Why this matters:

- concurrent Scavenger, Squeeze, and Mid-Tier runs would multiply RSS instead
  of capping it at the largest single lane
- the intended active-process target is the low-hundreds-of-MB range, roughly
  around `150 MB` where feasible, rather than a multi-process spike
- serialized execution leaves room for the compressor, shell, filesystem cache,
  and Telegram summary without courting OOM conditions

In practice the exact per-lane RSS varies with the live universe size and the
number of active partitions. The important architectural rule is not the exact
number; it is that only one memory-hungry lane is resident at any moment.

## Metadata Preflight And Self-Healing

The preflight script is:

- `scripts/refresh_shadow_metadata_cache.py`

Its job is to stop the shadow stack from going blind when the metadata cache no
longer covers the markets that are actually active in the live lake.

### Inputs

- live lake root
- existing shadow metadata cache
- today's UTC date partition by default

### Outputs

- refreshed cache at
  `/home/botuser/polymarket-bot/shadow_mode/artifacts/clob_arb_baseline_metadata.json`
- per-run summary JSON at
  `/home/botuser/polymarket-bot/shadow_logs/<run-stamp>/metadata_refresh_preflight.json`

### Refresh Algorithm

1. Scan today's live lake to build the active market catalog.
2. Fetch Gamma market rows in batches by `condition_ids`.
3. Build a condition-id keyed row map.
4. Reuse old market rows if Gamma fails for a subset of active ids.
5. Seed events from market payloads and then fetch full Gamma event rows.
6. Reuse old event rows when fresh event fetches fail.
7. Rebuild `markets_by_token` and `events_by_id`.
8. Write the cache atomically.
9. Emit a machine-readable summary and log lines.

### Status Model

The script can emit these refresh states:

- `success`: all active market ids were refreshed cleanly
- `fallback`: no ids are missing, but some rows were reused from the old cache
- `degraded`: some active market ids are still missing after refresh and reuse
- `cache_only`: refresh failed, but a previous cache file exists
- `unavailable`: refresh failed and no cache exists

### Coverage Alarm

The hard operator threshold is `95%` coverage.

If coverage drops below that threshold, the script emits:

- `WARNING SHADOW_METADATA_COVERAGE_LOW ...`

Otherwise it emits:

- `SHADOW_METADATA_COVERAGE_OK ...`

Every run also prints a `SHADOW_METADATA_PREFLIGHT` line that includes refresh
status, active market count, covered market count, coverage percentage, fetched
rows, reused rows, event count, and cache path.

This is the main blind-spot defense in the current architecture.

## Scavenger Architecture

The Scavenger lane runs through:

- `scripts/run_scavenger_protocol_historical_sweep.py`

The VPS deployment mirror sets its defaults to the live runtime:

- lake root: `/home/botuser/polymarket-bot/data/l2_book_live`
- metadata path:
  `/home/botuser/polymarket-bot/shadow_mode/artifacts/clob_arb_baseline_metadata.json`
- output root:
  `/home/botuser/polymarket-bot/shadow_logs/.../scavenger_protocol_historical_sweep`

Although the filename still says `historical_sweep`, the VPS copy is configured
for the live rolling lake.

### Strict T-72h Gate

The Scavenger time window is not approximate. It is strict.

The default argument is:

- `--resolution-window-hours 72`

The current candidate logic only admits rows where time to resolution is inside
the open interval:

- `0 < time_to_resolution_seconds < 72 * 3600`

Consequences:

- already-resolved markets are excluded
- markets beyond `72` hours are excluded
- markets exactly at the `72h` boundary are excluded

This same strict window now governs both candidate generation and the
price-distribution diagnostics, which prevents the reporting mismatch that had
previously shown false in-window opportunity counts.

## Squeeze Architecture

The Squeeze lane runs through:

- `scripts/run_conditional_probability_squeeze_batch.py`

The VPS deployment mirror defaults to:

- input root: `/home/botuser/polymarket-bot/data/l2_book_live`
- pairs config: `/home/botuser/polymarket-bot/shadow_mode/config/squeeze_pairs.json`
- output dir:
  `/home/botuser/polymarket-bot/shadow_logs/.../conditional_probability_squeeze_batch`

It produces:

- `batch_summary.json`
- `ranking.csv`

These outputs are later consumed by the Telegram summary script for pair counts,
signal counts, and FOK basket totals.

## Mid-Tier Architecture

The Mid-Tier lane runs through:

- `scripts/run_mid_tier_probability_compression_historical_sweep.py`

The VPS deployment mirror defaults to:

- input root: `/home/botuser/polymarket-bot/data/l2_book_live`
- output dir:
  `/home/botuser/polymarket-bot/shadow_logs/.../mid_tier_probability_compression_historical_sweep`

It runs a threshold and notional grid, writes the daily panel plus rankings, and
emits an `execution_summary.json` that includes measured memory fields such as:

- `sweep_peak_memory_mb`
- `sweep_peak_rss_mb`
- `reducer_peak_memory_mb`
- `reducer_peak_rss_mb`
- `sweep_within_750_mb`
- `reducer_within_750_mb`

That telemetry is part of the reason the overall scheduler stays serialized.

## Telegram Telemetry

The end-of-run summary script is:

- `scripts/send_shadow_hourly_telegram.py`

It reads the lane outputs from a completed run root and assembles one compact
operator message.

### Inputs

- `metadata_refresh_preflight.json`
- `scavenger_protocol_historical_sweep/summary.json`
- `conditional_probability_squeeze_batch/batch_summary.json`
- `conditional_probability_squeeze_batch/ranking.csv`
- `mid_tier_probability_compression_historical_sweep/execution_summary.json`
- `mid_tier_probability_compression_historical_sweep/daily_panel.parquet`

### Output

- `shadow_telegram_summary.json`

### Send Logic

1. Build a structured summary object.
2. Format a compact HTML Telegram message.
3. Resolve credentials from environment.
4. Send through `src.monitoring.telegram.TelegramAlerter`.
5. Persist whether the send was `sent`, `failed`, `disabled`, or `dry_run`.
6. Print `SHADOW_TELEGRAM_SUMMARY_SENT` or
   `WARNING SHADOW_TELEGRAM_SUMMARY_FAILED`.

The message intentionally compresses the entire hour into one operator-facing
surface instead of producing per-lane alert spam.

## Environment And Secret Resolution

The environment loader lives in:

- `src/core/config.py`

Its secret precedence is:

1. `/dev/shm/secrets/.env` if present
2. `/home/botuser/polymarket-bot/.env` as the repo-root fallback

For the shadow Telegram notifier, the required variables are:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Place them on the VPS in:

- `/home/botuser/polymarket-bot/.env`

`send_shadow_hourly_telegram.py` also supports optional shadow-only overrides:

- `SHADOW_TELEGRAM_BOT_TOKEN`
- `SHADOW_TELEGRAM_CHAT_ID`

If the shadow-specific overrides are unset, it falls back to the standard
Telegram variables.

## Failure Model

The architecture assumes partial failure is normal and should be surfaced, not
hidden.

Examples:

- If Gamma is partially unavailable, metadata refresh can fall back to old rows
  and still keep coverage high enough to run.
- If one lane fails, later lanes still run and Telegram still emits a summary.
- If Telegram credentials are missing, the summary JSON is still written with a
  `disabled` status.
- If coverage falls below `95%`, the operator gets a visible warning in
  `cron_runner.log` and the preflight summary JSON.

This is a shadow pipeline, not a silent batch job. The point is to know the
system's state every hour.

## Runtime Artifacts And Paths

Important live paths:

- live lake:
  `/home/botuser/polymarket-bot/data/l2_book_live`
- shadow root:
  `/home/botuser/polymarket-bot/shadow_mode`
- shadow logs:
  `/home/botuser/polymarket-bot/shadow_logs`
- cron log:
  `/home/botuser/polymarket-bot/shadow_logs/cron_runner.log`
- shadow cache:
  `/home/botuser/polymarket-bot/shadow_mode/artifacts/clob_arb_baseline_metadata.json`
- environment file:
  `/home/botuser/polymarket-bot/.env`

Each hourly run root typically contains:

- `metadata_refresh_preflight.json`
- `shadow_telegram_summary.json`
- `scavenger_protocol_historical_sweep/...`
- `conditional_probability_squeeze_batch/...`
- `mid_tier_probability_compression_historical_sweep/...`

## Repository Boundary

Keep this distinction clear:

- the repository stores code, docs, tests, config, and deploy mirrors
- the VPS stores the live lake, shadow outputs, and current runtime state

Do not treat checked-in artifacts, local CSVs, or local mirrors as production
truth.

## Recommended Reading Order

For a new engineer, the fastest accurate path is:

1. `README.md`
2. `scripts/polymarket-tick-compressor.service`
3. `scripts/live_tick_compressor.py`
4. `vps_shadow_mode/shadow_sentinel_cron.sh`
5. `scripts/refresh_shadow_metadata_cache.py`
6. `scripts/run_scavenger_protocol_historical_sweep.py`
7. `scripts/run_conditional_probability_squeeze_batch.py`
8. `scripts/run_mid_tier_probability_compression_historical_sweep.py`
9. `scripts/send_shadow_hourly_telegram.py`
10. `OPERATIONS.md`# Live Rolling Lake And VPS Shadow Mode

The production architecture is no longer centered on local historical replay.
After the Temporal Gap investigation, the repository moved to a live-data-first
model in which the Helsinki VPS is the canonical runtime and the local machine
is only a mirror, diagnostics host, and code authoring environment.

The system now has two authoritative planes:

- a continuous live L2 archive written to `data/l2_book_live/`
- an hourly shadow-mode evaluator that runs the current strategy wrappers
  against the live rolling lake

## System Contract

1. The Helsinki VPS is the source of truth.
2. `scripts/live_tick_compressor.py` is the always-on ingestion service.
3. The compressor writes hourly Parquet shards under
   `/home/botuser/polymarket-bot/data/l2_book_live/`.
4. The deployed cron runner invokes `shadow_sentinel_cron.sh` at minute `5` of
   every hour.
5. That cron job runs the shadow wrappers sequentially and writes outputs to
   `/home/botuser/polymarket-bot/shadow_logs/`.
6. The local host mirrors the VPS lake into
   `artifacts/l2_parquet_lake_rolling/` for inspection only.
7. Large historical lakes are no longer the deployment truth surface.

## Live Data Plane

The live data plane is owned by:

- `scripts/live_tick_compressor.py`
- `scripts/polymarket-tick-compressor.service`

The compressor contract is:

1. Discover the active tradeable universe.
2. Subscribe to live L2 books.
3. Maintain the clean best-bid, best-ask, and displayed-depth surface.
4. Persist hourly, `zstd`-compressed Parquet shards.

The physical quote schema is eight columns:

- `timestamp`
- `market_id`
- `event_id`
- `token_id`
- `best_bid`
- `best_ask`
- `bid_depth`
- `ask_depth`

When scanned from the partition root, the dataset is an effective ten-column
surface because the hive partition columns `date` and `hour` are materialized
alongside the eight stored quote fields.

Operational defaults currently matter:

- universe refresh every `7200` seconds
- heartbeat every `900` seconds
- hourly rotation
- `zstd` compression

This replaces the old raw JSON or CSV archival path. The archive that matters
now is the clean live lake.

## Shadow Evaluation Plane

The shadow evaluation plane is driven by the deployed `shadow_sentinel_cron.sh`
on the VPS.

At minute `5` of every hour it first refreshes the shadow metadata cache from
Gamma for the current live-lake universe, emits a coverage alarm if cache
coverage drops below `95%`, runs the three shadow sentinels, and then sends a
compact Telegram summary for the completed run.

1. `scripts/run_scavenger_protocol_historical_sweep.py`
2. `scripts/run_conditional_probability_squeeze_batch.py`
3. `scripts/run_mid_tier_probability_compression_historical_sweep.py`

The deployed script executes from the VPS shadow working tree, builds a fresh
timestamped run directory under `/home/botuser/polymarket-bot/shadow_logs/`,
and writes one subdirectory per wrapper.

The operator-facing log for the cron runner is:

- `/home/botuser/polymarket-bot/shadow_logs/cron_runner.log`

Important architectural note: the wrapper names still contain `historical` for
backward compatibility, but the deployed job is evaluating live rolling data
for the current UTC day.

## Local Mirror Plane

The local machine mirrors the live VPS archive into
`artifacts/l2_parquet_lake_rolling/`.

That mirror exists for:

- freshness checks
- post-run inspection
- lightweight local validation
- strategy diagnostics against the same live-shaped surface used by shadow mode

That mirror does not override the VPS. If local and VPS views disagree, the VPS
runtime wins.

## Git Boundary

The repository only tracks source code, tests, documentation, service files,
and static configuration.

The following are runtime surfaces and must not be committed:

- `artifacts/`
- `data/`
- `vps_shadow_mode/`
- parquet shards
- CSV exports
- logs
- generated state JSON
- generated summary JSON
- generated runtime configs and operator receipts

This boundary is part of the architecture, not just repository hygiene. The
runtime produces large, fast-moving operational state; the repository stores
the code and documentation needed to reproduce and operate that state.

## Operational Reading Order

If you need to understand the live system quickly, read in this order:

1. `scripts/live_tick_compressor.py`
2. `scripts/polymarket-tick-compressor.service`
3. `scripts/run_scavenger_protocol_historical_sweep.py`
4. `scripts/run_conditional_probability_squeeze_batch.py`
5. `scripts/run_mid_tier_probability_compression_historical_sweep.py`
6. `README.md`

Everything else in the repository is supporting code, offline research, or
legacy experimentation that no longer defines the production operating model.
- `scripts/polymarket-shield-paper.service`
- `scripts/polymarket-tick-compressor.service`

The Sword and Shield units:

- run as `botuser`;
- set the project working directory to `/home/botuser/polymarket-bot`;
- point `PATH` at the repo's virtual environment;
- use `Restart=always` with `RestartSec=10`;
- send stdout and stderr to journald;
- set `LimitNOFILE=65536`;
- set `NoNewPrivileges=true`;
- create `/dev/shm/secrets` before start;
- decrypt `.env.age` to `/dev/shm/secrets/.env` during `ExecStartPre`;
- remove `/dev/shm/secrets/.env` during `ExecStopPost`.

The compressor unit:

- runs as `botuser`;
- sets the same project working directory and virtualenv `PATH`;
- creates `/home/botuser/polymarket-bot/data/l2_book_live` during
   `ExecStartPre`;
- runs `live_tick_compressor.py` with hourly rotation, `zstd`,
   `market-limit 200`, `--universe-refresh-seconds 7200`,
   `--heartbeat-seconds 900`, and `--min-free-gb 10`;
- uses `Restart=on-failure` with `RestartSec=15`;
- lowers priority with `Nice=10` plus best-effort IO scheduling;
- sends stdout and stderr to journald;
- sets `LimitNOFILE=65536`;
- sets `NoNewPrivileges=true`.

The trading loop intervals come from the service environment:

- Sword sets `SWORD_INTERVAL_SECONDS=300`.
- Shield sets `SHIELD_INTERVAL_SECONDS=10800`.

The archiver cadence comes from the service arguments:

- Data Lake universe refresh every `7200` seconds.
- Data Lake heartbeat every `900` seconds.

### Installation Workflow

`scripts/install_paper_service.sh` is the repo's trading-service installer.

It:

1. stops ad-hoc paper processes;
2. sanitizes shell and service files for Linux line endings;
3. copies the two trading service units into `/etc/systemd/system/`;
4. disables the old single-service unit;
5. enables the two trading units;
6. restarts them and prints status plus recent journal output.

The tick compressor is installed separately as a standalone systemd unit
because it does not run through the trading scheduler.

### Log Rotation

`config/polymarket-bot.logrotate` is the checked-in log rotation policy.

It applies to `logs/*.log` and `logs/*.jsonl` and enforces:

- daily rotation;
- `14` retained rotations;
- compression with delayed compression;
- `copytruncate` for live writers;
- a `100M` max file size threshold;
- automatic file recreation as `0640 botuser:botuser`.

### Memory Guardrail

The checked-in `MemoryMax=750M` hard cap currently lives in the hardened master
service scaffold at `config/polymarket-master.service`.

That scaffold also carries:

- `EnvironmentFile` support;
- `TimeoutStopSec=30`;
- `KillSignal=SIGINT`;
- `Restart=always` with `RestartSec=15`.

The active paper units are separate templates focused on per-pipeline loop
execution. The repo's systemd memory-cap pattern is therefore present today,
but it is defined in the master-service scaffold rather than duplicated inside
both paper unit files.

## Files That Matter Most

If a new developer needs to understand the deployed fund quickly, start here in
order:

1. `scripts/vps_master_scheduler.sh`
2. `scripts/live_bbo_arb_scanner.py`
3. `scripts/launch_clob_arb.py`
4. `scripts/live_flb_scanner.py`
5. `scripts/launch_underwriter.py`
6. `scripts/live_tick_compressor.py`
7. `scripts/send_strategy_telegram_alert.py`
8. `scripts/polymarket-sword-paper.service`
9. `scripts/polymarket-shield-paper.service`
10. `scripts/polymarket-tick-compressor.service`
11. `config/polymarket-bot.logrotate`
12. `config/polymarket-master.service`
