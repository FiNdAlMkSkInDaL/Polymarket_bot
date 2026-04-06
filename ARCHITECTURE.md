# Live Rolling Lake And VPS Shadow Mode

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
coverage drops below `95%`, and then runs, in order:

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
