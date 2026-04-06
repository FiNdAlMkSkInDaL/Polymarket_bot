# Polymarket Bot

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
sequentially against the live rolling lake for the current UTC date:

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