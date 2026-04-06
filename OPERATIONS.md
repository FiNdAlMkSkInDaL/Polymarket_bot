# Shadow Mode Operations

This is the day-2 runbook for the deployed Helsinki VPS shadow stack.

Use this document when you need to verify health, investigate a blind spot,
rerun a step manually, or onboard a new operator.

## Runtime Paths

Core paths on the VPS:

- repo root: `/home/botuser/polymarket-bot`
- live lake: `/home/botuser/polymarket-bot/data/l2_book_live`
- live lake state: `/home/botuser/polymarket-bot/data/l2_book_live/_state`
- shadow root: `/home/botuser/polymarket-bot/shadow_mode`
- shadow logs: `/home/botuser/polymarket-bot/shadow_logs`
- cron log: `/home/botuser/polymarket-bot/shadow_logs/cron_runner.log`
- shadow metadata cache:
  `/home/botuser/polymarket-bot/shadow_mode/artifacts/clob_arb_baseline_metadata.json`
- environment file: `/home/botuser/polymarket-bot/.env`

## Required Environment

For Telegram delivery, the minimum required variables are:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Place them in:

- `/home/botuser/polymarket-bot/.env`

The config loader prefers `/dev/shm/secrets/.env` if it exists. If you later
adopt tmpfs secret injection, remember that the tmpfs file overrides the repo
root `.env`.

Optional shadow-only overrides are supported:

- `SHADOW_TELEGRAM_BOT_TOKEN`
- `SHADOW_TELEGRAM_CHAT_ID`

## Scheduler Contract

The installed cron job runs `shadow_sentinel_cron.sh` at minute `5` of every
hour.

That wrapper performs:

1. metadata preflight
2. Scavenger
3. Squeeze
4. Mid-Tier
5. Telegram summary

Each step writes into a timestamped run root under `shadow_logs/`.

## First Health Checks

### 1. Check The Live Compressor

```bash
sudo systemctl status polymarket-tick-compressor.service --no-pager
sudo journalctl -u polymarket-tick-compressor.service -n 100 --no-pager
```

What you want to see:

- unit is `active (running)`
- recent heartbeat lines
- recent shard flush lines
- no reconnect storm or disk pressure warnings

### 2. Check The Shadow Cron Log

```bash
tail -n 200 /home/botuser/polymarket-bot/shadow_logs/cron_runner.log
```

Healthy markers include:

- `SHADOW_METADATA_PREFLIGHT`
- `SHADOW_METADATA_COVERAGE_OK`
- `SHADOW_STEP_OK step=metadata_preflight`
- `SHADOW_STEP_OK step=scavenger`
- `SHADOW_STEP_OK step=squeeze`
- `SHADOW_STEP_OK step=mid_tier`
- `SHADOW_TELEGRAM_SUMMARY_SENT`

Degraded markers include:

- `WARNING SHADOW_METADATA_REFRESH_FALLBACK`
- `WARNING SHADOW_METADATA_COVERAGE_LOW`
- `WARNING SHADOW_STEP_FAILED ...`
- `WARNING SHADOW_TELEGRAM_SUMMARY_FAILED ...`

### 3. Inspect The Latest Run Root

```bash
ls -1 /home/botuser/polymarket-bot/shadow_logs | tail
```

Then inspect the newest folder for:

- `metadata_refresh_preflight.json`
- `shadow_telegram_summary.json`
- strategy subdirectories

## Manual Reruns

All manual reruns should be launched from the repo root unless there is a good
reason not to.

### Rerun The Metadata Preflight Only

```bash
cd /home/botuser/polymarket-bot/shadow_mode
/home/botuser/polymarket-bot/.venv/bin/python scripts/refresh_shadow_metadata_cache.py \
  --lake-root /home/botuser/polymarket-bot/data/l2_book_live \
  --cache-path /home/botuser/polymarket-bot/shadow_mode/artifacts/clob_arb_baseline_metadata.json \
  --summary-path /home/botuser/polymarket-bot/shadow_logs/manual_metadata_refresh.json
```

Use this when you suspect the cache has drifted away from the active live
universe.

### Rerun The Full Shadow Pipeline Once

```bash
cd /home/botuser/polymarket-bot
bash shadow_mode/shadow_sentinel_cron.sh >> shadow_logs/cron_runner.log 2>&1
```

This creates a fresh timestamped run and appends step markers into the normal
cron log.

### Send A Telegram Summary For An Existing Run

Dry-run first:

```bash
cd /home/botuser/polymarket-bot/shadow_mode
/home/botuser/polymarket-bot/.venv/bin/python scripts/send_shadow_hourly_telegram.py \
  --run-root /home/botuser/polymarket-bot/shadow_logs/<run-stamp> \
  --dry-run
```

Then send live:

```bash
cd /home/botuser/polymarket-bot/shadow_mode
/home/botuser/polymarket-bot/.venv/bin/python scripts/send_shadow_hourly_telegram.py \
  --run-root /home/botuser/polymarket-bot/shadow_logs/<run-stamp>
```

## Investigating Metadata Blindness

Symptoms:

- Scavenger universe count suddenly drops
- preflight reports low coverage
- active live markets are missing from the cache

What to check:

1. Open the latest `metadata_refresh_preflight.json`.
2. Compare `active_market_count` vs `covered_market_count`.
3. Check `refresh_status`, `reused_market_count`, and
   `missing_market_ids_sample`.
4. Confirm the live lake actually has current-day shards.

The current protection model is:

- fetch fresh Gamma rows by active condition ids
- reuse last known good rows on partial failure
- alarm if coverage falls below `95%`

If the script ends in `cache_only`, the old cache still exists but the refresh
itself failed. If it ends in `unavailable`, both refresh and usable cache are
gone and the lane surface cannot be trusted.

## Understanding The Scavenger Window

The Scavenger lane is strict about time to resolution.

It only admits opportunities where:

- time to resolution is positive
- time to resolution is strictly less than `72` hours

The exact `72h` boundary is excluded.

This is important when you investigate a surprising zero-target run. A zero can
be correct even when a market feels near-dated if it is still outside the open
interval.

## Understanding Telegram Outcomes

The summary script writes `shadow_telegram_summary.json` regardless of whether
Telegram actually sends.

Look at `telegram.status`:

- `sent`: Telegram acknowledged the message
- `failed`: send attempted but Telegram or network failed
- `disabled`: credentials were missing
- `dry_run`: message built but not sent by design

If status is `disabled`, check `/home/botuser/polymarket-bot/.env` first.

## Lake Freshness Checks

If the lake looks stale:

1. inspect the compressor service status and journal
2. list the latest `date=.../hour=...` partitions under `data/l2_book_live`
3. inspect `_state` handoff metadata if present

Quick check:

```bash
find /home/botuser/polymarket-bot/data/l2_book_live -name '*.parquet' | tail
```

If new shards are not arriving, the shadow lanes are evaluating stale data even
if the cron job itself is healthy.

## Memory Discipline

The shadow scheduler is intentionally sequential.

Do not parallelize Scavenger, Squeeze, and Mid-Tier on this host unless you
also redesign the memory budget. The reason is simple:

- the VPS is small
- each lane is analysis-heavy
- overlapping them converts a bounded single-process peak into an additive RSS
  spike

When adding a new lane, keep this contract intact:

1. run it after the existing metadata preflight
2. keep it serialized with the other lanes
3. write into the same run root
4. make its output easy for `send_shadow_hourly_telegram.py` to summarize

## Local Mirror Workflow

The local host is useful for analysis, but it is not authoritative.

Typical workflow:

```bash
python scripts/sync_lake_from_vps.py --local-root artifacts/l2_parquet_lake_rolling/l2_book --min-date 2026-04-04 --loop --interval-seconds 3600
python scripts/monitor_lake_health.py --sync-state artifacts/l2_parquet_lake_rolling/sync_state.json
```

Use the mirrored lake for diagnostics, plotting, and local validation. If the
mirror and the VPS disagree, trust the VPS.

## Deployment Notes For Future Changes

When changing the shadow stack:

1. update the top-level script if it is the canonical source
2. update the `vps_shadow_mode/` deployment mirror if the VPS path depends on it
3. verify the cron wrapper still writes step markers and still attempts the
   Telegram summary
4. verify any new environment variable is documented in both `README.md` and
   this file

The goal of the documentation set is operational continuity. A future engineer
should be able to recover the system quickly without rediscovering these paths
from scratch.