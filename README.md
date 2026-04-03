# Polymarket Bot

This repository's production deployment is a three-service Polymarket VPS
stack.

The VPS does not run a single monolithic strategy engine. It runs two separate
trading services plus one separate data-ingestion service:

- `polymarket-sword-paper.service`
- `polymarket-shield-paper.service`
- `polymarket-tick-compressor.service`

Sword and Shield use live market data and currently launch in PAPER mode. The
tick compressor uses the same live market surface to build the compressed
Parquet archive under `data/l2_archive/`.

## What Runs In Production

The production hot path is:

- `scripts/vps_master_scheduler.sh`
- `scripts/live_bbo_arb_scanner.py`
- `scripts/launch_clob_arb.py`
- `scripts/live_flb_scanner.py`
- `scripts/launch_underwriter.py`
- `scripts/live_tick_compressor.py`
- `scripts/send_strategy_telegram_alert.py`
- `scripts/polymarket-sword-paper.service`
- `scripts/polymarket-shield-paper.service`
- `scripts/polymarket-tick-compressor.service`

Everything else in the repo is support code, research, replay infrastructure,
or future work. Start with the files above if your goal is to understand the
VPS deployment.

## The Three Background Services

### Sword

Sword is the grouped-event CLOB arbitrage lane.

It works like this:

1. `live_bbo_arb_scanner.py` scans active Gamma events for executable
   Dutch-book strips using only live CLOB best bid, best ask, and displayed
   depth.
2. It writes the opportunity set to `config/live_executable_strips.json`.
3. `launch_clob_arb.py` converts each strip into a dynamically sized order
   basket.
4. Size is based on the thinnest executable leg, not a fixed ticket size.
5. The scheduler runs the launcher in PAPER mode and writes the JSON receipt to
   `data/clob_arb_launch_summary_paper.json`.
6. The launcher also writes a markdown receipt to `docs/clob_arb_receipt.md`.

Current risk handling:

- default time in force is `FOK`;
- all legs are submitted concurrently;
- if some legs fill and the strip is not complete, the launcher starts an
  anti-legging flatten workflow;
- Stage 1 flatten is IOC at the live BBO;
- Stage 2 flatten escalates to a more aggressive IOC repricing.

### Shield

Shield is the favorite-longshot underwriting lane.

It works like this:

1. `live_flb_scanner.py` scans live Gamma markets for YES longshots priced below
   five cents on the live book.
2. The scanner prefilters Gamma rows below eight cents before touching the CLOB.
3. Eligible targets are ranked by 24h volume, CLOB liquidity, and then cheaper
   entry price.
4. The target list is hard-capped at `100` names.
5. The scanner writes the target set to `data/flb_results_live.json`.
6. `launch_underwriter.py` posts one passive NO quote per target.
7. Each quote is currently hard-capped at `50` USD notional and priced at
   `0.95` NO.
8. The scheduler writes the JSON receipt to
   `data/underwriter_launch_summary_paper.json`.
9. The launcher also writes a markdown report to
   `docs/underwriter_launch_report_paper.md`.

Current execution posture:

- limit order only;
- `GTC` only;
- `post_only` only;
- paper mode intercepts signed live-shaped payloads locally.

### Data Lake

The Parquet Data Lake is the live ingestion lane.

It works like this:

1. `live_tick_compressor.py` replaces the old bloated JSON or CSV raw tick
   storage path for new live archival data.
2. It discovers the active tradeable universe and subscribes to live L2 books.
3. It shards subscriptions across websocket sockets, currently up to `50`
   asset ids per socket.
4. It writes normalized snapshot and delta rows into hourly, `zstd`-compressed
   Parquet chunks under `data/l2_archive/`.
5. It refreshes the active universe every `7200` seconds and rebinds sockets
   without restarting the full pool.
6. It emits `tick_compressor_heartbeat` every `900` seconds with
   `asset_count`, `socket_count`, `reconnect_count`, and refresh metadata.
7. Chunk creation is logged with `tick_parquet_chunk_written`.

## Scheduler And Services

`scripts/vps_master_scheduler.sh` owns the continuous loop behavior for Sword
and Shield only.

The checked-in intervals are:

- Sword every `300` seconds.
- Shield every `10800` seconds.

The checked-in VPS deployment now runs three independent, self-healing
services:

- Sword as the grouped-event arbitrage loop.
- Shield as the favorite-longshot underwriter loop.
- Data Lake as the standalone Parquet archiver with a `7200` second universe
  refresh cadence and a `900` second heartbeat cadence.

The scheduler can run `sword`, `shield`, or `all`, but the tick compressor is
a separate standalone unit. That isolation is intentional: one service can
restart without taking the other two down.

Install and enable the paper trading services on the VPS with:

```bash
cd /home/botuser/polymarket-bot
bash scripts/install_paper_service.sh
```

Install and enable the standalone tick archiver service with:

```bash
cd /home/botuser/polymarket-bot
sudo ln -sfn /home/botuser/polymarket-bot/scripts/polymarket-tick-compressor.service /etc/systemd/system/polymarket-tick-compressor.service
sudo systemctl daemon-reload
sudo systemctl enable --now polymarket-tick-compressor.service
```

Inspect the services with:

```bash
sudo systemctl status polymarket-sword-paper.service --no-pager
sudo systemctl status polymarket-shield-paper.service --no-pager
sudo systemctl status polymarket-tick-compressor.service --no-pager
sudo journalctl -u polymarket-sword-paper.service -n 100 --no-pager
sudo journalctl -u polymarket-shield-paper.service -n 100 --no-pager
sudo journalctl -u polymarket-tick-compressor.service -n 100 --no-pager
```

Run both pipelines once without installing systemd units:

```bash
bash scripts/vps_master_scheduler.sh --pipeline sword --run-once
bash scripts/vps_master_scheduler.sh --pipeline shield --run-once
```

Run the end-to-end VPS smoke test:

```bash
bash scripts/vps_smoke_test.sh
```

Run the Data Lake archiver directly without systemd:

```bash
python scripts/live_tick_compressor.py --output-dir data/l2_archive --rotation hourly --compression zstd
```

## Local Operator Commands

Run a one-off Sword scan:

```bash
python scripts/live_bbo_arb_scanner.py --output config/live_executable_strips.json
```

Run a one-off Sword PAPER launch:

```bash
python scripts/launch_clob_arb.py --env PAPER --input config/live_executable_strips.json --json-output data/clob_arb_launch_summary_paper.json
```

Run a one-off Shield scan:

```bash
python scripts/live_flb_scanner.py --output data/flb_results_live.json
```

Run a one-off Shield PAPER launch:

```bash
python scripts/launch_underwriter.py --env PAPER --input data/flb_results_live.json --json-output data/underwriter_launch_summary_paper.json
```

Run the tick compressor locally:

```bash
python scripts/live_tick_compressor.py --output-dir data/l2_archive
```

Important launcher detail: both launchers still require live credentials even
in PAPER mode because paper mode signs venue-authentic payloads before the
transport intercepts them.

## Artifacts

The current artifact contract is:

- `config/live_executable_strips.json`: Sword scan output.
- `data/clob_arb_launch_summary_paper.json`: Sword machine-readable receipt.
- `docs/clob_arb_receipt.md`: Sword markdown receipt.
- `data/flb_results_live.json`: Shield scan output.
- `data/underwriter_launch_summary_paper.json`: Shield machine-readable
  receipt.
- `docs/underwriter_launch_report_paper.md`: Shield markdown receipt.
- `data/l2_archive/`: Data Lake hourly `zstd`-compressed Parquet archive.

In LIVE mode only, Shield also persists a state file so it can avoid
duplicating still-live resting orders. PAPER mode skips that state write by
design.

## Alerts And Reporting

`scripts/send_strategy_telegram_alert.py` is the reporting bridge used by the
scheduler.

It supports:

- `sword` summaries
- `shield` summaries
- failure alerts
- strict communication checks

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