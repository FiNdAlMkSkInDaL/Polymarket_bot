# Three-Service Architecture

This repository's production path is a dual-strategy Polymarket fund plus a
dedicated compressed market-data archive, operated as three long-running
services on the VPS.

The hot path is deliberately small:

- Sword: scan grouped negative-risk events for executable Dutch-book strips,
  then fire concurrent CLOB strip orders.
- Shield: scan the full live market for sub-5c YES longshots, then underwrite
  them with resting NO quotes.
- Data Lake: capture live L2 order book updates into highly compressed Parquet
   archive chunks for replay, research, and audit.
- Control plane: one trading scheduler shell script, one Telegram bridge, one
   standalone compressor loop, and three systemd units plus a small artifact set
   in `config/`, `data/`, `docs/`, and `logs/`.

The repository still contains research, replay, and experimental modules, but
the deployed VPS fund does not need them to run. If a file is not named in
this document, treat it as support code or offline analysis rather than the
production loop.

## Production Contract

The currently deployed operating model is:

1. `scripts/polymarket-sword-paper.service` runs Sword continuously.
2. `scripts/polymarket-shield-paper.service` runs Shield continuously.
3. `scripts/polymarket-tick-compressor.service` runs the Data Lake
   continuously.
4. Sword and Shield call `scripts/vps_master_scheduler.sh` with a pipeline
   selector.
5. All three services consume live market data, but only Sword and Shield
   build PAPER trading payloads.
6. The trading launchers run in PAPER mode on the VPS today, but they still
   build live-shaped orders, signatures, and receipts.
7. The Data Lake writes hourly compressed Parquet archive chunks under
   `data/l2_archive/`.
8. The trading loops emit machine-readable JSON summaries, human-readable
   markdown receipts, and Telegram alerts.
9. The Data Lake emits journald heartbeats plus chunk-write telemetry for
   archive health and progress.

This is not a cron job, and it is not a single monolithic bot process. It is
two trading loops plus one independent data-ingestion loop.

## Control Plane

`scripts/vps_master_scheduler.sh` is the trading control plane.

It exposes three modes:

- `--pipeline sword`
- `--pipeline shield`
- `--pipeline all`

It also exposes `--run-once` for smoke tests and one-shot verification.

The checked-in intervals are:

- Sword every `300` seconds.
- Shield every `10800` seconds.

Each loop follows the same pattern:

1. Run the discovery scanner.
2. Read the scanner's JSON output.
3. Skip launch if no targets remain.
4. Run the strategy launcher in PAPER mode.
5. Send a strategy-specific Telegram summary.
6. If a scanner or launcher fails, send a pipeline failure alert and keep the
   loop alive.

Production isolation comes from the deployment shape rather than from complex
in-process orchestration: Sword and Shield are separate services, so one loop
can restart without taking the other down.

The Data Lake is intentionally outside this scheduler. It runs as its own
standalone systemd unit and owns its own websocket pool, timed universe
refresh, and heartbeat telemetry, so the archive can keep running even if a
trading loop is restarted or reconfigured.

## Sword: CLOB Arbitrage

Sword is the event-group arbitrage lane.

### Discovery

`scripts/live_bbo_arb_scanner.py` sweeps live Gamma grouped markets and asks a
single question: can the current best bid or best ask across every outcome be
traded as a risk-free Dutch book right now?

The scanner does the following:

1. Pull up to `40` Gamma pages of `500` active, open markets each.
2. Keep only markets that are active, accepting orders, and have an enabled
   order book.
3. Group legs by `event_id`.
4. Require at least `3` active outcomes in the event.
5. Require a binary YES/NO token shape for every grouped leg.
6. Require the grouped event to be flagged for negative-risk execution.
7. Reject cumulative threshold ladders such as overlapping "at least" or
   "above" markets.
8. Pull the live YES book for every candidate leg from the CLOB.
9. Record each leg's best bid, best ask, and displayed size.

The scanner emits two possible strip directions:

- `BUY_YES_STRIP`: sum of leg asks is less than `1.00 - fee_buffer`.
- `SELL_NO_STRIP`: sum of leg bids is greater than `1.00 + fee_buffer`.

The default fee buffer is `0.02`, so the raw boundary checks are:

- buy the full YES strip only if the strip costs less than `0.98`.
- sell the full NO strip only if the strip pays more than `1.02`.

Every leg must also clear the depth gate of `10` USD at the top of book.

The scanner writes `config/live_executable_strips.json` with:

- `gamma_markets_scanned`
- `grouped_events_considered`
- `executable_strips`
- filter settings
- grouping counters
- rejection counts
- the executable strip targets themselves

### Dynamic Strip Sizing

`scripts/launch_clob_arb.py` turns a scanner target into an executable strip.

Sizing is driven by the thinnest executable leg, not by a fixed order size.

For each strip:

1. Read `min_leg_depth_usd_observed` from the scanner output.
2. Round it down to a whole-dollar cap.
3. Divide that cap by the strip execution price sum to get a safe share count.
4. Clamp again by `strip_max_size_shares_at_bbo`.
5. Enforce the exchange minimum share and minimum USD checks.

In other words, the launcher never sizes the strip from the deepest leg. It
sizes from the weakest displayed leg because that is the only way to keep the
entire strip executable at the quoted edge.

### Concurrent FOK Execution

Sword launches one order per leg and signs them all before submission.

The scheduler currently invokes:

`launch_clob_arb.py --env PAPER --input config/live_executable_strips.json --json-output data/clob_arb_launch_summary_paper.json`

No `--time-in-force` override is supplied, so the runtime default remains
`FOK`.

That means the production Sword service currently attempts every strip as a
concurrent fill-or-kill basket.

Operational details:

- all leg payloads are prepared up front;
- all leg submissions are fired concurrently with `asyncio.gather(...)`;
- live mode checks wallet balance before buying a strip;
- paper mode intercepts the live-shaped payloads locally and records them as
  paper receipts.

### Anti-Legging Safety Handler

The anti-legging path lives inside `launch_clob_arb.py` and is triggered when
some legs fill but the full strip does not.

The flow is explicit:

1. If every leg is fully filled, the strip is marked `FULLY_FILLED`.
2. If no leg fills, the strip remains `NO_FILL` or `SUBMITTED` depending on the
   venue response.
3. If at least one leg fills but the strip is incomplete, the launcher enters a
   flatten workflow immediately.

Stage 1 flatten:

- calculate residual exposure per partially executed leg;
- look up the live BBO for the affected token;
- submit an IOC flatten order at the live BBO;
- use the best bid for sell-side flattening and the best ask for buy-side
  flattening;
- tag the result as `BBO_IOC`.

Stage 2 escalation:

- if inventory remains after Stage 1, reprice aggressively;
- move the price by the larger of `5%` of the Stage 1 price or an absolute
  `0.05`;
- clip the new price into the venue-safe range `0.001` to `0.999`;
- submit another IOC order;
- tag the result as `PANIC_IOC`.

The launcher records whether the strip required flattening, whether Stage 2 was
needed, and whether any residual inventory survived both stages.

### Sword Artifacts

The Sword pipeline persists three operator-facing artifacts:

- `config/live_executable_strips.json`: the current opportunity set.
- `data/clob_arb_launch_summary_paper.json`: machine-readable launch summary.
- `docs/clob_arb_receipt.md`: human-readable per-strip receipt.

The JSON summary includes:

- `paper_intercepted_payloads`
- `flatten_events`
- `flatten_stage2_events`
- `flatten_failures`
- `status_counts`
- `total_planned_notional_usd`
- per-strip leg results
- per-strip flatten leg results

## Shield: Favorite-Longshot Underwriter

Shield is the continuous longshot underwriting lane.

### Infinite Discovery Loop

`scripts/live_flb_scanner.py` performs the discovery pass for Shield.

It continuously sweeps live Gamma markets and keeps only markets that still
look like true YES longshots on the order book.

The scanner does the following:

1. Pull up to `40` Gamma pages of `500` active, open markets each.
2. Keep only markets that are active, accepting orders, and have an enabled
   order book.
3. Resolve YES and NO token ids from Gamma metadata.
4. Apply a cheap Gamma-side prefilter: YES outcome price must be below `0.08`
   before the CLOB is queried.
5. Pull the live YES order book for each surviving market.
6. Choose the live reference YES price in this order:
   - best ask if `0.001 < ask < 0.05`
   - midpoint if the book is two-sided and the midpoint falls in range
   - Gamma outcome price as the fallback midpoint proxy
7. Infer a display category from the event title, slug, and question text.
8. Rank all eligible targets by:
   - highest `market_volume_24h`
   - highest `liquidity_clob_usd`
   - lowest `entry_yes_ask`
   - alphabetical question text
9. Hard-cap the emitted target set at `100` markets.

That cap is not accidental. It is enforced directly by
`DEFAULT_MAX_SHIELD_TARGETS = 100` and by the default
`--max-shield-targets 100` argument.

The scanner writes `data/flb_results_live.json` with:

- `summary.active_bucket.count`
- category counts
- discovery stats
- rejection counts
- every selected live target

### Underwriter Execution

`scripts/launch_underwriter.py` converts the scanner output into passive NO
quotes.

The checked-in implementation expresses its Kelly sizing as a fixed per-name
cap:

- `MAX_NOTIONAL_PER_CONDITION = 50`
- `ENTRY_PRICE = 0.95`

So each selected market becomes one resting NO order worth `50` USD notional at
price `0.95`.

The share count is computed as:

- `order_size = 50 / 0.95`
- quantized to `0.000001` shares

The execution posture is always passive:

- side: `NO`
- order type: `LIMIT`
- time in force: `GTC`
- `post_only = True`

The launcher resolves token ids from the scanner output when present, and falls
back to Gamma when they are missing.

### PAPER Versus LIVE

Shield supports both PAPER and LIVE modes.

Important implementation detail: PAPER mode still validates live credentials
and still builds live-equivalent signed payloads. The only difference is the
transport layer:

- PAPER intercepts the payload locally.
- LIVE submits the order through the adapter and persists state.

State behavior:

- LIVE writes a state file to avoid duplicate resting orders.
- PAPER deliberately skips state persistence.

Default paper outputs are:

- `docs/underwriter_launch_report_paper.md`
- `data/underwriter_launch_summary_paper.json`

The JSON summary includes:

- `active_targets_loaded`
- `submitted_orders`
- `skipped_existing`
- `rejected_orders`
- `paper_intercepted_payloads`
- `submitted_notional_usd`
- category counts
- the submitted, skipped, rejected, and dry-run rows

## Data Lake: Compressed Tick Archive

`scripts/live_tick_compressor.py` is the live L2 archive service.

Its job is not trade execution. Its job is to replace the old bloated JSON or
CSV raw tick accumulation path with a compact, replayable archive.

### Storage Model

The compressor:

1. resolves the active tradeable universe with `fetch_active_markets(...)`;
2. builds YES and NO token subscriptions for the selected markets;
3. shards those subscriptions across websocket connections with up to `50`
   asset ids per socket;
4. normalizes snapshots and deltas into a fixed Parquet schema;
5. buffers rows in an asyncio queue;
6. writes immutable archive chunks into `data/l2_archive/`.

The checked-in service defaults are:

- output dir: `data/l2_archive/`
- rotation: `hourly`
- compression: `zstd`
- flush rows: `10000`
- flush seconds: `300`

That means the VPS now stores fresh L2 capture as hourly Parquet parts like:

- `data/l2_archive/YYYY-MM-DD/ticks_YYYY-MM-DD_HH_000001.parquet`

The retained schema keeps replay-critical columns such as `local_ts`,
`exchange_ts`, `msg_type`, `asset_id`, `market_id`, `outcome`, `price`,
`size`, `sequence_id`, `side`, and the compact raw `payload` JSON. The raw
payload is still preserved, but it now lives inside a compressed columnar file
instead of growing as standalone JSON or CSV dumps.

### Dynamic Universe Refresh

The compressor does not keep a startup-only subscription set.

The checked-in default `--universe-refresh-seconds 7200` means the service
re-resolves the active tradeable universe every two hours and diffs it against
the currently bound websocket pool.

The rebinding policy is:

- add and update subscriptions first;
- remove stale subscriptions second;
- fill existing sockets up to `max_assets_per_socket` before creating more;
- retire sockets that become empty.

That keeps the archive aligned with the live tradable universe without tearing
down the full socket pool on every refresh.

### Heartbeat and Archive Telemetry

The checked-in default `--heartbeat-seconds 900` makes the service emit
`tick_compressor_heartbeat` every fifteen minutes.

Each heartbeat records:

- `asset_count`
- `socket_count`
- `reconnect_count`
- `refresh_count`
- `last_refresh_reason`
- `last_refresh_age_s`

The service also emits `tick_parquet_chunk_written` whenever it flushes a new
archive part, so journald shows both pool health and archive progress.

Individual websocket shards reconnect with backoff, and the systemd unit
restarts the process on failure, so the archiver is self-healing at both the
socket and process layers.

## Telemetry and Reporting

The trading reporting contract is file-first and alert-second. The Data Lake
loop is archive-first and journal-first.

Each trading launcher produces:

- a JSON summary for machines and downstream scripts;
- a markdown receipt for humans.

`scripts/send_strategy_telegram_alert.py` turns those artifacts into operator
alerts.

It supports four commands:

- `shield`: load the Shield launch summary and send active target count,
  staged/intercepted counts, notional, top categories, and sample questions.
- `sword`: load the Sword scan summary plus optional launch summary and send
  executable strip count, grouped event count, per-launch statuses, and the top
  strip candidates.
- `failure`: emit a strategy, stage, and message failure alert.
- `comm-check`: perform a strict Telegram connectivity check and return a
  failing exit code if Telegram does not acknowledge the message.

`scripts/vps_smoke_test.sh` uses the same reporting surface to verify both
strategy loops and the Telegram channel end to end.

The compressor does not route through Telegram. Its operator surface is
journald plus the archive directory. The important events are
`tick_compressor_heartbeat`, `tick_parquet_chunk_written`,
`tick_universe_refresh_complete`, and the socket disconnect or silence-timeout
warnings.

## Service Model and OS Resilience

### Active VPS Services

The active deployment units are:

- `scripts/polymarket-sword-paper.service`
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
- creates `/home/botuser/polymarket-bot/data/l2_archive` during
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
