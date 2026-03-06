# VPS Data Dumps

This directory contains data pulled directly from the production VPS (`botuser@135.181.85.32`).

---

## `vps_march2026/` — March 1–3, 2026

Pulled on **2026-03-04** from `/home/botuser/polymarket-bot/` on the VPS.

### `logs/`
Rotated structured JSON log files from the bot's Python logger (`src/core/logger.py`).

| File | VPS Date | Description |
|------|----------|-------------|
| `bot.jsonl.5` | Mar 1 17:20 | Oldest rotation — covers early Mar 1 |
| `bot.jsonl.4` | Mar 2 15:34 | Mid-rotation — covers late Mar 1 → Mar 2 |
| `bot.jsonl.3` | Mar 3 07:36 | Rotation — covers Mar 2 → early Mar 3 |
| `bot.jsonl.2` | Mar 3 22:16 | Rotation — covers mid Mar 3 |
| `adverse_sel_outcomes.jsonl` | Mar 2 20:13 | Adverse-selection guard outcome events |

Each file is ~10 MB (rotation threshold). Events are newline-delimited JSON.

### `db/`
SQLite trade database with WAL (write-ahead log).

| File | Description |
|------|-------------|
| `trades.db` | Main SQLite database |
| `trades.db-shm` | Shared memory file (WAL index) |
| `trades.db-wal` | Write-ahead log (uncommitted transactions) |

To open cleanly, use: `sqlite3 trades.db 'PRAGMA wal_checkpoint(FULL)'` first.

### `journal/`
systemd `journalctl` output for `polymarket-bot.service` for Mar 1–3.
> Note: The bot routes all structured logs to rotating JSONL files rather than stdout, so the journal contains minimal entries. The JSONL files in `logs/` are the authoritative source.

### `ticks/`
Raw tick data recorded by `src/backtest/data_recorder.py`, organised by date.

| Directory | Size (approx) |
|-----------|---------------|
| `2026-03-01/` | ~1.3 GB |
| `2026-03-02/` | ~1.3 GB |
| `2026-03-03/` | in progress |

Each file is named by token ID (condition ID or token ID hex) and contains newline-delimited JSON tick events.

> ⚠️ The `ticks/` directory is **excluded from git** (see `.gitignore`) due to size.

---

## Re-pulling data

```powershell
$KEY = "$env:USERPROFILE\.ssh\id_ed25519"
$VPS  = "botuser@135.181.85.32"
$DEST = "data\vps_march2026"

# Logs
scp -i $KEY "${VPS}:/home/botuser/polymarket-bot/logs/bot.jsonl.*" "$DEST\logs\"

# Database
scp -i $KEY `
    "${VPS}:/home/botuser/polymarket-bot/logs/trades.db" `
    "${VPS}:/home/botuser/polymarket-bot/logs/trades.db-shm" `
    "${VPS}:/home/botuser/polymarket-bot/logs/trades.db-wal" `
    "$DEST\db\"

# Ticks (large — run as background job)
scp -i $KEY -r `
    "${VPS}:/home/botuser/polymarket-bot/data/raw_ticks/2026-03-01" `
    "${VPS}:/home/botuser/polymarket-bot/data/raw_ticks/2026-03-02" `
    "${VPS}:/home/botuser/polymarket-bot/data/raw_ticks/2026-03-03" `
    "$DEST\ticks\"
```

---

## Historical Data Backfill

Use `scripts/backfill_data.py` to download 60–90 days of historical trade data (and L2 deltas when a source adapter is available) for all tracked markets.

```bash
# Install dependencies
pip install -r scripts/requirements-backfill.txt

# Default 90-day backfill
python scripts/backfill_data.py

# Custom date range
python scripts/backfill_data.py --start-date 2025-12-06 --end-date 2026-03-04

# Force re-download with Parquet conversion
python scripts/backfill_data.py --lookback-days 30 --force --parquet
```

Output goes to `vps_march2026/ticks/YYYY-MM-DD/<market_id>.jsonl` — same format as live recorded ticks. Run `python scripts/backfill_data.py --help` for full options.
