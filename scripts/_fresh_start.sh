#!/usr/bin/env bash
# Reset all trade/position history for a fresh bot start.
# Runs on VPS as botuser. Kills the bot, cleans up, systemd restarts it.
set -e

BOT_DIR="$HOME/polymarket-bot"
LOGS="$BOT_DIR/logs"

echo "[1] Stopping bot processes..."
pkill -SIGTERM -u botuser -f 'src.cli run' 2>/dev/null && echo "  killed" || echo "  no processes found"
sleep 5  # wait for graceful shutdown (RestartSec=10 gives us 10s window)

echo "[2] Removing trade database and WAL files..."
rm -fv "$LOGS/trades.db" "$LOGS/trades.db-shm" "$LOGS/trades.db-wal"

echo "[3] Clearing bot log files..."
> "$LOGS/bot.jsonl"
for f in "$LOGS"/bot.jsonl.*; do
    [ -f "$f" ] && rm -fv "$f"
done

echo "[4] Clearing adverse selection history..."
> "$LOGS/adverse_sel_outcomes.jsonl"

echo "[5] Clearing system health snapshot..."
echo '{}' > "$LOGS/system_health.json"

echo "[6] Verifying cleanup..."
echo "  trades.db exists: $([ -f "$LOGS/trades.db" ] && echo YES || echo NO)"
echo "  bot.jsonl size:   $(wc -c < "$LOGS/bot.jsonl") bytes"
echo "  log rotations:    $(ls "$LOGS"/bot.jsonl.* 2>/dev/null | wc -l) files"

echo "[done] systemd will restart the bot in ~5s (RestartSec=10)."
