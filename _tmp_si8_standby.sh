#!/usr/bin/env bash
set -euo pipefail

cd /home/botuser/polymarket-bot

# Standby mode: throttle critical oracle polling to preserve API credits.
sed -i -E 's|^ORACLE_CRITICAL_POLL_MS=.*|ORACLE_CRITICAL_POLL_MS=300000|' .env

grep -n '^ORACLE_CRITICAL_POLL_MS=' .env

tmux kill-session -t bot_si8 2>/dev/null || true
rm -f /dev/shm/pmb_*
tmux new-session -d -s bot_si8 "cd /home/botuser/polymarket-bot && . .venv/bin/activate && python -m src.cli run --env PAPER >> logs/bot_si8.log 2>&1"

# Warmup then print relevant lines.
sleep 20
echo '---oracle-sync---'
grep -nF '[SI-8] Monitoring Knicks vs Pacers - Oracle Sync: OK' logs/bot_si8.log | tail -n 5 || true

echo '---recent-oracle-lines---'
grep -nE 'oracle_adapter_started|oracle_adapter_poll_error|Oracle Sync: OK|poll' logs/bot_si8.log | tail -n 40 || true
