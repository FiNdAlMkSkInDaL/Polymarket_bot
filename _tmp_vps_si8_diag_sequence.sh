#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/botuser/polymarket-bot"
ENV_FILE="$ROOT/.env"

ORACLE_JSON='[{"market_id":"0x7b02c10a310f38be83ae0dbbb5caa96722a69ae1488b1ac5c2003bccd70ac021","oracle_type":"crypto","external_id":"btcusdt","target_outcome":"YES","market_type":"threshold","goal_line":15000.0}]'

if grep -q '^ORACLE_MARKET_CONFIGS=' "$ENV_FILE"; then
  sed -i "s|^ORACLE_MARKET_CONFIGS=.*|ORACLE_MARKET_CONFIGS='$ORACLE_JSON'|" "$ENV_FILE"
else
  echo "ORACLE_MARKET_CONFIGS='$ORACLE_JSON'" >> "$ENV_FILE"
fi

grep '^ORACLE_MARKET_CONFIGS=' "$ENV_FILE"

pkill -f "python -m src.cli run"

rm -f /dev/shm/pmb_*

tmux send-keys -t bot_prod "cd /home/botuser/polymarket-bot && .venv/bin/python -m src.cli run --env PAPER > logs/bot_fresh.log 2>&1" C-m

sleep 20

grep -E "oracle_signal|fast_strike|oracle_rejected" /home/botuser/polymarket-bot/logs/bot_console.log | tail -n 10
