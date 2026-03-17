#!/usr/bin/env bash
set -euo pipefail

cd /home/botuser/polymarket-bot

echo "---adapter-check---"
grep -n "market_config.external_id" src/data/adapters/sports_adapter.py || true
grep -n "market_config.target_outcome" src/data/adapters/sports_adapter.py || true
grep -n "oracle_params" src/data/adapters/sports_adapter.py || true

sed -i -E 's|^ORACLE_SPORTS_API_URL=.*|ORACLE_SPORTS_API_URL=https://api.the-odds-api.com/v4|' .env
sed -i -E 's|^ORACLE_ARB_ENABLED=.*|ORACLE_ARB_ENABLED=True|' .env
sed -i -E 's|^ORACLE_CRITICAL_POLL_MS=.*|ORACLE_CRITICAL_POLL_MS=100|' .env
sed -i -E 's|^ORACLE_MARKET_CONFIGS=.*|ORACLE_MARKET_CONFIGS='\''[{"market_id":"0x62950ac7636e2f11ed8bc0eb8c00aa16bdc1884dd0ffccc1eaee1df682e0714b","oracle_type":"sports","external_id":"NBA_IND_NYK_2026_03_17","target_outcome":"Knicks","market_type":"winner"}]'\''|' .env

echo "---env-check---"
grep -n '^ORACLE_SPORTS_API_URL=\|^ORACLE_MARKET_CONFIGS=\|^ORACLE_ARB_ENABLED=\|^ORACLE_CRITICAL_POLL_MS=' .env

if tmux has-session -t bot_si8 2>/dev/null; then
  tmux kill-session -t bot_si8
fi

rm -f /dev/shm/pmb_*

tmux new-session -d -s bot_si8 "cd /home/botuser/polymarket-bot && . .venv/bin/activate && python -m src.cli run --env PAPER >> logs/bot_si8.log 2>&1"

sleep 10

echo "---log-tail---"
tail -n 120 logs/bot_si8.log
