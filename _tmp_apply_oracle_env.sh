#!/usr/bin/env bash
set -euo pipefail
cd /home/botuser/polymarket-bot

if ! grep -q '^ORACLE_SPORTS_API_KEY=' .env; then
	echo "ORACLE_SPORTS_API_KEY is missing from .env" >&2
	exit 1
fi

sed -i -E 's|^ORACLE_SPORTS_API_URL=.*|ORACLE_SPORTS_API_URL=https://api.the-odds-api.com/v4|' .env
sed -i -E 's|^ORACLE_MARKET_CONFIGS=.*|ORACLE_MARKET_CONFIGS='\''[{"market_id":"0x62950ac7636e2f11ed8bc0eb8c00aa16bdc1884dd0ffccc1eaee1df682e0714b","oracle_type":"sports","external_id":"9eba01d4cfa06caacc9d62a166b5a632","target_outcome":"Knicks","market_type":"winner"}]'\''|' .env

echo 'ORACLE_SPORTS_API_KEY is set'
grep -n '^ORACLE_SPORTS_API_URL=\|^ORACLE_MARKET_CONFIGS=' .env
