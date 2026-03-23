#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/botuser/polymarket-bot"
ENV_FILE="$ROOT/.env"

python3 - <<'PY'
from pathlib import Path

p = Path('/home/botuser/polymarket-bot/.env')
text = p.read_text(encoding='utf-8') if p.exists() else ''
lines = text.splitlines()
updates = {
    'DEPLOYMENT_ENV': 'PAPER',
    'ORACLE_ARB_ENABLED': 'True',
    'ORACLE_SHADOW_MODE': 'False',
    'ORACLE_CRITICAL_POLL_MS': '200',
    'ORACLE_MARKET_CONFIGS': "'[{\"market_id\":\"0x02deb9538f5c123373adaa4ee6217b01745f1662bc902e46ac92f3fe6f8741e8\",\"oracle_type\":\"crypto\",\"external_id\":\"btcusdt\",\"target_outcome\":\"YES\",\"market_type\":\"threshold\",\"goal_line\":150000.0}]'",
}
out = []
seen = set()
for line in lines:
    if '=' in line and not line.lstrip().startswith('#'):
        k = line.split('=', 1)[0].strip()
        if k in updates:
            out.append(f"{k}={updates[k]}")
            seen.add(k)
            continue
    out.append(line)
for k, v in updates.items():
    if k not in seen:
        out.append(f"{k}={v}")
p.write_text("\n".join(out) + "\n", encoding='utf-8')
print('env_updated_200ms')
PY

grep -n '^DEPLOYMENT_ENV=\|^ORACLE_ARB_ENABLED=\|^ORACLE_SHADOW_MODE=\|^ORACLE_CRITICAL_POLL_MS=\|^ORACLE_MARKET_CONFIGS=' "$ENV_FILE"

pkill -f "python -m src.cli run" || true
rm -f /dev/shm/pmb_*
tmux has-session -t bot_prod 2>/dev/null || tmux new-session -d -s bot_prod
tmux send-keys -t bot_prod "cd $ROOT && ulimit -n 65536 && .venv/bin/python -m src.cli run --env PAPER > logs/bot_fresh.log 2>&1" C-m

echo launched_bot_prod_200ms
