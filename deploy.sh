#!/usr/bin/env bash
# ============================================================
# deploy.sh — Strategic Integration Suite Deployment
# Date : 2026-03-09
# Target: botuser@135.181.85.32:/home/botuser/polymarket-bot
# ============================================================
set -euo pipefail

VPS_HOST="botuser@135.181.85.32"
VPS_DIR="/home/botuser/polymarket-bot"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "──── 1. Synchronize Codebase ────"
# rsync with explicit excludes — avoids the scp nesting bug
# (scp -r src/ host:dir/src/ creates dir/src/src/)
rsync -avz --delete \
    --exclude='.git/' \
    --exclude='logs/' \
    --exclude='data/vps_march2026/ticks/' \
    --exclude='.venv/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='polymarket_bot.egg-info/' \
    --exclude='__pycache__/' \
    --exclude='*.db' \
    --exclude='*.pyc' \
    "${LOCAL_DIR}/" "${VPS_HOST}:${VPS_DIR}/"

echo ""
echo "──── 2. Terminate Stale Session & Launch ────"
ssh -t "$VPS_HOST" bash -s <<'REMOTE'
set -euo pipefail

# Kill existing tmux session (ignore if absent)
tmux kill-session -t polybot 2>/dev/null || true

# Clean shared-memory allocations from prior L2 workers
# Use find instead of rm glob — avoids ARG_MAX with hundreds of segments
find /dev/shm -name 'pmb_*' -delete 2>/dev/null || true

# Reinstall package to pick up any pyproject.toml changes
cd /home/botuser/polymarket-bot
source .venv/bin/activate
pip install -e '.[dev]' --quiet

# Launch in detached tmux with raised fd limit (L2 workers open many shm segments)
tmux new-session -d -s polybot \
    "ulimit -n 65536; cd /home/botuser/polymarket-bot && source .venv/bin/activate && python -m src.cli run --confirm-production"

echo ""
echo "✅  polybot tmux session launched."
REMOTE

echo ""
echo "──── 3. Verification ────"
sleep 5
ssh "$VPS_HOST" "tmux capture-pane -t polybot -p | head -20"

echo ""
echo "──── 4. Log Check: Pillar 15.1 & SI-6 ────"
ssh "$VPS_HOST" "grep -E 'PCE|Pillar.?15|MetaStrategy|SI.?6|meta.controller' /home/botuser/polymarket-bot/logs/bot.jsonl | tail -10" || true

echo ""
echo "✅  Deployment complete."
