#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/botuser/polymarket-bot"
SERVICE_NAME="polymarket-bot.service"
UNIT_SRC="$PROJECT_DIR/scripts/$SERVICE_NAME"
UNIT_DST="/etc/systemd/system/$SERVICE_NAME"

echo "[1/6] stopping ad-hoc PAPER bot processes"
pkill -SIGTERM -u botuser -f 'python -m src.cli run --env PAPER' 2>/dev/null || true
sleep 5

echo "[2/6] installing systemd unit"
sudo cp "$UNIT_SRC" "$UNIT_DST"
sudo systemctl daemon-reload

echo "[3/6] enabling service"
sudo systemctl enable "$SERVICE_NAME"

echo "[4/6] restarting service"
sudo systemctl restart "$SERVICE_NAME"
sleep 8

echo "[5/6] service status"
sudo systemctl status "$SERVICE_NAME" --no-pager

echo "[6/6] forward data audit"
cd "$PROJECT_DIR"
.venv/bin/python scripts/audit_forward_data.py --data-dir data --date "$(date -u +%F)" --sample-limit 20000 --json || true

echo
echo "journal tail:"
sudo journalctl -u "$SERVICE_NAME" -n 50 --no-pager