#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/botuser/polymarket-bot"
SERVICE_NAME="polymarket-master.service"
SERVICE_SRC="$PROJECT_DIR/config/$SERVICE_NAME"
SERVICE_DST="/etc/systemd/system/$SERVICE_NAME"
LOGROTATE_SRC="$PROJECT_DIR/config/polymarket-bot.logrotate"
LOGROTATE_DST="/etc/logrotate.d/polymarket-bot"

echo "[1/7] ensuring runtime directories exist"
sudo install -d -m 0775 -o botuser -g botuser "$PROJECT_DIR/logs" "$PROJECT_DIR/data"

echo "[2/7] installing logrotate policy"
sudo install -m 0644 "$LOGROTATE_SRC" "$LOGROTATE_DST"

echo "[3/7] validating logrotate policy"
sudo logrotate -d "$LOGROTATE_DST" >/dev/null

echo "[4/7] installing master systemd unit"
sudo ln -sfn "$SERVICE_SRC" "$SERVICE_DST"

echo "[5/7] reloading systemd"
sudo systemctl daemon-reload

echo "[6/7] enabling master service for reboot persistence"
sudo systemctl enable "$SERVICE_NAME"

echo "[7/7] final status"
systemctl is-enabled "$SERVICE_NAME"
echo "Installed $SERVICE_NAME and logrotate policy."
echo "Service not started by this script. Start manually when the master scheduler entrypoint is present and validated."