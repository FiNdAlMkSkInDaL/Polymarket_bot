#!/usr/bin/env bash
# Watchdog cron script — restarts the bot if the systemd service is dead.
# Add to crontab: */5 * * * * /home/botuser/polymarket-bot/scripts/watchdog.sh
set -euo pipefail

SERVICE="polymarket-bot"

if ! systemctl is-active --quiet "$SERVICE"; then
    echo "$(date -Iseconds) — $SERVICE is down, restarting..." >> /var/log/botuser/watchdog.log
    systemctl restart "$SERVICE"
fi
