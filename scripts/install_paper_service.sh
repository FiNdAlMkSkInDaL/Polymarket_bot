#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/botuser/polymarket-bot"
OLD_SERVICE_NAME="polymarket-bot.service"
SERVICE_NAMES=("polymarket-sword-paper.service" "polymarket-shield-paper.service")

sanitize_linux_text_files() {
	find "$PROJECT_DIR/scripts" \( -name '*.sh' -o -name '*.service' \) -print0 | while IFS= read -r -d '' file_path; do
		sed -i 's/\r$//' "$file_path"
	done
}

echo "[1/6] stopping ad-hoc PAPER bot processes"
pkill -SIGTERM -u botuser -f 'python -m src.cli run --env PAPER' 2>/dev/null || true
pkill -SIGTERM -u botuser -f 'scripts/vps_master_scheduler.sh' 2>/dev/null || true
sleep 5

echo "[2/6] sanitizing shell and service files for Linux"
sanitize_linux_text_files

echo "[3/6] installing scheduler systemd units"
for service_name in "${SERVICE_NAMES[@]}"; do
	sudo cp "$PROJECT_DIR/scripts/$service_name" "/etc/systemd/system/$service_name"
done
sudo systemctl daemon-reload

echo "[4/6] disabling legacy hybrid service"
sudo systemctl disable "$OLD_SERVICE_NAME" 2>/dev/null || true
sudo systemctl stop "$OLD_SERVICE_NAME" 2>/dev/null || true

echo "[5/6] enabling new services"
for service_name in "${SERVICE_NAMES[@]}"; do
	sudo systemctl enable "$service_name"
done

echo "[6/6] restarting new services"
for service_name in "${SERVICE_NAMES[@]}"; do
	sudo systemctl restart "$service_name"
done
sleep 8

echo "service status"
for service_name in "${SERVICE_NAMES[@]}"; do
	sudo systemctl status "$service_name" --no-pager || true
done

echo "forward data audit"
cd "$PROJECT_DIR"
.venv/bin/python scripts/audit_forward_data.py --data-dir data --date "$(date -u +%F)" --sample-limit 20000 --json || true

echo
echo "journal tail:"
for service_name in "${SERVICE_NAMES[@]}"; do
	sudo journalctl -u "$service_name" -n 50 --no-pager || true
done