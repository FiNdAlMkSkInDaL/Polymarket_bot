#!/usr/bin/env bash
# ============================================================
# VPS Bootstrap Script — Ubuntu 24.04 LTS
# Run as root on a fresh Hetzner / Scaleway VPS.
# ============================================================
set -euo pipefail

BOTUSER="botuser"
PROJECT_DIR="/home/$BOTUSER/polymarket-bot"

echo "──── 1. System updates & hardening ────"
apt-get update && apt-get upgrade -y
apt-get install -y \
    ufw fail2ban git python3.12 python3.12-venv python3-pip \
    age jq curl unattended-upgrades

# Firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw --force enable

# fail2ban
systemctl enable fail2ban
systemctl start fail2ban

# Disable root SSH
sed -i 's/^PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

echo "──── 2. Create bot user ────"
if ! id "$BOTUSER" &>/dev/null; then
    adduser --disabled-password --gecos "" "$BOTUSER"
    mkdir -p /home/$BOTUSER/.ssh
    # Copy your SSH public key here:
    # echo "ssh-ed25519 AAAA..." > /home/$BOTUSER/.ssh/authorized_keys
    chmod 700 /home/$BOTUSER/.ssh
    chmod 600 /home/$BOTUSER/.ssh/authorized_keys 2>/dev/null || true
    chown -R $BOTUSER:$BOTUSER /home/$BOTUSER/.ssh
fi

echo "──── 3. Create tmpfs for secrets ────"
mkdir -p /dev/shm/secrets
chown $BOTUSER:$BOTUSER /dev/shm/secrets
chmod 700 /dev/shm/secrets

echo "──── 4. Clone / copy project ────"
if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
    chown $BOTUSER:$BOTUSER "$PROJECT_DIR"
    echo "⚠  Copy your project files to $PROJECT_DIR"
fi

echo "──── 5. Python venv ────"
su - $BOTUSER -c "
    cd $PROJECT_DIR
    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e '.[dev]'
"

echo "──── 6. Systemd service ────"
cp "$PROJECT_DIR/scripts/polymarket-bot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable polymarket-bot

echo "──── 7. Log directory ────"
mkdir -p /var/log/$BOTUSER
chown $BOTUSER:$BOTUSER /var/log/$BOTUSER

echo ""
echo "✅  VPS setup complete."
echo ""
echo "Next steps:"
echo "  1. Copy your .env.age to $PROJECT_DIR/"
echo "  2. Decrypt: su - $BOTUSER -c 'age -d -o /dev/shm/secrets/.env $PROJECT_DIR/.env.age'"
echo "  3. Start:   systemctl start polymarket-bot"
echo "  4. Logs:    journalctl -u polymarket-bot -f"
