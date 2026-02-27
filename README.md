# Polymarket Mean-Reversion Market Maker — PoC

A proof-of-concept automated trading bot that fades retail panic in Polymarket
prediction markets by buying discounted **NO** shares during spikes and exiting
via dynamic mean-reversion targets.

## Architecture

```
src/
├── core/           # Config, logging
├── data/           # WebSocket client, OHLCV aggregation, market discovery
├── signals/        # Panic spike detector, whale wallet monitor
├── trading/        # Order executor, position manager, take-profit calc
├── monitoring/     # SQLite trade store, Telegram alerts
├── bot.py          # Main orchestrator
└── cli.py          # Click CLI entry point

scripts/
├── vps_setup.sh            # Ubuntu VPS bootstrap
├── polymarket-bot.service  # systemd unit file
├── decrypt_secrets.sh      # Age-encrypted .env decryption
└── watchdog.sh             # Cron-based health check
```

## Quick Start (Local / Paper Trading)

```bash
# 1. Clone & setup
cd polymarket-bot
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env with your Polymarket API credentials

# 3. Run in paper mode (default)
python -m src.cli run --paper

# 4. Check stats
python -m src.cli stats
```

## Strategy Logic

1. **Signal**: Detect Z-score ≥ 2σ spike in YES price + 3× volume surge
2. **Entry**: GTC limit buy on NO shares at `best_ask - 1¢` (maker)
3. **Exit**: Dynamic target via `P_entry + α × (VWAP_no - P_entry)`
   - α ∈ [0.3, 0.7] adjusted by volatility, book depth, whale confluence, time to resolution
4. **Timeout**: Force market-sell after 30 min if target not hit

## Testing

```bash
pytest tests/ -v
```

## VPS Deployment

See `scripts/vps_setup.sh` for full bootstrap. Summary:

1. Provision Hetzner CX22 (Helsinki) — Ubuntu 24.04
2. Run `vps_setup.sh` as root
3. Copy `.env.age` encrypted secrets
4. `systemctl start polymarket-bot`

## Security

- Private keys stored in `.env`, encrypted at rest with `age`
- Decrypted only into tmpfs (`/dev/shm/secrets/`) — RAM only
- SSH key-only auth, `ufw` firewall, `fail2ban`
- systemd runs with `NoNewPrivileges`, `ProtectSystem=strict`
