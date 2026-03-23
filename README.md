# Polymarket HFT Market Maker And Latency Arb Engine

An automated Polymarket trading system centered on high-frequency pure market
making, maker-first combinatorial arbitrage, and zero-overhead latency
arbitrage.

This repo is no longer a mean-reversion panic bot as its primary live design.
Legacy directional modules still exist for research and compatibility, but the
current live architecture is driven by L2 market microstructure and external
latency edges.

## Live Architecture

```
src/
├── core/           # Config, latency/risk guards, process orchestration
├── data/           # CLOB streams, L2 books, oracle adapters, discovery
├── signals/        # RPE, oracle signals, SI-9 combinatorial arb logic
├── strategies/     # Pure market maker live strategy
├── trading/        # Executor, position manager, fast-strike path, risk
├── monitoring/     # SQLite trade store, Telegram alerts
├── bot.py          # Main orchestrator
└── cli.py          # CLI entry point

scripts/
├── vps_setup.sh            # Ubuntu VPS bootstrap
├── decrypt_secrets.sh      # Age-encrypted .env decryption
├── test_ws_oracles.py      # Tree News / oracle websocket smoke tests
└── watchdog.sh             # Cron-based health check
```

## Active Strategies

1. Pure Market Maker
   The passive NO-token quoting engine remains in the codebase, but PURE_MM is
   currently disabled on the Top 25 / high-volume live paper target map.
   Recent out-of-sample WFO on that universe produced OOS Sharpe -79.5565,
   confirming the books are currently too toxic for live passive deployment.

2. SI-9 Combinatorial Arbitrage
   Maker-first state machine for mutually exclusive event clusters. Works the
   bottleneck leg passively before completing the rest of the combo when a
   Dutch-book style mispricing exists. This strategy is currently live in paper
   mode. Current guardrails require minimum Σ bids of 0.85, maximum edge of
   0.15 dollars, and throttle repeated rejection logs to 1 hour unless Σ bids
   moves by more than 5 percentage points.

3. Latency Arbitrage
   SI-7 crypto fast-strike uses free BTC and ETH spot feeds plus
   Black-Scholes-style RPE pricing to snipe stale crypto-linked Polymarket
   orders.

   SI-8 news arbitrage uses the Tree News WebSocket firehose to front-run the
   book on breaking news. Paid non-crypto and paid sports feeds are outside the
   intended zero-overhead live architecture.

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

## Backtesting And WFO

The backtest and walk-forward stack has pivoted with the live strategy.

- Pure market maker replay requires L2 snapshots and deltas.
- Trade-only data is not sufficient to simulate passive maker fills, OFI, or
   depth evaporation defensibly.
- Pure-MM WFO intentionally fails fast when the dataset has no L2 book events.
- Recent pure-MM OOS WFO on the current high-volume universe recorded OOS
   Sharpe -79.5565, so PURE_MM is not the active live paper strategy on the
   current target map.

## Current Live Posture

- SI-9 combinatorial arbitrage is live in paper mode.
- Gamma discovery disables ONE_MARKET_PER_EVENT whenever SI-9 is enabled so all
   negRisk legs survive discovery.
- The live target map should stay below 25 markets in practice to maintain L2
   WebSocket stability and avoid packet drops.

If you are evaluating the current live architecture, assume L2 data is a hard
requirement rather than an optional enhancement.

## Offline Tools

`scripts/visualize_l2_wicks.py` is a standalone offline analysis tool for local
tick captures. It reconstructs the Level 2 top of book from snapshot and delta
events, plots best bid and best ask over time, overlays trades, and flags
potential liquidity vacuums or wick trades where executions occur more than 5%
away from the rolling 1-minute mid-price baseline.

Use it when you want to visually inspect market microstructure, deep sweeps,
and flash-move executions without running the live bot.

Example usage:

```bash
python scripts/visualize_l2_wicks.py data/vps_march2026/ticks/2026-03-18 <market_id>
python scripts/visualize_l2_wicks.py data/vps_march2026_parquet/2026-03-18 <market_id> --output wick_chart.png
python scripts/visualize_l2_wicks.py data/vps_march2026/ticks/2026-03-18 <market_id> --wick-threshold-pct 5 --window-seconds 60 --show
```

The script accepts either raw `.jsonl` captures or prepared `.parquet` files.
If the selected dataset contains only trade prints and no L2 snapshot or delta
events, the visualizer exits early because book reconstruction is not possible.

`scripts/screen_wick_markets.py` is a standalone universe-selection screener.
It queries the public Polymarket Gamma API for active markets, ranks them by
`24h volume / resting liquidity`, and prints the top high-volume, thin-book
candidates for next week's wick-catching watchlist.

Example usage:

```bash
python scripts/screen_wick_markets.py
python scripts/screen_wick_markets.py --top 25 --min-volume 1000
```

`scripts/screen_si9_clusters.py` is the offline SI-9 cluster screener.
It queries the public Polymarket Gamma `/events` API, filters to active
mutually exclusive negRisk clusters, groups markets by parent event, and ranks
those clusters by aggregate `24h volume / resting liquidity`.

Example usage:

```bash
python scripts/screen_si9_clusters.py
python scripts/screen_si9_clusters.py --top 10 --min-volume 5000
python scripts/screen_si9_clusters.py --export-json data/si9_clusters.json
```

Its JSON export preserves CLOB-compatible `conditionId` hex strings for every
leg in each cluster, rather than Gamma numeric market IDs.

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
