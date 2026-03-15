"""
CLI entry point for the Polymarket Mean-Reversion Market Maker.

Usage:
    polybot run                            # Paper trading (default)
    polybot run --env PENNY_LIVE           # Penny-live: real CLOB, $1 cap
    polybot run --env PRODUCTION --confirm-production  # Full production
    polybot stats                          # Print aggregate stats
    polybot scores                         # Print current market scores
    polybot kill                           # Emergency shutdown
"""

from __future__ import annotations

import asyncio
import json as _json
import sys
import time
from pathlib import Path

import click

from src.core.config import DeploymentEnv
from src.core.logger import setup_logging
from src.data.synthetic import (
    _DEFAULT_NO_ASSET as _SYNTH_NO,
    _DEFAULT_YES_ASSET as _SYNTH_YES,
)


def _startup_banner(env: DeploymentEnv, confirm: bool) -> None:
    """Print a loud startup banner and enforce a 5-second countdown
    when launching in any mode that touches real capital.

    In PAPER mode, prints a harmless one-liner.
    """
    if env == DeploymentEnv.PAPER:
        click.echo(
            click.style(
                "  ▸ Starting in PAPER mode — no real capital at risk.",
                fg="green",
            )
        )
        return

    from src.core.config import settings, PENNY_LIVE_MAX_TRADE_USD

    # Build wallet address display — NEVER show private key material
    wallet_key = settings.eoa_private_key
    if wallet_key:
        try:
            from eth_account import Account
            acct = Account.from_key(wallet_key)
            addr = acct.address
            wallet_display = addr[:6] + "\u2026" + addr[-4:]
        except Exception:
            wallet_display = "(key set, address derivation failed)"
    else:
        wallet_display = "(not set)"

    if env == DeploymentEnv.PENNY_LIVE:
        max_exposure = f"${PENNY_LIVE_MAX_TRADE_USD:.2f} USDC per trade"
        colour = "yellow"
    else:
        max_exposure = f"${settings.strategy.max_trade_size_usd:.2f} USDC (Kelly-derived)"
        colour = "red"

    width = 62
    border = "█" * width
    pad = "█" + " " * (width - 2) + "█"
    title = f"  ⚠️   LIVE CAPITAL MODE: {env.value}  ⚠️"

    click.echo()
    click.echo(click.style(border, fg=colour, bold=True))
    click.echo(click.style(pad, fg=colour))
    click.echo(click.style(f"█  {title:<{width - 4}}█", fg=colour, bold=True))
    click.echo(click.style(pad, fg=colour))
    click.echo(click.style(f"█  Wallet : {wallet_display:<{width - 14}}█", fg=colour))
    click.echo(click.style(f"█  Max exp: {max_exposure:<{width - 14}}█", fg=colour))
    click.echo(click.style(pad, fg=colour))
    click.echo(click.style(border, fg=colour, bold=True))
    click.echo()

    # 5-second countdown
    for remaining in range(5, 0, -1):
        click.echo(
            click.style(f"  Starting in {remaining}…", fg=colour),
            nl=True,
        )
        time.sleep(1)

    click.echo(click.style("  GO.", fg=colour, bold=True))
    click.echo()


@click.group()
def main() -> None:
    """Polymarket Mean-Reversion Market Maker PoC."""
    pass


@main.command()
@click.option(
    "--env",
    "deploy_env",
    type=click.Choice(["PAPER", "PENNY_LIVE", "PRODUCTION"], case_sensitive=False),
    default="PAPER",
    help="Deployment phase: PAPER (default), PENNY_LIVE, or PRODUCTION.",
)
@click.option(
    "--confirm-production",
    is_flag=True,
    default=False,
    help="Required safety flag for PRODUCTION deployment.",
)
@click.option("--log-dir", default="logs", help="Directory for log output.")
@click.option(
    "--paper/--live",
    default=None,
    hidden=True,
    help="(Deprecated) Use --env instead.",
)
def run(
    deploy_env: str,
    confirm_production: bool,
    log_dir: str,
    paper: bool | None,
) -> None:
    """Start the trading bot."""
    # Legacy --paper/--live flag support
    if paper is not None and deploy_env == "PAPER":
        # User used old-style flag without --env
        deploy_env = "PAPER" if paper else "PRODUCTION"

    env = DeploymentEnv(deploy_env.upper())

    # ── PRODUCTION gate ────────────────────────────────────────────────
    if env == DeploymentEnv.PRODUCTION and not confirm_production:
        raise RuntimeError(
            "PRODUCTION mode requires --confirm-production flag. "
            "This is not a drill.\n\n"
            "  Usage: polybot run --env PRODUCTION --confirm-production"
        )

    if env == DeploymentEnv.PENNY_LIVE:
        proceed = click.confirm(
            "PENNY_LIVE uses real wallet funds. Continue?",
            default=False,
        )
        if not proceed:
            click.echo("Aborted PENNY_LIVE launch.", err=True)
            sys.exit(1)

    # ── Credential validation ──────────────────────────────────────────
    if env != DeploymentEnv.PAPER:
        from src.core.config import settings

        missing = settings.validate_credentials()
        if missing:
            for msg in missing:
                click.echo(click.style(f"  ✗ {msg}", fg="red"), err=True)
            sys.exit(1)

    setup_logging(log_dir)

    # ── Startup banner + countdown ─────────────────────────────────────
    _startup_banner(env, confirm_production)

    from src.bot import TradingBot

    bot = TradingBot(
        deployment_env=env,
        confirmed_production=confirm_production,
    )

    if sys.platform == "win32":
        # Windows: ProactorEventLoop is the default on 3.10+ and supports
        # subprocesses.  Only override if running an older Python.
        pass
    else:
        # On Linux/macOS, install uvloop for ~2-4x event loop throughput.
        try:
            import uvloop  # type: ignore[import-untyped]
            uvloop.install()
        except ImportError:
            pass

    asyncio.run(bot.start())


@main.command()
@click.option("--db", default="logs/trades.db", help="Path to SQLite trade database.")
def stats(db: str) -> None:
    """Print aggregate trade statistics."""

    async def _stats() -> None:
        from src.monitoring.trade_store import TradeStore

        store = TradeStore(db)
        await store.init()
        s = await store.get_stats()
        await store.close()

        click.echo("\n📊  Trade Statistics")
        click.echo("─" * 40)
        for k, v in s.items():
            click.echo(f"  {k:.<30} {v}")

        from src.monitoring.trade_store import TradeStore as TS

        store2 = TradeStore(db)
        await store2.init()
        ready, _ = await store2.passes_go_live_criteria()
        await store2.close()
        status = "✅ READY" if ready else "❌ NOT READY"
        click.echo(f"\n  Go-live readiness: {status}\n")

    asyncio.run(_stats())


@main.command()
def kill() -> None:
    """Send a kill signal (placeholder — in production, signal via Telegram or PID)."""
    click.echo("🛑  Kill signal sent.  If running as systemd, use:")
    click.echo("    sudo systemctl stop polymarket-bot")


@main.command("shadow-report")
@click.option("--db", default="logs/trades.db", help="Path to SQLite trade database.")
@click.option("--source", default=None, help="Filter to a specific signal source (e.g. SI-3).")
def shadow_report(db: str, source: str | None) -> None:
    """Generate a tearsheet for shadow strategies and evaluate go-live readiness."""

    async def _report() -> None:
        from src.monitoring.trade_store import TradeStore

        store = TradeStore(db)
        await store.init()

        if source:
            sources = [source]
        else:
            sources = await store.get_all_shadow_sources()

        if not sources:
            click.echo("\n⚠️  No shadow trades found.\n")
            await store.close()
            return

        click.echo("\n👻  Shadow Strategy Performance Report")
        click.echo("═" * 60)

        for src_name in sources:
            ready, stats = await store.passes_shadow_go_live(src_name)

            total = stats.get("total_trades", 0)
            wr = stats.get("win_rate", 0.0)
            ev = stats.get("expectancy_cents", 0.0)
            total_pnl = stats.get("total_pnl_cents", 0.0)
            avg_win = stats.get("avg_win_cents", 0.0)
            avg_loss = stats.get("avg_loss_cents", 0.0)
            max_dd = stats.get("max_drawdown_cents", 0.0)
            target_exits = stats.get("target_exits", 0)
            stop_exits = stats.get("stop_exits", 0)
            avg_hold = stats.get("avg_hold_seconds", 0)

            click.echo(f"\n📊  Signal Source: {src_name}")
            click.echo("─" * 50)
            click.echo(f"  {'Total trades':<30} {total}")
            click.echo(f"  {'Win rate':<30} {wr:.1%}")
            click.echo(f"  {'Expectancy (cents)':<30} {ev:+.2f}")
            click.echo(f"  {'Total PnL (cents)':<30} {total_pnl:+.2f}")
            click.echo(f"  {'Avg win (cents)':<30} {avg_win:+.2f}")
            click.echo(f"  {'Avg loss (cents)':<30} {avg_loss:+.2f}")
            click.echo(f"  {'Max drawdown (cents)':<30} {max_dd:.2f}")
            click.echo(f"  {'TP exits':<30} {target_exits}")
            click.echo(f"  {'SL exits':<30} {stop_exits}")
            click.echo(f"  {'Avg hold (seconds)':<30} {avg_hold:.0f}")

            # Go-live criteria evaluation
            criteria_lines = []
            criteria_lines.append(
                f"    {'≥20 trades':<26} {'✅' if total >= 20 else '❌'}  ({total})"
            )
            criteria_lines.append(
                f"    {'WR ≥ 55%':<26} {'✅' if wr >= 0.55 else '❌'}  ({wr:.1%})"
            )
            criteria_lines.append(
                f"    {'Positive EV':<26} {'✅' if ev > 0 else '❌'}  ({ev:+.2f}¢)"
            )

            click.echo(f"\n  Go-live criteria:")
            for line in criteria_lines:
                click.echo(line)

            if ready:
                click.echo(
                    click.style(
                        f"\n  ✅ {src_name}: READY FOR DEPLOYMENT",
                        fg="green",
                        bold=True,
                    )
                )
            else:
                click.echo(
                    click.style(
                        f"\n  ❌ {src_name}: NOT READY",
                        fg="red",
                    )
                )

        click.echo()
        await store.close()

    asyncio.run(_report())


@main.command()
def scores() -> None:
    """Discover markets and print their quality scores (no trading)."""

    async def _scores() -> None:
        from src.data.market_lifecycle import MarketLifecycleManager

        lm = MarketLifecycleManager()
        active = await lm.initial_discovery()

        click.echo("\n📈  Market Quality Scores")
        click.echo("═" * 80)

        # Active tier
        click.echo(f"\n🟢  Active ({len(lm.active)} markets)")
        click.echo("─" * 80)
        for am in sorted(lm.active.values(), key=lambda x: x.score.total, reverse=True):
            bd = am.score
            q = am.info.question[:55].ljust(55)
            click.echo(
                f"  {q}  "
                f"TOTAL={bd.total:5.1f}  "
                f"vol={bd.volume:4.0f} liq={bd.liquidity:4.0f} "
                f"sprd={bd.spread:4.0f} ttr={bd.time_to_resolve:4.0f} "
                f"prng={bd.price_range:4.0f} freq={bd.trade_freq:4.0f} "
                f"whale={bd.whale_interest:4.0f}"
            )

        # Observing tier
        if lm.observing:
            click.echo(f"\n🟡  Observing ({len(lm.observing)} markets)")
            click.echo("─" * 80)
            for om in sorted(
                lm.observing.values(), key=lambda x: x.score.total, reverse=True
            ):
                bd = om.score
                q = om.info.question[:55].ljust(55)
                click.echo(
                    f"  {q}  "
                    f"TOTAL={bd.total:5.1f}  "
                    f"vol={bd.volume:4.0f} liq={bd.liquidity:4.0f}"
                )

        click.echo()

    asyncio.run(_scores())


@main.command()
@click.option("--data-dir", required=True, help="Path to recorded tick data directory.")
@click.option("--asset-id", default=None, help="Filter to a single asset ID.")
@click.option("--initial-cash", default=1000.0, help="Starting cash balance (USD).")
@click.option("--latency-ms", default=150.0, help="Simulated latency in milliseconds.")
@click.option("--fee-max-pct", default=1.56, help="Max fee rate percentage.")
@click.option("--no-fees", is_flag=True, help="Disable dynamic fee curve.")
@click.option("--market-id", default="BACKTEST", help="Market condition ID.")
@click.option("--yes-asset", default=_SYNTH_YES, help="YES token asset ID.")
@click.option("--no-asset", default=_SYNTH_NO, help="NO token asset ID.")
def backtest(
    data_dir: str,
    asset_id: str | None,
    initial_cash: float,
    latency_ms: float,
    fee_max_pct: float,
    no_fees: bool,
    market_id: str,
    yes_asset: str,
    no_asset: str,
) -> None:
    """Run a backtest on recorded historical data."""
    from src.backtest.data_loader import DataLoader
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.backtest.strategy import BotReplayAdapter

    asset_ids = {asset_id} if asset_id else None
    loader = DataLoader.from_directory(data_dir, asset_ids=asset_ids)

    config = BacktestConfig(
        initial_cash=initial_cash,
        latency_ms=latency_ms,
        fee_max_pct=fee_max_pct,
        fee_enabled=not no_fees,
    )

    strategy = BotReplayAdapter(
        market_id=market_id,
        yes_asset_id=yes_asset,
        no_asset_id=no_asset,
        fee_enabled=not no_fees,
        initial_bankroll=initial_cash,
    )

    engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
    result = engine.run()

    click.echo(result.summary())


@main.command()
@click.option("--data-dir", required=True, help="Path to recorded tick data directory.")
@click.option("--train-days", default=30, help="In-Sample training window (days).")
@click.option("--test-days", default=7, help="Out-of-Sample testing window (days).")
@click.option("--step-days", default=7, help="Step size between folds (days).")
@click.option("--n-trials", default=100, help="Optuna trials per fold.")
@click.option("--max-workers", default=None, type=int, help="Parallel processes (default: cpu_count - 1). Use -1 for all cores.")
@click.option("--max-drawdown", default=0.15, help="Max acceptable drawdown (fraction).")
@click.option("--initial-cash", default=1000.0, help="Starting cash balance (USD).")
@click.option("--market-id", default="BACKTEST", help="Market condition ID.")
@click.option("--yes-asset", default=_SYNTH_YES, help="YES token asset ID.")
@click.option("--no-asset", default=_SYNTH_NO, help="NO token asset ID.")
@click.option("--latency-ms", default=150.0, help="Simulated latency in milliseconds.")
@click.option("--fee-max-pct", default=1.56, help="Max fee rate percentage.")
@click.option("--no-fees", is_flag=True, help="Disable dynamic fee curve.")
@click.option("--storage", default="sqlite:///wfo_optuna.db", help="Optuna RDB storage URL.")
@click.option("--anchored", is_flag=True, help="Use expanding (anchored) IS window.")
@click.option("--embargo-days", default=1, help="Gap in calendar days between IS/OOS windows.")
@click.option("--output-params", default=None, type=str, help="Path to export champion parameters JSON.")
@click.option("--max-markets", default=None, type=int, help="Limit multi-market universe to N markets (default: all).")
@click.option("--search-space-bounds", default=None, type=str, help="Path to JSON file with narrowed search-space bounds.")
def wfo(
    data_dir: str,
    train_days: int,
    test_days: int,
    step_days: int,
    n_trials: int,
    max_workers: int | None,
    max_drawdown: float,
    initial_cash: float,
    market_id: str,
    yes_asset: str,
    no_asset: str,
    latency_ms: float,
    fee_max_pct: float,
    no_fees: bool,
    storage: str,
    anchored: bool,
    embargo_days: int,
    output_params: str | None,
    max_markets: int | None,
    search_space_bounds: str | None,
) -> None:
    """Run Walk-Forward Optimization on recorded historical data."""
    import os

    from src.core.logger import setup_logging

    setup_logging(log_dir="logs", log_file="wfo.jsonl")

    from src.backtest.wfo_optimizer import WfoConfig, run_wfo

    # Resolve max_workers: -1 means all cores, None means cpu_count - 1
    if max_workers is not None and max_workers == -1:
        resolved_workers = os.cpu_count() or 1
    else:
        resolved_workers = max_workers or max((os.cpu_count() or 2) - 1, 1)

    cfg = WfoConfig(
        data_dir=data_dir,
        market_id=market_id,
        yes_asset_id=yes_asset,
        no_asset_id=no_asset,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        n_trials=n_trials,
        max_workers=resolved_workers,
        max_acceptable_drawdown=max_drawdown,
        initial_cash=initial_cash,
        storage_url=storage,
        latency_ms=latency_ms,
        fee_max_pct=fee_max_pct,
        fee_enabled=not no_fees,
        anchored=anchored,
        embargo_days=embargo_days,
        output_params_path=output_params,
        max_markets=max_markets,
        search_space_bounds_path=search_space_bounds,
    )

    report = run_wfo(cfg)
    try:
        click.echo(report.summary())
    except UnicodeEncodeError:
        # Windows console may choke on box-drawing chars when piped
        click.echo(report.summary().encode("ascii", errors="replace").decode())


# ═══════════════════════════════════════════════════════════════════════════
#  Data pipeline commands (mock / process)
# ═══════════════════════════════════════════════════════════════════════════


@main.group()
def data() -> None:
    """Data pipeline utilities (generate mock data, convert to Parquet)."""
    pass


@data.command("mock")
@click.option(
    "--output-dir",
    default="data",
    help="Root output directory (files go into <dir>/raw_ticks/...).",
)
@click.option("--num-rows", default=100_000, type=int, help="Total rows to generate.")
@click.option(
    "--duration-hours", default=24.0, type=float, help="Simulated time span in hours."
)
@click.option("--num-assets", default=2, type=int, help="Number of distinct assets.")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--gap-prob", default=0.0, type=float, help="Probability of injecting sequence gaps.")
@click.option("--spike-prob", default=0.0, type=float, help="Probability of injecting price spikes.")
@click.option("--spread-compress-prob", default=0.0, type=float, help="Probability of injecting spread compressions.")
def data_mock(
    output_dir: str,
    num_rows: int,
    duration_hours: float,
    num_assets: int,
    seed: int | None,
    gap_prob: float,
    spike_prob: float,
    spread_compress_prob: float,
) -> None:
    """Generate synthetic Polymarket L2/trade JSONL data for testing."""
    from src.data.synthetic import SyntheticGenerator

    gen = SyntheticGenerator(
        seed=seed,
        gap_probability=gap_prob,
        spike_probability=spike_prob,
        spread_compress_probability=spread_compress_prob,
    )
    ts_start = time.time()
    raw_dir = gen.generate(
        output_dir,
        num_rows=num_rows,
        duration_hours=duration_hours,
        num_assets=num_assets,
    )
    elapsed = time.time() - ts_start

    # Count files
    files = list(raw_dir.rglob("*.jsonl"))

    click.echo(f"\n  Synthetic data generated in {elapsed:.1f}s")
    click.echo(f"  Rows: {num_rows:,}  |  Assets: {num_assets}  |  Duration: {duration_hours}h")
    click.echo(f"  Files: {len(files)}")
    for f in files:
        click.echo(f"    → {f}")
    click.echo()


@data.command("process")
@click.option(
    "--input-dir",
    required=True,
    help="Path to raw JSONL directory (e.g. data/raw_ticks).",
)
@click.option(
    "--output-dir",
    default="data/processed",
    help="Root output directory for Parquet files.",
)
@click.option(
    "--category-map",
    default=None,
    type=click.Path(exists=True),
    help='Path to JSON file mapping asset_id → category (e.g. {"0xabc": "crypto"}).',
)
def data_process(
    input_dir: str,
    output_dir: str,
    category_map: str | None,
) -> None:
    """Convert raw JSONL tick data into optimised Parquet files."""
    from src.data.prep_data import ParquetConverter

    cat_map: dict[str, str] | None = None
    if category_map:
        with open(category_map, "r", encoding="utf-8") as fh:
            cat_map = _json.load(fh)

    converter = ParquetConverter(category_map=cat_map)
    ts_start = time.time()
    report = converter.convert([Path(input_dir)], output_dir)
    elapsed = time.time() - ts_start

    click.echo(report.summary())
    click.echo(f"  Completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
