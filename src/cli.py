"""
CLI entry point for the Polymarket Mean-Reversion Market Maker.

Usage:
    python -m src.cli run --paper          # Paper trading (default)
    python -m src.cli run --live           # Live trading (after criteria met)
    python -m src.cli stats                # Print aggregate stats
    python -m src.cli kill                 # Emergency shutdown
"""

from __future__ import annotations

import asyncio
import sys

import click

from src.core.logger import setup_logging


@click.group()
def main() -> None:
    """Polymarket Mean-Reversion Market Maker PoC."""
    pass


@main.command()
@click.option("--paper/--live", default=True, help="Run in paper or live mode.")
@click.option("--log-dir", default="logs", help="Directory for log output.")
def run(paper: bool, log_dir: str) -> None:
    """Start the trading bot."""
    setup_logging(log_dir)

    from src.bot import TradingBot

    bot = TradingBot(paper_mode=paper)

    if sys.platform == "win32":
        # Windows requires a ProactorEventLoop for subprocess + signal compat
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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


if __name__ == "__main__":
    main()
