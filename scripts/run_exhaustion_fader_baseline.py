from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_exhaustion_fader import load_fade_fills, reconstruct_markouts, summarize
from scripts.run_universal_backtest import UniversalReplayEngine, load_market_catalog


DEFAULT_START_DATE = "2026-03-15"
DEFAULT_END_DATE = "2026-03-19"
DEFAULT_PROGRESS_EVERY_EVENTS = 200_000
STRATEGY_PATH = "src.signals.exhaustion_fader.ExhaustionFader"


@dataclass(slots=True)
class BaselineDayResult:
    date: str
    trigger_count_per_day: int
    total_simulated_fills: int
    pnl_usd: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-day strict ExhaustionFader baseline with visible progress.")
    parser.add_argument("--input-dir", default="logs/local_snapshot/l2_data", help="Replay source root containing the raw tick archive.")
    parser.add_argument("--market-map", default="data/market_map.json", help="Market map used to resolve YES/NO token ids for matching.")
    parser.add_argument("--db-dir", default="logs", help="Directory receiving per-day replay SQLite databases.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--output", default=None, help="Optional markdown output path. Defaults to a range-stamped artifact path.")
    parser.add_argument(
        "--progress-every-events",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_EVENTS,
        help="Heartbeat interval for long replay and analysis loops.",
    )
    return parser.parse_args()


def iter_dates(start_date: str, end_date: str) -> list[str]:
    current = date.fromisoformat(start_date)
    final = date.fromisoformat(end_date)
    if final < current:
        raise ValueError("end_date must be on or after start_date")
    dates: list[str] = []
    while current <= final:
        dates.append(current.isoformat())
        current += timedelta(days=1)
    return dates


def default_output_path(start_date: str, end_date: str) -> Path:
    return Path("artifacts") / f"exhaustion_fader_baseline_{start_date}_{end_date}.md"


def render_markdown(
    results: list[BaselineDayResult],
    *,
    start_date: str,
    end_date: str,
    input_dir: Path,
) -> str:
    total_triggers = sum(result.trigger_count_per_day for result in results)
    total_fills = sum(result.total_simulated_fills for result in results)
    total_pnl_usd = sum(result.pnl_usd for result in results)
    lines = [
        f"# ExhaustionFader Strict Baseline ({start_date} to {end_date})",
        "",
        "## Setup",
        "",
        f"- Strategy: `{STRATEGY_PATH}`",
        f"- Replay source: `{input_dir}`",
        "- Trigger count proxy: replay dispatch count, because ExhaustionFader does not expose a separate trigger diagnostics counter.",
        "- Fill and PnL figures: 60-second mark-to-mid values reconstructed from the existing analyzer without modifying the strategy.",
        "",
        "## Baseline Table",
        "",
        "| Date | trigger_count_per_day | Total Simulated Fills | PnL (USD) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {date} | {triggers} | {fills} | {pnl:.2f} |".format(
                date=result.date,
                triggers=result.trigger_count_per_day,
                fills=result.total_simulated_fills,
                pnl=result.pnl_usd,
            )
        )
    lines.extend(
        [
            "| Total | {triggers} | {fills} | {pnl:.2f} |".format(
                triggers=total_triggers,
                fills=total_fills,
                pnl=total_pnl_usd,
            ),
            "",
            "## Verdict",
            "",
            "Across the measured range, the strict unmodified ExhaustionFader generated `{triggers}` total triggers, `{fills}` simulated fills, and `${pnl:.2f}` of aggregated 60-second mark-to-mid PnL.".format(
                triggers=total_triggers,
                fills=total_fills,
                pnl=total_pnl_usd,
            ),
        ]
    )
    return "\n".join(lines) + "\n"


async def run_day(
    day: str,
    *,
    input_dir: Path,
    db_dir: Path,
    market_catalog,
    progress_every_events: int | None,
) -> BaselineDayResult:
    db_path = db_dir / f"universal_backtest_exhaustion_{day}.db"
    if db_path.exists():
        db_path.unlink()

    print(f"baseline_day_start date={day} phase=replay db={db_path}", flush=True)
    engine = UniversalReplayEngine(
        input_dir=input_dir,
        db_path=db_path,
        strategy_path=STRATEGY_PATH,
        market_catalog=market_catalog,
        start_date=day,
        end_date=day,
        progress_every_events=progress_every_events,
        progress_label=f"exhaustion-replay:{day}",
    )
    replay_summary = await engine.run()
    print(
        "baseline_day_complete date={day} phase=replay dispatches={dispatches} maker_fills={maker_fills} persisted_shadow_rows={rows}".format(
            day=day,
            dispatches=replay_summary.dispatches,
            maker_fills=replay_summary.maker_fills,
            rows=replay_summary.persisted_shadow_rows,
        ),
        flush=True,
    )

    print(f"baseline_day_start date={day} phase=analysis db={db_path}", flush=True)
    fills = load_fade_fills(db_path, start_date=day, end_date=day)
    reconstruct_markouts(
        fills,
        input_dir=input_dir,
        start_date=day,
        end_date=day,
        progress_every_events=progress_every_events,
        progress_label=f"exhaustion-analysis:{day}",
    )
    analysis_summary = summarize(fills)
    total_simulated_fills = int(analysis_summary["total_fades"])
    if total_simulated_fills != replay_summary.maker_fills:
        raise RuntimeError(
            "Fill count mismatch for {day}: replay maker_fills={maker_fills}, analyzer total_fades={total_fades}".format(
                day=day,
                maker_fills=replay_summary.maker_fills,
                total_fades=total_simulated_fills,
            )
        )
    pnl_usd = float(analysis_summary["total_pnl_cents"]) / 100.0
    print(
        "baseline_day_complete date={day} phase=analysis triggers={triggers} fills={fills} pnl_usd={pnl:.2f}".format(
            day=day,
            triggers=replay_summary.dispatches,
            fills=total_simulated_fills,
            pnl=pnl_usd,
        ),
        flush=True,
    )
    return BaselineDayResult(
        date=day,
        trigger_count_per_day=replay_summary.dispatches,
        total_simulated_fills=total_simulated_fills,
        pnl_usd=pnl_usd,
    )


async def _main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else default_output_path(args.start_date, args.end_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    market_catalog = load_market_catalog(Path(args.market_map) if args.market_map else None)

    results: list[BaselineDayResult] = []
    for day in iter_dates(args.start_date, args.end_date):
        results.append(
            await run_day(
                day,
                input_dir=input_dir,
                db_dir=db_dir,
                market_catalog=market_catalog,
                progress_every_events=args.progress_every_events,
            )
        )

    markdown = render_markdown(
        results,
        start_date=args.start_date,
        end_date=args.end_date,
        input_dir=input_dir,
    )
    output_path.write_text(markdown, encoding="utf-8")
    print("\n---MARKDOWN---\n", flush=True)
    print(markdown, flush=True)
    print(f"baseline_output={output_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(_main())