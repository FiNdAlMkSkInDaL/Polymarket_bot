from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_universal_backtest import (
    UniversalReplayEngine,
    iter_file_events,
    load_market_catalog,
    load_strategy_config,
)
from src.data.orderbook import OrderbookTracker
from src.trading.fees import get_fee_rate


DEFAULT_DB_PATH = Path("logs/batch_vacuum_backtest.db")
DEFAULT_MARKDOWN_PATH = Path("artifacts/vacuum_batch_backtest_scorecard.md")
STRATEGY_PATH = "src.signals.vacuum_maker.VacuumMaker"


@dataclass(slots=True)
class BatchRunResult:
    batch_label: str
    order_id_prefix: str
    file_path: Path
    file_size_bytes: int
    total_events: int
    book_events: int
    trade_events: int
    dispatches: int
    rejections: int
    maker_fills: int
    persisted_shadow_rows: int


@dataclass(slots=True)
class FillMarkout:
    fill_id: str
    market_id: str
    asset_id: str
    quote_side: str
    entry_price: float
    entry_size: float
    entry_time_ms: int
    fill_mid_yes: float | None = None
    future_mid_yes: float | None = None

    @property
    def covered(self) -> bool:
        return self.future_mid_yes is not None

    @property
    def gross_edge_cents(self) -> float | None:
        if self.future_mid_yes is None:
            return None
        if self.quote_side == "ASK":
            return (self.entry_price - self.future_mid_yes) * 100.0
        return (self.future_mid_yes - self.entry_price) * 100.0

    @property
    def net_edge_cents(self) -> float | None:
        gross_edge = self.gross_edge_cents
        if gross_edge is None or self.future_mid_yes is None:
            return None
        return gross_edge - (get_fee_rate(self.future_mid_yes) * 100.0)

    @property
    def pnl_cents(self) -> float:
        net_edge = self.net_edge_cents
        if net_edge is None:
            return 0.0
        return net_edge * self.entry_size

    @property
    def is_win(self) -> int:
        net_edge = self.net_edge_cents
        return 1 if net_edge is not None and net_edge > 0.0 else 0


@dataclass(slots=True)
class MarketSummary:
    batch_label: str
    horizon_seconds: int
    file_path: Path
    asset_id: str
    market_id: str
    file_size_bytes: int
    fill_count: int
    covered_fill_count: int
    win_rate: float
    avg_net_edge_cents: float
    total_pnl_cents: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VacuumMaker across the top N largest historical JSONL files and aggregate markout PnL.")
    parser.add_argument("--input-dir", default=None, help="Directory containing raw L2 JSONL files for one historical slice.")
    parser.add_argument("--file-list", default=None, help="Optional text file listing raw JSONL files to replay, one per line.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database that collects fills and batch analysis tables.")
    parser.add_argument("--market-map", default="data/market_map.json", help="Optional market map used to resolve YES/NO token ids.")
    parser.add_argument("--strategy-config", default=None, help="Optional JSON object or JSON file with VacuumMaker overrides.")
    parser.add_argument("--top-n", type=int, default=50, help="Replay the top N largest .jsonl files in the directory.")
    parser.add_argument("--horizon-seconds", type=int, default=60, help="Forward markout horizon used for PnL aggregation.")
    parser.add_argument("--batch-label", default=None, help="Optional batch label stored in the analysis tables.")
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_PATH), help="Markdown scorecard output path.")
    parser.add_argument("--reset-db", action="store_true", help="Delete the output DB before running the batch.")
    return parser.parse_args()


def select_top_files(input_dir: Path, *, top_n: int) -> list[Path]:
    candidates = [path for path in input_dir.glob("*.jsonl") if path.is_file()]
    ranked = sorted(candidates, key=lambda path: (-path.stat().st_size, path.name))
    return ranked[:top_n]


def select_files_from_list(file_list_path: Path) -> list[Path]:
    selected_files: list[Path] = []
    base_dir = file_list_path.resolve().parent
    for line in file_list_path.read_text(encoding="utf-8").splitlines():
        raw_path = line.strip()
        if not raw_path or raw_path.startswith("#"):
            continue
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Listed replay file does not exist: {candidate}")
        selected_files.append(candidate)
    deduped = sorted(set(selected_files), key=lambda path: str(path))
    return deduped


def resolve_batch_input_dir(*, input_dir_arg: str | None, selected_files: list[Path]) -> Path:
    if input_dir_arg:
        resolved = Path(input_dir_arg)
        if not resolved.exists() or not resolved.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {resolved}")
        return resolved
    if not selected_files:
        raise FileNotFoundError("No replay files selected")
    try:
        common_path = Path(os.path.commonpath([str(path.parent) for path in selected_files]))
    except ValueError:
        common_path = selected_files[0].parent
    return common_path


def normalize_batch_label(raw_label: str | None) -> str:
    if raw_label:
        return raw_label.strip()
    return datetime.now(timezone.utc).strftime("vacuum-batch-%Y%m%dT%H%M%SZ")


def order_prefix_for(batch_label: str, index: int) -> str:
    sanitized = "".join(char if char.isalnum() else "-" for char in batch_label.lower()).strip("-") or "vacuum-batch"
    return f"{sanitized}-{index:02d}"


async def run_batch(args: argparse.Namespace) -> tuple[list[BatchRunResult], dict[str, object]]:
    file_list_path = Path(args.file_list) if args.file_list else None
    if file_list_path is not None:
        if not file_list_path.exists() or not file_list_path.is_file():
            raise FileNotFoundError(f"File list does not exist: {file_list_path}")
        selected_files = select_files_from_list(file_list_path)
    else:
        if not args.input_dir:
            raise FileNotFoundError("--input-dir is required when --file-list is not provided")
        selected_files = select_top_files(Path(args.input_dir), top_n=args.top_n)
    if not selected_files:
        source_label = str(file_list_path) if file_list_path is not None else str(args.input_dir)
        raise FileNotFoundError(f"No replay files found for batch source: {source_label}")
    input_dir = resolve_batch_input_dir(input_dir_arg=args.input_dir, selected_files=selected_files)

    db_path = Path(args.db)
    if args.reset_db and db_path.exists():
        db_path.unlink()

    batch_label = normalize_batch_label(args.batch_label)
    strategy_config = load_strategy_config(args.strategy_config)
    market_map_path = Path(args.market_map) if args.market_map else None
    market_catalog = load_market_catalog(market_map_path if market_map_path and market_map_path.exists() else None)

    results: list[BatchRunResult] = []
    aggregate_strategy_diagnostics: dict[str, int] = {}
    for index, file_path in enumerate(selected_files, start=1):
        order_id_prefix = order_prefix_for(batch_label, index)
        engine = UniversalReplayEngine(
            input_dir=file_path,
            db_path=db_path,
            strategy_path=STRATEGY_PATH,
            market_catalog=market_catalog,
            strategy_config=strategy_config,
            order_id_prefix=order_id_prefix,
        )
        summary = await engine.run()
        strategy_diagnostics = engine.strategy_diagnostics()
        result = BatchRunResult(
            batch_label=batch_label,
            order_id_prefix=order_id_prefix,
            file_path=file_path,
            file_size_bytes=file_path.stat().st_size,
            total_events=summary.total_events,
            book_events=summary.book_events,
            trade_events=summary.trade_events,
            dispatches=summary.dispatches,
            rejections=summary.rejections,
            maker_fills=summary.maker_fills,
            persisted_shadow_rows=summary.persisted_shadow_rows,
        )
        results.append(result)
        for key, value in strategy_diagnostics.items():
            if isinstance(value, bool):
                aggregate_strategy_diagnostics[key] = aggregate_strategy_diagnostics.get(key, 0) + int(value)
            elif isinstance(value, int):
                aggregate_strategy_diagnostics[key] = aggregate_strategy_diagnostics.get(key, 0) + value
        print(
            "[{current}/{total}] {name} events={events} trades={trades} dispatches={dispatches} maker_fills={fills}".format(
                current=index,
                total=len(selected_files),
                name=file_path.name,
                events=summary.total_events,
                trades=summary.trade_events,
                dispatches=summary.dispatches,
                fills=summary.maker_fills,
            )
        )
        if strategy_diagnostics:
            print(
                "[{current}/{total}] diagnostics {diagnostics}".format(
                    current=index,
                    total=len(selected_files),
                    diagnostics=" ".join(f"{key}={value}" for key, value in strategy_diagnostics.items()),
                )
            )

    analysis = analyze_batch(
        db_path=db_path,
        batch_label=batch_label,
        run_results=results,
        horizon_seconds=args.horizon_seconds,
    )
    analysis["strategy_diagnostics"] = aggregate_strategy_diagnostics
    if aggregate_strategy_diagnostics:
        print(
            "batch_diagnostics " + " ".join(
                f"{key}={value}" for key, value in sorted(aggregate_strategy_diagnostics.items())
            )
        )
    return results, analysis


def analyze_batch(
    *,
    db_path: Path,
    batch_label: str,
    run_results: list[BatchRunResult],
    horizon_seconds: int,
) -> dict[str, object]:
    ensure_analysis_tables(db_path)
    insert_run_results(db_path, run_results)

    markouts: list[tuple[BatchRunResult, FillMarkout]] = []
    market_summaries: list[MarketSummary] = []
    for run_result in run_results:
        fills = load_fill_markouts_for_prefix(db_path, order_id_prefix=run_result.order_id_prefix)
        reconstruct_file_markouts(fills, file_path=run_result.file_path, horizon_seconds=horizon_seconds)
        persist_fill_markouts(
            db_path,
            batch_label=batch_label,
            order_id_prefix=run_result.order_id_prefix,
            file_path=run_result.file_path,
            horizon_seconds=horizon_seconds,
            fills=fills,
        )
        market_summary = build_market_summary(
            batch_label=batch_label,
            horizon_seconds=horizon_seconds,
            run_result=run_result,
            fills=fills,
        )
        market_summaries.append(market_summary)
        markouts.extend((run_result, fill) for fill in fills)

    persist_market_summaries(db_path, market_summaries)
    return query_batch_rollup(db_path, batch_label=batch_label, horizon_seconds=horizon_seconds)


def ensure_analysis_tables(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS batch_vacuum_runs (
                batch_label TEXT NOT NULL,
                order_id_prefix TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                total_events INTEGER NOT NULL,
                book_events INTEGER NOT NULL,
                trade_events INTEGER NOT NULL,
                dispatches INTEGER NOT NULL,
                rejections INTEGER NOT NULL,
                maker_fills INTEGER NOT NULL,
                persisted_shadow_rows INTEGER NOT NULL,
                created_at REAL NOT NULL DEFAULT (unixepoch()),
                PRIMARY KEY (batch_label, order_id_prefix)
            );
            CREATE TABLE IF NOT EXISTS batch_vacuum_fill_markouts (
                batch_label TEXT NOT NULL,
                order_id_prefix TEXT NOT NULL,
                fill_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                horizon_seconds INTEGER NOT NULL,
                market_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                quote_side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_size REAL NOT NULL,
                entry_time_ms INTEGER NOT NULL,
                fill_mid_yes REAL,
                future_mid_yes REAL,
                gross_edge_cents REAL,
                net_edge_cents REAL,
                pnl_cents REAL NOT NULL,
                is_win INTEGER NOT NULL,
                PRIMARY KEY (batch_label, fill_id, horizon_seconds)
            );
            CREATE TABLE IF NOT EXISTS batch_vacuum_market_summary (
                batch_label TEXT NOT NULL,
                horizon_seconds INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                fill_count INTEGER NOT NULL,
                covered_fill_count INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                avg_net_edge_cents REAL NOT NULL,
                total_pnl_cents REAL NOT NULL,
                PRIMARY KEY (batch_label, horizon_seconds, file_path)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_run_results(db_path: Path, run_results: Iterable[BatchRunResult]) -> None:
    rows = [
        (
            result.batch_label,
            result.order_id_prefix,
            str(result.file_path),
            result.file_size_bytes,
            result.total_events,
            result.book_events,
            result.trade_events,
            result.dispatches,
            result.rejections,
            result.maker_fills,
            result.persisted_shadow_rows,
        )
        for result in run_results
    ]
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO batch_vacuum_runs
            (batch_label, order_id_prefix, file_path, file_size_bytes, total_events, book_events, trade_events,
             dispatches, rejections, maker_fills, persisted_shadow_rows)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def load_fill_markouts_for_prefix(db_path: Path, *, order_id_prefix: str) -> list[FillMarkout]:
    conn = sqlite3.connect(db_path)
    try:
        table_row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'shadow_trades'"
        ).fetchone()
        if table_row is None:
            return []
        rows = conn.execute(
            """
            SELECT id, market_id, asset_id, reference_price_band, entry_price, entry_size, entry_time
            FROM shadow_trades
            WHERE id LIKE ? AND exit_reason = 'MAKER_FILL' AND reference_price_band LIKE 'MAKER:%'
            ORDER BY entry_time, id
            """,
            (f"{order_id_prefix}-%",),
        ).fetchall()
    finally:
        conn.close()

    fills: list[FillMarkout] = []
    for row in rows:
        band = str(row[3])
        quote_side = band.split(":", 1)[1].upper() if ":" in band else "BID"
        fills.append(
            FillMarkout(
                fill_id=str(row[0]),
                market_id=str(row[1]),
                asset_id=str(row[2]),
                quote_side=quote_side,
                entry_price=float(row[4]),
                entry_size=float(row[5]),
                entry_time_ms=int(round(float(row[6]) * 1000)),
            )
        )
    return fills


def reconstruct_file_markouts(fills: list[FillMarkout], *, file_path: Path, horizon_seconds: int) -> None:
    if not fills:
        return
    tracker: OrderbookTracker | None = None
    fill_index = 0
    future_index = 0

    for event in iter_file_events(file_path):
        if event.event_type not in {"BOOK", "PRICE_CHANGE"}:
            continue
        if tracker is None:
            tracker = OrderbookTracker(event.asset_id)
        if event.event_type == "BOOK":
            tracker.on_book_snapshot(event.payload)
        else:
            tracker.on_price_change(event.payload)
        snapshot = tracker.snapshot()
        if snapshot.best_bid <= 0 or snapshot.best_ask <= 0:
            continue
        mid_yes = snapshot.mid_price

        while fill_index < len(fills) and fills[fill_index].entry_time_ms <= event.timestamp_ms:
            if fills[fill_index].fill_mid_yes is None:
                fills[fill_index].fill_mid_yes = mid_yes
            fill_index += 1

        while future_index < len(fills):
            target_time_ms = fills[future_index].entry_time_ms + (horizon_seconds * 1000)
            if target_time_ms > event.timestamp_ms:
                break
            if fills[future_index].future_mid_yes is None:
                fills[future_index].future_mid_yes = mid_yes
            future_index += 1


def persist_fill_markouts(
    db_path: Path,
    *,
    batch_label: str,
    order_id_prefix: str,
    file_path: Path,
    horizon_seconds: int,
    fills: Iterable[FillMarkout],
) -> None:
    rows = [
        (
            batch_label,
            order_id_prefix,
            fill.fill_id,
            str(file_path),
            horizon_seconds,
            fill.market_id,
            fill.asset_id,
            fill.quote_side,
            fill.entry_price,
            fill.entry_size,
            fill.entry_time_ms,
            fill.fill_mid_yes,
            fill.future_mid_yes,
            fill.gross_edge_cents,
            fill.net_edge_cents,
            fill.pnl_cents,
            fill.is_win,
        )
        for fill in fills
    ]
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO batch_vacuum_fill_markouts
            (batch_label, order_id_prefix, fill_id, file_path, horizon_seconds, market_id, asset_id, quote_side,
             entry_price, entry_size, entry_time_ms, fill_mid_yes, future_mid_yes, gross_edge_cents,
             net_edge_cents, pnl_cents, is_win)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def build_market_summary(
    *,
    batch_label: str,
    horizon_seconds: int,
    run_result: BatchRunResult,
    fills: list[FillMarkout],
) -> MarketSummary:
    covered = [fill for fill in fills if fill.covered]
    net_edges = [fill.net_edge_cents for fill in covered if fill.net_edge_cents is not None]
    total_wins = sum(fill.is_win for fill in covered)
    first_fill = fills[0] if fills else None
    return MarketSummary(
        batch_label=batch_label,
        horizon_seconds=horizon_seconds,
        file_path=run_result.file_path,
        asset_id=first_fill.asset_id if first_fill is not None else run_result.file_path.stem,
        market_id=first_fill.market_id if first_fill is not None else run_result.file_path.stem,
        file_size_bytes=run_result.file_size_bytes,
        fill_count=len(fills),
        covered_fill_count=len(covered),
        win_rate=(total_wins / len(covered)) if covered else 0.0,
        avg_net_edge_cents=(sum(net_edges) / len(net_edges)) if net_edges else 0.0,
        total_pnl_cents=sum(fill.pnl_cents for fill in covered),
    )


def persist_market_summaries(db_path: Path, market_summaries: Iterable[MarketSummary]) -> None:
    rows = [
        (
            summary.batch_label,
            summary.horizon_seconds,
            str(summary.file_path),
            summary.asset_id,
            summary.market_id,
            summary.file_size_bytes,
            summary.fill_count,
            summary.covered_fill_count,
            summary.win_rate,
            summary.avg_net_edge_cents,
            summary.total_pnl_cents,
        )
        for summary in market_summaries
    ]
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO batch_vacuum_market_summary
            (batch_label, horizon_seconds, file_path, asset_id, market_id, file_size_bytes,
             fill_count, covered_fill_count, win_rate, avg_net_edge_cents, total_pnl_cents)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def query_batch_rollup(db_path: Path, *, batch_label: str, horizon_seconds: int) -> dict[str, object]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        totals = conn.execute(
            """
            WITH rollup AS (
                SELECT *
                FROM batch_vacuum_market_summary
                WHERE batch_label = ? AND horizon_seconds = ?
            )
            SELECT
                COALESCE(SUM(total_pnl_cents), 0.0) AS total_pnl_cents,
                COALESCE(SUM(fill_count), 0) AS total_fills,
                COALESCE(SUM(covered_fill_count), 0) AS covered_fills,
                COALESCE(AVG(win_rate), 0.0) AS avg_market_win_rate,
                COUNT(*) AS markets,
                COALESCE(SUM(CASE WHEN total_pnl_cents > 0 THEN 1 ELSE 0 END), 0) AS profitable_markets
            FROM rollup
            """,
            (batch_label, horizon_seconds),
        ).fetchone()
        top_markets = conn.execute(
            """
            SELECT file_path, fill_count, covered_fill_count, win_rate, avg_net_edge_cents, total_pnl_cents
            FROM batch_vacuum_market_summary
            WHERE batch_label = ? AND horizon_seconds = ?
            ORDER BY total_pnl_cents DESC, fill_count DESC, file_path ASC
            LIMIT 10
            """,
            (batch_label, horizon_seconds),
        ).fetchall()
        bottom_markets = conn.execute(
            """
            SELECT file_path, fill_count, covered_fill_count, win_rate, avg_net_edge_cents, total_pnl_cents
            FROM batch_vacuum_market_summary
            WHERE batch_label = ? AND horizon_seconds = ?
            ORDER BY total_pnl_cents ASC, fill_count DESC, file_path ASC
            LIMIT 10
            """,
            (batch_label, horizon_seconds),
        ).fetchall()
        run_totals = conn.execute(
            """
            SELECT
                COALESCE(SUM(total_events), 0) AS total_events,
                COALESCE(SUM(book_events), 0) AS book_events,
                COALESCE(SUM(trade_events), 0) AS trade_events,
                COALESCE(SUM(dispatches), 0) AS dispatches,
                COALESCE(SUM(rejections), 0) AS rejections,
                COALESCE(SUM(maker_fills), 0) AS maker_fills,
                COALESCE(SUM(file_size_bytes), 0) AS total_bytes
            FROM batch_vacuum_runs
            WHERE batch_label = ?
            """,
            (batch_label,),
        ).fetchone()
    finally:
        conn.close()

    return {
        "batch_label": batch_label,
        "horizon_seconds": horizon_seconds,
        "totals": dict(totals),
        "run_totals": dict(run_totals),
        "top_markets": [dict(row) for row in top_markets],
        "bottom_markets": [dict(row) for row in bottom_markets],
    }


def render_markdown(
    *,
    analysis: dict[str, object],
    input_dir: Path,
    db_path: Path,
    strategy_config: dict[str, object],
) -> str:
    totals = analysis["totals"]
    run_totals = analysis["run_totals"]
    top_markets = analysis["top_markets"]
    bottom_markets = analysis["bottom_markets"]
    horizon_seconds = analysis["horizon_seconds"]
    total_pnl_usd = float(totals["total_pnl_cents"]) / 100.0
    avg_market_win_rate = float(totals["avg_market_win_rate"])
    profitable_markets = int(totals["profitable_markets"])
    market_count = int(totals["markets"])
    verdict = "supports horizontal scaling" if total_pnl_usd > 0 and profitable_markets >= max(1, market_count // 2) else "does not yet prove horizontal scaling"
    crash_abs_obi = strategy_config.get("crash_abs_obi", "0.95")
    lines = [
        f"# VacuumMaker Batch Backtest Scorecard ({analysis['batch_label']})",
        "",
        "## Setup",
        "",
        f"- Strategy: `{STRATEGY_PATH}`",
        f"- Replay directory: `{input_dir}`",
        f"- Aggregated SQLite: `{db_path}`",
        f"- Files processed: `{market_count}`",
        f"- Forward markout horizon: `{horizon_seconds}s`",
        f"- OBI crash threshold: `{crash_abs_obi}`",
        "",
        "## Batch Totals",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total PnL (USD) | {total_pnl_usd:.2f} |",
        f"| Total fills | {int(totals['total_fills']):,} |",
        f"| Covered fills | {int(totals['covered_fills']):,} |",
        f"| Average market win rate | {avg_market_win_rate:.2%} |",
        f"| Profitable markets | {profitable_markets:,} |",
        f"| Total normalized events | {int(run_totals['total_events']):,} |",
        f"| Total trade events | {int(run_totals['trade_events']):,} |",
        f"| Total dispatches | {int(run_totals['dispatches']):,} |",
        f"| Total maker fills | {int(run_totals['maker_fills']):,} |",
        "",
        "## SQL Rollup",
        "",
        "```sql",
        "SELECT SUM(total_pnl_cents) / 100.0 AS total_pnl_usd,",
        "       SUM(fill_count) AS total_fills,",
        "       AVG(win_rate) AS avg_market_win_rate",
        "FROM batch_vacuum_market_summary",
        f"WHERE batch_label = '{analysis['batch_label']}' AND horizon_seconds = {horizon_seconds};",
        "```",
        "",
        "## Top Markets",
        "",
        "| File | Fills | Covered | Win Rate | Avg Net Edge (c/share) | Total PnL (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_markets:
        lines.append(
            "| {file_name} | {fills:,} | {covered:,} | {win_rate:.2%} | {avg_edge:.3f} | {pnl:.2f} |".format(
                file_name=Path(str(row['file_path'])).name,
                fills=int(row["fill_count"]),
                covered=int(row["covered_fill_count"]),
                win_rate=float(row["win_rate"]),
                avg_edge=float(row["avg_net_edge_cents"]),
                pnl=float(row["total_pnl_cents"]) / 100.0,
            )
        )
    lines.extend(
        [
            "",
            "## Bottom Markets",
            "",
            "| File | Fills | Covered | Win Rate | Avg Net Edge (c/share) | Total PnL (USD) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bottom_markets:
        lines.append(
            "| {file_name} | {fills:,} | {covered:,} | {win_rate:.2%} | {avg_edge:.3f} | {pnl:.2f} |".format(
                file_name=Path(str(row['file_path'])).name,
                fills=int(row["fill_count"]),
                covered=int(row["covered_fill_count"]),
                win_rate=float(row["win_rate"]),
                avg_edge=float(row["avg_net_edge_cents"]),
                pnl=float(row["total_pnl_cents"]) / 100.0,
            )
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "The batch run {verdict}: the {horizon}s fee-adjusted markout rollup produced `${pnl:.2f}` across `{markets}` markets with an average per-market win rate of `{win_rate:.2%}`.".format(
                verdict=verdict,
                horizon=horizon_seconds,
                pnl=total_pnl_usd,
                markets=market_count,
                win_rate=avg_market_win_rate,
            ),
        ]
    )
    return "\n".join(lines) + "\n"


async def _main() -> None:
    args = parse_args()
    results, analysis = await run_batch(args)
    markdown = render_markdown(
        analysis=analysis,
        input_dir=resolve_batch_input_dir(input_dir_arg=args.input_dir, selected_files=[result.file_path for result in results]),
        db_path=Path(args.db),
        strategy_config=load_strategy_config(args.strategy_config),
    )
    output_path = Path(args.markdown_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print("\n---MARKDOWN---\n")
    print(markdown)
    print(f"Wrote {len(results)} run summaries to {args.db}")
    print(f"Wrote markdown scorecard to {output_path}")


if __name__ == "__main__":
    asyncio.run(_main())