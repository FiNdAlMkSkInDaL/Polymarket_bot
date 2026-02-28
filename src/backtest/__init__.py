"""
Institutional-grade event-driven backtesting framework.

Replays historical L2 order book deltas and public trades through a
pessimistic CLOB matching engine with Polymarket's dynamic fee curve.

Public API
──────────
    BacktestEngine   – top-level replay orchestrator
    BacktestConfig   – engine configuration
    BacktestResult   – post-run analytics container
    MatchingEngine   – pessimistic CLOB simulator
    SimClock         – simulated wall-clock (patches time.time)
    DataLoader       – chronological event replay from JSONL files
    MarketEvent      – single timestamped market event
    StrategyABC      – callback-based strategy interface
    BotReplayAdapter – adapter wiring live-bot components into backtest
    MarketDataRecorder – live WS event capture to disk
    Telemetry        – post-run PnL / drawdown / Sharpe analytics
    WfoConfig        – Walk-Forward Optimization configuration
    WfoReport        – WFO aggregated results
    FoldResult       – per-fold WFO metrics
    run_wfo          – WFO pipeline entry point
    generate_folds   – time-series CV window generator
"""

from src.backtest.clock import SimClock
from src.backtest.data_loader import DataLoader, MarketEvent
from src.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from src.backtest.matching_engine import Fill, MatchingEngine, SimOrder
from src.backtest.strategy import BotReplayAdapter, StrategyABC
from src.backtest.telemetry import Telemetry
from src.backtest.wfo_optimizer import (
    FoldResult,
    WfoConfig,
    WfoReport,
    generate_folds,
    run_wfo,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BotReplayAdapter",
    "DataLoader",
    "Fill",
    "FoldResult",
    "MarketEvent",
    "MatchingEngine",
    "SimClock",
    "SimOrder",
    "StrategyABC",
    "Telemetry",
    "WfoConfig",
    "WfoReport",
    "generate_folds",
    "run_wfo",
]
