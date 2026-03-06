"""Quick diagnostic: does the backtest engine now produce fills?"""
import sys, json, pathlib, os

# Suppress structured logging noise
os.environ["LOG_LEVEL"] = "WARNING"

sys.path.insert(0, ".")

from src.backtest.data_loader import DataLoader
from src.backtest.data_recorder import MarketDataRecorder
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.strategy import BotReplayAdapter
from src.core.config import StrategyParams

data_dir = pathlib.Path("data/vps_march2026")
market_map = json.loads((pathlib.Path("data/market_map.json")).read_text())
m = market_map[0]
market_id = m["market_id"]
yes_id = str(m["yes_id"])
no_id = str(m["no_id"])

print(f"Market: {market_id[:30]}...")
print(f"YES: {yes_id[:15]}...")
print(f"YES full: '{yes_id}'")
print(f"NO full: '{no_id}'")

rec = MarketDataRecorder(str(data_dir))
dates = rec.available_dates(str(data_dir))[:35]
print(f"Days: {len(dates)}")

files = []
for d in dates:
    files.extend(rec.data_files_for_date(str(data_dir), d))
print(f"Files: {len(files)}")

loader = DataLoader(files, asset_ids={yes_id, no_id})
# Force lenient params to verify end-to-end fill pipeline works
params = StrategyParams(zscore_threshold=0.5, volume_ratio_threshold=0.05, min_edge_score=1.0, min_spread_cents=0.5)
strategy = BotReplayAdapter(market_id=market_id, yes_asset_id=yes_id, no_asset_id=no_id, params=params)
config = BacktestConfig(initial_cash=1000.0)
engine = BacktestEngine(strategy=strategy, data_loader=loader, config=config)
result = engine.run()

print(f"\n=== RESULTS ===")
print(f"Events processed : {engine._events_processed}")
print(f"Total fills      : {len(result.all_fills)}")
print(f"Positions opened : {len(strategy._positions)}")
print(f"PnL              : {result.metrics.total_pnl:.4f}")
print(f"Sharpe           : {result.metrics.sharpe_ratio:.4f}")
print(f"Final best_bid   : {engine.matching_engine.best_bid}")
print(f"Final best_ask   : {engine.matching_engine.best_ask}")
print(f"has_real_book    : {engine._has_real_book}")

# Aggregator stats
if strategy._yes_agg:
    agg = strategy._yes_agg
    print(f"\n--- YES aggregator ---")
    print(f"bars             : {len(agg.bars)}")
    print(f"rolling_vwap     : {agg.rolling_vwap:.6f}")
    print(f"rolling_volatility : {agg.rolling_volatility:.6f}")
    print(f"avg_bar_volume   : {agg.avg_bar_volume:.4f}")
    if agg.bars:
        last_bar = agg.bars[-1]
        print(f"last_bar close   : {last_bar.close}")
        print(f"last_bar volume  : {last_bar.volume}")
