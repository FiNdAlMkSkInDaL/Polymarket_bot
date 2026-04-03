from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.reward_replay import ReplayConfig, RewardReplayEngine, load_reward_markets, replay_anchor_ms_for_range
from src.rewards.reward_selector import RewardSelectorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay historical L2 ticks into RewardPosterSidecar.")
    parser.add_argument(
        "--input-dir",
        default="logs/local_snapshot/l2_data/data/raw_ticks",
        help="Directory containing date-partitioned raw L2 JSONL files.",
    )
    parser.add_argument("--db", default="logs/backtest.db", help="SQLite output path.")
    parser.add_argument(
        "--reward-universe",
        default="data/mid_tier_reward_audit_2026_03_23.json",
        help="Reward-enriched market universe JSON.",
    )
    parser.add_argument(
        "--market-map",
        default="data/market_map.json",
        help="Optional market map used to backfill missing token ids.",
    )
    parser.add_argument("--start-date", default=None, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--activation-latency-ms", type=int, default=50, help="Quote activation delay in simulation time.")
    parser.add_argument("--max-events", type=int, default=None, help="Stop after processing this many normalized events.")
    parser.add_argument("--min-reward-usd", type=float, default=None, help="Override minimum daily reward required for admission.")
    parser.add_argument("--min-daily-volume-usd", type=float, default=None, help="Override minimum daily volume required for admission.")
    parser.add_argument("--max-daily-volume-usd", type=float, default=None, help="Override maximum daily volume allowed for admission.")
    parser.add_argument("--min-days-to-resolution", type=int, default=None, help="Override minimum days to resolution.")
    parser.add_argument("--max-days-to-resolution", type=int, default=None, help="Override maximum days to resolution.")
    parser.add_argument("--max-spread-cents", type=float, default=None, help="Override maximum allowed spread in cents.")
    parser.add_argument("--min-reward-to-competition", type=float, default=None, help="Override minimum reward-to-competition score.")
    parser.add_argument("--min-mid-price", type=float, default=None, help="Override lower safe-mid bound.")
    parser.add_argument("--max-mid-price", type=float, default=None, help="Override upper safe-mid bound.")
    parser.add_argument("--jump-risk-move-pct", type=float, default=None, help="Override recent-move veto threshold.")
    parser.add_argument("--jump-risk-volatility-pct", type=float, default=None, help="Override volatility-jump veto threshold.")
    parser.add_argument("--market-cap", type=int, default=None, help="Override reward market cap.")
    parser.add_argument("--quote-cap", type=int, default=None, help="Override reward quote cap.")
    parser.add_argument("--quote-notional-cap", type=float, default=None, help="Override reward quote notional cap.")
    parser.add_argument("--inventory-cap", type=float, default=None, help="Override reward inventory cap.")
    parser.add_argument("--cancel-on-stale-ms", type=int, default=None, help="Override reward stale cancel window.")
    parser.add_argument("--replace-only-if-price-moves-ticks", type=int, default=None, help="Override quote replace threshold in ticks.")
    parser.add_argument("--refresh-interval-ms", type=int, default=None, help="Override reward universe refresh interval.")
    parser.add_argument(
        "--condition-id",
        action="append",
        default=[],
        help="Optional condition id filter. Repeat for multiple markets.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    db_path = Path(args.db)
    reward_universe_path = Path(args.reward_universe)
    market_map_path = Path(args.market_map)
    replay_anchor_ms = replay_anchor_ms_for_range(input_dir, start_date=args.start_date)
    condition_ids = frozenset(str(value).strip() for value in args.condition_id if str(value).strip())
    selector_config = RewardSelectorConfig(
        min_reward_usd=args.min_reward_usd if args.min_reward_usd is not None else RewardSelectorConfig().min_reward_usd,
        min_daily_volume_usd=args.min_daily_volume_usd if args.min_daily_volume_usd is not None else RewardSelectorConfig().min_daily_volume_usd,
        max_daily_volume_usd=args.max_daily_volume_usd if args.max_daily_volume_usd is not None else RewardSelectorConfig().max_daily_volume_usd,
        min_days_to_resolution=args.min_days_to_resolution if args.min_days_to_resolution is not None else RewardSelectorConfig().min_days_to_resolution,
        max_days_to_resolution=args.max_days_to_resolution if args.max_days_to_resolution is not None else RewardSelectorConfig().max_days_to_resolution,
        max_spread_cents=args.max_spread_cents if args.max_spread_cents is not None else RewardSelectorConfig().max_spread_cents,
        min_reward_to_competition=args.min_reward_to_competition if args.min_reward_to_competition is not None else RewardSelectorConfig().min_reward_to_competition,
        min_mid_price=args.min_mid_price if args.min_mid_price is not None else RewardSelectorConfig().min_mid_price,
        max_mid_price=args.max_mid_price if args.max_mid_price is not None else RewardSelectorConfig().max_mid_price,
        jump_risk_move_pct=args.jump_risk_move_pct if args.jump_risk_move_pct is not None else RewardSelectorConfig().jump_risk_move_pct,
        jump_risk_volatility_pct=args.jump_risk_volatility_pct if args.jump_risk_volatility_pct is not None else RewardSelectorConfig().jump_risk_volatility_pct,
        quote_notional_cap=args.quote_notional_cap if args.quote_notional_cap is not None else RewardSelectorConfig().quote_notional_cap,
        cancel_on_stale_ms=args.cancel_on_stale_ms if args.cancel_on_stale_ms is not None else RewardSelectorConfig().cancel_on_stale_ms,
        replace_only_if_price_moves_ticks=args.replace_only_if_price_moves_ticks if args.replace_only_if_price_moves_ticks is not None else RewardSelectorConfig().replace_only_if_price_moves_ticks,
    )
    markets = load_reward_markets(
        reward_universe_path,
        replay_anchor_ms=replay_anchor_ms,
        market_map_path=market_map_path,
        condition_ids=condition_ids,
    )
    if not markets:
        raise SystemExit("No reward markets loaded for backtest.")

    config = ReplayConfig(
        input_dir=input_dir,
        db_path=db_path,
        reward_universe_path=reward_universe_path,
        market_map_path=market_map_path,
        activation_latency_ms=args.activation_latency_ms,
        max_events=args.max_events,
        start_date=args.start_date,
        end_date=args.end_date,
        condition_ids=condition_ids,
        selector_config=selector_config,
        reward_market_cap=args.market_cap,
        reward_quote_cap=args.quote_cap,
        reward_quote_notional_cap=args.quote_notional_cap,
        reward_inventory_cap=args.inventory_cap,
        reward_cancel_on_stale_ms=args.cancel_on_stale_ms,
        reward_replace_only_if_price_moves_ticks=args.replace_only_if_price_moves_ticks,
        reward_refresh_interval_ms=args.refresh_interval_ms,
        markout_horizons_seconds=(5, 15, 60),
    )
    summary = await RewardReplayEngine(config, markets).run()
    print(
        "events={events} book={book} trades={trades} matched_fills={fills} persisted_shadow_rows={rows} db={db}".format(
            events=summary.total_events,
            book=summary.book_events,
            trades=summary.trade_events,
            fills=summary.matched_fills,
            rows=summary.persisted_shadow_rows,
            db=db_path,
        )
    )


if __name__ == "__main__":
    asyncio.run(_main())