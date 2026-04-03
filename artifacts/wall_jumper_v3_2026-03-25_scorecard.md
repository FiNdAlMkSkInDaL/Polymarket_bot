# WallJumper Toxic-Day Scorecard

## Run

- Date: `2026-03-25`
- Strategy: `src.signals.wall_jumper.WallJumper`
- Replay DB: `logs\universal_backtest_wall_jumper_v3_2026-03-25.db`
- Replay source: `logs\local_snapshot\l2_data`
- Backtest log: `logs\universal_backtest_wall_jumper_v3_2026-03-25.log`

## Method

- Fills were read from the universal backtest database and filtered to `signal_metadata.strategy = wall_jumper`.
- Fill-time spread capture and 5s/15s/60s forward markouts were reconstructed from the same raw tick stream used by the replay engine.
- Net edge per share was computed as signed fill-to-mid markout at each horizon, net of the simulated exit fee.
- Emergency wall collapses were counted from the strategy diagnostics emitted by the replay run.
- WallJumper v3 only jumps walls that are both structurally deep and structurally large; no time-of-day overlay is applied.

## Counter Block

- Walls identified: `20`
- Walls aged past `0`ms: `20`
- Min distance from mid: `2` ticks
- Min wall size: `$100000`
- Jump quotes emitted: `20`
- Emergency `CANCEL_ALL` triggers: `8`
- Maker fills: `11`
- Filtered-set wall-pull rate: `40.00%`
- Unique markets touched: `11`
- Total filled shares: `110.0`
- Average fill size: `10.0000`
- Bid fills: `0`
- Ask fills: `11`
- Average observed full spread at fill: `5.36364c/share`
- Average spread captured: `2.68182c/share`
- Median spread captured: `3.20000c/share`

## Markout Table

| Horizon | Covered Fills | Win Rate | Avg Markout (c/share) | Avg Adverse Selection (c/share) | Avg Net Edge (c/share) | Total PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5s | 11 | 90.91% | 2.64091 | 0.04091 | 2.41822 | 2.66 |
| 15s | 11 | 90.91% | 2.64091 | 0.04091 | 2.41822 | 2.66 |
| 60s | 11 | 90.91% | 1.27727 | 1.40455 | 1.15642 | 1.27 |

## Daily Breakdown

| Day | Total Fills | Covered 60s Fills | Total Shares | Deployed Notional (USD) | 60s Avg Net Edge (c/share) | 60s PnL (USD) | Daily Return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-03-25 | 11 | 11 | 110.0 | 109.69 | 1.15642 | 1.27 | 1.15969% |

## OOS Diagnostics

- 5-day average 60s net edge: `1.15642c/share`
- Aggregate filtered-set wall-pull rate: `40.00%`
- Aggregate 60s PnL across the sweep: `$1.27`
- Estimated OOS daily-return Sharpe: `n/a`

## Verdict

- The v3 structural gate was **positive** over `2026-03-25`: 60s net edge was `1.15642c/share` and total 60s simulated PnL was `$1.27`.
- Whale support still failed `8` times across `20` filtered jump quotes, for a filtered-set wall-pull rate of `40.00%`.
- The realized 60s markout averaged `1.27727c/share` while immediate spread capture averaged `2.68182c/share`, which indicates the post-fill drift preserved the captured edge.
