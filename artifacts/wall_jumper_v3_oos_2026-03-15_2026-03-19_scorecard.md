# WallJumper OOS Scorecard

## Run

- Date: `2026-03-15 to 2026-03-19`
- Strategy: `src.signals.wall_jumper.WallJumper`
- Replay DB: `logs\universal_backtest_wall_jumper_v3_oos_2026-03-15_2026-03-19.db`
- Replay source: `logs\local_snapshot\l2_data`
- Backtest log: `logs\universal_backtest_wall_jumper_v3_oos_2026-03-15_2026-03-19.log`

## Method

- Fills were read from the universal backtest database and filtered to `signal_metadata.strategy = wall_jumper`.
- Fill-time spread capture and 5s/15s/60s forward markouts were reconstructed from the same raw tick stream used by the replay engine.
- Net edge per share was computed as signed fill-to-mid markout at each horizon, net of the simulated exit fee.
- Emergency wall collapses were counted from the strategy diagnostics emitted by the replay run.
- WallJumper v3 only jumps walls that are both structurally deep and structurally large; no time-of-day overlay is applied.

## Counter Block

- Walls identified: `270668`
- Walls aged past `0`ms: `270668`
- Min distance from mid: `2` ticks
- Min wall size: `$100000`
- Jump quotes emitted: `270583`
- Emergency `CANCEL_ALL` triggers: `270565`
- Maker fills: `12`
- Filtered-set wall-pull rate: `99.99%`
- Unique markets touched: `4`
- Total filled shares: `60.1`
- Average fill size: `5.0119`
- Bid fills: `0`
- Ask fills: `12`
- Average observed full spread at fill: `1.33333c/share`
- Average spread captured: `1.00000c/share`
- Median spread captured: `1.00000c/share`

## Markout Table

| Horizon | Covered Fills | Win Rate | Avg Markout (c/share) | Avg Adverse Selection (c/share) | Avg Net Edge (c/share) | Total PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5s | 12 | 8.33% | 1.00000 | 0.00000 | -0.46297 | -0.45 |
| 15s | 12 | 8.33% | 1.00000 | 0.00000 | -0.46297 | -0.45 |
| 60s | 12 | 8.33% | 0.91667 | 0.08333 | -0.54547 | -0.50 |

## Daily Breakdown

| Day | Total Fills | Covered 60s Fills | Total Shares | Deployed Notional (USD) | 60s Avg Net Edge (c/share) | 60s PnL (USD) | Daily Return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-03-15 | 0 | 0 | 0.0 | 0.00 | 0.00000 | 0.00 | 0.00000% |
| 2026-03-16 | 0 | 0 | 0.0 | 0.00 | 0.00000 | 0.00 | 0.00000% |
| 2026-03-17 | 2 | 2 | 10.0 | 8.10 | -0.28000 | -0.03 | -0.34568% |
| 2026-03-18 | 8 | 8 | 31.1 | 20.50 | -0.48565 | -0.27 | -1.33263% |
| 2026-03-19 | 2 | 2 | 19.0 | 13.80 | -1.05020 | -0.20 | -1.42426% |

## OOS Diagnostics

- 5-day average 60s net edge: `-0.54547c/share`
- Aggregate filtered-set wall-pull rate: `99.99%`
- Aggregate 60s PnL across the sweep: `$-0.50`
- Estimated OOS daily-return Sharpe: `-16.771`

## Verdict

- The v3 structural gate was **negative** over `2026-03-15 to 2026-03-19`: 60s net edge was `-0.54547c/share` and total 60s simulated PnL was `$-0.50`.
- Whale support still failed `270565` times across `270583` filtered jump quotes, for a filtered-set wall-pull rate of `99.99%`.
- The realized 60s markout averaged `0.91667c/share` while immediate spread capture averaged `1.00000c/share`, which indicates the post-fill drift overwhelmed the captured edge.
