# WallJumper Toxic-Day Scorecard

## Run

- Date: `2026-03-25`
- Strategy: `src.signals.wall_jumper.WallJumper`
- Replay DB: `logs\universal_backtest_wall_jumper_age15s_2026-03-25.db`
- Replay source: `logs\local_snapshot\l2_data`
- Backtest log: `logs\universal_backtest_wall_jumper_age15s_2026-03-25.log`

## Method

- Fills were read from the universal backtest database and filtered to `signal_metadata.strategy = wall_jumper`.
- Fill-time spread capture and 5s/15s/60s forward markouts were reconstructed from the same raw tick stream used by the replay engine.
- Net edge per share was computed as signed fill-to-mid markout at each horizon, net of the simulated exit fee.
- Emergency wall collapses were counted from the strategy diagnostics emitted by the replay run.

## Counter Block

- Walls identified: `12540`
- Walls aged past `15000`ms: `986`
- Jump quotes emitted: `60`
- Emergency `CANCEL_ALL` triggers: `42`
- Maker fills: `17`
- Filtered-set wall-pull rate: `70.00%`
- Unique markets touched: `14`
- Total filled shares: `140.0`
- Average fill size: `8.2353`
- Bid fills: `10`
- Ask fills: `7`
- Average observed full spread at fill: `3.67059c/share`
- Average spread captured: `1.83529c/share`
- Median spread captured: `0.45000c/share`

## Markout Table

| Horizon | Covered Fills | Win Rate | Avg Markout (c/share) | Avg Adverse Selection (c/share) | Avg Net Edge (c/share) | Total PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5s | 17 | 94.12% | 1.82647 | 0.00882 | 1.66145 | 2.74 |
| 15s | 17 | 94.12% | 1.80882 | 0.02647 | 1.64507 | 2.71 |
| 60s | 17 | 88.24% | 0.72059 | 1.11471 | 0.63822 | 1.00 |

## Verdict

- The `15000`ms wall-age gate was **positive** on `2026-03-25`: 60s net edge was `0.63822c/share` and total 60s simulated PnL was `$1.00`.
- Whale support still failed `42` times across `60` filtered jump quotes, for a filtered-set wall-pull rate of `70.00%`.
- The realized 60s markout averaged `0.72059c/share` while immediate spread capture averaged `1.83529c/share`, which indicates the post-fill drift preserved the captured edge.
