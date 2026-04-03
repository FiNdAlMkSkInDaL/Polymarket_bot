# WallJumper Toxic-Day Scorecard

## Run

- Date: `2026-03-25`
- Strategy: `src.signals.wall_jumper.WallJumper`
- Replay DB: `logs\universal_backtest_wall_jumper_2026-03-25.db`
- Replay source: `logs\local_snapshot\l2_data`
- Backtest log: `logs\universal_backtest_wall_jumper_2026-03-25.log`

## Method

- Fills were read from the universal backtest database and filtered to `signal_metadata.strategy = wall_jumper`.
- Fill-time spread capture and 5s/15s/60s forward markouts were reconstructed from the same raw tick stream used by the replay engine.
- Net edge per share was computed as signed fill-to-mid markout at each horizon, net of the simulated exit fee.
- Emergency wall collapses were counted from the strategy diagnostics emitted by the replay run.

## Counter Block

- Walls identified: `12540`
- Walls aged past `0`ms: `12540`
- Jump quotes emitted: `116`
- Emergency `CANCEL_ALL` triggers: `64`
- Maker fills: `27`
- Filtered-set wall-pull rate: `55.17%`
- Unique markets touched: `24`
- Total filled shares: `240.0`
- Average fill size: `8.8889`
- Bid fills: `15`
- Ask fills: `12`
- Average observed full spread at fill: `2.59259c/share`
- Average spread captured: `1.29630c/share`
- Median spread captured: `0.05000c/share`

## Markout Table

| Horizon | Covered Fills | Win Rate | Avg Markout (c/share) | Avg Adverse Selection (c/share) | Avg Net Edge (c/share) | Total PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5s | 27 | 88.89% | 1.29074 | 0.00556 | 1.16827 | 3.07 |
| 15s | 27 | 88.89% | 1.27963 | 0.01667 | 1.15796 | 3.04 |
| 60s | 27 | 85.19% | 0.59444 | 0.70185 | 0.52402 | 1.33 |

## Verdict

- The `0`ms wall-age gate was **positive** on `2026-03-25`: 60s net edge was `0.52402c/share` and total 60s simulated PnL was `$1.33`.
- Whale support still failed `64` times across `116` filtered jump quotes, for a filtered-set wall-pull rate of `55.17%`.
- The realized 60s markout averaged `0.59444c/share` while immediate spread capture averaged `1.29630c/share`, which indicates the post-fill drift preserved the captured edge.
