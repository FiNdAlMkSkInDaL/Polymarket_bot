# WallJumper Toxic-Day Scorecard

## Run

- Date: `2026-03-25`
- Strategy: `src.signals.wall_jumper.WallJumper`
- Replay DB: `logs\universal_backtest_wall_jumper_age20s_2026-03-25.db`
- Replay source: `logs\local_snapshot\l2_data`
- Backtest log: `logs\universal_backtest_wall_jumper_age20s_2026-03-25.log`

## Method

- Fills were read from the universal backtest database and filtered to `signal_metadata.strategy = wall_jumper`.
- Fill-time spread capture and 5s/15s/60s forward markouts were reconstructed from the same raw tick stream used by the replay engine.
- Net edge per share was computed as signed fill-to-mid markout at each horizon, net of the simulated exit fee.
- Emergency wall collapses were counted from the strategy diagnostics emitted by the replay run.

## Counter Block

- Walls identified: `12540`
- Walls aged past `20000`ms: `940`
- Jump quotes emitted: `50`
- Emergency `CANCEL_ALL` triggers: `36`
- Maker fills: `13`
- Filtered-set wall-pull rate: `72.00%`
- Unique markets touched: `10`
- Total filled shares: `100.0`
- Average fill size: `7.6923`
- Bid fills: `8`
- Ask fills: `5`
- Average observed full spread at fill: `3.65385c/share`
- Average spread captured: `1.82692c/share`
- Median spread captured: `0.45000c/share`

## Markout Table

| Horizon | Covered Fills | Win Rate | Avg Markout (c/share) | Avg Adverse Selection (c/share) | Avg Net Edge (c/share) | Total PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5s | 13 | 92.31% | 1.81538 | 0.01154 | 1.64607 | 2.06 |
| 15s | 13 | 92.31% | 1.79231 | 0.03462 | 1.62466 | 2.03 |
| 60s | 13 | 84.62% | 0.92692 | 0.90000 | 0.82290 | 0.99 |

## Verdict

- The `20000`ms wall-age gate was **positive** on `2026-03-25`: 60s net edge was `0.82290c/share` and total 60s simulated PnL was `$0.99`.
- Whale support still failed `36` times across `50` filtered jump quotes, for a filtered-set wall-pull rate of `72.00%`.
- The realized 60s markout averaged `0.92692c/share` while immediate spread capture averaged `1.82692c/share`, which indicates the post-fill drift preserved the captured edge.
