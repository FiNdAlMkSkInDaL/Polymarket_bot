# ExhaustionFader Toxic-Day Scorecard (2026-03-25 to 2026-03-25)

## Run

- Strategy: `src.signals.exhaustion_fader.ExhaustionFader`
- Replay source: `logs\local_snapshot\l2_data`
- Replay DB: `logs\universal_backtest_exhaustion_2026-03-25.db`
- Markout horizon: `60s`

## Method

- Maker fills were loaded from the universal replay database.
- Fill-time spread capture and 60-second forward markouts were reconstructed from the same raw tick stream used by the replay engine.
- Signed 60-second markout is defined as `future edge - fill-time edge`; positive means the post-fill move mean-reverted in the fade's favor.
- Total simulated PnL is the 60-second mark-to-mid edge times filled size; no additional exit-fee adjustment is applied in this scorecard.

## Performance

- Total fades executed: `0`
- Fades with full 60s horizon: `0`
- Total filled shares: `0.0`
- Average fill size: `0.0000`
- Bid fills: `0`
- Ask fills: `0`
- Win rate of fades: `0.00%`
- Mean-reversion success rate: `0.00%`
- Average observed full spread at fill: `0.00000c/share`
- Average spread captured at fill: `0.00000c/share`
- Median spread captured at fill: `0.00000c/share`
- Average 60s signed markout: `0.00000c/share`
- Median 60s signed markout: `0.00000c/share`
- Average 60s adverse-selection loss: `0.00000c/share`
- Average 60s net edge: `0.00000c/share`
- Median 60s net edge: `0.00000c/share`
- Total simulated PnL: `$0.00`

## Verdict

Flat-OBI retail-spike fading produced **no demonstrated edge** on `2026-03-25 to 2026-03-25`: the 60-second mark-to-mid result was `0.00000c/share` on average with a `0.00%` win rate across `0` completed fades.
The replay produced zero dispatches and zero fills, so this partition does not provide evidence of a positive mean-reversion edge for the current trigger thresholds.
