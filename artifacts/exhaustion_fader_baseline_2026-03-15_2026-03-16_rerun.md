# ExhaustionFader Strict Baseline (2026-03-15 to 2026-03-16)

## Setup

- Strategy: `src.signals.exhaustion_fader.ExhaustionFader`
- Replay source: `logs\local_snapshot\l2_data`
- Trigger count proxy: replay dispatch count, because ExhaustionFader does not expose a separate trigger diagnostics counter.
- Fill and PnL figures: 60-second mark-to-mid values reconstructed from the existing analyzer without modifying the strategy.

## Baseline Table

| Date | trigger_count_per_day | Total Simulated Fills | PnL (USD) |
| --- | ---: | ---: | ---: |
| 2026-03-15 | 13 | 7 | 0.01 |
| 2026-03-16 | 19 | 14 | 0.20 |
| Total | 32 | 21 | 0.21 |

## Verdict

Across the measured range, the strict unmodified ExhaustionFader generated `32` total triggers, `21` simulated fills, and `$0.21` of aggregated 60-second mark-to-mid PnL.
