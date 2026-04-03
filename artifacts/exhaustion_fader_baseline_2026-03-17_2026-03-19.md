# ExhaustionFader Strict Baseline (2026-03-17 to 2026-03-19)

## Setup

- Strategy: `src.signals.exhaustion_fader.ExhaustionFader`
- Replay source: `logs\local_snapshot\l2_data`
- Trigger count proxy: replay dispatch count, because ExhaustionFader does not expose a separate trigger diagnostics counter.
- Fill and PnL figures: 60-second mark-to-mid values reconstructed from the existing analyzer without modifying the strategy.

## Baseline Table

| Date | trigger_count_per_day | Total Simulated Fills | PnL (USD) |
| --- | ---: | ---: | ---: |
| 2026-03-17 | 8 | 5 | 0.11 |
| 2026-03-18 | 20 | 11 | 0.67 |
| 2026-03-19 | 18 | 4 | 0.71 |
| Total | 46 | 20 | 1.50 |

## Verdict

Across the measured range, the strict unmodified ExhaustionFader generated `46` total triggers, `20` simulated fills, and `$1.50` of aggregated 60-second mark-to-mid PnL.
