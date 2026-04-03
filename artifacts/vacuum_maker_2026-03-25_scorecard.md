# VacuumMaker Toxic-Day Scorecard

## Run

- Date: `2026-03-25`
- Strategy: `src.signals.vacuum_maker.VacuumMaker`
- Replay DB: `logs/universal_backtest_vacuum_2026-03-25.db`
- Baseline DB: `logs/universal_backtest_obi_evader_2026-03-25.db`

## Method

- Fills were read from the universal backtest databases.
- Fill-time spread capture and 60-second markout were reconstructed from the same `2026-03-25` raw tick stream used by the replay engine.
- Net edge per share was computed as signed fill-to-mid markout after 60 seconds.
- Adverse selection was measured as the unfavorable portion of the 60-second post-fill drift.

## VacuumMaker

- Total simulated fills: `40`
- Fills with full 60s horizon: `40`
- Total filled shares: `330.0`
- Average fill size: `8.25`
- Bid fills: `18`
- Ask fills: `22`
- Win rate: `77.50%`
- Average spread captured: `0.03875c/share`
- Median spread captured: `0.05000c/share`
- Average 60s signed markout: `0.00000c/share`
- Average 60s adverse-selection loss: `0.00125c/share`
- Average net edge: `0.03875c/share`
- Median net edge: `0.05000c/share`
- P10/P90 net edge: `0.00000c/share` / `0.05000c/share`
- Total PnL: `$0.15`

## ObiEvader Baseline

- Total simulated fills: `4732`
- Fills with full 60s horizon: `4729`
- Total filled shares: `7573.0`
- Average fill size: `1.6014`
- Bid fills: `183`
- Ask fills: `4546`
- Win rate: `99.45%`
- Average spread captured: `0.06131c/share`
- Average 60s signed markout: `-0.00093c/share`
- Average 60s adverse-selection loss: `0.00097c/share`
- Average net edge: `0.06038c/share`
- Total PnL: `$6.21`

## Comparison

- VacuumMaker minus ObiEvader net edge: `-0.02163c/share`
- VacuumMaker minus ObiEvader total PnL: `-$6.06`
- VacuumMaker minus ObiEvader signed 60s markout: `+0.00093c/share`

## Verdict

- The 500ms delay did appear to protect VacuumMaker from the toxic tail. Its average 60-second adverse-selection loss was only `0.00125c/share`, effectively zero.
- The strategy was mathematically positive on this day, but only barely: `0.03875c/share` average net edge and `$0.15` total PnL across `330` shares.
- VacuumMaker was **not** superior to the ObiEvader baseline. ObiEvader delivered higher spread capture, higher average net edge, much higher win rate, and materially larger total PnL on the same dataset.
- Conclusion: stepping into the post-crash vacuum was viable, but on `2026-03-25` it was weaker than simply evading the crash.