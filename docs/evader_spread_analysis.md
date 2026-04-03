# ObiEvader Spread-Width Analysis (2026-03-15 to 2026-03-19)

## Scope

- Existing maker fills analyzed from `logs/universal_backtest.db`: `57,748`
- No new backtests were run; fill-time spread and 60s markouts were reconstructed from the archived L2 tape.
- Buckets are based on total quoted spread at the moment the maker fill was observed.

## 60s Performance by Fill-Time Spread

| Spread Bucket | Total Fills | 60s Coverage | 60s Win Rate | Avg 60s Conservative Net Edge (c/share) | Total 60s PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| <= 0.5c | 48,428 | 48,414 | 29.88% | -0.286 | -566.15 |
| 0.5c - 1.0c | 6,439 | 6,108 | 6.96% | -0.807 | -183.37 |
| 1.0c - 2.0c | 1,701 | 1,701 | 28.81% | -0.479 | -32.47 |
| > 2.0c | 1,172 | 1,172 | 60.84% | 1.004 | 52.94 |

## Verdict

Only the widest spread buckets were profitable at 60s.
Profitable buckets: `> 2.0c`.
