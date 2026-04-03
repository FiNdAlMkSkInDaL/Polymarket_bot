# OBI Evader Maker Backtest Report (2026-03-25 to 2026-03-25)

## Setup

- Strategy: `src.signals.obi_evader.ObiEvader`
- Replay source: `logs\local_snapshot\l2_data`
- Fill database: `logs\universal_backtest.db`
- Maker fills analyzed: `8,893`
- Passive bid fills: `3,364`
- Passive ask fills: `5,529`
- Average observed full spread at fill: `59.727` cents
- Average captured edge at fill vs mid: `22.460` cents/share

## Maker Markouts

| Horizon | Coverage | Win Rate | Gross Maker Edge (c/share) | Conservative Net Edge (c/share) | Mean Adverse Selection (c/share) | Total Conservative PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5s | 8,893 | 72.88% | 21.781 | 20.619 | 0.679 | 2056.70 |
| 15s | 8,893 | 70.28% | 19.866 | 18.774 | 2.595 | 1887.13 |
| 60s | 8,889 | 68.35% | 18.667 | 17.665 | 3.804 | 1666.19 |

## Comparison vs Prior Taker OBI

| Horizon | Taker Net Edge (c/share) | Maker Net Edge (c/share) | Improvement (c/share) | Taker Total PnL (USD) | Maker Total PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5s | -16.256 | 20.619 | 36.875 | -88101.37 | 2056.70 |
| 15s | -16.138 | 18.774 | 34.912 | -87440.86 | 1887.13 |
| 60s | -16.040 | 17.665 | 33.705 | -86795.44 | 1666.19 |

## Verdict

The OBI evasion layer produces **positive maker edge** at 60 seconds: conservative net edge is `17.665` cents/share versus `-16.040` cents/share for the prior taker variant.
Average adverse selection at 60 seconds is `3.804` cents/share, which is below the captured spread and consistent with viable passive quoting.
