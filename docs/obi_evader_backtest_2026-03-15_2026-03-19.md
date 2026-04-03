# OBI Evader Maker Backtest Report (2026-03-15 to 2026-03-19)

## Setup

- Strategy: `src.signals.obi_evader.ObiEvader`
- Replay source: `logs\local_snapshot\l2_data`
- Fill database: `logs\universal_backtest.db`
- Maker fills analyzed: `57,748`
- Passive bid fills: `24,520`
- Passive ask fills: `33,228`
- Average observed full spread at fill: `-0.549` cents
- Average captured edge at fill vs mid: `0.251` cents/share

## Maker Markouts

| Horizon | Coverage | Win Rate | Gross Maker Edge (c/share) | Conservative Net Edge (c/share) | Mean Adverse Selection (c/share) | Total Conservative PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5s | 57,737 | 27.94% | 0.244 | -0.313 | 0.007 | -692.24 |
| 15s | 57,726 | 27.94% | 0.238 | -0.319 | 0.013 | -709.18 |
| 60s | 57,395 | 28.04% | 0.228 | -0.320 | 0.021 | -729.05 |

## Daily 60s Breakdown

| Day | Fills | 60s Coverage | 60s Win Rate | 60s Net Edge (c/share) | 60s Net PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-03-15 | 1,126 | 1,125 | 25.24% | -0.402 | -15.35 |
| 2026-03-16 | 11,413 | 11,409 | 32.83% | -0.366 | -150.54 |
| 2026-03-17 | 12,033 | 12,032 | 25.82% | -0.315 | -157.19 |
| 2026-03-18 | 18,055 | 18,049 | 21.34% | -0.253 | -200.39 |
| 2026-03-19 | 15,121 | 14,780 | 34.54% | -0.365 | -205.57 |

## Comparison vs Prior Taker OBI

| Horizon | Taker Net Edge (c/share) | Maker Net Edge (c/share) | Improvement (c/share) | Taker Total PnL (USD) | Maker Total PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5s | -16.256 | -0.313 | 15.943 | -88101.37 | -692.24 |
| 15s | -16.138 | -0.319 | 15.819 | -87440.86 | -709.18 |
| 60s | -16.040 | -0.320 | 15.720 | -86795.44 | -729.05 |

## Verdict

The OBI evasion layer produces **still toxic** at 60 seconds: conservative net edge is `-0.320` cents/share versus `-16.040` cents/share for the prior taker variant.
Average adverse selection at 60 seconds is `0.021` cents/share, which is still overwhelming the captured spread.
