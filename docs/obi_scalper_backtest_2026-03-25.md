# OBI Scalper Backtest Report (2026-03-25 to 2026-03-25)

## Setup

- Strategy: `src.signals.obi_scalper.ObiScalper`
- Replay source: `logs\local_snapshot\l2_data`
- Persisted fills DB: `logs\universal_backtest.db`
- Simulated fills: `108,459`
- YES fills: `54,227`
- NO fills: `54,232`

## Execution Cost

- Average observed full spread: `18.263` cents
- Average taker crossing cost (half-spread): `9.132` cents/share
- Average fill minus prevailing instrument mid: `15.058` cents/share

## Forward Markouts

| Horizon | Coverage | Win Rate | Gross Edge (mean, c/share) | Net Edge After Fees (mean, c/share) | Net Edge Median (c/share) | Mean Net PnL (c/fill) | Total Net PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5s | 108,395 | 6.48% | -15.231 | -16.256 | -2.187 | -81.278 | -88101.37 |
| 15s | 108,369 | 8.15% | -15.117 | -16.138 | -2.124 | -80.688 | -87440.86 |
| 60s | 108,221 | 11.89% | -15.021 | -16.040 | -2.019 | -80.202 | -86795.44 |

## Verdict

The 60-second post-fill markout is **toxic** on a fee-adjusted basis: mean net edge is `-16.040` cents/share with a `11.89%` win rate.
If the 60-second horizon is the decision horizon, the signal is not viable as a taker strategy at the current threshold.
