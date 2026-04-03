# OBI Evader Maker Backtest Report (2026-03-25 to 2026-03-25)

## Setup

- Strategy: `src.signals.obi_evader_v2.ObiEvaderV2`
- Replay source: `logs\local_snapshot\l2_data`
- Fill database: `logs\universal_backtest_obi_evader_v2_2026-03-25.db`
- Maker fills analyzed: `0`
- Passive bid fills: `0`
- Passive ask fills: `0`
- Average observed full spread at fill: `0.000` cents
- Average captured edge at fill vs mid: `0.000` cents/share

## Maker Markouts

| Horizon | Coverage | Win Rate | Gross Maker Edge (c/share) | Conservative Net Edge (c/share) | Mean Adverse Selection (c/share) | Total Conservative PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5s | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.00 |
| 15s | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.00 |
| 60s | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.00 |

## Daily 60s Breakdown

| Day | Fills | 60s Coverage | 60s Win Rate | 60s Net Edge (c/share) | 60s Net PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0 | 0 | 0.00% | 0.000 | 0.00 |

## Comparison vs Prior Taker OBI

| Horizon | Taker Net Edge (c/share) | Maker Net Edge (c/share) | Improvement (c/share) | Taker Total PnL (USD) | Maker Total PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5s | -16.256 | 0.000 | 16.256 | -88101.37 | 0.00 |
| 15s | -16.138 | 0.000 | 16.138 | -87440.86 | 0.00 |
| 60s | -16.040 | 0.000 | 16.040 | -86795.44 | 0.00 |

## Verdict

The spread-gated replay produced **no maker fills** over this window, so there is no realized 60s edge to score.
On this baseline day the `> 2.0c` gate filtered the strategy down to zero executed fills, which means the offline wide-spread result did not translate into realized activity on this run.
