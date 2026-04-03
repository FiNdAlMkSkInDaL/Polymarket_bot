# Lead/Lag Maker Scorecard

## Scope

- Replay date: `2026-03-03`
- Primary condition: `0xaeea5f917fc5746387b5f9c0a4263dba035dbb3f0ac6ad72bf92183d21e26739`
- Secondary condition: `0xcfbe2de215f38407317a7ecfff4c13aa591eca2fe428fc203e5669a936009e78`
- Primary input: token-level YES stream `27911616648163853231017805596118911526202185567944724228908312975649746722206.jsonl`
- Secondary input: token-level YES stream `10852332002112773763359204434610715394710971181223888185945987959566837502210.jsonl`

The checked-in `data/market_map.json` does not include human-readable titles for this pair. The pair was selected from the archived contagion study because it had the strongest saved matched-return relationship in `_tmp_contagion_revision_analysis.json`:

- `pearson_matched_returns = 0.4758`
- `same_direction_rate = 1.0000`
- `matched_moves = 80`
- `sync_p50_ms = 289254.115`

## Replay Setup

- Strategy: `src.signals.lead_lag_maker.LeadLagMaker`
- Contagion trigger: `0.5` cents
- Lookback window: `60_000 ms`
- Cooldown: `5_000 ms`
- Order size: `5`
- Baseline: identical replay with contagion threshold disabled (`9999` cents)

## Required Engine Fixes

Two code-path fixes were required before the backtest reflected the real archive correctly:

1. `scripts/run_universal_backtest.py` now expands `--strategy-config` keys into the strategy constructor, so `LeadLagMaker` can receive `primary_market_id` and `secondary_market_id` from JSON config.
2. `src/data/orderbook.py` now sorts full `book` snapshots before truncating to `_MAX_LEVELS`. The previous implementation truncated first, which discarded the true best bid/ask on unsorted Polymarket snapshots and caused the strategy to quote non-fillable extremes.

## Replay Summary

| Metric                         | Evasion Replay | No-Evasion Control |
| ------------------------------ | -------------: | -----------------: |
| Total events                   |            477 |                477 |
| Book events                    |            404 |                404 |
| Trade events                   |             73 |                 73 |
| Dispatches                     |            293 |                286 |
| Maker fills                    |             11 |                 11 |
| Persisted rows                 |             11 |                 11 |
| Contagion fires (`CANCEL_ALL`) |              7 |                  0 |
| Cancelled resting orders       |             14 |                  0 |

## Secondary-Market Edge Scorecard

PnL is measured as 5-second forward markout on each secondary-market maker fill:

- For `BID` fills: `(future_mid_5s - fill_price) * size * 100`
- For `ASK` fills: `(fill_price - future_mid_5s) * size * 100`

| Metric                       | Evasion Replay | No-Evasion Control |
| ---------------------------- | -------------: | -----------------: |
| Fill count                   |             11 |                 11 |
| Win rate                     |         90.91% |             90.91% |
| Adverse-selection rate at 5s |          0.00% |              0.00% |
| Total 5s PnL                 |  15.3680 cents |      15.3680 cents |
| Average 5s PnL per fill      |   1.3971 cents |       1.3971 cents |

## Cooldown Effectiveness

- Contagion detection fired `7` times.
- The strategy cancelled `14` resting secondary orders across those `7` events.
- Control fills falling inside evasion cooldown windows: `0`

That means the 5-second cooldown never intercepted a fill-bearing toxic window on this pair/day. The cancellation logic executed, but it did not remove any fills that the control otherwise would have taken.

## Interpretation

1. Cross-market reflexivity exists at the trigger level. The primary market produced `7` contagion detections once the lookback was aligned to the real archive cadence.
2. The latency gap is not monetizable on this replay slice. Secondary fills, fill timestamps, win rate, and 5-second markout PnL were identical between evasion and control.
3. The cooldown did not protect against toxic flow in a measurable way. No control fills occurred during the windows where the evasion strategy was flat.
4. Relative to the passive no-evasion baseline, measured maker edge uplift was exactly `0.0000` cents on this dataset.

## Verdict

For the tested pair and date, the historical Polymarket latency gap is **not wide enough to prove positive incremental maker edge from contagion evasion**.

The lead market did move first often enough to trigger cancellations, but those cancellations did not overlap with harmful secondary fills. On this evidence, Level 0 quote cancellation is operationally possible, yet not economically justified for this pair/day.