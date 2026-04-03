# Favorite-Longshot Bias Yield Analysis

## Scope

- SQLite audit target: `logs\trades.db`
- Raw tick archive used for BBO reconstruction: `logs\local_snapshot\l2_data\data\raw_ticks`
- Replay window: `earliest available` to `latest available`
- Market files scanned: `100`
- Token candidates that ever satisfied the sustained 1c-5c filter before side resolution: `8`
- Condition ids resolved against Gamma for YES/NO token mapping: `8`

## Data Quality Notes

- SQLite tables present: `live_orders, live_positions, shadow_trades, trade_persistence_journal, trades`
- No SQLite table storing historical bid/ask or orderbook snapshots was found; quote history was reconstructed from the raw tick archive instead.
- Qualification is strict: a token must stay inside the YES ask band continuously for at least 24h, with observation gaps larger than 6h treated as broken continuity.
- Theoretical PnL follows the PM directive exactly: buy NO at 0.95 once the YES contract has already spent 24h in the 1c-5c band, then assume a +5c payoff unless the market is now explicitly resolved YES.

## Headline Results

| Metric | Value |
| --- | ---: |
| Qualified YES longshots | 8 |
| Avg entry YES ask | 0.032 |
| Median entry YES ask | 0.034 |
| Avg longest sub-5c window | 45.7h |
| Median longest sub-5c window | 32.7h |
| Avg observed post-entry life | 148.2h |
| Median observed post-entry life | 151.7h |
| Avg terminal YES ask | 0.037 |
| Median terminal YES ask | 0.032 |
| Avg max YES ask after qualification | 0.057 |
| Avg max NO mark-to-market drawdown | 1.96 cents |
| YES ask ever spiked above 10c | 1 (12.5%) |
| YES ask ever spiked above 25c | 0 (0.0%) |
| YES ask ever spiked above 50c | 0 (0.0%) |
| Terminal YES ask <= 1c | 0 (0.0%) |
| Markets currently resolved YES | 0 |
| Markets currently resolved NO | 2 |
| Assumed NO wins under PM rule | 8 |
| Theoretical gross ROC at NO 0.95 | 5.26% |

## Bucket Breakdown

| Entry Band | Count | Resolved YES | Theoretical ROC | Avg Observed Life | Spike >10c | Spike >25c |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-2c | 2 | 0 | 5.26% | 155.4h | 0.0% | 0.0% |
| 2-3c | 2 | 0 | 5.26% | 9.1h | 0.0% | 0.0% |
| 3-4c | 1 | 0 | 5.26% | 539.3h | 0.0% | 0.0% |
| 4-5c | 3 | 0 | 5.26% | 105.8h | 33.3% | 0.0% |

## Interpretation

- On this archive slice, no qualified YES longshot has yet gone on to resolve YES. Under the PM's assumption set, that leaves a mechanical gross ROC of 5.26% on capital deployed at 0.95 per contract.
- The bigger practical question is path risk, not terminal win rate: 1 of 8 contracts (12.5%) traded above 10c after already spending 24h in the longshot zone, and 0 (0.0%) traded above 25c.
- Average observed max adverse mark-to-market for a standardized NO 0.95 entry was 1.96 cents per share, with worst cases listed below. That makes this more of a slow-carry short-vol / structural-bias harvest than a low-latency hedge.
- 0 contracts (0.0%) simply decayed back to <=1c by the end of observation, which is the behavioral pattern the FLB thesis needs.

## Worst Post-Qualification Spikes

| Question | Entry YES Ask | Max YES Ask | Terminal YES Ask | Max NO Drawdown | Resolved State | Qualified At |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Will US withdraw from NATO before 2027? | 0.042 | 0.180 | 0.084 | 13.00c | open/assumed NO | 2026-03-12 17:30:03Z |
| Will Bodo Glimt win the 2025–26 Champions League? | 0.018 | 0.077 | 0.026 | 2.70c | NO | 2026-03-10 04:39:09Z |
| Will the Detroit Red Wings win the Eastern Conference? | 0.048 | 0.049 | 0.047 | 0.00c | open/assumed NO | 2026-03-19 00:30:58Z |
| Will Ukraine recapture Crimean territory by June 30, 2026? | 0.042 | 0.044 | 0.035 | 0.00c | open/assumed NO | 2026-03-12 08:12:24Z |
| Will Jesus Christ return before 2027? | 0.039 | 0.039 | 0.039 | 0.00c | open/assumed NO | 2026-03-01 02:12:47Z |
| Will Han Jun-ho win the 2026 Gyeonggi Province Gubernatorial Election? | 0.029 | 0.029 | 0.029 | 0.00c | open/assumed NO | 2026-03-19 15:26:24Z |
| Will Russia test a nuclear weapon by March 31 2026? | 0.020 | 0.020 | 0.019 | 0.00c | NO | 2026-03-12 00:01:42Z |
| Will Andy Beshear win the 2028 US Presidential Election? | 0.017 | 0.017 | 0.016 | 0.00c | open/assumed NO | 2026-03-10 02:31:58Z |

## Quiet Decays

| Question | Entry YES Ask | Max YES Ask | Terminal YES Ask | Longest Sub-5c Window | Resolved State |
| --- | ---: | ---: | ---: | ---: | --- |
