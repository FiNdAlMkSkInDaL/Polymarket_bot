# Favorite-Longshot Bias Yield Analysis

## Scope

- SQLite audit target: `logs\trades.db`
- Raw tick archive used for BBO reconstruction: `logs\local_snapshot\l2_data\data\raw_ticks`
- Replay window: `earliest available` to `latest available`
- Market files scanned: `4500`
- Token candidates that ever satisfied the sustained 1c-5c filter before side resolution: `252`
- Condition ids resolved against Gamma for YES/NO token mapping: `133`

## Data Quality Notes

- SQLite tables present: `live_orders, live_positions, shadow_trades, trade_persistence_journal, trades`
- No SQLite table storing historical bid/ask or orderbook snapshots was found; quote history was reconstructed from the raw tick archive instead.
- Qualification is strict: a token must stay inside the YES ask band continuously for at least 24h, with observation gaps larger than 6h treated as broken continuity.
- Theoretical PnL follows the PM directive exactly: buy NO at 0.95 once the YES contract has already spent 24h in the 1c-5c band, then assume a +5c payoff unless the market is now explicitly resolved YES.

## Headline Results

| Metric | Value |
| --- | ---: |
| Qualified YES longshots | 127 |
| Avg entry YES ask | 0.025 |
| Median entry YES ask | 0.025 |
| Avg longest sub-5c window | 41.4h |
| Median longest sub-5c window | 33.1h |
| Avg observed post-entry life | 104.8h |
| Median observed post-entry life | 61.0h |
| Avg terminal YES ask | 0.023 |
| Median terminal YES ask | 0.019 |
| Avg max YES ask after qualification | 0.043 |
| Avg max NO mark-to-market drawdown | 1.26 cents |
| YES ask ever spiked above 10c | 4 (3.1%) |
| YES ask ever spiked above 25c | 3 (2.4%) |
| YES ask ever spiked above 50c | 1 (0.8%) |
| Terminal YES ask <= 1c | 22 (17.3%) |
| Markets currently resolved YES | 0 |
| Markets currently resolved NO | 38 |
| Resolved bucket size | 38 |
| Active bucket size | 89 |
| Assumed NO wins under PM rule | 127 |
| Theoretical gross ROC at NO 0.95 | 5.26% |

## Resolution Filter

| Bucket | Count | YES Resolutions | NO Resolutions | Gross ROC | Net ROC After Fees | Avg Entry Fee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Resolved | 38 | 0 | 38 | 5.26% | 5.17% | 0.090c |
| Active | 89 | 0 | 0 | 5.26% | 5.13% | 0.128c |

## Bucket Breakdown

| Entry Band | Count | Resolved YES | Theoretical ROC | Avg Observed Life | Spike >10c | Spike >25c |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-2c | 46 | 0 | 5.26% | 98.0h | 0.0% | 0.0% |
| 2-3c | 41 | 0 | 5.26% | 119.8h | 7.3% | 7.3% |
| 3-4c | 22 | 0 | 5.26% | 89.8h | 0.0% | 0.0% |
| 4-5c | 18 | 0 | 5.26% | 106.2h | 5.6% | 0.0% |

## Interpretation

- On this archive slice, no qualified YES longshot has yet gone on to resolve YES. Under the PM's assumption set, that leaves a mechanical gross ROC of 5.26% on capital deployed at 0.95 per contract.
- Ground-truth realized yield on already closed markets is 5.17% net of modeled Polymarket entry fees, across 38 resolved contracts. That is the underwriter-grade ROC to trust rather than the full-sample 5.26% gross carry figure.
- The bigger practical question is path risk, not terminal win rate: 4 of 127 contracts (3.1%) traded above 10c after already spending 24h in the longshot zone, and 3 (2.4%) traded above 25c.
- Average observed max adverse mark-to-market for a standardized NO 0.95 entry was 1.26 cents per share, with worst cases listed below. That makes this more of a slow-carry short-vol / structural-bias harvest than a low-latency hedge.
- 22 contracts (17.3%) simply decayed back to <=1c by the end of observation, which is the behavioral pattern the FLB thesis needs.

## Worst Post-Qualification Spikes

| Question | Category | Entry YES Ask | Max YES Ask | Terminal YES Ask | Max NO Drawdown | Resolved State | Qualified At |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Kraken IPO by March 31, 2026? | crypto | 0.025 | 0.529 | 0.007 | 47.90c | NO | 2026-03-12 15:58:14Z |
| GTA VI released before June 2026? | culture | 0.021 | 0.404 | 0.028 | 35.40c | open/assumed NO | 2026-03-01 02:12:52Z |
| Cap on gambling loss deductions repealed by March 31? | policy | 0.027 | 0.399 | 0.022 | 34.90c | NO | 2026-03-18 20:57:02Z |
| Will US withdraw from NATO before 2027? | geopolitics | 0.042 | 0.180 | 0.084 | 13.00c | open/assumed NO | 2026-03-12 17:30:03Z |
| Will Hezbollah disarm by March 31? | geopolitics | 0.023 | 0.097 | 0.015 | 4.70c | NO | 2026-03-10 16:13:44Z |
| Will Fannie Mae’s market cap be between $200B and $250B at market close on IPO day? | business | 0.031 | 0.081 | 0.064 | 3.10c | open/assumed NO | 2026-03-17 01:33:37Z |
| Will Dplus win the LCK 2026 season playoffs? | sports | 0.037 | 0.080 | 0.051 | 3.00c | open/assumed NO | 2026-03-16 21:22:59Z |
| Will the People Power Party (PPP) win the 2026 South Korean local elections? | sports | 0.039 | 0.079 | 0.048 | 2.90c | open/assumed NO | 2026-03-10 05:39:51Z |
| Will Bodo Glimt win the 2025–26 Champions League? | sports | 0.018 | 0.077 | 0.026 | 2.70c | NO | 2026-03-10 04:39:09Z |
| Will the Toronto Raptors win the NBA Eastern Conference Finals? | sports | 0.021 | 0.075 | 0.023 | 2.50c | open/assumed NO | 2026-03-10 13:47:25Z |

## Narrative Shock Sectors (>10c)

| Question | Category | Max YES Ask | Resolved State |
| --- | --- | ---: | --- |
| Kraken IPO by March 31, 2026? | crypto | 0.529 | NO |
| GTA VI released before June 2026? | culture | 0.404 | ACTIVE |
| Cap on gambling loss deductions repealed by March 31? | policy | 0.399 | NO |
| Will US withdraw from NATO before 2027? | geopolitics | 0.180 | ACTIVE |

## Quiet Decays

| Question | Entry YES Ask | Max YES Ask | Terminal YES Ask | Longest Sub-5c Window | Resolved State |
| --- | ---: | ---: | ---: | ---: | --- |
| Will the Fed decrease interest rates by 50+ bps after the April 2026 meeting? | 0.010 | 0.010 | 0.005 | 27.1h | open/assumed NO |
| Will LNG Esports win the LPL 2026 season? | 0.010 | 0.010 | 0.006 | 31.8h | open/assumed NO |
| Will Brian Kemp win the 2028 Republican presidential nomination? | 0.010 | 0.010 | 0.009 | 30.0h | open/assumed NO |
| Will Ann Diener win the Alaska Senate race in 2026? | 0.010 | 0.010 | 0.010 | 27.3h | open/assumed NO |
| Will Vivek Ramaswamy win the 2028 US Presidential Election? | 0.011 | 0.011 | 0.007 | 24.4h | open/assumed NO |
| Will LeBron James win the 2028 Democratic presidential nomination? | 0.011 | 0.011 | 0.010 | 30.7h | open/assumed NO |
| AI Industry Downturn by March 31, 2026? | 0.012 | 0.012 | 0.010 | 24.7h | NO |
| Will Fulham finish in 3rd place in the 2025-26 English Premier League? | 0.013 | 0.013 | 0.003 | 27.0h | open/assumed NO |
| Nicolas Sarkozy in jail by March 31? | 0.013 | 0.013 | 0.008 | 29.8h | NO |
| Will Tottenham win the 2025–26 Champions League? | 0.017 | 0.017 | 0.004 | 25.7h | NO |
