# Contagion Detector Revision Plan

## Executive Verdict

Do not rerun the current detector unchanged.

The dominant failure is not downstream execution and not same-market book quality. It is the combination of:

1. An inter-market sync gate that assumes near-simultaneous updates across different markets.
2. A synthetic 5-market universe whose shared tags and event IDs create structural prior overlap without real causal linkage.
3. Upstream detector gating that leaves too few leader events after spike and impulse filtering.

From the champion OOS replay on 2026-03-03:

- Leader evaluations with previous snapshot: 3034
- Toxicity spikes: 619
- Leader impulse survivors: 224
- Leader yes/no sync pass rate at 1132 ms: 73.46%
- Leader-to-lagger sync pass rate at 1132 ms: 10.34%
- Leader-to-lagger sync pass rate at 5000 ms: 11.11%

That asymmetry is the core diagnosis.

## Empirical Findings

### 1. Same-market sync is mostly acceptable

Leader yes/no book divergence statistics:

- median: 1.28 ms
- p90: 241,998 ms
- p95: 594,643 ms
- pass rate at 1132 ms: 73.46%
- pass rate at 2000 ms: 74.43%
- pass rate at 5000 ms: 75.13%

Interpretation:

- Most same-market evaluations are locally coherent.
- There is a long tail of stale book combinations, but the central tendency is low.
- This does not look like a universal archive corruption problem.

### 2. Inter-market sync is catastrophically mismatched to the detector design

Leader-to-lagger divergence statistics:

- median: 443,782 ms
- p90: 1,448,271 ms
- p95: 1,715,462 ms
- pass rate at 1132 ms: 10.34%
- pass rate at 2000 ms: 10.34%
- pass rate at 5000 ms: 11.11%

Interpretation:

- The detector is effectively demanding a sub-second synchronization regime from markets that update minutes apart.
- Widening the current threshold from 1132 ms to 5000 ms is functionally irrelevant.
- This is not a tuning issue. It is a model-contract issue.

### 3. Toxicity and impulse gates are secondary blockers, not the primary one

From the same replay:

- 619 spike events from 3034 leader evaluations with previous snapshots: 20.40%
- 224 impulse survivors from 619 spikes: 36.19%

Interpretation:

- The percentile-based toxicity gate is selective but not dead.
- The leader impulse floor removes another large fraction.
- These gates matter, but even if they were relaxed, the current inter-market sync contract would still kill most candidate evaluations.

### 4. Correlation is currently misleading

Runtime correlation matrix for all 10 pairs ended at 0.85 for every pair, but matched-return analysis does not support that as a real empirical relationship.

Representative matched-return results:

- best pair Pearson on matched returns: 0.476
- many pairs are near 0 or negative
- several pairs show 0% same-direction rate where non-zero overlaps exist

Interpretation:

- Structural priors are dominating the runtime correlation view.
- The synthetic shared metadata in the current 5-market universe is overstating relationship strength.
- This makes the current correlation gate look harmless while the universe itself is weak.

## Root Cause Ranking

### 1. Inter-market sync gate contract is wrong

Current logic treats cross-market contagion like a near-simultaneous book alignment problem.

That is incompatible with this archive and likely incompatible with real cross-market Polymarket microstructure outside the most active clusters.

### 2. Universe construction is not real

The current fast micro universe uses placeholder questions with identical synthetic event and tag metadata.

That is useful for plumbing isolation but not for signal discovery.

### 3. Structural priors are masking weak empirical co-movement

The detector sees correlation support where matched return behavior suggests little or no stable same-direction propagation.

### 4. Toxicity spike and leader impulse thresholds still leave a thin upstream candidate set

This matters, but it is downstream of the larger sync/universe problem.

## Highest-Leverage Change

Replace the inter-market sync gate with a causal lag/freshness gate.

### Current contract

- Require leader and lagger snapshots to be almost simultaneous.
- Result: only about 10% of candidate cross-market comparisons survive.

### Proposed contract

Keep two different synchronization contracts:

1. Intra-market yes/no integrity gate
- Continue using a tight same-market desync threshold.
- Suggested search range: 100 ms to 2000 ms.

2. Inter-market causal lag gate
- Replace symmetric max-minus-min timestamp divergence with an asymmetric causal window.
- Accept lagger snapshots when:
  - `0 <= leader_ts - lagger_ts <= causal_max_lag_ms`
  - lagger last-trade age is below a separate freshness ceiling
- Suggested search range for `causal_max_lag_ms`: 30,000 ms to 900,000 ms
- Suggested search range for lagger freshness: 30 s to 300 s

Reasoning:

- Same-market books need local coherence.
- Cross-market contagion needs recency and ordering, not simultaneity.
- The current execution path already contains a last-trade-age filter, so this aligns detector logic with execution reality.

## Detector Revision Plan

### Phase 1. Fix the sync model

Add separate parameters:

- `contagion_arb_leader_book_max_desync_ms`
- `contagion_arb_intermarket_causal_lag_ms`
- `contagion_arb_lagger_snapshot_max_age_s`

Behavior:

- Leader yes/no snapshots must satisfy the tight same-market gate.
- Lagger comparison must satisfy causal ordering and freshness, not simultaneity.
- Record telemetry for:
  - causal lag accepted
  - causal lag too large
  - lagger newer than leader
  - lagger stale

### Phase 2. Stop using synthetic theme overlap as a proxy for tradable relatedness

Use real cluster metadata only.

Rules:

- Prefer same-event mutually exclusive markets.
- Then allow tightly curated same-theme clusters with proven archive coverage and empirical co-movement.
- Do not let identical synthetic tags manufacture relationship strength.

### Phase 3. Rebalance spike and impulse gates after sync refactor

Suggested revised search ranges:

- `contagion_arb_trigger_percentile`: 0.60 to 0.90
- `contagion_arb_min_history`: 4 to 32
- `contagion_arb_min_leader_shift`: 0.00025 to 0.01
- `contagion_arb_min_residual_shift`: 0.00025 to 0.01
- `contagion_arb_toxicity_impulse_scale`: 0.01 to 0.15

Reasoning:

- The current leader impulse floor still leaves only 224 events after spikes.
- Once inter-market gating is repaired, these settings should be re-optimized on a real universe instead of the current synthetic cluster.

### Phase 4. Add better detector telemetry

Required counters beyond the current autopsy:

- `reject_lagger_newer_than_leader`
- `reject_lagger_snapshot_stale`
- `reject_causal_lag_too_large`
- `accepted_causal_lag_count`
- per-pair accepted lag quantiles
- per-pair residual distribution at pre-signal stage
- per-pair matched-return correlation and same-direction rate

## Universe Recommendation

## Current answer

Do not expand the current March archive universe in-place.

The cluster scan over `data/si9_clusters_monday.json` found no real replacement clusters with token coverage in the recorded March 2 to March 4 archive.

That means the blocker is not only selection logic. The archive itself does not currently contain the real 15 to 20 market universe needed for a proper contagion experiment.

## Recommended next universe

Backfill or capture a real 15 to 20 market universe built from concrete mutually exclusive clusters.

Recommended candidate set from existing cluster inventory:

1. Fed decision in April
- 4 markets
- high liquidity
- mutually exclusive outcomes

2. La Liga Winner
- 4 markets
- liquid multi-leg winner market

3. Next Prime Minister of Slovenia
- 4 markets
- cohesive election-style cluster

4. Who will win the Lyon mayoral election?
- 4 markets
- real same-event multi-runner market

5. Slovenian Parliamentary Election Winner
- 2 markets
- compact, high-volume same-event pair

Total target size: 18 markets

Important caveat:

- These are recommended cluster shapes, not immediately runnable archive inputs.
- They require fresh token-level backfill or a new recorded archive window.

## Estimated Signal-Rate Improvement

### Sync refactor only

Current cross-market pass rate is 10.34% at 1132 ms and only 11.11% at 5000 ms.

If the detector moves from strict simultaneity to a causal lag/freshness model and that lifts usable pair evaluations from roughly 10% to even 50%, the evaluable pair count rises by about 4.8x.

If most fresh causal laggers become admissible, the increase is closer to 9x.

### Combined with gate rebalance

If spike and impulse tuning lifts the leader event pool from 224 to something closer to the full spike set of 619, candidate pair evaluations could increase by roughly another 2x to 3x on top of the sync fix.

### Practical estimate

Expected improvement in pre-signal evaluable opportunities:

- conservative: 5x
- aggressive: 10x+

Expected improvement in actual fired signals:

- from zero to non-zero is realistic if the universe is real and archive coverage is fixed
- zero to meaningful still depends on replacing the synthetic universe with actual event-linked markets

## Next WFO Configuration

Use the next WFO only after the sync refactor and real-universe backfill.

Recommended optimization set:

- `contagion_arb_leader_book_max_desync_ms`
- `contagion_arb_intermarket_causal_lag_ms`
- `contagion_arb_lagger_snapshot_max_age_s`
- `contagion_arb_trigger_percentile`
- `contagion_arb_min_history`
- `contagion_arb_min_leader_shift`
- `contagion_arb_min_residual_shift`
- `contagion_arb_toxicity_impulse_scale`
- `contagion_arb_min_correlation`

Recommended WFO acceptance criteria:

- OOS signals per fold greater than 20
- OOS fills per fold greater than 5
- positive OOS PnL in at least half of folds
- no dominant single suppressor above 60% after refactor
- cross-market causal lag telemetry showing a stable accepted-lag distribution, not a threshold-edge artifact

## Final Recommendation

The next engineering cycle should not be “try wider thresholds.”

It should be:

1. Replace inter-market simultaneity with causal lag plus freshness.
2. Backfill a real cluster universe.
3. Re-run WFO with telemetry that distinguishes archive quality, causal lag, and actual signal scarcity.

Without those three changes, another contagion WFO is likely to produce a cleaner version of the same zero-signal result.

## Implementation Results

### Scope completed

- Added `CausalLagConfig`, `CausalLagAssessment`, and `CausalLagGate`.
- Refactored `ContagionArbDetector` so cross-market comparisons use causal lag plus freshness while same-market yes/no integrity still uses `CrossBookSyncGate`.
- Extended detector telemetry with:
  - `leader_age_ms`
  - `lagger_age_ms`
  - `causal_lag_ms`
  - `causal_gate_result`
  - causal/legacy pass counters and rejection counters
- Added unit coverage for the new gate and contagion refactor behavior.

### Validation configuration

Archived-data validation was run on `2026-03-03` using the champion March micro parameters plus this fixed causal configuration:

- `max_leader_age_ms=5000`
- `max_lagger_age_ms=30000`
- `max_causal_lag_ms=600000`
- `allow_negative_lag=False`

### Validation outcome

Summary from `_tmp_contagion_causal_validation.json`:

- leader events evaluated: 48,728
- events reaching spike check: 11,132
- cross-market pairs evaluated: 70,749
- legacy sync pass rate on the same evaluated pair set: 3.91%
- causal gate pass rate on the same evaluated pair set: 18.52%
- signals fired: 0
- fills executed: 0

Interpretation:

- The refactor materially improved admissible cross-market pair flow, lifting pair survival by about 4.74x versus the old simultaneity gate on the same evaluated set.
- The new dominant suppressor is now `lagger_snapshot_stale`, with 57,647 rejects.
- Correlation then becomes the next blocker on the surviving causal set, with 13,102 rejects.
- Residual shift is no longer the active blocker in this fixed-config replay.

### Updated conclusion

The causal refactor was directionally correct and measurably improved detector admissibility, but it did not convert this archive/universe into a tradable contagion setup.

The blocker has moved from impossible simultaneity toward lagger freshness and weak usable cross-market linkage. That is progress in diagnosis and model correctness, but not yet enough for deployment or WFO rerun.
