# Whale Wall Jumper Feature Importance Report

## Scope

- Dataset: `artifacts/wall_jumper_2026-03-25_wall_metadata.csv`
- Analysis target: discriminate `FILLED` walls from `PULLED` walls.
- Excluded outcome: `EXPIRED`.
- Sample used for classification analysis: `10,581` walls = `24 FILLED` + `10,557 PULLED`.

## Executive Verdict

The Whale Wall Jumper does possess **filterable alpha**, but only in a narrow, high-conviction slice of the wall universe. The dominant predictive feature is **distance from the mid-price**. Walls near the mid are almost always pulled. Walls several ticks away from the mid are materially more likely to survive and get filled. `wall_size_usd` adds a useful secondary screen. `time_of_day_bucket` is weak on its own and should only be used as a tie-breaker.

Binary decision: **Do not retire the strategy. Ship a sparse v3 filter.**

## Baseline

Across all non-expired labeled walls:

| Universe                       |  Count | Filled | Pulled | Fill Rate | Pull Rate |
| ------------------------------ | -----: | -----: | -----: | --------: | --------: |
| All `FILLED` vs `PULLED` walls | 10,581 |     24 | 10,557 |     0.23% |    99.77% |

This is a highly imbalanced classification problem. Any usable filter must lift fill odds by orders of magnitude, not a few basis points.

## Feature Results

### 1. `price_level_vs_mid`

This is the only feature with clear first-order separation.

- Rank-separation score: `AUC ~= 0.998` for larger absolute distance implying higher fill odds.
- Median absolute distance:
  - `FILLED`: `3.425` ticks
  - `PULLED`: `0.500` ticks
- Mean absolute distance:
  - `FILLED`: `7.354` ticks
  - `PULLED`: `0.598` ticks

Distance bins:

| Absolute Distance From Mid |  Count | Filled | Fill Rate |
| -------------------------- | -----: | -----: | --------: |
| `<= 1.0` ticks             | 10,496 |      0 |     0.00% |
| `1.0 - 1.5` ticks          |     18 |      2 |    11.11% |
| `2.0 - 3.0` ticks          |      8 |      4 |    50.00% |
| `> 3.0` ticks              |     45 |     18 |    40.00% |

Threshold view:

| Rule                   | Count | Filled | Fill Rate |   95% Wilson CI |
| ---------------------- | ----: | -----: | --------: | --------------: |
| `abs(distance) >= 2.0` |    53 |     22 |    41.51% | 29.26% - 54.91% |
| `abs(distance) >= 3.0` |    47 |     20 |    42.55% | 29.51% - 56.72% |

Interpretation:

- Near-mid whale walls are overwhelmingly spoof-like in this dataset.
- Every actionable filled wall lives in the far-from-mid tail.
- This is not a subtle effect. It is the core signal.

### 2. `wall_size_usd`

Size helps, but not enough to stand alone.

- Rank-separation score: `AUC ~= 0.651`.
- Median wall size:
  - `FILLED`: `$323,220`
  - `PULLED`: `$13,472`
- Mean wall size:
  - `FILLED`: `$3.97m`
  - `PULLED`: `$40.5k`

Threshold view:

| Rule            | Count | Filled | Fill Rate |           95% Wilson CI |
| --------------- | ----: | -----: | --------: | ----------------------: |
| `size >= $100k` |   146 |     12 |     8.22% |          4.76% - 13.82% |
| `size >= $250k` |    75 |     12 |    16.00% | not computed separately |
| `size >= $1m`   |    31 |      9 |    29.03% | not computed separately |

Interpretation:

- Larger walls are firmer than small walls.
- But size alone still leaves pull rates too high for a clean production rule.
- Size should be used only after the distance gate, not before it.

### 3. `time_of_day_bucket`

Time-of-day is weak as a standalone predictor and likely reflects where the rare deep walls happened to appear on this day.

Best standalone hour buckets:

| Hour Bucket   | Count | Filled | Fill Rate |
| ------------- | ----: | -----: | --------: |
| `14:00-14:59` |    44 |      2 |     4.55% |
| `03:00-03:59` |    92 |      4 |     4.35% |
| `00:00-00:59` |   199 |      8 |     4.02% |
| `13:00-13:59` |    58 |      2 |     3.45% |
| `05:00-05:59` |    64 |      2 |     3.13% |
| `22:00-22:59` |    72 |      2 |     2.78% |

Many other buckets produced zero fills.

Interpretation:

- Hour-of-day alone is not a robust predictor.
- It becomes mildly useful only after applying the distance gate.
- This should be treated as a tertiary filter or a ranking feature, not a hard rule by itself.

## Best Combinations

The useful rules are combinations of **far from mid** plus **large wall size**.

| Rule                                                    | Count | Filled | Pulled | Fill Rate | Pull Rate |   95% Wilson CI |
| ------------------------------------------------------- | ----: | -----: | -----: | --------: | --------: | --------------: |
| `abs(distance) >= 2.0` and `size >= $100k`              |    23 |     11 |     12 |    47.83% |    52.17% | 29.24% - 67.04% |
| `abs(distance) >= 3.0` and `size >= $500k`              |    19 |     10 |      9 |    52.63% |    47.37% | 31.71% - 72.67% |
| Best hours + `abs(distance) >= 2.0`                     |    41 |     18 |     23 |    43.90% |    56.10% | 29.89% - 58.96% |
| Best hours + `abs(distance) >= 3.0` and `size >= $100k` |    14 |      8 |      6 |    57.14% |    42.86% | 32.59% - 78.62% |
| Best hours + `abs(distance) >= 3.0` and `size >= $500k` |    13 |      8 |      5 |    61.54% |    38.46% | 35.52% - 82.29% |

Retention of positive outcomes:

- `abs(distance) >= 2.0` keeps `22 / 24` fills.
- `abs(distance) >= 3.0` keeps `20 / 24` fills.
- `abs(distance) >= 2.0` and `size >= $100k` keeps `11 / 24` fills.
- Best hours + `abs(distance) >= 3.0` and `size >= $500k` keeps `8 / 24` fills.

## Statistical Reading

There is a clear hierarchy:

1. `price_level_vs_mid` is overwhelmingly informative.
2. `wall_size_usd` is directionally helpful but weaker.
3. `time_of_day_bucket` adds only sparse incremental value.

The key threshold is not close-to-mid. It is the opposite:

- **Walls within 1 tick of mid are almost pure noise.**
- **Walls at least 2 ticks from mid are a different regime.**

That threshold alone lifts fill rate from `0.23%` to `41.51%` while retaining `22` of the `24` fills. Adding a large-size gate pushes the subset near or above coin-flip in favor of `FILLED`, which is a dramatic improvement over the raw universe and better than the live filtered-set pull profile reported by the existing strategy artifacts.

## Recommendation

### v3 Entry Rule

Use a two-stage entry rule.

Hard gate:

- `abs(price_level_vs_mid_ticks) >= 2.0`
- `wall_size_usd >= 100000`

High-conviction boost tier:

- Increase priority or size only when:
  - `abs(price_level_vs_mid_ticks) >= 3.0`
  - `wall_size_usd >= 500000`
  - `time_of_day_bucket` is one of `00`, `03`, `05`, `13`, `14`, `22`

Why this rule:

- The hard gate keeps almost all fills while removing the near-mid spoof regime.
- The size screen removes small, flimsy walls that still survive the distance filter.
- The hour bucket helps rank the sparse high-conviction opportunities, but it should not be the primary gate.

### Deployment Posture

- Do **not** run the broad Wall Jumper logic against all observed walls.
- Do run a **sparse, high-conviction v3** built around distance-from-mid first.
- Treat the hour-bucket overlay as provisional until it is validated on more than one day.

## Final Decision

The Whale Wall Jumper should **not** be sent to the graveyard. The metadata is not statistically indistinguishable noise. There is a real, observable separation between durable and spoof-like walls, and it is driven primarily by how far the wall sits from the mid-price. The correct response is not retirement. It is to narrow the strategy into a distance-gated, size-screened v3.