# VacuumMaker Batch Backtest Scorecard (vacuum-batch-2026-03-25-top50)

## Setup

- Strategy: `src.signals.vacuum_maker.VacuumMaker`
- Replay directory: `logs\local_snapshot\l2_data\data\raw_ticks\2026-03-25`
- Aggregated SQLite: `logs\batch_vacuum_backtest.db`
- Files processed: `50`
- Forward markout horizon: `60s`
- OBI crash threshold: `0.95`

## Batch Totals

| Metric | Value |
| --- | ---: |
| Total PnL (USD) | 0.00 |
| Total fills | 0 |
| Covered fills | 0 |
| Average market win rate | 0.00% |
| Profitable markets | 0 |
| Total normalized events | 1,083,987 |
| Total trade events | 34,592 |
| Total dispatches | 0 |
| Total maker fills | 0 |

## SQL Rollup

```sql
SELECT SUM(total_pnl_cents) / 100.0 AS total_pnl_usd,
       SUM(fill_count) AS total_fills,
       AVG(win_rate) AS avg_market_win_rate
FROM batch_vacuum_market_summary
WHERE batch_label = 'vacuum-batch-2026-03-25-top50' AND horizon_seconds = 60;
```

## Top Markets

| File | Fills | Covered | Win Rate | Avg Net Edge (c/share) | Total PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0x0c4cd2055d6ea89354ffddc55d6dbcef9355748112ea952fc925f3db6a5c457f.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x0f49db97f71c68b1e42a6d16e3de93d85dbf7d4148e3f018eb79e88554be9f75.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x1945a8b23e313ed7423b6b6fd556f9ab5578900376b565a61dc480a5f4f35d21.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x2d3c4fc5cde6dfb43448402b912e41bd4453e3f030448ed026bff8f1a0bc072e.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x30cfb887558b20373a984da60c372fe5a90c0296aa6d8bb413a8aa7543846da2.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x30d55d8124ee1e12dabe89201badc45669b81dff69e4ce44d961f32878ec178a.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x375409bc5eeeff961e82b479caeccc20f33d15738e5bce1186d628aa3d9dfb1f.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x46dbd48d6bde5b81edb480e0f676a2cdda6c6b592c4d86a9367c7ad5a9870195.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x5a59d269c2b5108cd2f64c624e46ee2c8b5cfd88b882582565f927918315b6aa.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x6331a779482df72d904c3c1e12b6409ff836bc06f8c97945cba9b25ada2c605c.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |

## Bottom Markets

| File | Fills | Covered | Win Rate | Avg Net Edge (c/share) | Total PnL (USD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0x0c4cd2055d6ea89354ffddc55d6dbcef9355748112ea952fc925f3db6a5c457f.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x0f49db97f71c68b1e42a6d16e3de93d85dbf7d4148e3f018eb79e88554be9f75.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x1945a8b23e313ed7423b6b6fd556f9ab5578900376b565a61dc480a5f4f35d21.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x2d3c4fc5cde6dfb43448402b912e41bd4453e3f030448ed026bff8f1a0bc072e.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x30cfb887558b20373a984da60c372fe5a90c0296aa6d8bb413a8aa7543846da2.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x30d55d8124ee1e12dabe89201badc45669b81dff69e4ce44d961f32878ec178a.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x375409bc5eeeff961e82b479caeccc20f33d15738e5bce1186d628aa3d9dfb1f.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x46dbd48d6bde5b81edb480e0f676a2cdda6c6b592c4d86a9367c7ad5a9870195.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x5a59d269c2b5108cd2f64c624e46ee2c8b5cfd88b882582565f927918315b6aa.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |
| 0x6331a779482df72d904c3c1e12b6409ff836bc06f8c97945cba9b25ada2c605c.jsonl | 0 | 0 | 0.00% | 0.000 | 0.00 |

## Verdict

The 50-market batch does not yet prove horizontal scaling: the 60s fee-adjusted markout rollup produced `$0.00` across `50` markets with an average per-market win rate of `0.00%`.
