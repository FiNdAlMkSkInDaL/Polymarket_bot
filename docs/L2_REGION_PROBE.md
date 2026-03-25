# L2 Region Probe

This workflow measures feed stability against the same WebSocket endpoint shape used by `src/data/l2_websocket.py`:

- URL: `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- Subscription payload: `{"type": "subscribe", "channel": "book", "assets_ids": [...]}`

It is intentionally isolated from the trading runtime so it can run on small regional instances with only Python and `websockets` installed.

## Metrics

Each regional summary records:

- `disconnect_count`: total connection drops or connection failures during the run.
- `silence_gap_count`: number of inter-frame gaps at or above the configured threshold, default `1500 ms`.
- `frame_gap_ms.stdev`: variability of local receive gaps between frames.
- `exchange_lag_ms.mean`, `exchange_lag_ms.stdev`, `exchange_lag_ms.p95`: local arrival time minus exchange event timestamp when the payload exposes `timestamp`.

Absolute exchange-lag levels assume reasonably synchronized clocks on every host. If a region reports negative lag medians, treat variance and disconnect behavior as more trustworthy than the mean.

For region selection, prioritize:

1. Lowest `disconnect_count`
2. Lowest `silence_gap_count`
3. Lowest `frame_gap_ms.stdev`
4. Lowest `exchange_lag_ms.stdev`
5. Lowest `exchange_lag_ms.p95`

## Recommended Regions To Test

- Virginia: AWS `us-east-1` or GCP `us-east4`
- Frankfurt: AWS `eu-central-1` or GCP `europe-west3`
- London: AWS `eu-west-2` or GCP `europe-west2`
- Tokyo: AWS `ap-northeast-1` or GCP `asia-northeast1`

## Suggested Asset Selection

Use the same liquid asset ids across all regions. Prefer a small fixed basket of 3-10 active assets so the comparison is driven by network stability rather than one inactive market.

## Local Validation

Example single-host run from the current machine:

```powershell
c:/Users/finph/OneDrive/Desktop/PMB/Polymarket_bot/.venv/Scripts/python.exe scripts/l2_region_probe.py --label local --asset-id 95344728193244020854716971584323868633691370479151373416918812532794690491367 --duration-s 30 --output artifacts/l2_probe/local.json
```

## Remote Multi-Region Run

1. Copy `scripts/l2_region_inventory.example.json` to a real inventory file and replace placeholder hosts.
2. Provide one or more asset ids directly or through a file.
3. Launch the remote orchestration script.

Example:

```powershell
c:/Users/finph/OneDrive/Desktop/PMB/Polymarket_bot/.venv/Scripts/python.exe scripts/run_l2_region_probe_remote.py --inventory scripts/l2_region_inventory.example.json --output-dir artifacts/l2_regions --asset-id 95344728193244020854716971584323868633691370479151373416918812532794690491367 --duration-s 600
```

## Build The Matrix

After the regional summaries are pulled back locally:

```powershell
c:/Users/finph/OneDrive/Desktop/PMB/Polymarket_bot/.venv/Scripts/python.exe scripts/l2_region_matrix.py artifacts/l2_regions/aws-us-east-1.json artifacts/l2_regions/aws-eu-central-1.json artifacts/l2_regions/gcp-europe-west2.json artifacts/l2_regions/gcp-asia-northeast1.json --output artifacts/l2_regions/matrix.json --markdown-output artifacts/l2_regions/matrix.md
```

The first row in the output ranking is the best region under the current measurement window.