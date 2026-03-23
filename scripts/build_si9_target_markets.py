from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "si9_clusters_monday.json"
OUTPUT_PATH = ROOT / "data" / "si9_target_markets.json"
MAX_MARKETS = 25


def _cluster_condition_ids(cluster: dict) -> list[str]:
    condition_ids = cluster.get("condition_ids")
    if isinstance(condition_ids, list) and condition_ids:
        return [str(condition_id) for condition_id in condition_ids]

    markets = cluster.get("markets", [])
    ids: list[str] = []
    for market in markets:
        condition_id = market.get("condition_id")
        if condition_id is not None:
            ids.append(str(condition_id))
    return ids


def main() -> int:
    clusters = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(clusters, list):
        raise ValueError(f"Expected top-level list in {INPUT_PATH}")

    selected_condition_ids: list[str] = []
    included_clusters = 0

    for cluster in clusters:
        cluster_ids = _cluster_condition_ids(cluster)
        if not cluster_ids:
            continue

        next_total = len(selected_condition_ids) + len(cluster_ids)
        if next_total > MAX_MARKETS:
            break

        selected_condition_ids.extend(cluster_ids)
        included_clusters += 1

    OUTPUT_PATH.write_text(
        json.dumps(selected_condition_ids, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"included_clusters={included_clusters}")
    print(f"total_legs={len(selected_condition_ids)}")
    print(f"output_path={OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())