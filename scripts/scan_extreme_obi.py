from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT_PATH = Path("logs/candidate_crash_markets.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan raw L2 JSONL files for extreme top-3 OBI book events.")
    parser.add_argument("--input-dir", required=True, help="Directory of raw JSONL files to scan.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Text file receiving one crash-candidate path per line.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Absolute OBI threshold used to flag candidate files.")
    return parser.parse_args()


def compute_top3_abs_obi_from_book(payload: dict) -> float | None:
    bids_raw = payload.get("bids") or []
    asks_raw = payload.get("asks") or []
    bids: list[tuple[float, float]] = []
    asks: list[tuple[float, float]] = []

    for level in bids_raw:
        try:
            price = float(level["price"])
            size = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if price <= 0.0 or size <= 0.0:
            continue
        bids.append((price, size))

    for level in asks_raw:
        try:
            price = float(level["price"])
            size = float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if price <= 0.0 or size <= 0.0:
            continue
        asks.append((price, size))

    if not bids or not asks:
        return None

    bids.sort(key=lambda level: level[0], reverse=True)
    asks.sort(key=lambda level: level[0])
    top_bid_size = sum(size for _, size in bids[:3])
    top_ask_size = sum(size for _, size in asks[:3])
    total_size = top_bid_size + top_ask_size
    if total_size <= 0.0:
        return None
    return abs((top_bid_size - top_ask_size) / total_size)


def scan_file_for_max_abs_obi(file_path: Path) -> float:
    max_abs_obi = 0.0
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            payload = raw.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            if str(payload.get("event_type") or "").strip().lower() != "book":
                continue
            abs_obi = compute_top3_abs_obi_from_book(payload)
            if abs_obi is None:
                continue
            if abs_obi > max_abs_obi:
                max_abs_obi = abs_obi
    return max_abs_obi


def scan_directory(input_dir: Path, *, threshold: float) -> list[tuple[Path, float]]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    candidates: list[tuple[Path, float]] = []
    for file_path in sorted(path for path in input_dir.glob("*.jsonl") if path.is_file()):
        max_abs_obi = scan_file_for_max_abs_obi(file_path)
        if max_abs_obi > threshold:
            candidates.append((file_path.resolve(), max_abs_obi))
    return candidates


def write_candidate_file(output_path: Path, candidates: list[tuple[Path, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(str(path) for path, _ in candidates) + ("\n" if candidates else ""),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    threshold = float(args.threshold)
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    candidates = scan_directory(input_dir, threshold=threshold)
    write_candidate_file(output_path, candidates)
    print(
        "scanned_files={scanned} candidate_files={candidates_count} threshold={threshold:.4f} output={output}".format(
            scanned=len(list(path for path in input_dir.glob("*.jsonl") if path.is_file())),
            candidates_count=len(candidates),
            threshold=threshold,
            output=output_path,
        )
    )


if __name__ == "__main__":
    main()