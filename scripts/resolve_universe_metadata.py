from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_universal_backtest import iter_raw_file_records
from scripts.scan_long_tail_markets import MarketSummary, apply_goldilocks_filter, scan_markets


GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
USER_AGENT = "polymarket-universe-resolver/1.0"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "will",
    "with",
}

CRYPTO_KEYWORDS = {
    "airdrop",
    "altcoin",
    "bitcoin",
    "btc",
    "crypto",
    "defi",
    "eth",
    "ethereum",
    "launch",
    "memecoin",
    "sol",
    "solana",
    "token",
}

THEME_KEYWORDS = {
    "crypto": {
        *CRYPTO_KEYWORDS,
        "consensys",
        "kraken",
        "metamask",
        "microstrategy",
        "ostium",
        "pump",
        "theo",
        "usdai",
    },
    "geopolitics": {
        "annex",
        "capture",
        "china",
        "clash",
        "cuba",
        "hezbollah",
        "india",
        "iran",
        "israel",
        "kostyantynivka",
        "nuke",
        "president",
        "qassem",
        "rodynske",
        "russia",
        "secretary",
        "sumy",
        "territory",
        "west",
    },
    "sports": {
        "fifa",
        "gaming",
        "lpl",
        "ncaa",
        "qualify",
        "season",
        "tournament",
        "winner",
        "world",
    },
    "policy": {
        "canada",
        "nato",
        "province",
        "referendum",
        "withdraw",
    },
    "macro": {
        "economy",
        "gdp",
        "inflation",
        "rates",
        "recession",
        "unemployment",
    },
    "politics": {
        "bill",
        "candidate",
        "ciotti",
        "davidson",
        "democratic",
        "election",
        "enacts",
        "eric",
        "impeach",
        "impeached",
        "mayoral",
        "nice",
        "nominee",
        "ny",
        "safety",
        "starmer",
        "trump",
    },
}

DISPLAY_CATEGORIES = {"crypto", "geopolitics", "sports", "politics", "policy", "macro", "business", "technology"}
SUBTHEME_FALLBACK_STOPWORDS = {
    "airdrop",
    "before",
    "bill",
    "capture",
    "closing",
    "counties",
    "december",
    "dip",
    "general",
    "hit",
    "in",
    "ipo",
    "january",
    "june",
    "launch",
    "march",
    "market",
    "nominee",
    "october",
    "out",
    "perform",
    "president",
    "price",
    "qualify",
    "secretary",
    "sell",
    "sells",
    "september",
    "safety",
    "token",
    "tournament",
    "winner",
    "women",
    "world",
}


@dataclass(slots=True)
class RankedUniverseEntry:
    rank: int
    avg_daily_trade_count: float
    avg_daily_volume_usd: float
    avg_time_weighted_spread_cents: float
    active_days: int
    entry: UniverseEntry


@dataclass(slots=True)
class UniverseEntry:
    market_id: str
    clob_token_ids: list[str]
    question: str
    category: str
    end_date_iso: str
    event_id: str
    event_title: str
    slug: str
    condition_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve Gamma metadata for the top markets in a long-tail universe report.")
    parser.add_argument("--selection-mode", choices=("report", "balanced"), default="report")
    parser.add_argument("--report", default="docs/long_tail_relaxed_universe.md")
    parser.add_argument("--input-dir", default="logs/local_snapshot/l2_data")
    parser.add_argument("--json-output", default="config/hybrid_arb_universe.json")
    parser.add_argument("--markdown-output", default="docs/arb_universe_summary.md")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--target-count", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--max-market-pages", type=int, default=40)
    parser.add_argument("--max-event-pages", type=int, default=40)
    parser.add_argument("--max-daily-volume-usd", type=float, default=100_000.0)
    parser.add_argument("--min-daily-trade-count", type=float, default=10.0)
    parser.add_argument("--min-time-weighted-spread-cents", type=float, default=3.0)
    parser.add_argument("--require-spread-filter", action="store_true")
    parser.add_argument("--max-subtheme-markets", type=int, default=2)
    parser.add_argument("--max-category-markets", type=int, default=6)
    parser.add_argument("--start-date", default="2026-03-15")
    parser.add_argument("--end-date", default="2026-03-19")
    return parser.parse_args()


def _parse_listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _clean_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _normalise_category(event_payload: dict[str, Any], market_payload: dict[str, Any]) -> str:
    inferred = _infer_category(market_payload, event_payload)
    if inferred != "unknown":
        return inferred
    for source in (
        event_payload.get("category"),
        market_payload.get("category"),
        market_payload.get("seriesSlug"),
        event_payload.get("slug"),
    ):
        text = _clean_text(source).lower()
        if text in DISPLAY_CATEGORIES:
            return text
    return inferred


def _infer_category(market_payload: dict[str, Any], event_payload: dict[str, Any]) -> str:
    tokens = _tokenise(
        " ".join(
            filter(
                None,
                [
                    _clean_text(market_payload.get("question")),
                    _clean_text(market_payload.get("slug")),
                    _clean_text(event_payload.get("title")),
                    _clean_text(event_payload.get("slug")),
                ],
            )
        )
    )
    best_category = "unknown"
    best_score = 0
    for category, keywords in THEME_KEYWORDS.items():
        score = len(tokens & keywords)
        if score > best_score:
            best_category = category
            best_score = score
    if best_score > 0:
        return best_category
    return "unknown"


def _read_target_market_ids(report_path: Path, top: int) -> list[str]:
    pattern = re.compile(r"^\|\s*\d+\s*\|\s*(0x[a-f0-9]+)\s*\|", re.IGNORECASE)
    market_ids: list[str] = []
    for line in report_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        market_ids.append(match.group(1).lower())
        if len(market_ids) >= top:
            break
    if len(market_ids) < top:
        raise ValueError(f"Expected at least {top} market IDs in {report_path}, found {len(market_ids)}")
    return market_ids


def _rank_relaxed_candidates(
    input_dir: Path,
    *,
    start_date: str,
    end_date: str,
    max_daily_volume_usd: float,
    min_daily_trade_count: float,
    min_time_weighted_spread_cents: float,
    require_spread_filter: bool,
) -> list[MarketSummary]:
    summaries = scan_markets(input_dir, start_date=start_date, end_date=end_date)
    return apply_goldilocks_filter(
        summaries,
        max_daily_volume_usd=max_daily_volume_usd,
        min_daily_trade_count=min_daily_trade_count,
        min_time_weighted_spread_cents=min_time_weighted_spread_cents,
        require_spread_filter=require_spread_filter,
    )


def _fetch_paginated(
    client: httpx.Client,
    url: str,
    *,
    page_size: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    offset = 0
    for _ in range(max_pages):
        response = client.get(url, params={"limit": page_size, "offset": offset})
        response.raise_for_status()
        payload = response.json()
        page_items = payload if isinstance(payload, list) else payload.get("data", [])
        if not isinstance(page_items, list) or not page_items:
            break
        items.extend(item for item in page_items if isinstance(item, dict))
        if len(page_items) < page_size:
            break
        offset += page_size
    return items


def _fetch_markets_by_condition_ids(
    client: httpx.Client,
    condition_ids: list[str],
) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    seen_condition_ids: set[str] = set()
    for condition_id in condition_ids:
        response = client.get(
            GAMMA_MARKETS_URL,
            params={"condition_ids": condition_id, "limit": 10},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            resolved_condition_id = _market_condition_id(item)
            if not resolved_condition_id or resolved_condition_id in seen_condition_ids:
                continue
            seen_condition_ids.add(resolved_condition_id)
            markets.append(item)
    return markets


def _scan_archive_token_ids(input_dir: Path, target_ids: list[str]) -> dict[str, list[str]]:
    targets = set(target_ids)
    token_ids: dict[str, set[str]] = {market_id: set() for market_id in target_ids}
    for path in input_dir.rglob("*.jsonl"):
        for raw in iter_raw_file_records(path):
            payload = raw.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            market_id = _clean_text(payload.get("market")).lower()
            if market_id not in targets:
                continue
            for candidate in (
                payload.get("asset_id"),
                raw.get("asset_id"),
            ):
                asset_id = _clean_text(candidate)
                if asset_id and not asset_id.startswith("0x"):
                    token_ids[market_id].add(asset_id)
            for change in payload.get("price_changes") or []:
                if not isinstance(change, dict):
                    continue
                asset_id = _clean_text(change.get("asset_id"))
                if asset_id and not asset_id.startswith("0x"):
                    token_ids[market_id].add(asset_id)
        if all(len(token_ids[market_id]) >= 2 for market_id in target_ids):
            break
    return {market_id: sorted(values) for market_id, values in token_ids.items()}


def _event_index(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(event.get("id") or ""): event for event in events if str(event.get("id") or "")}


def _market_condition_id(payload: dict[str, Any]) -> str:
    return _clean_text(payload.get("conditionId") or payload.get("condition_id")).lower()


def _market_clob_token_ids(payload: dict[str, Any]) -> list[str]:
    return [str(token_id).strip() for token_id in _parse_listish(payload.get("clobTokenIds")) if str(token_id).strip()]


def _resolve_entries(
    target_ids: list[str],
    markets: list[dict[str, Any]],
    events: list[dict[str, Any]],
    archive_token_ids: dict[str, list[str]],
) -> tuple[list[UniverseEntry], list[str]]:
    targets = set(target_ids)
    event_lookup = _event_index(events)
    matched: dict[str, UniverseEntry] = {}
    token_to_market_id = {
        token_id: market_id
        for market_id, token_ids in archive_token_ids.items()
        for token_id in token_ids
    }

    for market in markets:
        condition_id = _market_condition_id(market)
        matched_market_id = condition_id if condition_id in targets else ""
        if not matched_market_id:
            for token_id in _market_clob_token_ids(market):
                matched_market_id = token_to_market_id.get(token_id, "")
                if matched_market_id:
                    break
        if not matched_market_id or matched_market_id in matched:
            continue
        event_id = _clean_text(market.get("eventId") or market.get("event_id"))
        market_events = market.get("events") or []
        if not event_id and isinstance(market_events, list) and market_events:
            first_event = market_events[0] if isinstance(market_events[0], dict) else {}
            event_id = _clean_text(first_event.get("id"))
        event_payload = event_lookup.get(event_id, {})
        if not event_payload and isinstance(market_events, list) and market_events:
            first_event = market_events[0]
            event_payload = first_event if isinstance(first_event, dict) else {}
        matched[matched_market_id] = UniverseEntry(
            market_id=matched_market_id,
            clob_token_ids=_market_clob_token_ids(market),
            question=_clean_text(market.get("question"), "<untitled market>"),
            category=_normalise_category(event_payload, market),
            end_date_iso=_clean_text(market.get("endDate") or market.get("end_date_iso") or market.get("endDateIso")),
            event_id=event_id,
            event_title=_clean_text(event_payload.get("title") or market.get("eventTitle") or market.get("question")),
            slug=_clean_text(market.get("slug")),
            condition_id=condition_id,
        )

    unresolved = [market_id for market_id in target_ids if market_id not in matched]
    ordered = [matched[market_id] for market_id in target_ids if market_id in matched]
    return ordered, unresolved


def _tokenise(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token and token not in STOPWORDS and len(token) > 2
    }


def _resolve_target_entries(
    client: httpx.Client,
    *,
    target_ids: list[str],
    input_dir: Path,
    page_size: int,
    max_market_pages: int,
    max_event_pages: int,
) -> tuple[list[UniverseEntry], list[str], dict[str, list[str]]]:
    markets = _fetch_markets_by_condition_ids(client, target_ids)
    events: list[dict[str, Any]] = []

    archive_token_ids = {
        condition_id: _market_clob_token_ids(market)
        for market in markets
        for condition_id in [_market_condition_id(market)]
        if condition_id
    }
    unresolved_exact = [market_id for market_id in target_ids if market_id not in archive_token_ids]
    if unresolved_exact:
        fallback_markets = _fetch_paginated(
            client,
            GAMMA_MARKETS_URL,
            page_size=page_size,
            max_pages=max_market_pages,
        )
        fallback_events = _fetch_paginated(
            client,
            GAMMA_EVENTS_URL,
            page_size=min(page_size, 100),
            max_pages=max_event_pages,
        )
        markets.extend(fallback_markets)
        events = fallback_events
        archive_token_ids.update(_scan_archive_token_ids(input_dir, unresolved_exact))

    entries, unresolved = _resolve_entries(target_ids, markets, events, archive_token_ids)
    return entries, unresolved, archive_token_ids


def _subtheme_key(entry: UniverseEntry) -> str:
    slug = _clean_text(entry.slug).lower()
    if slug:
        chain = re.sub(r"-(?:by|before|after)-.*$", "", slug)
        chain = re.sub(r"-(?:january|february|march|april|may|june|july|august|september|october|november|december)-.*$", "", chain)
        chain = re.sub(r"-20\d{2}.*$", "", chain)
        chain = chain.strip("-")
        if chain:
            return chain

    text = " ".join(filter(None, [entry.event_title, entry.question]))
    tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS and token not in SUBTHEME_FALLBACK_STOPWORDS]
    return "-".join(tokens[:4]) or entry.market_id


def _select_balanced_entries(
    ranked_entries: list[RankedUniverseEntry],
    *,
    target_count: int,
    max_subtheme_markets: int,
    max_category_markets: int,
) -> tuple[list[RankedUniverseEntry], list[dict[str, Any]], Counter[str], Counter[str]]:
    selected: list[RankedUniverseEntry] = []
    skipped: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    subtheme_counts: Counter[str] = Counter()

    for candidate in ranked_entries:
        category = candidate.entry.category or "unknown"
        subtheme = _subtheme_key(candidate.entry)
        reasons: list[str] = []
        if category_counts[category] >= max_category_markets:
            reasons.append(f"category_cap:{category}")
        if subtheme_counts[subtheme] >= max_subtheme_markets:
            reasons.append(f"subtheme_cap:{subtheme}")
        if reasons:
            skipped.append(
                {
                    "rank": candidate.rank,
                    "market_id": candidate.entry.market_id,
                    "category": category,
                    "subtheme": subtheme,
                    "reason": ",".join(reasons),
                }
            )
            continue

        selected.append(candidate)
        category_counts[category] += 1
        subtheme_counts[subtheme] += 1
        if len(selected) >= target_count:
            break

    return selected, skipped, category_counts, subtheme_counts


def _ranked_entry_payload(candidate: RankedUniverseEntry) -> dict[str, Any]:
    payload = asdict(candidate.entry)
    payload.update(
        {
            "source_rank": candidate.rank,
            "avg_daily_trade_count": candidate.avg_daily_trade_count,
            "avg_daily_volume_usd": candidate.avg_daily_volume_usd,
            "avg_time_weighted_spread_cents": candidate.avg_time_weighted_spread_cents,
            "active_days": candidate.active_days,
            "subtheme": _subtheme_key(candidate.entry),
        }
    )
    return payload


def _render_summary(entries: list[UniverseEntry], unresolved: list[str]) -> str:
    category_counts = Counter(entry.category.lower() for entry in entries)
    event_counts = Counter(entry.event_title for entry in entries if entry.event_title)
    token_counts = Counter(token for entry in entries for token in _tokenise(f"{entry.event_title} {entry.question}"))
    dominant_category, dominant_category_count = category_counts.most_common(1)[0] if category_counts else ("unknown", 0)
    repeat_events = [title for title, count in event_counts.items() if count > 1]
    repeat_tokens = [token for token, count in token_counts.most_common(8) if count >= 3]

    lines = [
        "# Arbitrage Universe Summary",
        "",
        "## Top 25 Resolved Markets",
        "",
        "| Rank | Market ID | Category | End Date | Question |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for index, entry in enumerate(entries, start=1):
        lines.append(
            "| {rank} | {market_id} | {category} | {end_date} | {question} |".format(
                rank=index,
                market_id=entry.market_id,
                category=entry.category,
                end_date=entry.end_date_iso or "unknown",
                question=entry.question.replace("|", "/"),
            )
        )

    lines.extend(["", "## Category Mix", ""])
    if category_counts:
        for category, count in category_counts.most_common():
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- No categories resolved")

    lines.extend(["", "## Clustering Risks", ""])
    if entries:
        lines.append(
            "- Dominant category: `{category}` with `{count}` of `{total}` markets.".format(
                category=dominant_category,
                count=dominant_category_count,
                total=len(entries),
            )
        )
        if repeat_events:
            lines.append("- Repeated event clusters detected: " + ", ".join(f"`{title}`" for title in repeat_events[:8]) + ".")
        else:
            lines.append("- No repeated event-title clusters were obvious in the top 25.")
        if repeat_tokens:
            lines.append("- Recurrent theme tokens: " + ", ".join(f"`{token}`" for token in repeat_tokens) + ".")
        else:
            lines.append("- No high-frequency theme tokens stood out across questions and event titles.")
        if dominant_category_count >= max(10, len(entries) // 2):
            lines.append("- Concentration risk is material: more than half of the universe sits in one category.")
        else:
            lines.append("- Category concentration is moderate rather than single-theme dominated.")
    else:
        lines.append("- No markets were resolved, so clustering risk could not be assessed.")

    if unresolved:
        lines.extend(["", "## Unresolved IDs", ""])
        for market_id in unresolved:
            lines.append(f"- {market_id}")

    return "\n".join(lines) + "\n"


def _render_balanced_summary(
    selected: list[RankedUniverseEntry],
    *,
    category_counts: Counter[str],
    subtheme_counts: Counter[str],
    skipped: list[dict[str, Any]],
    unresolved: list[str],
    target_count: int,
    max_subtheme_markets: int,
    max_category_markets: int,
) -> str:
    lines = [
        "# Balanced Arbitrage Universe Summary",
        "",
        "## Constraints",
        "",
        f"- Target markets: `{target_count}`",
        f"- Max per sub-theme: `{max_subtheme_markets}`",
        f"- Max per high-level category: `{max_category_markets}`",
        f"- Selected markets: `{len(selected)}`",
        "",
        "## Selected Markets",
        "",
        "| Slot | Source Rank | Market ID | Category | Sub-theme | Avg Spread (cents) | Question |",
        "| ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    for index, candidate in enumerate(selected, start=1):
        lines.append(
            "| {slot} | {rank} | {market_id} | {category} | {subtheme} | {spread:.3f} | {question} |".format(
                slot=index,
                rank=candidate.rank,
                market_id=candidate.entry.market_id,
                category=candidate.entry.category,
                subtheme=_subtheme_key(candidate.entry),
                spread=candidate.avg_time_weighted_spread_cents,
                question=candidate.entry.question.replace("|", "/"),
            )
        )

    lines.extend(["", "## Category Mix", ""])
    for category, count in category_counts.most_common():
        lines.append(f"- {category}: {count}")

    repeated_subthemes = [(subtheme, count) for subtheme, count in subtheme_counts.most_common() if count > 1]
    lines.extend(["", "## Sub-theme Mix", ""])
    if repeated_subthemes:
        for subtheme, count in repeated_subthemes:
            lines.append(f"- {subtheme}: {count}")
    else:
        lines.append("- No sub-theme was used more than once")

    if selected:
        lines.extend(["", "## Selection Depth", ""])
        lines.append(f"- Deepest selected source rank: `{max(candidate.rank for candidate in selected)}`")
        lines.append(f"- Backfill skips due to caps: `{len(skipped)}`")

    if unresolved:
        lines.extend(["", "## Unresolved IDs", ""])
        for market_id in unresolved:
            lines.append(f"- {market_id}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    input_dir = Path(args.input_dir)
    ranked_candidates: list[MarketSummary] = []
    target_ids: list[str]

    if args.selection_mode == "balanced":
        ranked_candidates = _rank_relaxed_candidates(
            input_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            max_daily_volume_usd=args.max_daily_volume_usd,
            min_daily_trade_count=args.min_daily_trade_count,
            min_time_weighted_spread_cents=args.min_time_weighted_spread_cents,
            require_spread_filter=args.require_spread_filter,
        )
        target_ids = [summary.market_id for summary in ranked_candidates]
    else:
        target_ids = _read_target_market_ids(report_path, args.top)

    if args.selection_mode == "balanced":
        selected: list[RankedUniverseEntry] = []
        skipped: list[dict[str, Any]] = []
        category_counts: Counter[str] = Counter()
        subtheme_counts: Counter[str] = Counter()
        unresolved: list[str] = []
        archive_token_ids: dict[str, list[str]] = {}
        resolved_candidate_count = 0
        chunk_size = 40

        timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 10.0))
        with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
            for chunk_start in range(0, len(ranked_candidates), chunk_size):
                chunk = ranked_candidates[chunk_start : chunk_start + chunk_size]
                chunk_ids = [summary.market_id for summary in chunk]
                chunk_entries, chunk_unresolved, chunk_archive_token_ids = _resolve_target_entries(
                    client,
                    target_ids=chunk_ids,
                    input_dir=input_dir,
                    page_size=args.page_size,
                    max_market_pages=args.max_market_pages,
                    max_event_pages=args.max_event_pages,
                )
                unresolved.extend(chunk_unresolved)
                archive_token_ids.update(chunk_archive_token_ids)
                chunk_lookup = {entry.market_id: entry for entry in chunk_entries}

                for rank, summary in enumerate(chunk, start=chunk_start + 1):
                    entry = chunk_lookup.get(summary.market_id)
                    if entry is None:
                        continue
                    resolved_candidate_count += 1
                    candidate = RankedUniverseEntry(
                        rank=rank,
                        avg_daily_trade_count=summary.avg_daily_trade_count,
                        avg_daily_volume_usd=summary.avg_daily_volume_usd,
                        avg_time_weighted_spread_cents=summary.avg_time_weighted_spread_cents,
                        active_days=summary.active_days,
                        entry=entry,
                    )
                    category = entry.category or "unknown"
                    subtheme = _subtheme_key(entry)
                    reasons: list[str] = []
                    if category_counts[category] >= args.max_category_markets:
                        reasons.append(f"category_cap:{category}")
                    if subtheme_counts[subtheme] >= args.max_subtheme_markets:
                        reasons.append(f"subtheme_cap:{subtheme}")
                    if reasons:
                        skipped.append(
                            {
                                "rank": candidate.rank,
                                "market_id": candidate.entry.market_id,
                                "category": category,
                                "subtheme": subtheme,
                                "reason": ",".join(reasons),
                            }
                        )
                        continue

                    selected.append(candidate)
                    category_counts[category] += 1
                    subtheme_counts[subtheme] += 1
                    if len(selected) >= args.target_count:
                        break

                if len(selected) >= args.target_count:
                    break

        json_payload = {
            "selection_mode": args.selection_mode,
            "source_archive": str(input_dir),
            "source_report": str(report_path),
            "scan_window": {"start_date": args.start_date, "end_date": args.end_date},
            "scan_filters": {
                "max_daily_volume_usd": args.max_daily_volume_usd,
                "min_daily_trade_count": args.min_daily_trade_count,
                "min_time_weighted_spread_cents": args.min_time_weighted_spread_cents,
                "require_spread_filter": args.require_spread_filter,
            },
            "constraints": {
                "target_count": args.target_count,
                "max_subtheme_markets": args.max_subtheme_markets,
                "max_category_markets": args.max_category_markets,
            },
            "candidate_count": len(ranked_candidates),
            "resolved_candidate_count": resolved_candidate_count,
            "selected_market_count": len(selected),
            "markets": [_ranked_entry_payload(candidate) for candidate in selected],
            "category_counts": dict(category_counts),
            "subtheme_counts": dict(subtheme_counts),
            "archive_token_ids": archive_token_ids,
            "skipped_candidates": skipped,
            "unresolved_market_ids": unresolved,
        }
        markdown = _render_balanced_summary(
            selected,
            category_counts=category_counts,
            subtheme_counts=subtheme_counts,
            skipped=skipped,
            unresolved=unresolved,
            target_count=args.target_count,
            max_subtheme_markets=args.max_subtheme_markets,
            max_category_markets=args.max_category_markets,
        )
    else:
        timeout = httpx.Timeout(args.timeout, connect=min(args.timeout, 10.0))
        with httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT}) as client:
            entries, unresolved, archive_token_ids = _resolve_target_entries(
                client,
                target_ids=target_ids,
                input_dir=input_dir,
                page_size=args.page_size,
                max_market_pages=args.max_market_pages,
                max_event_pages=args.max_event_pages,
            )
        json_payload = {
            "selection_mode": args.selection_mode,
            "source_report": str(report_path),
            "source_archive": str(input_dir),
            "market_count": len(entries),
            "markets": [asdict(entry) for entry in entries],
            "archive_token_ids": archive_token_ids,
            "unresolved_market_ids": unresolved,
        }
        markdown = _render_summary(entries, unresolved)

    json_output_path = Path(args.json_output)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    markdown_output_path = Path(args.markdown_output)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_output_path.write_text(markdown, encoding="utf-8")

    print(
        json.dumps(
            {
                "selection_mode": args.selection_mode,
                "resolved": json_payload.get("resolved_candidate_count", json_payload.get("market_count", 0)),
                "selected": json_payload.get("selected_market_count", json_payload.get("market_count", 0)),
                "unresolved": unresolved,
            },
            indent=2,
        )
    )
    print("\n---MARKDOWN---\n")
    print(markdown)


if __name__ == "__main__":
    main()