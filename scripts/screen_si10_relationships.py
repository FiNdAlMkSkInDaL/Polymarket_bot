#!/usr/bin/env python3
"""Screen Polymarket Gamma for SI-10 base/base/joint relationship candidates.

This script queries both Gamma ``/events`` and ``/markets`` to discover likely
joint-probability markets and their candidate base legs using lightweight text
and tag heuristics. The output is intended for offline review and optional
validation with ``scripts/validate_si10_relationships.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_S = 0.5
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
JOINT_CONNECTOR_PATTERNS = (
    " and ",
    " & ",
    " plus ",
    " along with ",
)
JOINT_HINT_TERMS = (
    "trifecta",
    "same game parlay",
    "combo",
    "double result",
    "both to happen",
    "parlay",
)
NON_JOINT_RANGE_PATTERNS = (
    r"\bbetween\b.+\band\b",
    r"\bfrom\b.+\bto\b",
    r"\bover/under\b",
    r"\bo/u\b",
    r"\bodd/even\b",
)
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


@dataclass(slots=True)
class GammaMarketRecord:
    market_id: str
    condition_id: str
    question: str
    event_id: str
    event_title: str
    tags: tuple[str, ...]
    category: str
    end_date: str
    yes_token_id: str
    no_token_id: str
    volume_24h: float
    liquidity: float
    outcomes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class RelationshipCandidate:
    relationship_id: str
    label: str
    heuristic_score: float
    heuristic_reasons: list[str]
    shared_tags: list[str]
    base_a_condition_id: str
    base_b_condition_id: str
    joint_condition_id: str
    base_a_question: str
    base_b_question: str
    joint_question: str
    base_a_event_title: str
    base_b_event_title: str
    joint_event_title: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen Polymarket Gamma for SI-10 base/base/joint candidates using "
            "text and tag heuristics."
        )
    )
    parser.add_argument("--top", type=int, default=50, help="Number of candidates to print/export.")
    parser.add_argument("--events-page-size", type=int, default=100)
    parser.add_argument("--events-max-pages", type=int, default=10)
    parser.add_argument("--markets-page-size", type=int, default=500)
    parser.add_argument("--markets-max-pages", type=int, default=10)
    parser.add_argument("--min-volume", type=float, default=0.0)
    parser.add_argument("--min-liquidity", type=float, default=0.0)
    parser.add_argument("--min-score", type=float, default=1.4)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--backoff-base-s", type=float, default=DEFAULT_BACKOFF_BASE_S)
    parser.add_argument("--export-json", type=Path, default=None, metavar="FILE")
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


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


def _condition_id(value: Any) -> str:
    condition_id = str(value or "").strip()
    return condition_id if condition_id.startswith("0x") else ""


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token and token not in STOPWORDS and len(token) > 1
    }


def _normalise_tags(value: Any) -> tuple[str, ...]:
    tags: list[str] = []
    for item in _parse_listish(value) if not isinstance(value, str) else [value]:
        if isinstance(item, dict):
            for key in ("label", "slug", "name", "id"):
                text = _clean_text(item.get(key))
                if text:
                    tags.append(text.lower())
        else:
            text = _clean_text(item)
            if text:
                tags.append(text.lower())
    if isinstance(value, str):
        for chunk in value.split(","):
            text = _clean_text(chunk)
            if text:
                tags.append(text.lower())
    return tuple(dict.fromkeys(tags))


def _parse_tokens(raw_market: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    native_tokens = raw_market.get("tokens")
    if isinstance(native_tokens, list) and native_tokens:
        tokens = [token for token in native_tokens if isinstance(token, dict)]
        yes = next((str(token.get("token_id") or "") for token in tokens if str(token.get("outcome") or "").upper() == "YES"), "")
        no = next((str(token.get("token_id") or "") for token in tokens if str(token.get("outcome") or "").upper() == "NO"), "")
        outcomes = tuple(str(token.get("outcome") or "").strip() for token in tokens if str(token.get("outcome") or "").strip())
        return yes, no, outcomes

    clob_ids = _parse_listish(raw_market.get("clobTokenIds"))
    outcomes = tuple(str(item).strip() for item in _parse_listish(raw_market.get("outcomes")) if str(item).strip())
    yes = str(clob_ids[0]).strip() if len(clob_ids) >= 1 else ""
    no = str(clob_ids[1]).strip() if len(clob_ids) >= 2 else ""
    return yes, no, outcomes


def _is_binary_market(raw_market: dict[str, Any]) -> bool:
    yes_token_id, no_token_id, outcomes = _parse_tokens(raw_market)
    return bool(yes_token_id and no_token_id and len(outcomes) == 2)


def _market_record(raw_market: dict[str, Any], event_index: dict[str, dict[str, Any]]) -> GammaMarketRecord | None:
    if not isinstance(raw_market, dict):
        return None
    if not raw_market.get("active", False) or raw_market.get("closed", True):
        return None
    if not raw_market.get("acceptingOrders", True) or not raw_market.get("enableOrderBook", True):
        return None
    if not _is_binary_market(raw_market):
        return None

    condition_id = _condition_id(raw_market.get("conditionId") or raw_market.get("condition_id"))
    if not condition_id:
        return None

    yes_token_id, no_token_id, outcomes = _parse_tokens(raw_market)
    event_id = _clean_text(raw_market.get("eventId") or raw_market.get("event_id"))
    event_payload = event_index.get(event_id, {})
    event_title = _clean_text(event_payload.get("title") or raw_market.get("eventTitle") or raw_market.get("question"))
    tags = tuple(dict.fromkeys((*_normalise_tags(event_payload.get("tags")), *_normalise_tags(raw_market.get("tags")))))
    category = _clean_text(event_payload.get("category") or raw_market.get("category") or raw_market.get("slug"))

    return GammaMarketRecord(
        market_id=_clean_text(raw_market.get("id")),
        condition_id=condition_id,
        question=_clean_text(raw_market.get("question"), "<untitled market>"),
        event_id=event_id,
        event_title=event_title,
        tags=tags,
        category=category.lower(),
        end_date=_clean_text(raw_market.get("endDate")),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        volume_24h=_safe_float(raw_market.get("volume24hr") or raw_market.get("volume24hrClob") or raw_market.get("volumeNum24hr")),
        liquidity=_safe_float(raw_market.get("liquidity") or raw_market.get("liquidityClob") or raw_market.get("liquidityNum")),
        outcomes=outcomes,
    )


def _is_joint_market_text(question: str, event_title: str = "") -> bool:
    text = f" {question.lower()} | {event_title.lower()} "
    if any(re.search(pattern, text) for pattern in NON_JOINT_RANGE_PATTERNS):
        return False
    return any(pattern in text for pattern in JOINT_CONNECTOR_PATTERNS) or any(term in text for term in JOINT_HINT_TERMS)


def _split_joint_fragments(question: str) -> list[str]:
    text = f" {question.strip()} "
    for pattern in JOINT_CONNECTOR_PATTERNS:
        if pattern in text.lower():
            parts = re.split(re.escape(pattern), text, flags=re.IGNORECASE)
            cleaned = [_clean_text(part) for part in parts if _clean_text(part)]
            if len(cleaned) >= 2:
                return cleaned[:2]
    return []


def _shared_tag_count(left: GammaMarketRecord, right: GammaMarketRecord) -> int:
    return len(set(left.tags) & set(right.tags))


def _date_proximity_bonus(left: GammaMarketRecord, right: GammaMarketRecord) -> float:
    if not left.end_date or not right.end_date:
        return 0.0
    try:
        left_dt = datetime.fromisoformat(left.end_date.replace("Z", "+00:00"))
        right_dt = datetime.fromisoformat(right.end_date.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    delta_days = abs((left_dt - right_dt).total_seconds()) / 86400.0
    if delta_days <= 1.0:
        return 0.35
    if delta_days <= 7.0:
        return 0.15
    return 0.0


def _text_match_score(fragment: str, market: GammaMarketRecord) -> float:
    fragment_tokens = _tokenize(fragment)
    market_tokens = _tokenize(f"{market.question} {market.event_title}")
    if not fragment_tokens or not market_tokens:
        return 0.0
    overlap = len(fragment_tokens & market_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(fragment_tokens))


def _pair_score(joint: GammaMarketRecord, base_a: GammaMarketRecord, base_b: GammaMarketRecord) -> tuple[float, list[str]]:
    reasons: list[str] = []
    fragments = _split_joint_fragments(joint.question)
    fragment_a_score = _text_match_score(fragments[0], base_a) if len(fragments) >= 2 else 0.0
    fragment_b_score = _text_match_score(fragments[1], base_b) if len(fragments) >= 2 else 0.0
    reverse_a_score = _text_match_score(fragments[0], base_b) if len(fragments) >= 2 else 0.0
    reverse_b_score = _text_match_score(fragments[1], base_a) if len(fragments) >= 2 else 0.0
    ordered_score = fragment_a_score + fragment_b_score
    reverse_score = reverse_a_score + reverse_b_score
    if reverse_score > ordered_score:
        base_a, base_b = base_b, base_a
        ordered_score = reverse_score

    score = ordered_score
    if ordered_score > 0:
        reasons.append("joint_text_split_match")

    shared_tags = len(set(joint.tags) & set(base_a.tags) & set(base_b.tags))
    if shared_tags > 0:
        score += 0.30 * shared_tags
        reasons.append("shared_tags")

    if base_a.event_id and base_a.event_id == base_b.event_id:
        score += 0.25
        reasons.append("same_event_pair")

    if joint.category and joint.category == base_a.category == base_b.category:
        score += 0.15
        reasons.append("same_category")

    score += _date_proximity_bonus(joint, base_a)
    score += _date_proximity_bonus(joint, base_b)
    if _date_proximity_bonus(joint, base_a) > 0 or _date_proximity_bonus(joint, base_b) > 0:
        reasons.append("near_resolution_dates")

    score += min(base_a.volume_24h, 10000.0) / 100000.0
    score += min(base_b.volume_24h, 10000.0) / 100000.0
    score += min(joint.volume_24h, 10000.0) / 100000.0

    shared_joint_tokens = _tokenize(joint.question) & (_tokenize(base_a.question) | _tokenize(base_b.question))
    if len(shared_joint_tokens) >= 2:
        score += 0.20
        reasons.append("shared_joint_tokens")

    return score, reasons


def build_relationship_candidates(
    markets: list[GammaMarketRecord],
    *,
    min_score: float,
    min_volume: float,
    min_liquidity: float,
) -> list[RelationshipCandidate]:
    eligible = [
        market
        for market in markets
        if market.volume_24h >= min_volume and market.liquidity >= min_liquidity
    ]
    joint_markets = [market for market in eligible if _is_joint_market_text(market.question, market.event_title)]
    base_markets = [market for market in eligible if not _is_joint_market_text(market.question, market.event_title)]

    candidates: list[RelationshipCandidate] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for joint in joint_markets:
        related_bases = [
            market
            for market in base_markets
            if market.condition_id != joint.condition_id
            and (_shared_tag_count(joint, market) > 0 or _text_match_score(joint.question, market) > 0)
        ]
        for index, base_a in enumerate(related_bases):
            for base_b in related_bases[index + 1 :]:
                if base_a.condition_id == base_b.condition_id:
                    continue
                score, reasons = _pair_score(joint, base_a, base_b)
                if score < min_score:
                    continue
                ordered_bases = sorted((base_a, base_b), key=lambda market: market.condition_id)
                key = (ordered_bases[0].condition_id, ordered_bases[1].condition_id, joint.condition_id)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                shared_tags = sorted(set(joint.tags) & set(ordered_bases[0].tags) & set(ordered_bases[1].tags))
                candidates.append(
                    RelationshipCandidate(
                        relationship_id=_relationship_id(ordered_bases[0], ordered_bases[1], joint),
                        label=f"{ordered_bases[0].question} AND {ordered_bases[1].question}",
                        heuristic_score=round(score, 4),
                        heuristic_reasons=sorted(set(reasons)),
                        shared_tags=shared_tags,
                        base_a_condition_id=ordered_bases[0].condition_id,
                        base_b_condition_id=ordered_bases[1].condition_id,
                        joint_condition_id=joint.condition_id,
                        base_a_question=ordered_bases[0].question,
                        base_b_question=ordered_bases[1].question,
                        joint_question=joint.question,
                        base_a_event_title=ordered_bases[0].event_title,
                        base_b_event_title=ordered_bases[1].event_title,
                        joint_event_title=joint.event_title,
                    )
                )

    return sorted(candidates, key=lambda candidate: candidate.heuristic_score, reverse=True)


def _relationship_id(base_a: GammaMarketRecord, base_b: GammaMarketRecord, joint: GammaMarketRecord) -> str:
    left = re.sub(r"[^a-z0-9]+", "-", base_a.question.lower()).strip("-")[:24]
    right = re.sub(r"[^a-z0-9]+", "-", base_b.question.lower()).strip("-")[:24]
    joint_slug = re.sub(r"[^a-z0-9]+", "-", joint.question.lower()).strip("-")[:24]
    return f"si10-{left}-{right}-{joint_slug}"[:96]


def fetch_event_index(
    *,
    page_size: int,
    max_pages: int,
    timeout: float,
    max_retries: int,
    backoff_base_s: float,
) -> dict[str, dict[str, Any]]:
    event_index: dict[str, dict[str, Any]] = {}
    client_timeout = httpx.Timeout(timeout, connect=min(timeout, 10.0))
    with httpx.Client(timeout=client_timeout, headers={"User-Agent": "polymarket-si10-screener/1.0"}) as client:
        for page_index in range(max_pages):
            response = _get_with_retries(
                client,
                GAMMA_EVENTS_URL,
                {
                    "active": "true",
                    "closed": "false",
                    "limit": page_size,
                    "offset": page_index * page_size,
                },
                attempts=max_retries,
                backoff_base_s=backoff_base_s,
            )
            payload = response.json()
            items = payload if isinstance(payload, list) else payload.get("data", [])
            if not isinstance(items, list) or not items:
                break
            for item in items:
                if not isinstance(item, dict):
                    continue
                event_id = _clean_text(item.get("id"))
                if not event_id:
                    continue
                event_index[event_id] = {
                    "title": _clean_text(item.get("title") or item.get("question")),
                    "tags": item.get("tags") or [],
                    "category": item.get("category") or item.get("seriesSlug") or "",
                }
            if len(items) < page_size:
                break
    return event_index


def fetch_markets(
    *,
    event_index: dict[str, dict[str, Any]],
    page_size: int,
    max_pages: int,
    timeout: float,
    max_retries: int,
    backoff_base_s: float,
) -> list[GammaMarketRecord]:
    markets: list[GammaMarketRecord] = []
    client_timeout = httpx.Timeout(timeout, connect=min(timeout, 10.0))
    with httpx.Client(timeout=client_timeout, headers={"User-Agent": "polymarket-si10-screener/1.0"}) as client:
        offset = 0
        for _page in range(max_pages):
            response = _get_with_retries(
                client,
                GAMMA_MARKETS_URL,
                {
                    "active": "true",
                    "closed": "false",
                    "limit": page_size,
                    "offset": offset,
                },
                attempts=max_retries,
                backoff_base_s=backoff_base_s,
            )
            payload = response.json()
            if not isinstance(payload, list) or not payload:
                break
            for item in payload:
                record = _market_record(item, event_index)
                if record is not None:
                    markets.append(record)
            if len(payload) < page_size:
                break
            offset += page_size
    return markets


def _get_with_retries(
    client: httpx.Client,
    url: str,
    params: dict[str, Any],
    attempts: int = DEFAULT_MAX_RETRIES,
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url, params=params)
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise httpx.HTTPStatusError(
                    f"retryable Gamma status {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response
        except (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.HTTPStatusError,
        ) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(backoff_base_s * (2 ** (attempt - 1)))
    if last_error is None:
        raise RuntimeError("Gamma request failed without an exception")
    raise RuntimeError(f"Gamma request failed after {attempts} attempts: {last_error}") from last_error


def _payload(candidates: list[RelationshipCandidate], *, top: int) -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in candidates[:top]]


def print_summary(candidates: list[RelationshipCandidate], *, top: int) -> None:
    selected = candidates[:top]
    if not selected:
        print("No SI-10 relationship candidates matched the requested filters.")
        return
    print("Top SI-10 relationship candidates")
    print("Score combines joint-text split matches, shared tags, category alignment, and date proximity.")
    for index, candidate in enumerate(selected, start=1):
        print(f"{index:>2}. score={candidate.heuristic_score:.4f}  joint={candidate.joint_condition_id}")
        print(f"    joint: {candidate.joint_question}")
        print(f"    baseA: {candidate.base_a_condition_id}  {candidate.base_a_question}")
        print(f"    baseB: {candidate.base_b_condition_id}  {candidate.base_b_question}")
        print(f"    reasons: {', '.join(candidate.heuristic_reasons) or 'none'}")
        if candidate.shared_tags:
            print(f"    shared tags: {', '.join(candidate.shared_tags)}")


def main() -> int:
    args = parse_args()
    event_index = fetch_event_index(
        page_size=args.events_page_size,
        max_pages=args.events_max_pages,
        timeout=args.timeout,
        max_retries=args.max_retries,
        backoff_base_s=args.backoff_base_s,
    )
    markets = fetch_markets(
        event_index=event_index,
        page_size=args.markets_page_size,
        max_pages=args.markets_max_pages,
        timeout=args.timeout,
        max_retries=args.max_retries,
        backoff_base_s=args.backoff_base_s,
    )
    candidates = build_relationship_candidates(
        markets,
        min_score=args.min_score,
        min_volume=args.min_volume,
        min_liquidity=args.min_liquidity,
    )
    print_summary(candidates, top=args.top)

    if args.export_json is not None:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(json.dumps(_payload(candidates, top=args.top), indent=2), encoding="utf-8")
        print(f"\nExported {min(len(candidates), args.top)} SI-10 candidates to {args.export_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())