from __future__ import annotations

import json
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import requests
from dotenv import load_dotenv


POLYMARKET_EVENTS_URL = "https://gamma-api.polymarket.com/events"
ODDS_SCORES_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/scores/"
SIMILARITY_THRESHOLD = 0.80
REQUEST_TIMEOUT_SECONDS = 30
POLYMARKET_PAGE_SIZE = 100
POLYMARKET_MAX_PAGES = 10


@dataclass
class PolymarketMarket:
    condition_id: str
    title: str
    outcomes: list[str]


@dataclass
class OddsMatch:
    match_id: str
    home_team: str
    away_team: str


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _extract_text_values(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            if isinstance(v, str):
                out.append(v)
    elif isinstance(value, list):
        for item in value:
            out.extend(_extract_text_values(item))
    return out


def _parse_outcomes(raw_outcomes: Any) -> list[str]:
    if isinstance(raw_outcomes, list):
        return [str(x).strip() for x in raw_outcomes if str(x).strip()]

    if isinstance(raw_outcomes, str):
        text = raw_outcomes.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return [str(x).strip() for x in decoded if str(x).strip()]
        except json.JSONDecodeError:
            pass
        return [text]

    return []


def _is_sports_event(event: dict[str, Any]) -> bool:
    candidate_texts: list[str] = []
    candidate_texts.extend(_extract_text_values(event.get("tags")))
    candidate_texts.extend(_extract_text_values(event.get("category")))
    candidate_texts.extend(_extract_text_values(event.get("categories")))
    candidate_texts.extend(_extract_text_values(event.get("sport")))

    joined = " ".join(candidate_texts).lower()
    return ("sports" in joined) or ("basketball" in joined) or ("nba" in joined)


def fetch_polymarket_sports_markets() -> list[PolymarketMarket]:
    extracted: list[PolymarketMarket] = []
    for page in range(POLYMARKET_MAX_PAGES):
        response = requests.get(
            POLYMARKET_EVENTS_URL,
            params={
                "active": "true",
                "closed": "false",
                "limit": str(POLYMARKET_PAGE_SIZE),
                "offset": str(page * POLYMARKET_PAGE_SIZE),
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()

        events: list[dict[str, Any]]
        if isinstance(payload, list):
            events = payload
        elif isinstance(payload, dict):
            data = payload.get("data")
            events = data if isinstance(data, list) else []
        else:
            events = []

        if not events:
            break

        for event in events:
            if not isinstance(event, dict) or not _is_sports_event(event):
                continue

            event_title = str(event.get("title") or event.get("question") or "").strip()
            markets = _as_list(event.get("markets"))
            for market in markets:
                if not isinstance(market, dict):
                    continue

                condition_id = str(
                    market.get("conditionId")
                    or market.get("condition_id")
                    or event.get("conditionId")
                    or event.get("condition_id")
                    or ""
                ).strip()
                if not condition_id:
                    continue

                title = str(market.get("question") or market.get("title") or event_title).strip()
                outcomes = _parse_outcomes(market.get("outcomes"))
                if not title and not outcomes:
                    continue

                extracted.append(
                    PolymarketMarket(
                        condition_id=condition_id,
                        title=title,
                        outcomes=outcomes,
                    )
                )

        if len(events) < POLYMARKET_PAGE_SIZE:
            break

    return extracted


def fetch_nba_scores(api_key: str) -> list[OddsMatch]:
    response = requests.get(
        ODDS_SCORES_URL,
        params={"apiKey": api_key, "daysFrom": "1"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"Odds API request failed: {exc}")
        return []
    payload = response.json()

    if not isinstance(payload, list):
        return []

    matches: list[OddsMatch] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        match_id = str(row.get("id") or "").strip()
        home_team = str(row.get("home_team") or "").strip()
        away_team = str(row.get("away_team") or "").strip()
        if not (match_id and home_team and away_team):
            continue
        matches.append(OddsMatch(match_id=match_id, home_team=home_team, away_team=away_team))

    return matches


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _best_team_match_for_market(
    market: PolymarketMarket,
    odds_matches: list[OddsMatch],
) -> tuple[OddsMatch, str, float] | None:
    text_pool = [market.title] + market.outcomes
    text_pool = [x for x in text_pool if x]
    if not text_pool:
        return None

    best_result: tuple[OddsMatch, str, float] | None = None

    for odds_match in odds_matches:
        team_candidates = [odds_match.home_team, odds_match.away_team]
        best_local_team = ""
        best_local_score = 0.0

        # Compare every market text fragment to each team name.
        for market_text in text_pool:
            for team in team_candidates:
                score = _sim(market_text, team)
                if score > best_local_score:
                    best_local_score = score
                    best_local_team = team

        if best_local_score >= SIMILARITY_THRESHOLD:
            if best_result is None or best_local_score > best_result[2]:
                best_result = (odds_match, best_local_team, best_local_score)

    return best_result


def build_si8_payload(
    polymarket_markets: list[PolymarketMarket],
    odds_matches: list[OddsMatch],
) -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for market in polymarket_markets:
        best = _best_team_match_for_market(market, odds_matches)
        if best is None:
            continue

        matched_game, matched_team, _ = best
        pair_key = (market.condition_id, matched_game.match_id)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        payloads.append(
            {
                "market_id": market.condition_id,
                "oracle_type": "sports",
                "external_id": matched_game.match_id,
                "target_outcome": matched_team,
                "market_type": "winner",
            }
        )

    return payloads


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ORACLE_SPORTS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing ORACLE_SPORTS_API_KEY in local environment.")

    polymarket_markets = fetch_polymarket_sports_markets()
    odds_matches = fetch_nba_scores(api_key)
    payloads = build_si8_payload(polymarket_markets, odds_matches)

    print(f"Polymarket sports markets scanned: {len(polymarket_markets)}")
    print(f"Odds API NBA matches scanned: {len(odds_matches)}")
    print(f"Successful SI-8 mappings: {len(payloads)}")
    print(json.dumps(payloads, indent=2))


if __name__ == "__main__":
    main()