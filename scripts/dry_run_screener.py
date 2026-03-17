"""
Dry run screener to verify the real distribution of zero-fee markets.
Makes a direct GET request over the internet to Polymarket's Gamma API.
"""

import asyncio
import httpx

def is_fee_bearing_market(market_data: dict) -> bool:
    """Return True if the market is fee-bearing."""
    taker_fee = market_data.get("takerFeeBps")
    if taker_fee is None:
        taker_fee = market_data.get("taker_fee_bps")
        
    if taker_fee is not None:
        try:
            return float(taker_fee) != 0.0
        except ValueError:
            return True
    return True

async def main():
    print("Fetching active markets from Polymarket API...")
    url = "https://gamma-api.polymarket.com/events"
    params = {"closed": "false", "active": "true", "limit": 1000}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        events = resp.json()
        
    markets = []
    if not isinstance(events, list):
        events = events.get("data", []) if isinstance(events, dict) else []
        
    for event in events:
        sub_markets = event.get("markets", [])
        if not isinstance(sub_markets, list):
            continue
        event_tags = event.get("tags") or event.get("category", "")
        for m in sub_markets:
            if event_tags and not m.get("tags") and not m.get("category"):
                m["category"] = event_tags
            markets.append(m)

    zero_fee_markets = []
    total_markets = len(markets)
    
    for m in markets:
        if not is_fee_bearing_market(m):
            zero_fee_markets.append(m)

    fee_bearing_count = total_markets - len(zero_fee_markets)
    print(f"\\n--- Metrics ---")
    print(f"Total active markets: {total_markets}")
    print(f"Zero-fee markets: {len(zero_fee_markets)}")
    print(f"Fee-bearing markets: {fee_bearing_count}")
    
    print("\\n--- First 10 Zero-Fee Markets ---")
    for i, m in enumerate(zero_fee_markets[:10], start=1):
        question = m.get("question", "Unknown")
        tags = m.get("tags", m.get("category", ""))
        print(f"{i}. {question}")
        print(f"   Tags: {tags}")

if __name__ == "__main__":
    asyncio.run(main())
