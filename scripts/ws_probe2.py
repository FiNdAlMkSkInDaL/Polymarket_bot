"""Probe price_change message structure."""
import asyncio
import json
import websockets

async def probe():
    url = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    async with websockets.connect(url, ping_interval=20) as ws:
        msg = {
            "type": "subscribe",
            "channel": "market",
            "assets_ids": [
                "95344728193244020854716971584323868633691370479151373416918812532794690491367"
            ],
        }
        await ws.send(json.dumps(msg))
        count = 0
        while count < 3:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            data = json.loads(raw)
            if isinstance(data, dict) and data.get("event_type") == "price_change":
                print(json.dumps(data, indent=2)[:2000])
                print("---")
                count += 1

asyncio.run(probe())
