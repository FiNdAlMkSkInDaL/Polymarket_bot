"""Quick probe to see raw Polymarket WS message format."""
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
        for i in range(15):
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            data = json.loads(raw)
            if isinstance(data, list):
                for d in data:
                    et = d.get("event_type", "")
                    aid = d.get("asset_id", "")[:20]
                    print(f"MSG {i}: event_type={et}, asset_id={aid}..., keys={list(d.keys())[:10]}")
            else:
                et = data.get("event_type", "")
                aid = data.get("asset_id", "")[:20]
                print(f"MSG {i}: event_type={et}, asset_id={aid}..., keys={list(data.keys())[:10]}")

asyncio.run(probe())
