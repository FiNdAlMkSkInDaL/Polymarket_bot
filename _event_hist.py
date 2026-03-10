import json, collections, sys
c = collections.Counter()
with open("/home/botuser/polymarket-bot/logs/bot.jsonl") as f:
    for line in f:
        try:
            d = json.loads(line)
            c[d.get("event","?")] += 1
        except:
            pass
for ev, cnt in c.most_common(60):
    print(f"{cnt:>6}  {ev}")
