import json
import re
import urllib.request

BASE_URL = "https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=1000&offset="


def fetch(offset: int):
    req = urllib.request.Request(BASE_URL + str(offset), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


all_markets = []
for offset in range(0, 8000, 1000):
    page = fetch(offset)
    if not page:
        break
    all_markets.extend(page)
    if len(page) < 1000:
        break

print("TOTAL", len(all_markets))

hits = []
for m in all_markets:
    blob = json.dumps(m).lower()
    if "bitcoin" in blob or '"btc' in blob or " btc" in blob:
        cid = m.get("conditionId") or m.get("condition_id") or ""
        q = str(m.get("question") or m.get("title") or "")
        nums = []
        for match in re.finditer(r"\$?([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,7})([kKmM]?)", blob):
            n = float(match.group(1).replace(",", ""))
            s = match.group(2).lower()
            if s == "k":
                n *= 1000
            elif s == "m":
                n *= 1000000
            if 70000 <= n <= 200000:
                nums.append(n)
        goal = max(nums) if nums else None
        hits.append((goal or 0, cid, q, m.get("active"), m.get("closed"), m.get("slug") or ""))

hits.sort(key=lambda x: (x[0], x[2]))
print("HITS", len(hits))
for row in hits[:120]:
    print(f"{row[0]}|{row[1]}|{row[3]}|{row[4]}|{row[5]}|{row[2]}")
