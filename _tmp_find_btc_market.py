import json
import re
import urllib.request

BASE_URL = "https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=1000&offset="


def fetch(offset: int):
    url = BASE_URL + str(offset)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


data = []
for offset in range(0, 8000, 1000):
    page = fetch(offset)
    if not page:
        break
    data.extend(page)
    if len(page) < 1000:
        break

rows = []
raw_rows = []
seen = set()
for m in data:
    market_id = m.get("id") or m.get("marketId") or m.get("slug")
    if market_id in seen:
        continue
    seen.add(market_id)

    question = str(m.get("question", ""))
    lower_q = question.lower()
    if "bitcoin" not in lower_q and " btc" not in lower_q and "btc " not in lower_q and not lower_q.startswith("btc"):
        continue

    condition_id = m.get("conditionId") or m.get("condition_id")
    if not condition_id:
        continue
    raw_rows.append((condition_id, question, m.get("active", True), m.get("closed", False)))

    nums = []
    for match in re.finditer(r"\$?([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,7})([kKmM]?)", question):
        raw_num = match.group(1).replace(",", "")
        suffix = match.group(2).lower()
        value = float(raw_num)
        if suffix == "k":
            value *= 1000
        elif suffix == "m":
            value *= 1000000
        nums.append(value)

    goal_line = max(nums) if nums else None
    if goal_line and 72000 < goal_line <= 200000 and m.get("active", True) and not m.get("closed", False):
        rows.append((goal_line, condition_id, question))

rows.sort(key=lambda r: r[0])
print(f"FOUND {len(rows)}")
for goal_line, condition_id, question in rows[:30]:
    print(f"{goal_line}|{condition_id}|{question}")

print("RAW_ROWS", len(raw_rows))
for condition_id, question, active, closed in raw_rows[:40]:
    print(f"RAW|{active}|{closed}|{condition_id}|{question}")
