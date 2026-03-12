import sys, json
from collections import Counter

cats = Counter()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        ev = d.get("event", "")
        if ev == "order_place_failed":
            err = d.get("error", "")
            if "lower than the minimum" in err:
                cats["MIN_SHARES (size < 5)"] += 1
            elif "min size: $1" in err:
                cats["MIN_DOLLAR (< $1)"] += 1
            elif "not enough balance" in err:
                cats["NO_BALANCE"] += 1
            elif "crosses book" in err:
                cats["CROSSES_BOOK (post-only)"] += 1
            else:
                cats["OTHER"] += 1
    except:
        pass

print("=== ORDER FAILURE BREAKDOWN ===")
for k, v in cats.most_common():
    print(f"  {v:>4}  {k}")
print(f"  {'─'*30}")
print(f"  {sum(cats.values()):>4}  TOTAL")
