import sys, json

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        ev = d.get("event", "")
        if ev == "periodic_stats":
            print(json.dumps(d))
        elif ev == "order_place_failed":
            err = d.get("error", "")
            ts = d.get("timestamp", "")
            # categorise
            if "lower than the minimum" in err:
                cat = "MIN_SHARES"
            elif "min size: $1" in err:
                cat = "MIN_DOLLAR"
            elif "not enough balance" in err:
                cat = "NO_BALANCE"
            elif "crosses book" in err:
                cat = "CROSSES_BOOK"
            else:
                cat = "OTHER"
            print(f"[{ts}] {cat}: {err[:120]}")
    except:
        pass
