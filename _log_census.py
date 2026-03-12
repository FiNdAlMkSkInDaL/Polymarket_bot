import sys, json

counts = {}
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        ev = d.get("event", "UNKNOWN")
        counts[ev] = counts.get(ev, 0) + 1
    except:
        pass

for k, v in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"{v:>6}  {k}")
