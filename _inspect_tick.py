import json, os
date_dir = 'data/vps_march2026/ticks/2026-01-15'
for fname in sorted(os.listdir(date_dir))[:3]:
    fpath = date_dir + '/' + fname
    print(f"\n=== {fname[:40]} ===")
    sources_seen = set()
    with open(fpath) as f:
        for line in f:
            rec = json.loads(line)
            src = rec.get('source', '')
            if src not in sources_seen:
                sources_seen.add(src)
                print(f"source={src}")
                print(f"  top-level asset_id: {rec.get('asset_id', 'NONE')!r}")
                p = rec.get('payload', {})
                paid = p.get('asset_id', 'NONE')
                print(f"  payload.asset_id:   {paid!r}")
                print(f"  payload keys: {list(p.keys())[:8]}")
            if len(sources_seen) >= 3:
                break
