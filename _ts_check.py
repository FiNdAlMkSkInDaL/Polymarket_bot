import json, pathlib, datetime

ticks_dir = pathlib.Path('data/vps_march2026/ticks')
for date_dir in sorted(ticks_dir.iterdir())[:5]:
    for f in sorted(date_dir.glob('*.jsonl'))[:1]:
        lines = list(open(f))
        if not lines:
            continue
        r0 = json.loads(lines[0])
        rl = json.loads(lines[-1])
        t0 = float(r0.get('local_ts', 0))
        tl = float(rl.get('local_ts', 0))
        pt0 = int(r0.get('payload', {}).get('timestamp', t0 * 1000)) / 1000
        ptl = int(rl.get('payload', {}).get('timestamp', tl * 1000)) / 1000
        print(f"{date_dir.name} | n={len(lines)} | local_span={tl-t0:.0f}s | payload_span={ptl-pt0:.0f}s")
        print(f"  local_ts[0]  = {t0} => {datetime.datetime.utcfromtimestamp(t0)}")
        print(f"  payload_ts[0]= {pt0} => {datetime.datetime.utcfromtimestamp(pt0)}")
