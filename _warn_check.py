import re, collections
with open('/home/botuser/polymarket-bot/logs/bot_fresh.log') as f:
    lines = [l for l in f if 'warning' in l and 'l2_crossed' not in l and 'l2_desync' not in l and 'ws_disconnect' not in l]
evts = []
for l in lines:
    m = re.search(r'"event": "(\w+)"', l)
    if m:
        evts.append(m.group(1))
c = collections.Counter(evts)
for k, v in c.most_common():
    print(f'{v:>5} {k}')
