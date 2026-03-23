from pathlib import Path

env_path = Path('/home/botuser/polymarket-bot/.env')
text = env_path.read_text(encoding='utf-8')

updates = {
    'DEPLOYMENT_ENV': 'PAPER',
    'ORACLE_ARB_ENABLED': 'True',
    'ORACLE_SHADOW_MODE': 'False',
    'ORACLE_MARKET_CONFIGS': '[{"market_id":"0xbb57ccf5853a85487bc3d83d04d669310d28c6c810758953b9d9b91d1aee89d2","oracle_type":"crypto","external_id":"btcusdt","target_outcome":"YES","market_type":"threshold","goal_line":80000.0}]',
}

lines = text.splitlines()
seen = set()
out = []
for line in lines:
    if '=' not in line:
        out.append(line)
        continue
    key = line.split('=', 1)[0]
    if key in updates:
        out.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        out.append(line)

for key, value in updates.items():
    if key not in seen:
        out.append(f"{key}={value}")

env_path.write_text('\n'.join(out) + '\n', encoding='utf-8')

for key in ('DEPLOYMENT_ENV', 'ORACLE_ARB_ENABLED', 'ORACLE_SHADOW_MODE', 'ORACLE_MARKET_CONFIGS'):
    print(f"{key}={updates[key]}")
