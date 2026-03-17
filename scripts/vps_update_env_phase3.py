import re

path = '/home/botuser/polymarket-bot/.env'
with open(path, 'r') as f:
    content = f.read()

updates = {
    'ORACLE_MARKET_CONFIGS': '\'[{"market_id":"0xNBA_KNICKS_PACERS_ID","oracle_type":"sports","external_id":"NBA_IND_NYK_2026_03_17","target_outcome":"Knicks","market_type":"winner"}]\'',
    'ORACLE_CRITICAL_POLL_MS': '100'
}

for k, v in updates.items():
    if re.search(f'^{k}=.*', content, flags=re.MULTILINE):
        content = re.sub(f'^{k}=.*', f'{k}={v}', content, flags=re.MULTILINE)
    else:
        content += f'\n{k}={v}'

with open(path, 'w') as f:
    f.write(content.strip() + '\n')
