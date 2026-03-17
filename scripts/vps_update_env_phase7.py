import re

path = '/home/botuser/polymarket-bot/.env'
with open(path, 'r') as f:
    content = f.read()

updates = {
    'ORACLE_MARKET_CONFIGS': '\'[{"market_id":"0x62950ac7636e2f11ed8bc0eb8c00aa16bdc1884dd0ffccc1eaee1df682e0714b","oracle_type":"sports","external_id":"NBA_IND_NYK_2026_03_17","target_outcome":"Knicks","market_type":"winner"}]\'',
    'ORACLE_SPORTS_API_URL': '"https://api.the-odds-api.com/v4"'
}

for k, v in updates.items():
    if re.search(f'^{k}=.*', content, flags=re.MULTILINE):
        content = re.sub(f'^{k}=.*', f'{k}={v}', content, flags=re.MULTILINE)
    else:
        content += f'\n{k}={v}'

with open(path, 'w') as f:
    f.write(content.strip() + '\n')
