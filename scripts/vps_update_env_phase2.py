import re

path = '/home/botuser/polymarket-bot/.env'
with open(path, 'r') as f:
    content = f.read()

updates = {
    'ORACLE_SPORTS_API_URL': '"https://api.betspredict.io/v1"',
    'ORACLE_MARKET_CONFIGS': '\'[{"market_id":"0xREAL_ID_HERE","oracle_type":"sports","external_id":"12345","target_outcome":"Lakers","market_type":"winner"}]\''
}

for k, v in updates.items():
    if re.search(f'^{k}=.*', content, flags=re.MULTILINE):
        content = re.sub(f'^{k}=.*', f'{k}={v}', content, flags=re.MULTILINE)
    else:
        content += f'\n{k}={v}'

with open(path, 'w') as f:
    f.write(content.strip() + '\n')
