import re

path = '/home/botuser/polymarket-bot/.env'
with open(path, 'r') as f:
    content = f.read()

updates = {
    'ORACLE_ARB_ENABLED': 'True',
    'ORACLE_SHADOW_MODE': 'False',
    'ORACLE_MARKET_CONFIGS': '\'[{"market_id":"0xTestMarket123","oracle_type":"sports","external_id":"TEST_MATCH_01","target_outcome":"Team A","market_type":"winner"}]\''
}

for k, v in updates.items():
    if re.search(f'^{k}=.*', content, flags=re.MULTILINE):
        content = re.sub(f'^{k}=.*', f'{k}={v}', content, flags=re.MULTILINE)
    else:
        content += f'\n{k}={v}'

with open(path, 'w') as f:
    f.write(content.strip() + '\n')
