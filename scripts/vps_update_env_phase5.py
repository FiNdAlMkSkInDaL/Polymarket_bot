import re

path = '/home/botuser/polymarket-bot/.env'
with open(path, 'r') as f:
    content = f.read()

updates = {
    'ORACLE_SPORTS_API_URL': '"https://api.the-odds-api.com/v4"'
}

for k, v in updates.items():
    if re.search(f'^{k}=.*', content, flags=re.MULTILINE):
        content = re.sub(f'^{k}=.*', f'{k}={v}', content, flags=re.MULTILINE)
    else:
        content += f'\n{k}={v}'

with open(path, 'w') as f:
    f.write(content.strip() + '\n')
