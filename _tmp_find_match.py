import requests
u='https://api.the-odds-api.com/v4/sports/basketball_nba/scores/?apiKey=869484bcf56587d333eabda03e41dd2d&daysFrom=1'
r=requests.get(u,timeout=20); r.raise_for_status()
g=[x for x in r.json() if {'Indiana Pacers','New York Knicks'}=={x.get('home_team'),x.get('away_team')}]
print(g[0]['id'] if g else '')
