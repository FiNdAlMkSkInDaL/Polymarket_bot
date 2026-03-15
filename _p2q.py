import sqlite3
c = sqlite3.connect("/home/botuser/polymarket-bot/logs/wfo_phase2.db")
rows = c.execute("SELECT state, COUNT(*) FROM trials GROUP BY state").fetchall()
for state, cnt in rows:
    print(f"{state}: {cnt}")
c.close()
