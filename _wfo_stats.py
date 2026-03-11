import sqlite3
c = sqlite3.connect("/home/botuser/polymarket-bot/logs/wfo_phase1.db").cursor()
c.execute("SELECT COUNT(*) FROM trial_values WHERE value=-10.0")
neg = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM trial_values WHERE value>0")
pos = c.fetchone()[0]
c.execute("SELECT AVG(value) FROM trial_values WHERE value>0")
avg = c.fetchone()[0]
c.execute("SELECT MAX(value) FROM trial_values")
mx = c.fetchone()[0]
print(f"Rejected(-10): {neg}, Positive: {pos}, AvgPositive: {avg:.3f}, Best: {mx:.3f}")
