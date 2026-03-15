import sqlite3, statistics
from datetime import datetime, timezone, timedelta

con = sqlite3.connect('/home/botuser/polymarket-bot/logs/wfo_phase2.db')

rows = con.execute('SELECT state, COUNT(*) FROM trials GROUP BY state').fetchall()
print('=== TRIAL STATE COUNTS ===')
for r in rows:
    print('  %s: %d' % (r[0], r[1]))

scores = con.execute(
    'SELECT t.trial_id, tv.value, t.datetime_start, t.datetime_complete '
    'FROM trials t JOIN trial_values tv ON t.trial_id = tv.trial_id '
    "WHERE t.state = 'COMPLETE' ORDER BY t.trial_id DESC LIMIT 50"
).fetchall()

print()
print('=== LAST 20 COMPLETED TRIAL SCORES ===')
for s in scores[:20]:
    print('  trial %-6d  score=%-10.4f  start=%s  end=%s' % (
        s[0], s[1] if s[1] is not None else 0.0, str(s[2])[:19], str(s[3])[:19]))

vals = [s[1] for s in scores if s[1] is not None]
if vals:
    print()
    print('=== SCORE STATISTICS (last %d completed) ===' % len(vals))
    print('  Min:   %.4f' % min(vals))
    print('  Max:   %.4f' % max(vals))
    print('  Mean:  %.4f' % statistics.mean(vals))
    print('  Stdev: %.4f' % (statistics.stdev(vals) if len(vals) > 1 else 0.0))
    penalty = sum(1 for v in vals if v <= -9.0)
    zeros = sum(1 for v in vals if v == 0.0)
    print('  Penalty (<=-9.0): %d/%d (%.1f%%)' % (penalty, len(vals), 100 * penalty / len(vals)))
    print('  Zero (==0.0):     %d/%d (%.1f%%)' % (zeros, len(vals), 100 * zeros / len(vals)))

timing = con.execute(
    'SELECT datetime_start, datetime_complete FROM trials '
    "WHERE state='COMPLETE' AND datetime_start IS NOT NULL "
    'AND datetime_complete IS NOT NULL ORDER BY trial_id DESC LIMIT 30'
).fetchall()

durations = []
for s, e in timing:
    try:
        ds = datetime.strptime(str(s)[:26], '%Y-%m-%d %H:%M:%S.%f')
        de = datetime.strptime(str(e)[:26], '%Y-%m-%d %H:%M:%S.%f')
        durations.append((de - ds).total_seconds())
    except Exception:
        pass

completed = sum(r[1] for r in rows if r[0] == 'COMPLETE')
running = sum(r[1] for r in rows if r[0] == 'RUNNING')

if durations:
    avg_s = statistics.mean(durations)
    remaining = max(0, 100 - completed)
    eta_s = (remaining / 2.0) * avg_s
    now = datetime.now(timezone.utc)
    eta = now + timedelta(seconds=eta_s)
    print()
    print('=== TIMING & ETA ===')
    print('  Avg s/trial:  %.1f s' % avg_s)
    print('  Completed:    %d/100' % completed)
    print('  Running now:  %d' % running)
    print('  Remaining:    %d' % remaining)
    print('  Now UTC:      %s' % now.strftime('%Y-%m-%d %H:%M:%S UTC'))
    print('  ETA UTC:      %s' % eta.strftime('%Y-%m-%d %H:%M:%S UTC'))
    print('  ETA in:       %dh %dm' % (int(eta_s // 3600), int((eta_s % 3600) // 60)))

con.close()
