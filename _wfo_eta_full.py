import sqlite3, statistics
from datetime import datetime, timezone, timedelta

def trial_durations(db_path):
    try:
        con = sqlite3.connect(db_path)
        rows = con.execute(
            'SELECT state, COUNT(*) FROM trials GROUP BY state'
        ).fetchall()
        timing = con.execute(
            'SELECT datetime_start, datetime_complete FROM trials '
            "WHERE state='COMPLETE' AND datetime_start IS NOT NULL "
            'AND datetime_complete IS NOT NULL ORDER BY trial_id DESC LIMIT 20'
        ).fetchall()
        last3 = con.execute(
            'SELECT trial_id, datetime_complete FROM trials '
            "WHERE state='COMPLETE' ORDER BY trial_id DESC LIMIT 3"
        ).fetchall()
        con.close()
        durations = []
        for s, e in timing:
            try:
                ds = datetime.strptime(str(s)[:26], '%Y-%m-%d %H:%M:%S.%f')
                de = datetime.strptime(str(e)[:26], '%Y-%m-%d %H:%M:%S.%f')
                durations.append((de - ds).total_seconds())
            except Exception:
                pass
        return rows, durations, last3
    except Exception as ex:
        return None, None, str(ex)

print('=== PHASE 1 DB ===')
rows, durs, last3 = trial_durations('/home/botuser/polymarket-bot/logs/wfo_phase1.db')
if rows:
    for r in rows:
        print('  %s: %d' % (r[0], r[1]))
    if durs:
        print('  Avg s/trial (last 20): %.1f' % statistics.mean(durs))
        print('  Min/Max: %.1f / %.1f' % (min(durs), max(durs)))
    print('  Last 3: %s' % str(last3))
else:
    print('  Error:', last3)

print()
print('=== PHASE 2 DB ===')
rows, durs, last3 = trial_durations('/home/botuser/polymarket-bot/logs/wfo_phase2.db')
if rows:
    for r in rows:
        print('  %s: %d' % (r[0], r[1]))
    completed = sum(r[1] for r in rows if r[0] == 'COMPLETE')
    running = sum(r[1] for r in rows if r[0] == 'RUNNING')
    if durs:
        avg = statistics.mean(durs)
        remaining = max(0, 100 - completed)
        eta_s = (remaining / 2.0) * avg
        now = datetime.now(timezone.utc)
        eta_p2 = now + timedelta(seconds=eta_s)
        print('  Avg s/trial (last 20): %.1f' % avg)
        print('  Completed: %d/100, Running: %d, Remaining: %d' % (completed, running, remaining))
        print('  Phase2 ETA UTC: %s (in %dh %dm)' % (
            eta_p2.strftime('%Y-%m-%d %H:%M:%S UTC'),
            int(eta_s // 3600), int((eta_s % 3600) // 60)))
        # Phase3 estimate: 1 trial, 31 markets, 9 folds
        # Phase2 trial = 10 markets x 9 folds = 90 market-folds in avg seconds
        # Phase3 = 31 markets x 9 folds = 279 market-folds (no max_workers)
        # Assume same per-market-fold speed, sequential (1 worker default for n_trials=1)
        phase3_s = (279.0 / 90.0) * avg  # scale by market-folds, sequential
        eta_p3 = eta_p2 + timedelta(seconds=phase3_s)
        print('  Phase3 est (31 markets, 1 trial): %.0fs (%.1f min)' % (phase3_s, phase3_s/60))
        print('  FULL PIPELINE ETA UTC: %s (in %dh %dm from now)' % (
            eta_p3.strftime('%Y-%m-%d %H:%M:%S UTC'),
            int((eta_s + phase3_s) // 3600), int(((eta_s + phase3_s) % 3600) // 60)))
    print('  Last 3: %s' % str(last3))
else:
    print('  Error:', last3)

print()
print('=== PHASE 3 DB ===')
rows, durs, last3 = trial_durations('/home/botuser/polymarket-bot/logs/wfo_phase3.db')
if rows:
    for r in rows:
        print('  %s: %d' % (r[0], r[1]))
else:
    print('  Not yet created (Phase 3 has not started)')
