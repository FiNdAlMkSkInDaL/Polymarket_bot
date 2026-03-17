import optuna
import json

storage = 'sqlite:////home/botuser/polymarket-bot/logs/wfo_phase2.db'
try:
    summaries = optuna.get_all_study_summaries(storage=storage)
    res = {}
    for s in summaries:
        study = optuna.load_study(study_name=s.study_name, storage=storage)
        try:
            res[s.study_name] = {
                'best_value': study.best_value,
                'best_params': study.best_trial.params,
                'trials_completed': len([t for t in study.trials if t.state.name == 'COMPLETE']),
                'trials_total': len(study.trials)
            }
        except Exception as e:
            res[s.study_name] = {'error': str(e)}
    with open('/tmp/wfo_export.json', 'w') as f:
        json.dump(res, f, indent=2)
except Exception as e:
    with open('/tmp/wfo_export.json', 'w') as f:
        json.dump({'error': str(e)}, f)
