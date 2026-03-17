import optuna
storage = 'sqlite:////home/botuser/polymarket-bot/logs/wfo_phase2.db'
try:
    summaries = optuna.get_all_study_summaries(storage=storage)
    for s in summaries:
        study = optuna.load_study(study_name=s.study_name, storage=storage)
        try:
            print(f'Study: {s.study_name}')
            print(f'Trials: {len(study.trials)}')
            print(f'Best Value: {study.best_value}')
            print(f'Best Params: {study.best_trial.params}')
        except Exception as e:
            print(f'No best trial for {s.study_name}')
except Exception as e:
    print('Failed to read db:', e)
