import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
try:
    study = optuna.load_study('polymarket_wfo_fold_0', storage='sqlite:///wfo_optuna.db')
    print('n_trials:', len(study.trials))
    print('best_value:', study.best_value)
except Exception as e:
    print('Error:', e)
