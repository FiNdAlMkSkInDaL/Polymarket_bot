tmux capture-pane -p -S -20000 -t polybot_wfo | grep -iE 'Trial.*finished|fold_start|fold|pipeline' | tail -n 20
sqlite3 ~/polymarket-bot/logs/wfo_phase2.db "SELECT count(*) FROM trials;"
sqlite3 ~/polymarket-bot/logs/wfo_phase2.db "SELECT study_name, count(*) FROM trials JOIN studies ON trials.study_id = studies.study_id GROUP BY study_name;"
