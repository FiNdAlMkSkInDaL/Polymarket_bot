#!/bin/bash
pkill -f "python -m src.cli run"
sleep 1
rm -f /dev/shm/pmb_*
tmux send-keys -t bot_fresh "cd /home/botuser/polymarket-bot && ulimit -n 65536 && .venv/bin/python -m src.cli run --env PAPER > logs/bot_fresh.log 2>&1" C-m
