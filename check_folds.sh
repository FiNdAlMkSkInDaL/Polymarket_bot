cd ~/polymarket-bot
find . -name "*.py" | xargs grep -i -E "fold.*=|n_folds|num_folds" | head -n 20
cat .env | grep -i fold
