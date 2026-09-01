#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# --- CONFIG ---
# Set your football-data.org API key here:
export FOOTBALL_API_KEY="ccd7cab5cda94525912e531edc479e22"
# --------------

# Activate virtualenv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=== Running migrations ==="
python manage.py migrate

echo "=== Running full pipeline (fetch + backtest + predict) ==="
python manage.py shell < scripts/run_full_pipeline.py

echo "=== Starting Django server ==="
python manage.py runserver 0.0.0.0:8000