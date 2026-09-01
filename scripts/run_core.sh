#!/usr/bin/env bash
# Run the core backtest → predict pipeline.
# Usage: bash scripts/run_core.sh

set -e

cd "$(dirname "$0")/.."

python manage.py shell < scripts/run_pipeline.py
