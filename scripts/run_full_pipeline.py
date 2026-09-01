"""
Full pipeline:
  1. Fetch current fixtures (football-data.org) -> data/current.csv
  2. Load history, fit model
  3. Predict on current fixtures
  4. Save predictions to DB
"""

import os
import subprocess
from pathlib import Path

from core.logic import run_pipeline

# Derive project root from Django settings module
settings_module = os.environ["DJANGO_SETTINGS_MODULE"]  # e.g. "sports_pred.settings"
project_package = settings_module.split(".")[0]         # "sports_pred"
project_package_dir = Path(__import__(project_package).__file__).parent
BASE = project_package_dir.parent                       # project root

FETCH_SCRIPT = BASE / "fetch_current.py"
HISTORY_CSV = BASE / "data" / "history.csv"
CURRENT_CSV = BASE / "data" / "current.csv"

# Ensure API key is set
if "FOOTBALL_API_KEY" not in os.environ:
    raise RuntimeError("Set FOOTBALL_API_KEY env var before running the pipeline.")

# 1. Fetch current fixtures
print("Fetching current fixtures from football-data.org...")
subprocess.run(["python", str(FETCH_SCRIPT)], check=True)

if not CURRENT_CSV.exists():
    raise FileNotFoundError("current.csv was not created by fetch_current.py")

# 2. Run backtest + prediction
print("Running backtest → predict pipeline...")
count = run_pipeline(str(HISTORY_CSV), str(CURRENT_CSV))
print(f"Generated predictions for {count} current games.")