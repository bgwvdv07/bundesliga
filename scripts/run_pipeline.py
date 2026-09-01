"""
Run this via:
  python manage.py shell < scripts/run_pipeline.py

It executes the full backtest → predict pipeline.
"""

import os
from pathlib import Path

from core.logic import run_pipeline

# DJANGO_SETTINGS_MODULE is like "sports_pred.settings"
settings_module = os.environ["DJANGO_SETTINGS_MODULE"]
project_package = settings_module.split(".")[0]  # "sports_pred"

# Directory containing the project package (where settings.py lives)
project_package_dir = Path(__import__(project_package).__file__).parent

# Project root: parent of the package dir (contains manage.py, data/, core/, scripts/)
BASE_DIR = project_package_dir.parent

HISTORY_CSV = BASE_DIR / "data" / "history.csv"
CURRENT_CSV = BASE_DIR / "data" / "current.csv"

# Debug: remove after confirming it works
print("BASE_DIR:", BASE_DIR)
print("HISTORY_CSV exists:", HISTORY_CSV.exists())
print("CURRENT_CSV exists:", CURRENT_CSV.exists())

print("Running backtest → predict pipeline...")
count = run_pipeline(str(HISTORY_CSV), str(CURRENT_CSV))
print(f"Generated predictions for {count} current games.")