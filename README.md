# Sports Prediction API (Django + SQLite) – Backtest → Predict

Minimal Django project that:

- Loads **historical game data** with results
- **Fits / backtests** a simple model on history
- Uses the fitted model to compute **win probabilities for current/upcoming games**
- Saves predictions to SQLite via Django ORM
- Exposes predictions via a simple JSON API

## Project structure

```text
sports_pred_project/
  README.md
  requirements.txt
  manage.py
  sports_pred/
    __init__.py
    settings.py
    urls.py
    asgi.py
    wsgi.py
  core/
    __init__.py
    models.py
    admin.py
    apps.py
    logic.py
    ingest.py
    views.py
    urls.py
  data/
    history.csv
    current.csv
  scripts/
    run_pipeline.py
    run_core.sh
```

## Requirements

- Python 3.10+ recommended
- `pip` and `venv`

## Setup

```bash
# 1. Create virtualenv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Correct sequence to init DB and run core logic

### 1. Initialize the database

From the project root (directory containing `manage.py`):

```bash
# Run migrations to create tables
python manage.py migrate

# (Optional) Create admin user
python manage.py createsuperuser
```

This creates `db.sqlite3` and all tables defined in `core/models.py`.

### 2. Run the core logic (backtest + predict)

Core logic =  
- Load historical games  
- Fit / backtest model  
- Load current/upcoming games  
- Compute win probabilities  
- Save predictions

```bash
# Run the pipeline
python manage.py shell < scripts/run_pipeline.py
```

Or:

```bash
bash scripts/run_core.sh
```

`scripts/run_pipeline.py` will:

- Load `data/history.csv` (with results)
- Fit a simple model on history
- Load `data/current.csv` (upcoming games)
- Compute win probabilities for each current game
- Save `Prediction` objects to the database

### 3. Start the API server

```bash
python manage.py runserver 0.0.0.0:8000
```

Then visit:

- `http://localhost:8000/api/predictions/` – list of predictions
- `http://localhost:8000/api/predictions/<id>/` – single prediction
- `http://localhost:8000/admin/` – Django admin (create a superuser with `createsuperuser`)

## Data format

### `data/history.csv` (historical games)

Columns:

- `game_id`
- `date` (YYYY-MM-DD)
- `home_team`
- `away_team`
- `home_odds`
- `away_odds`
- `home_score`
- `away_score`

You can add more features later (shots, xG, rest days, etc.) and extend the model.

### `data/current.csv` (upcoming games)

Columns:

- `game_id`
- `date` (YYYY-MM-DD)
- `home_team`
- `away_team`
- `home_odds`
- `away_odds`

No score columns here; these are the events you want predictions for.

## Core logic overview

- `core/ingest.load_games_from_csv(path, with_results: bool)` → list of game dicts
- `core/logic.fit_model_on_history(history_games)` → trained model object
- `core/logic.predict_current_games(model, current_games)` → list of `{game, home_win_prob, away_win_prob}`
- `core/logic.run_pipeline(history_path, current_path)` → full backtest → predict → save workflow

The default model is a simple **logistic regression** trained on:

- `home_odds`, `away_odds`
- Derived features (implied probabilities, normalized odds)

You can replace this with your own model (ELO, gradient boosting, neural net, etc.) inside `core/logic.py`.

## Notes

- All logic is in `core/logic.py` and `core/ingest.py` for easy extension.
- To add more features, extend the CSVs and update `build_features()` in `logic.py`.
