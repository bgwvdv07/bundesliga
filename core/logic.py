"""
Core prediction logic with backtesting.

Flow:
1. Load historical games (with results)
2. Fit / backtest a model on history
3. Load current/upcoming games
4. Compute win probabilities using the fitted model
5. Save predictions to DB

Replace the model with your own as needed.
"""

from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from typing import Any
from .models import Game, Prediction


def build_features(game: Dict[str, Any]) -> np.ndarray:
    """
    Build feature vector for a single game.

    Default simple features:
      - home_implied = 1 / home_odds
      - away_implied = 1 / away_odds
      - home_norm = home_implied / (home_implied + away_implied)
      - away_norm = 1 - home_norm

    Extend this with more features (team strength, rest days, etc.).
    """
    home_odds = game.get("home_odds")
    away_odds = game.get("away_odds")

    if not home_odds or not away_odds:
        # Fallback features if odds missing
        return np.array([0.5, 0.5, 0.5, 0.5])

    home_implied = 1.0 / home_odds
    away_implied = 1.0 / away_odds

    total = home_implied + away_implied
    home_norm = home_implied / total
    away_norm = 1.0 - home_norm

    return np.array([home_implied, away_implied, home_norm, away_norm])


def build_target(game: Dict[str, Any]) -> int:
    """
    Build binary target for historical game:
      1 if home team won, 0 otherwise (draw or away win).

    Adjust if you want a 3-class model (home/draw/away).
    """
    home_score = game.get("home_score")
    away_score = game.get("away_score")

    if home_score is None or away_score is None:
        # Should not happen for history; default to 0
        return 0

    return 1 if home_score > away_score else 0


def fit_model_on_history(history_games: List[Dict[str, Any]]):
    X = []
    y = []

    for g in history_games:
        feats = build_features(g)
        target = build_target(g)
        X.append(feats)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    base = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_depth=4,
    l2_regularization=1.0,
)

    # Calibrate probabilities using 3-fold CV
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X, y)
    return model


def predict_current_games(
    model: Any,
    current_games: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Use the fitted model to predict win probabilities for current games.

    Returns a list of dicts:
      {
        "game": game_dict,
        "home_win_prob": float,
        "away_win_prob": float,
      }
    """
    preds = []

    for g in current_games:
        feats = build_features(g).reshape(1, -1)
        # predict_proba returns [[p_0, p_1]] where p_1 = P(home_win=1)
        proba = model.predict_proba(feats)[0]
        home_win_prob = float(proba[1])
        away_win_prob = 1.0 - home_win_prob

        preds.append(
            {
                "game": g,
                "home_win_prob": home_win_prob,
                "away_win_prob": away_win_prob,
            }
        )

    return preds


def save_game_and_prediction(
    game: Dict[str, Any],
    probs: Dict[str, float],
    model_version: str = "logreg_odds_v1",
) -> Tuple[Game, Prediction]:
    game_obj, _ = Game.objects.update_or_create(
        game_id=game["game_id"],
        defaults={
            "date": game["date"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "home_odds": game["home_odds"],
            "away_odds": game["away_odds"],
        },
    )

    # Implied probabilities from odds
    home_odds = game.get("home_odds")
    away_odds = game.get("away_odds")

    home_implied = (1.0 / home_odds) if home_odds else None
    away_implied = (1.0 / away_odds) if away_odds else None

    # Normalize if you want (optional, since 1/odds already sum >1 due to margin)
    # Here we just use raw implied.

    home_edge = (probs["home_win_prob"] - home_implied) if home_implied else None
    away_edge = (probs["away_win_prob"] - away_implied) if away_implied else None

    prediction_obj, _ = Prediction.objects.update_or_create(
        game=game_obj,
        defaults={
            "home_win_prob": probs["home_win_prob"],
            "away_win_prob": probs["away_win_prob"],
            "model_version": model_version,
            "home_implied_prob": home_implied,
            "away_implied_prob": away_implied,
            "home_edge": home_edge,
            "away_edge": away_edge,
        },
    )

    return game_obj, prediction_obj


def run_pipeline(history_csv_path: str, current_csv_path: str) -> int:
    """
    Full pipeline:
      - Load historical games
      - Fit model on history (backtest / calibration step)
      - Load current/upcoming games
      - Compute win probabilities
      - Save predictions to DB

    Returns number of current games processed.
    """
    from .ingest import load_games_from_csv

    # 1. Load and fit on history
    history_games = load_games_from_csv(history_csv_path, with_results=True)
    if len(history_games) == 0:
        raise ValueError("No historical games loaded; check history CSV.")

    model = fit_model_on_history(history_games)

    # 2. Predict on current events
    current_games = load_games_from_csv(current_csv_path, with_results=False)
    if len(current_games) == 0:
        raise ValueError("No current games loaded; check current CSV.")

    preds = predict_current_games(model, current_games)

    # 3. Save to DB
    count = 0
    for p in preds:
        save_game_and_prediction(p["game"], p)
        count += 1

    return count
