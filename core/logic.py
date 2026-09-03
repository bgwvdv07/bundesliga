"""
Core prediction logic with time-aware 3-class model.

Flow:
1. Load historical games (with results)
2. Fit time-aware 3-class model on history
3. Load current/upcoming games
4. Compute win/draw/away probabilities using the fitted model
5. Derive likely_winner and definite_winner
6. Save predictions to DB
"""

from typing import Dict, Any, List, Tuple
from datetime import date

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
from .models import Game, Prediction

import csv

# ---------------- Config ----------------

DEF_PROB_THRESH = 0.65   # minimum probability for definite winner
DEF_MARGIN_THRESH = 0.15 # minimum margin over other outcomes


# ---------------- Target ----------------



def load_injuries(injuries_csv_path: str):
    injuries = []
    with open(injuries_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            injuries.append({
                "team": row["team"].strip(),
                "date": row["date"],  # keep as string or parse to date if you prefer
                "position": row["position"].strip(),
                "importance": row["importance"].strip(),
            })
    return injuries

def build_target_3class(game: Dict[str, Any]) -> int:
    """
    0 = home win, 1 = draw, 2 = away win
    """
    home_score = game.get("home_score")
    away_score = game.get("away_score")
    if home_score is None or away_score is None:
        return 0
    if home_score > away_score:
        return 0
    elif home_score == away_score:
        return 1
    else:
        return 2


# ---------------- Team features ----------------

def build_team_features(
    past_games: List[Dict[str, Any]],
    game: Dict[str, Any],
    window: int = 10,
) -> Dict[str, float]:
    home_team = game["home_team"]
    away_team = game["away_team"]
    game_date = game["date"]

    past = [g for g in past_games if g["date"] < game_date]

    home_recent = []
    away_recent = []

    for g in past:
        if g["home_team"] == home_team or g["away_team"] == home_team:
            home_recent.append(g)
        if g["home_team"] == away_team or g["away_team"] == away_team:
            away_recent.append(g)

    home_recent = sorted(home_recent, key=lambda x: x["date"], reverse=True)[:window]
    away_recent = sorted(away_recent, key=lambda x: x["date"], reverse=True)[:window]

    def team_stats(games, team):
        if not games:
            return {"win_rate": 0.5, "goals_for_avg": 1.0, "goals_against_avg": 1.0}
        wins = 0
        gf = 0
        ga = 0
        for g in games:
            is_home = g["home_team"] == team
            if is_home:
                team_score = g["home_score"]
                opp_score = g["away_score"]
            else:
                team_score = g["away_score"]
                opp_score = g["home_score"]

            gf += team_score
            ga += opp_score
            if team_score > opp_score:
                wins += 1

        n = len(games)
        return {
            "win_rate": wins / n,
            "goals_for_avg": gf / n,
            "goals_against_avg": ga / n,
        }

    home_stats = team_stats(home_recent, home_team)
    away_stats = team_stats(away_recent, away_team)

    return {
        "home_win_rate": home_stats["win_rate"],
        "away_win_rate": away_stats["win_rate"],
        "home_gf_avg": home_stats["goals_for_avg"],
        "home_ga_avg": home_stats["goals_against_avg"],
        "away_gf_avg": away_stats["goals_for_avg"],
        "away_ga_avg": away_stats["goals_against_avg"],
    }


# ---------------- Features ----------------

def build_features(
    game: Dict[str, Any],
    past_games: List[Dict[str, Any]],
    injuries: List[Dict[str, Any]] | None = None,
) -> np.ndarray:
    base = []

    home_odds = game.get("home_odds")
    away_odds = game.get("away_odds")

    

    if home_odds and away_odds:
        home_implied = 1.0 / home_odds
        away_implied = 1.0 / away_odds
        total = home_implied + away_implied
        home_norm = home_implied / total
        away_norm = 1.0 - home_norm
        base += [home_implied, away_implied, home_norm, away_norm]
    else:
        base += [0.5, 0.5, 0.5, 0.5]

    tf = build_team_features(past_games, game, window=10)
    base += [
        tf["home_win_rate"],
        tf["away_win_rate"],
        tf["home_gf_avg"],
        tf["home_ga_avg"],
        tf["away_gf_avg"],
        tf["away_ga_avg"],
    ]

    return np.array(base)


# ---------------- Model ----------------

def fit_model_on_history(
    history_games: List[Dict[str, Any]],
    injuries: List[Dict[str, Any]] | None = None,
):
    history_sorted = sorted(history_games, key=lambda g: g["date"])
    X = []
    y = []

    for i, g in enumerate(history_sorted):
        past_games = history_sorted[:i]
        feats = build_features(g, past_games, injuries=injuries)
        target = build_target_3class(g)
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

    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X, y)
    return model


# ---------------- Prediction ----------------

def predict_current_games(
    model,
    history_games: List[Dict[str, Any]],
    current_games: List[Dict[str, Any]],
    injuries: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    preds = []

    for g in current_games:
        feats = build_features(g, history_games, injuries=injuries).reshape(1, -1)
        proba = model.predict_proba(feats)[0]
        p_home, p_draw, p_away = float(proba[0]), float(proba[1]), float(proba[2])

        probs = {"home": p_home, "draw": p_draw, "away": p_away}
        likely_winner = max(probs, key=probs.get)

        definite_winner = None
        if (
            p_home > DEF_PROB_THRESH
            and p_home > p_away + DEF_MARGIN_THRESH
            and p_home > p_draw + DEF_MARGIN_THRESH
        ):
            definite_winner = "home"
        elif (
            p_away > DEF_PROB_THRESH
            and p_away > p_home + DEF_MARGIN_THRESH
            and p_away > p_draw + DEF_MARGIN_THRESH
        ):
            definite_winner = "away"

        preds.append(
            {
                "game": g,
                "home_win_prob": p_home,
                "draw_prob": p_draw,
                "away_win_prob": p_away,
                "likely_winner": likely_winner,
                "definite_winner": definite_winner,
            }
        )

    return preds


# ---------------- Save to DB ----------------

def save_game_and_prediction(
    game: Dict[str, Any],
    probs: Dict[str, float],
    model_version: str = "gbc_3class_v1",
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

    home_odds = game.get("home_odds")
    away_odds = game.get("away_odds")

    home_implied = (1.0 / home_odds) if home_odds else None
    away_implied = (1.0 / away_odds) if away_odds else None
    total = (home_implied or 0) + (away_implied or 0)
    draw_implied = (1.0 - total) if total else None
    if draw_implied is not None and draw_implied < 0:
        draw_implied = 0.0

    home_edge = (probs["home_win_prob"] - home_implied) if home_implied else None
    away_edge = (probs["away_win_prob"] - away_implied) if away_implied else None
    draw_edge = (probs["draw_prob"] - draw_implied) if draw_implied is not None else None

    prediction_obj, _ = Prediction.objects.update_or_create(
        game=game_obj,
        defaults={
            "home_win_prob": probs["home_win_prob"],
            "draw_prob": probs["draw_prob"],
            "away_win_prob": probs["away_win_prob"],
            "model_version": model_version,
            "home_implied_prob": home_implied,
            "draw_implied_prob": draw_implied,
            "away_implied_prob": away_implied,
            "home_edge": home_edge,
            "draw_edge": draw_edge,
            "away_edge": away_edge,
            "likely_winner": probs.get("likely_winner"),
            "definite_winner": probs.get("definite_winner"),
        },
    )

    return game_obj, prediction_obj


# ---------------- Pipeline ----------------

def run_pipeline(
    history_csv_path: str,
    current_csv_path: str,
    injuries_csv_path: str | None = None,
) -> int:
    from .ingest import load_games_from_csv

    history_games = load_games_from_csv(history_csv_path, with_results=True)
    current_games = load_games_from_csv(current_csv_path, with_results=False)

    injuries = None
    if injuries_csv_path is not None and Path(injuries_csv_path).exists():
        injuries = load_injuries(injuries_csv_path)

    if len(history_games) == 0:
        raise ValueError("No historical games loaded; check history CSV.")
    if len(current_games) == 0:
        raise ValueError("No current games loaded; check current CSV.")

    model = fit_model_on_history(history_games, injuries=injuries)
    preds = predict_current_games(model, history_games, current_games, injuries=injuries)

    count = 0
    for p in preds:
        save_game_and_prediction(p["game"], p)
        count += 1

    return count