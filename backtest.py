"""
Time-aware backtest of the 3-class model and definite_winner rule.

For each season:
  - Train on all games before that season
  - Predict games in that season using only past data
  - Evaluate:
      * 3-class log loss / accuracy
      * Definite_winner calls: count, hit rate, avg odds, simulated P/L

Also:
  - Run a simple threshold sweep for definite_winner
  - Save per-season results to CSV
  - Save a summary text file
"""

import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score

from core.ingest import load_games_from_csv


# ---------- Model functions ----------

def build_target_3class(game):
    home_score = game.get("home_score")
    away_score = game.get("away_score")
    if home_score is None or away_score is None:
        return 0
    if home_score > away_score:
        return 0  # home win
    elif home_score == away_score:
        return 1  # draw
    else:
        return 2  # away win


def build_team_features(past_games, game, window=10):
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


def build_features(game, past_games):
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


def fit_model(history_games):
    history_sorted = sorted(history_games, key=lambda g: g["date"])
    X = []
    y = []

    for i, g in enumerate(history_sorted):
        past_games = history_sorted[:i]
        feats = build_features(g, past_games)
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


def predict_games(model, history_games, games, prob_thresh=0.65, margin_thresh=0.15):
    preds = []
    for g in games:
        feats = build_features(g, history_games).reshape(1, -1)
        proba = model.predict_proba(feats)[0]
        p_home, p_draw, p_away = float(proba[0]), float(proba[1]), float(proba[2])

        probs = {"home": p_home, "draw": p_draw, "away": p_away}
        likely_winner = max(probs, key=probs.get)

        definite_winner = None
        if (
            p_home > prob_thresh
            and p_home > p_away + margin_thresh
            and p_home > p_draw + margin_thresh
        ):
            definite_winner = "home"
        elif (
            p_away > prob_thresh
            and p_away > p_home + margin_thresh
            and p_away > p_draw + margin_thresh
        ):
            definite_winner = "away"

        preds.append(
            {
                "game": g,
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "likely_winner": likely_winner,
                "definite_winner": definite_winner,
            }
        )
    return preds


# ---------- Backtest logic ----------

def get_season(date_obj):
    year = date_obj.year
    month = date_obj.month
    if month >= 7:
        return f"{year}/{str(year+1)[-2:]}"
    else:
        return f"{year-1}/{str(year)[-2:]}"


def run_backtest(
    history_csv_path: str,
    prob_thresh=0.65,
    margin_thresh=0.15,
    save_dir: Path | None = None,
):
    history = load_games_from_csv(history_csv_path, with_results=True)
    history_sorted = sorted(history, key=lambda g: g["date"])

    seasons = defaultdict(list)
    for g in history_sorted:
        seasons[get_season(g["date"])].append(g)

    season_names = sorted(seasons.keys())

    print("Seasons found:", season_names)
    print()

    all_y_true = []
    all_y_pred = []
    all_probs = []

    definite_stats = {
        "count": 0,
        "hits": 0,
        "total_odds_sum": 0.0,
        "profit": 0.0,
    }

    season_rows = []

    for i, s in enumerate(season_names):
        train_games = []
        for prev in season_names[:i]:
            train_games.extend(seasons[prev])

        test_games = seasons[s]

        if len(train_games) == 0:
            print(f"Season {s}: no prior training data, skipping.")
            continue

        if len(test_games) == 0:
            continue

        model = fit_model(train_games)
        preds = predict_games(
            model, train_games, test_games,
            prob_thresh=prob_thresh,
            margin_thresh=margin_thresh,
        )

        y_true = []
        y_pred = []
        probs = []

        season_def = {
            "count": 0,
            "hits": 0,
            "total_odds_sum": 0.0,
            "profit": 0.0,
        }

        for p in preds:
            g = p["game"]
            true_label = build_target_3class(g)
            y_true.append(true_label)
            pred_label = {"home": 0, "draw": 1, "away": 2}[p["likely_winner"]]
            y_pred.append(pred_label)
            probs.append([p["p_home"], p["p_draw"], p["p_away"]])

            dw = p["definite_winner"]
            if dw in ("home", "away"):
                definite_stats["count"] += 1
                season_def["count"] += 1

                actual = true_label
                won = (dw == "home" and actual == 0) or (dw == "away" and actual == 2)
                if won:
                    definite_stats["hits"] += 1
                    season_def["hits"] += 1
                    odds = g["home_odds"] if dw == "home" else g["away_odds"]
                    definite_stats["total_odds_sum"] += odds
                    season_def["total_odds_sum"] += odds
                    definite_stats["profit"] += (odds - 1.0)
                    season_def["profit"] += (odds - 1.0)
                else:
                    definite_stats["profit"] -= 1.0
                    season_def["profit"] -= 1.0

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_probs.extend(probs)

        acc = accuracy_score(y_true, y_pred)
        ll = log_loss(y_true, probs, labels=[0, 1, 2])

        season_rows.append(
            {
                "season": s,
                "games": len(test_games),
                "acc": acc,
                "log_loss": ll,
                "def_count": season_def["count"],
                "def_hits": season_def["hits"],
                "def_hit_rate": season_def["hits"] / season_def["count"] if season_def["count"] > 0 else None,
                "def_avg_odds": (
                    season_def["total_odds_sum"] / season_def["count"]
                    if season_def["count"] > 0
                    else None
                ),
                "def_profit": season_def["profit"],
            }
        )

        print(
            f"Season {s}: "
            f"games={len(test_games)}, "
            f"acc={acc:.3f}, "
            f"log_loss={ll:.3f}, "
            f"def_count={season_def['count']}, "
            f"def_hit_rate={season_def['hits']/season_def['count'] if season_def['count'] else 0:.3f}"
        )

    if len(all_y_true) == 0:
        print("No predictions made; check history CSV and season logic.")
        return

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_ll = log_loss(all_y_true, all_probs, labels=[0, 1, 2])

    print()
    print("Overall 3-class metrics:")
    print(f"  Accuracy: {overall_acc:.3f}")
    print(f"  Log loss: {overall_ll:.3f}")

    print()
    print("Definite winner rule (prob_thresh={:.2f}, margin_thresh={:.2f}):".format(
        prob_thresh, margin_thresh
    ))
    print(f"  Count: {definite_stats['count']}")
    if definite_stats["count"] > 0:
        hit_rate = definite_stats["hits"] / definite_stats["count"]
        avg_odds = definite_stats["total_odds_sum"] / definite_stats["count"]
        print(f"  Hits: {definite_stats['hits']}")
        print(f"  Hit rate: {hit_rate:.3f}")
        print(f"  Avg odds: {avg_odds:.2f}")
        print(f"  Net profit (1u per bet): {definite_stats['profit']:.2f}")
    else:
        print("  No definite_winner signals generated with current thresholds.")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / "backtest_by_season.csv"
        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "season", "games", "acc", "log_loss",
                "def_count", "def_hits", "def_hit_rate", "def_avg_odds", "def_profit",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(season_rows)

        summary_path = save_dir / "backtest_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Thresholds: prob={prob_thresh:.2f}, margin={margin_thresh:.2f}\n")
            f.write(f"Seasons: {season_names}\n\n")
            f.write(f"Overall accuracy: {overall_acc:.3f}\n")
            f.write(f"Overall log_loss: {overall_ll:.3f}\n\n")
            f.write(f"Definite winner count: {definite_stats['count']}\n")
            if definite_stats["count"] > 0:
                hit_rate = definite_stats["hits"] / definite_stats["count"]
                avg_odds = definite_stats["total_odds_sum"] / definite_stats["count"]
                f.write(f"Definite winner hit rate: {hit_rate:.3f}\n")
                f.write(f"Definite winner avg odds: {avg_odds:.2f}\n")
                f.write(f"Definite winner net profit: {definite_stats['profit']:.2f}\n")

        print()
        print(f"Saved per-season CSV to: {csv_path}")
        print(f"Saved summary to: {summary_path}")


def threshold_sweep(history_csv_path: str, save_dir: Path):
    """
    Simple sweep over prob_thresh and margin_thresh.
    Prints best combination by net profit.
    """
    history = load_games_from_csv(history_csv_path, with_results=True)
    history_sorted = sorted(history, key=lambda g: g["date"])

    seasons = defaultdict(list)
    for g in history_sorted:
        seasons[get_season(g["date"])].append(g)
    season_names = sorted(seasons.keys())

    prob_threshs = [0.55, 0.60, 0.65, 0.70]
    margin_threshs = [0.10, 0.15, 0.20]

    best = None

    print("Threshold sweep:")
    for pt in prob_threshs:
        for mt in margin_threshs:
            definite_stats = {
                "count": 0,
                "hits": 0,
                "profit": 0.0,
            }

            for i, s in enumerate(season_names):
                train_games = []
                for prev in season_names[:i]:
                    train_games.extend(seasons[prev])

                test_games = seasons[s]
                if len(train_games) == 0 or len(test_games) == 0:
                    continue

                model = fit_model(train_games)
                preds = predict_games(
                    model, train_games, test_games,
                    prob_thresh=pt,
                    margin_thresh=mt,
                )

                for p in preds:
                    g = p["game"]
                    true_label = build_target_3class(g)
                    dw = p["definite_winner"]
                    if dw in ("home", "away"):
                        definite_stats["count"] += 1
                        actual = true_label
                        won = (dw == "home" and actual == 0) or (dw == "away" and actual == 2)
                        if won:
                            definite_stats["hits"] += 1
                            odds = g["home_odds"] if dw == "home" else g["away_odds"]
                            definite_stats["profit"] += (odds - 1.0)
                        else:
                            definite_stats["profit"] -= 1.0

            hit_rate = definite_stats["hits"] / definite_stats["count"] if definite_stats["count"] else 0
            print(
                f"prob={pt:.2f}, margin={mt:.2f} -> "
                f"count={definite_stats['count']}, "
                f"hit_rate={hit_rate:.3f}, "
                f"profit={definite_stats['profit']:.2f}"
            )

            if best is None or definite_stats["profit"] > best["profit"]:
                best = {
                    "prob_thresh": pt,
                    "margin_thresh": mt,
                    "count": definite_stats["count"],
                    "hit_rate": hit_rate,
                    "profit": definite_stats["profit"],
                }

    print()
    print("Best thresholds by profit:")
    print(
        f"prob={best['prob_thresh']:.2f}, margin={best['margin_thresh']:.2f} -> "
        f"count={best['count']}, hit_rate={best['hit_rate']:.3f}, profit={best['profit']:.2f}"
    )

    return best


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    HISTORY_CSV = BASE / "data" / "history.csv"
    OUTPUT_DIR = BASE / "backtest_output"

    print("Running time-aware backtest with default thresholds...")
    run_backtest(
        str(HISTORY_CSV),
        prob_thresh=0.65,
        margin_thresh=0.15,
        save_dir=OUTPUT_DIR,
    )

    print()
    print("=" * 60)
    print("Threshold sweep")
    print("=" * 60)
    best = threshold_sweep(str(HISTORY_CSV), OUTPUT_DIR)

    # Optionally re-run backtest with best thresholds and save
    if best:
        print()
        print("Re-running backtest with best thresholds...")
        run_backtest(
            str(HISTORY_CSV),
            prob_thresh=best["prob_thresh"],
            margin_thresh=best["margin_thresh"],
            save_dir=OUTPUT_DIR / "best",
        )