"""
Data ingestion utilities.

Loads game data from CSV into simple dicts.
Supports:
  - Historical games (with results)
  - Current/upcoming games (no results)
"""

from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


def load_games_from_csv(csv_path: str, with_results: bool = False) -> List[Dict[str, Any]]:
    """
    Load games from a CSV file.

    If with_results=True (history):
      Expected columns:
        - game_id
        - date (YYYY-MM-DD)
        - home_team
        - away_team
        - home_odds
        - away_odds
        - home_score
        - away_score

    If with_results=False (current/upcoming):
      Expected columns:
        - game_id
        - date (YYYY-MM-DD)
        - home_team
        - away_team
        - home_odds
        - away_odds

    Returns a list of dicts ready to be used by the logic layer.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(path)

    base_cols = ["game_id", "date", "home_team", "away_team", "home_odds", "away_odds"]
    result_cols = ["home_score", "away_score"] if with_results else []

    required_cols = base_cols + result_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.date

    games = []
    for _, row in df.iterrows():
        game = {
            "game_id": str(row["game_id"]).strip(),
            "date": row["date"],
            "home_team": str(row["home_team"]).strip(),
            "away_team": str(row["away_team"]).strip(),
            "home_odds": float(row["home_odds"]) if pd.notna(row["home_odds"]) else None,
            "away_odds": float(row["away_odds"]) if pd.notna(row["away_odds"]) else None,
        }
        if with_results:
            game["home_score"] = int(row["home_score"]) if pd.notna(row["home_score"]) else None
            game["away_score"] = int(row["away_score"]) if pd.notna(row["away_score"]) else None

        games.append(game)

    return games
