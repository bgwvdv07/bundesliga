"""
Fetch Bundesliga team injuries from api-football and write data/injuries.csv.

Required env var:
  FOOTBALL_API_KEY  (api-football.org key)

Output:
  data/injuries.csv with columns:
    team,date,position,importance

You can then pass this file into the prediction pipeline.
"""

import os
import csv
import requests
from pathlib import Path
from datetime import datetime

API_KEY = os.environ["FOOTBALL_API_KEY"]
BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY,
}

# Bundesliga league ID
LEAGUE_ID = 78
CURRENT_SEASON = 2026  # adjust if needed


def get_teams():
    """Get list of Bundesliga teams for the current season."""
    url = f"{BASE_URL}/teams"
    params = {
        "league": LEAGUE_ID,
        "season": CURRENT_SEASON,
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    teams = []
    for item in data.get("response", []):
        team = item.get("team", {})
        teams.append({
            "team_id": team.get("id"),
            "team_name": team.get("name"),
        })
    return teams


def get_team_injuries(team_id: int):
    """Get injuries for a given team_id."""
    url = f"{BASE_URL}/players/injuries"
    params = {
        "team": team_id,
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    injuries = []
    for item in data.get("response", []):
        player = item.get("player", {})
        injury = item.get("injury", {})

        # We only care about currently injured (no return date or future return)
        reason = injury.get("reason", "")
        start_str = injury.get("start")  # "YYYY-MM-DD"
        end_str = injury.get("end")      # "YYYY-MM-DD" or null

        if not start_str:
            continue

        try:
            start_date = datetime.fromisoformat(start_str).date()
        except Exception:
            continue

        end_date = None
        if end_str:
            try:
                end_date = datetime.fromisoformat(end_str).date()
            except Exception:
                pass

        injuries.append({
            "player_name": player.get("name"),
            "position": player.get("position"),  # Goalkeeper, Defender, Midfielder, Attacker
            "reason": reason,
            "start": start_date,
            "end": end_date,
        })

    return injuries


def map_position_to_importance(position: str, reason: str) -> str:
    """
    Simple heuristic to map position + reason to importance.
    You can refine this later.
    """
    pos = (position or "").lower()
    reason_lower = (reason or "").lower()

    # Goalkeepers and defenders often more critical for clean sheets
    if "goalkeeper" in pos:
        return "high"
    if "defender" in pos:
        return "medium"
    if "midfielder" in pos:
        return "medium"
    if "attacker" in pos or "forward" in pos:
        return "high"

    # Default
    return "low"


def main():
    teams = get_teams()
    print(f"Found {len(teams)} Bundesliga teams.")

    rows = []
    for t in teams:
        team_name = t["team_name"]
        team_id = t["team_id"]
        print(f"Fetching injuries for {team_name} (id={team_id})...")
        try:
            injuries = get_team_injuries(team_id)
        except Exception as e:
            print(f"  Error fetching injuries for {team_name}: {e}")
            continue

        for inj in injuries:
            # For now, use start date as the 'date' column
            # You could also use today's date if you want a snapshot
            rows.append({
                "team": team_name,
                "date": inj["start"].isoformat(),
                "position": inj["position"],
                "importance": map_position_to_importance(inj["position"], inj["reason"]),
            })

    # Write CSV
    BASE = Path(__file__).resolve().parent
    out_path = BASE / "data" / "injuries.csv"
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w", newline="") as f:
        fieldnames = ["team", "date", "position", "importance"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} injury records to {out_path}")


if __name__ == "__main__":
    main()