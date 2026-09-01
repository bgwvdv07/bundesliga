import os
import csv
import requests
from pathlib import Path

# Your football-data.org API key
API_KEY = os.environ["FOOTBALL_API_KEY"]

# Bundesliga competition code
COMP = "BL1"

URL = f"https://api.football-data.org/v4/competitions/{COMP}/matches"

params = {
    "status": "SCHEDULED",
    # No matchday to avoid 400 if that matchday doesn't exist yet
}

headers = {"X-Auth-Token": API_KEY}

r = requests.get(URL, params=params, headers=headers, timeout=10)
print("API status:", r.status_code)
if r.status_code != 200:
    print("Response:", r.text[:500])
    r.raise_for_status()

data = r.json()
matches = data.get("matches", []) or []
print("Found scheduled matches:", len(matches))

rows = []
for i, m in enumerate(matches, 1):
    date_str = m.get("utcDate", "")[:10]  # "2026-09-05"
    home = m.get("homeTeam", {}).get("name", "")
    away = m.get("awayTeam", {}).get("name", "")

    # Odds: may be missing; use defaults if so
    odds = m.get("odds", {}) or {}
    home_odds = odds.get("home", 2.0)
    away_odds = odds.get("away", 2.0)

    if not date_str or not home or not away:
        continue

    rows.append({
        "game_id": f"CURR_BL_{i:02d}",
        "date": date_str,
        "home_team": home,
        "away_team": away,
        "home_odds": home_odds,
        "away_odds": away_odds,
    })

# Project root: parent of this script's directory
BASE = Path(__file__).resolve().parent
data_dir = BASE / "data"
data_dir.mkdir(exist_ok=True)
out = data_dir / "current.csv"

with open(out, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["game_id", "date", "home_team", "away_team", "home_odds", "away_odds"],
    )
    writer.writeheader()
    writer.writerows(rows)

print("Wrote", len(rows), "fixtures to", out)