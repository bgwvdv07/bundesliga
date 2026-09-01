from django.shortcuts import render
from django.http import HttpResponse

from .models import Prediction


def fmt3(x):
    """Format float to 3 decimals or 'n/a' if None."""
    return f"{x:.3f}" if x is not None else "n/a"


def predictions_html(request):
    sort_by = request.GET.get("sort", "home_edge")  # home_edge, away_edge, home_win_prob

    if sort_by == "home_win_prob":
        qs = Prediction.objects.select_related("game").order_by("-home_win_prob")
    elif sort_by == "away_edge":
        qs = Prediction.objects.select_related("game").order_by("-away_edge")
    else:
        qs = Prediction.objects.select_related("game").order_by("-home_edge")

    rows = []
    for p in qs:
        rows.append({
            "game_id": p.game.game_id,
            "date": str(p.game.date),
            "home_team": p.game.home_team,
            "away_team": p.game.away_team,
            "home_odds": p.game.home_odds,
            "away_odds": p.game.away_odds,
            "home_win_prob": p.home_win_prob,
            "away_win_prob": p.away_win_prob,
            "home_implied_prob": p.home_implied_prob,
            "away_implied_prob": p.away_implied_prob,
            "home_edge": p.home_edge,
            "away_edge": p.away_edge,
        })

    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Bundesliga Predictions</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { font-size: 1.4em; }
        table { border-collapse: collapse; width: 100%; font-size: 0.9em; }
        th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }
        th { background: #f5f5f5; }
        a { text-decoration: none; color: #0066cc; }
        .pos { color: #008800; font-weight: bold; }
        .neg { color: #cc0000; }
      </style>
    </head>
    <body>
      <h1>Bundesliga Predictions (sorted by {{ sort_by }}</h1>
      <p>
        Sort by:
        <a href="?sort=home_edge">home_edge</a> |
        <a href="?sort=away_edge">away_edge</a> |
        <a href="?sort=home_win_prob">home_win_prob</a>
      </p>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Game</th>
            <th>Home odds</th>
            <th>Away odds</th>
            <th>Home prob</th>
            <th>Away prob</th>
            <th>Home implied</th>
            <th>Away implied</th>
            <th>Home edge</th>
            <th>Away edge</th>
          </tr>
        </thead>
        <tbody>
    """

    for r in rows:
        home_edge_cls = "pos" if (r["home_edge"] or 0) > 0 else "neg"
        away_edge_cls = "pos" if (r["away_edge"] or 0) > 0 else "neg"

        html += "<tr>"
        html += f"<td>{r['date']}</td>"
        html += f"<td>{r['home_team']} vs {r['away_team']} ({r['game_id']})</td>"
        html += f"<td>{r['home_odds']}</td>"
        html += f"<td>{r['away_odds']}</td>"
        html += f"<td>{fmt3(r['home_win_prob'])}</td>"
        html += f"<td>{fmt3(r['away_win_prob'])}</td>"
        html += f"<td>{fmt3(r['home_implied_prob'])}</td>"
        html += f"<td>{fmt3(r['away_implied_prob'])}</td>"
        html += f"<td class='{home_edge_cls}'>{fmt3(r['home_edge'])}</td>"
        html += f"<td class='{away_edge_cls}'>{fmt3(r['away_edge'])}</td>"
        html += "</tr>"

    html += """
        </tbody>
      </table>
    </body>
    </html>
    """

    # Simple string replace for sort_by (avoiding Django template engine here)
    html = html.replace("{{ sort_by }}", sort_by)

    return HttpResponse(html)