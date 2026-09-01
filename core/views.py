"""
Minimal API views for predictions.
"""

from django.http import JsonResponse
from django.views import View

from .models import Game, Prediction


class PredictionsListView(View):
    def get(self, request):
        # Sort by home_edge descending
        preds = (
            Prediction.objects
            .select_related("game")
            .order_by("-home_edge")
        )
        data = [
            {
                "id": p.id,
                "game_id": p.game.game_id,
                "date": str(p.game.date),
                "home_team": p.game.home_team,
                "away_team": p.game.away_team,
                "home_win_prob": p.home_win_prob,
                "away_win_prob": p.away_win_prob,
                "home_implied_prob": p.home_implied_prob,
                "away_implied_prob": p.away_implied_prob,
                "home_edge": p.home_edge,
                "away_edge": p.away_edge,
                "model_version": p.model_version,
                "created_at": p.created_at.isoformat(),
            }
            for p in preds
        ]
        return JsonResponse({"predictions": data})


class PredictionDetailView(View):
    """
    GET /api/predictions/<id>/
    Returns a single prediction.
    """

    def get(self, request, prediction_id: int):
        try:
            p = Prediction.objects.select_related("game").get(id=prediction_id)
        except Prediction.DoesNotExist:
            return JsonResponse({"error": "Prediction not found"}, status=404)

        data = {
            "id": p.id,
            "game_id": p.game.game_id,
            "date": str(p.game.date),
            "home_team": p.game.home_team,
            "away_team": p.game.away_team,
            "home_odds": p.game.home_odds,
            "away_odds": p.game.away_odds,
            "home_win_prob": p.home_win_prob,
            "away_win_prob": p.away_win_prob,
            "model_version": p.model_version,
            "created_at": p.created_at.isoformat(),
        }
        return JsonResponse(data)
