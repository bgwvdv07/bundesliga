from django.contrib import admin

from .models import Game, Prediction


@admin.register(Game)
class GameAdmin(admin.ModelAdmin):
    list_display = ("game_id", "date", "home_team", "away_team", "home_odds", "away_odds")
    list_filter = ("date", "home_team", "away_team")
    search_fields = ("game_id", "home_team", "away_team")


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("game", "home_win_prob", "away_win_prob", "model_version", "created_at")
    list_filter = ("model_version", "created_at")
    search_fields = ("game__game_id", "game__home_team", "game__away_team")
