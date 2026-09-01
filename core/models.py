from django.db import models


class Game(models.Model):
    """
    Represents a single sports game.
    """

    game_id = models.CharField(max_length=64, unique=True, db_index=True)
    date = models.DateField()
    home_team = models.CharField(max_length=128)
    away_team = models.CharField(max_length=128)
    home_odds = models.FloatField(null=True, blank=True)
    away_odds = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["date", "home_team", "away_team"]

    def __str__(self):
        return f"{self.game_id}: {self.home_team} vs {self.away_team} ({self.date})"


class Prediction(models.Model):
    game = models.OneToOneField(
        Game, on_delete=models.CASCADE, related_name="prediction"
    )
    home_win_prob = models.FloatField()
    away_win_prob = models.FloatField()

    model_version = models.CharField(max_length=64, default="logreg_odds_v1")

    # Market-implied probabilities
    home_implied_prob = models.FloatField(null=True, blank=True)
    away_implied_prob = models.FloatField(null=True, blank=True)

    # Edge = model - implied
    home_edge = models.FloatField(null=True, blank=True)
    away_edge = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-home_edge"]

    def __str__(self):
        return (
            f"Prediction for {self.game.game_id}: "
            f"home={self.home_win_prob:.3f}, away={self.away_win_prob:.3f}"
        )
