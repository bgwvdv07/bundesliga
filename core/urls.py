"""
URL configuration for core app (API).
"""

from django.urls import path

from .views import PredictionsListView, PredictionDetailView
from .views_html import predictions_html

urlpatterns = [
    path("predictions/", PredictionsListView.as_view(), name="predictions-list"),
    path("predictions/<int:prediction_id>/", PredictionDetailView.as_view(), name="prediction-detail"),
    path("predictions-html/", predictions_html, name="predictions-html"),
]