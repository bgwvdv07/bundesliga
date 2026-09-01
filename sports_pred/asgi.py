"""
ASGI config for sports_pred project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sports_pred.settings")

application = get_asgi_application()
