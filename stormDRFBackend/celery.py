# your_project/celery.py

import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stormDRFBackend.settings")

app = Celery("stormDRFBackend")
app.config_from_object("django.conf:settings", namespace="CELERY")

# Autodiscover tasks in your installed apps
app.autodiscover_tasks()
