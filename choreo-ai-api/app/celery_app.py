"""Celery app with Redis broker."""
from celery import Celery

from app.config import REDIS_URL

celery_app = Celery(
    "choreo_ai",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks"],
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
