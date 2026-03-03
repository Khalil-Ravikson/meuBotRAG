import os
from celery import Celery

# Força o uso do DB 2 no Redis para a fila (separado do RAG no DB 0)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0").replace("/0", "/2")

celery_app = Celery(
    "bot_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    # CRÍTICO: Regista o módulo de tarefas para evitar o KeyError
    include=["src.application.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="America/Sao_Paulo",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)