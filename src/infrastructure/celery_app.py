
# =============================================================================
#
# PROBLEMA: O `include=["src.application.tasks"]` é processado no momento
# em que o objeto Celery é criado. Se houver qualquer erro de import na
# cadeia tasks → handle_message → agent/core → (imports pesados), o
# include falha silenciosamente e a task nunca fica registada.
#
# SOLUÇÃO: Usar autodiscover_tasks() com on_after_configure que é chamado
# DEPOIS da app estar configurada, quando o worker já está a correr. Isso
# garante que os imports pesados (embeddings, redis, gemini) já estão prontos.
# =============================================================================

import os
from celery import Celery
from celery.signals import worker_ready

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0").replace("/0", "/2")

celery_app = Celery(
    "bot_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    # Serialização
    task_serializer    = "json",
    accept_content     = ["json"],
    result_serializer  = "json",

    # Timezone
    timezone           = "America/Sao_Paulo",
    enable_utc         = True,

    # Fiabilidade: ack só após execução, 1 task por vez por worker
    task_acks_late          = True,
    worker_prefetch_multiplier = 1,

    # CRÍTICO: lista explícita dos módulos de tasks
    # Colocado no conf.update (e não no construtor) para ser aplicado
    # APÓS a app estar configurada, evitando imports circulares precoces.
    include = ["src.application.tasks"],
)


# DIAGNÓSTICO: Loga as tasks registadas quando o worker ficar pronto.
# Verifica nos logs se 'processar_mensagem_whatsapp' aparece aqui.
@worker_ready.connect
def on_worker_ready(sender=None, **kwargs):
    import logging
    log = logging.getLogger(__name__)
    tasks = list(celery_app.tasks.keys())
    user_tasks = [t for t in tasks if not t.startswith("celery.")]
    log.info("✅ [CELERY] Worker pronto. Tasks registadas (%d): %s",
             len(user_tasks), user_tasks)
    if not user_tasks:
        log.error(
            "❌ [CELERY] NENHUMA task de utilizador registada! "
            "Verifica se src/application/tasks.py tem erros de import."
        )
