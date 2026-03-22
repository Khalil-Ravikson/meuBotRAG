"""
infrastructure/celery_app.py — Celery + Beat Schedule
=======================================================

MUDANÇA vs versão anterior:
  ADICIONADO beat_schedule com duas entradas:
    1. verificar_prazos_diario  → 08:00 BRT todo dia de semana
    2. verificar_prazos_fim_semana → 09:00 BRT sáb/dom (horário mais tardio)

  MANTIDO: lock de ingestão, task processar_mensagem_whatsapp, diagnóstico.

POR QUE DOIS SCHEDULES?
  Em dias úteis alunos checam WhatsApp mais cedo (8h).
  Aos fins de semana, 9h é menos intrusivo. A distinção é feita
  no beat_schedule usando crontab — o Celery Worker normal
  não precisa de nenhuma mudança.

INFRAESTRUTURA:
  Broker/Backend: Redis DB 2 (separado do DB 0 usado pela app)
  Beat state:     Redis (redbeat) — persiste o estado entre restarts
  Timezone:       America/Sao_Paulo (UTC-3 / BRT)
"""

import os
from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_ready

# Redis DB 2 para o Celery (isolado do DB 0 da app e DB 1 da Evolution)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0").replace("/0", "/2")

celery_app = Celery(
    "bot_tasks",
    broker  = REDIS_URL,
    backend = REDIS_URL,
)

celery_app.conf.update(
    # ── Serialização ──────────────────────────────────────────────────────────
    task_serializer   = "json",
    accept_content    = ["json"],
    result_serializer = "json",

    # ── Timezone (OBRIGATÓRIO para crontab funcionar no horário certo) ────────
    timezone   = "America/Sao_Paulo",
    enable_utc = True,

    # ── Confiabilidade ────────────────────────────────────────────────────────
    task_acks_late             = True,
    worker_prefetch_multiplier = 1,

    # ── Tasks registradas ─────────────────────────────────────────────────────
    include = [
        "src.application.tasks",             # processar_mensagem_whatsapp
        "src.application.tasks_notificacao", # verificar_e_notificar_prazos
        "src.application.tasks_admin",       # ingerir_documento_whatsapp
    ],

    # ── Beat Schedule — Notificações Proativas ────────────────────────────────
    # Celery Beat precisa estar rodando como serviço separado (ver docker-compose).
    # O worker normal NÃO executa o beat — são dois processos distintos.
    beat_schedule = {
        # Segunda a sexta: 08:00 BRT
        "verificar_prazos_dias_uteis": {
            "task":     "verificar_e_notificar_prazos",
            "schedule": crontab(hour=8, minute=0, day_of_week="1-5"),
            "options":  {"queue": "notificacoes"},
        },
        # Sábado e domingo: 09:00 BRT (menos intrusivo)
        "verificar_prazos_fim_semana": {
            "task":     "verificar_e_notificar_prazos",
            "schedule": crontab(hour=9, minute=0, day_of_week="0,6"),
            "options":  {"queue": "notificacoes"},
        },
    },

    # ── Filas separadas por prioridade ────────────────────────────────────────
    # "default"      → mensagens WhatsApp em tempo real (alta prioridade)
    # "notificacoes" → envios em lote pelo Beat (pode aguardar)
    # "admin"        → tarefas admin (ingestão, comandos)
    task_default_queue = "default",
    task_routes = {
        "processar_mensagem_whatsapp":  {"queue": "default"},
        "verificar_e_notificar_prazos": {"queue": "notificacoes"},
        "notificar_evento_especifico":  {"queue": "notificacoes"},
        "ingerir_documento_whatsapp":   {"queue": "admin"},
        "executar_comando_admin":       {"queue": "admin"},
    },

    # ── Beat: armazena estado no Redis (sobrevive a restarts) ─────────────────
    # Requer: pip install celery[redis] (já incluso no requirements.txt)
    beat_scheduler = "celery.beat:PersistentScheduler",
    beat_schedule_filename = "/tmp/celery_beat_schedule",
)


# ── Diagnóstico no startup do worker ──────────────────────────────────────────

@worker_ready.connect
def on_worker_ready(sender=None, **kwargs):
    import logging
    log = logging.getLogger(__name__)
    tasks = list(celery_app.tasks.keys())
    user_tasks = [t for t in tasks if not t.startswith("celery.")]
    log.info(
        "✅ [CELERY] Worker pronto. Tasks (%d): %s",
        len(user_tasks), user_tasks,
    )
    if not user_tasks:
        log.error(
            "❌ [CELERY] NENHUMA task registrada! "
            "Verifica imports em src/application/tasks*.py"
        )
    # Avisa se beat_schedule está configurado
    schedules = list(celery_app.conf.beat_schedule.keys())
    log.info("📅 Beat schedules configurados: %s", schedules)