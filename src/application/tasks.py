"""
application/tasks.py — Celery Worker Tasks (V4.1 — AgentCore auto-inicializado)
=================================================================================

PROBLEMA RESOLVIDO:
────────────────────
  O AgentCore é um singleton de processo inicializado no startup() do FastAPI.
  O worker Celery é um processo separado — nunca passa pelo startup() do FastAPI.
  Resultado: agent_core._inicializado = False sempre no worker → resposta de erro.

SOLUÇÃO — Lazy Initialization com threading.Lock:
──────────────────────────────────────────────────
  _garantir_agente_inicializado() verifica se o AgentCore está pronto.
  Se não estiver, inicializa-o (índices Redis + ingestão + tools + registo).
  Um threading.Lock garante que a inicialização só acontece uma vez,
  mesmo com concurrency=2 (dois ForkPoolWorkers a correr em paralelo).

  CICLO DE VIDA NO WORKER:
    1ª task recebida  → _garantir_agente_inicializado() detecta _inicializado=False
                      → adquire lock → inicializa → libera lock
    2ª+ tasks         → _garantir_agente_inicializado() vê _inicializado=True
                      → passa diretamente (custo ~0ms)
    Restart do worker → processo novo → _inicializado=False → re-inicializa

  PORQUÊ NÃO USAR @worker_init SIGNAL DO CELERY?
    O sinal worker_init corre antes dos imports pesados (sentence-transformers,
    redis) estarem prontos no contexto do ForkPoolWorker. A inicialização lazy
    na primeira task garante que o ambiente está completamente pronto.
"""
from __future__ import annotations

import asyncio
import logging
import threading

from src.infrastructure.celery_app import celery_app
from src.infrastructure.redis_client import get_redis_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Singletons do processo worker (inicializados uma vez, reutilizados sempre)
# ─────────────────────────────────────────────────────────────────────────────
_init_lock         = threading.Lock()
_evolution_service = None   # Instância cacheada do EvolutionService


async def _get_evolution() -> "EvolutionService":
    """
    Retorna uma instância do EvolutionService já inicializada.
    Na 1ª chamada: faz a verificação HTTP (~60ms).
    Nas seguintes: devolve a instância cacheada (~0ms).
    """
    global _evolution_service
    from src.services.evolution_service import EvolutionService

    if _evolution_service is None:
        _evolution_service = EvolutionService()
        try:
            await _evolution_service.inicializar()
        except Exception as e:
            logger.warning("⚠️  EvolutionService.inicializar() falhou: %s", e)
            # Mantém a instância mesmo sem inicializar — enviar_mensagem faz retry

    return _evolution_service


def _garantir_agente_inicializado() -> None:
    """
    Verifica se o AgentCore está inicializado no processo do worker.
    Se não estiver, executa a sequência de inicialização completa.

    Idêntico ao startup() do FastAPI, mas síncrono e thread-safe.
    Custo na 1ª chamada: ~10-30s (carrega modelos de embeddings).
    Custo nas chamadas seguintes: ~0ms (apenas checa o flag).
    """
    # Import local — pesado, só carrega uma vez quando necessário
    from src.agent.core import agent_core

    # Fast path: já inicializado — retorna imediatamente
    if agent_core._inicializado:
        return

    # Slow path: adquire lock para inicialização exclusiva
    with _init_lock:
        # Double-check: outro thread pode ter inicializado enquanto esperávamos
        if agent_core._inicializado:
            return

        logger.info("🔧 [CELERY] Inicializando AgentCore no processo worker...")

        try:
            # Passo 1: Garante que os índices Redis existem
            from src.infrastructure.redis_client import inicializar_indices
            inicializar_indices()
            logger.info("✅ [CELERY] Índices Redis prontos.")

            # Passo 2: Ingere PDFs se necessário (idempotente — verifica hash)
            from src.rag.ingestion import Ingestor
            ingestor = Ingestor()
            ingestor.ingerir_se_necessario()
            logger.info("✅ [CELERY] Ingestão verificada.")

            # Passo 3: Instancia tools e inicializa o AgentCore
            # Isto regista as tools no Redis para o semantic router
            from src.tools import get_tools_ativas
            tools = get_tools_ativas()
            agent_core.inicializar(tools)

            logger.info(
                "✅ [CELERY] AgentCore inicializado com %d tools. "
                "Pipeline pronta para processar mensagens.",
                len(tools),
            )

        except Exception as e:
            # Log detalhado mas não re-lança — a task tratará o estado não-inicializado
            # com a mensagem amigável _MSG_AQUECENDO ao utilizador
            logger.exception(
                "❌ [CELERY] Falha ao inicializar AgentCore: %s. "
                "As tasks irão responder com mensagem de aquecimento.",
                e,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Coroutine assíncrona principal
# ─────────────────────────────────────────────────────────────────────────────

async def _processar_async(identity: dict) -> None:
    """
    Núcleo assíncrono. Imports pesados feitos aqui — nunca no topo do ficheiro.
    """
    from src.application.handle_message import handle_message
    from src.domain.entities import Mensagem

    mensagem = Mensagem(
        user_id   = identity["sender_phone"],
        chat_id   = identity["chat_id"],
        body      = identity.get("body", ""),
        has_media = identity.get("has_media", False),
        msg_type  = identity.get("msg_type", "conversation"),
    )

    # Usa instância cacheada — sem chamada HTTP repetida a cada mensagem
    evolution = await _get_evolution()
    await handle_message(mensagem, evolution)


# ─────────────────────────────────────────────────────────────────────────────
# Task principal
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(
    name        = "processar_mensagem_whatsapp",
    bind        = True,
    max_retries = 3,
)
def processar_mensagem_task(self, identity: dict) -> None:
    """
    Task Celery: garante inicialização → adquire lock → processa → libera lock.
    """
    chat_id = identity.get("chat_id", "desconhecido")

    # ── PASSO 0: Garante que o AgentCore está pronto neste processo ───────────
    # Na 1ª execução: ~10-30s (carrega sentence-transformers)
    # Nas seguintes: ~0ms (flag já True)
    _garantir_agente_inicializado()

    # ── PASSO 1: Distributed Lock ─────────────────────────────────────────────
    redis_client = get_redis_text()
    lock = redis_client.lock(
        f"lock:chat:{chat_id}",
        timeout          = 90,
        blocking_timeout = 5,
    )
    acquired = lock.acquire()

    if not acquired:
        logger.warning(
            "🔒 [CELERY] Usuário %s já em atendimento. "
            "Retry em 5s (tentativa %d/%d)...",
            chat_id, self.request.retries + 1, self.max_retries,
        )
        raise self.retry(countdown=5)

    # ── PASSO 2: Processamento exclusivo ──────────────────────────────────────
    try:
        logger.info("👷 [CELERY] Processamento exclusivo iniciado para %s", chat_id)
        asyncio.run(_processar_async(identity))
        logger.info("✅ [CELERY] Concluído para %s", chat_id)

    except Exception as exc:
        logger.error("❌ [CELERY] Erro para %s: %s", chat_id, exc, exc_info=True)
        countdown = 5 ** (self.request.retries + 1)
        raise self.retry(exc=exc, countdown=countdown)

    finally:
        try:
            lock.release()
            logger.debug("🔓 [CELERY] Lock liberado para %s", chat_id)
        except Exception:
            pass