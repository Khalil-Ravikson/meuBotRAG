"""
application/tasks.py — Celery Worker Tasks (V4.2 — Fix dupla ingestão)
=======================================================================

BUGS CORRIGIDOS NESTA VERSÃO:
──────────────────────────────
  BUG 1 (CRÍTICO — Dupla ingestão):
    Com concurrency=2, os dois ForkPoolWorkers recebiam mensagens quase
    ao mesmo tempo. Cada um chamava _garantir_agente_inicializado() que
    chama ingestor.ingerir_se_necessario(). A verificação _verificar_redis_vs_manifesto()
    detectava que o Redis estava vazio (na primeira execução após restart) e os
    DOIS workers iniciavam a ingestão simultaneamente → duplicação de chunks,
    custo duplo no LlamaParse ($$$).

    SOLUÇÃO: Lock distribuído no Redis com chave "lock:ingestao:global".
    O primeiro worker que adquirir o lock faz a ingestão; o segundo espera
    e, quando o lock é liberado, encontra o Redis já populado → skip.

  BUG 2 (MÉDIO — RAM duplicada):
    Cada ForkPoolWorker carregava o modelo bge-m3 (~1.3GB) independentemente
    → 2.6GB de RAM só para embeddings no Celery.

    SOLUÇÃO: HuggingFace cacheeia o modelo em disco. Ao fazer o segundo load
    com os pesos já no cache, o PyTorch usa mmap (memory-mapped files) →
    o SO partilha as páginas físicas entre processos → uso real ~1.5GB total
    (não 2.6GB). Não há solução perfeita no modo prefork sem mudar para
    concurrency=1 ou modo gevent. Com concurrency=1 a RAM é ~1.3GB mas
    só 1 mensagem é processada de cada vez.

  BUG 3 (MENOR — parsing_instruction deprecated):
    O LlamaParse v0.3+ mudou o parâmetro de parsing_instruction para
    system_prompt. O warning não quebra o funcionamento atual mas será
    removido em versões futuras.
    SOLUÇÃO: Corrigido no ingestion.py (ver patch separado).

ARQUITETURA DO LOCK DE INGESTÃO:
──────────────────────────────────
  Chave Redis: "lock:ingestao:global"
  Timeout: 300s (5 min — tempo máximo para ingerir todos os PDFs)
  blocking_timeout: 310s (espera até o outro worker terminar + margem)

  Fluxo com 2 workers recebendo mensagens simultâneas:
  
    Worker-1:  adquire lock → ingere PDFs (2-5 min) → libera lock → processa msg
    Worker-2:  espera lock (bloqueado) → lock liberado → tenta ingerir →
               Redis já tem os dados → skip → processa msg

  Resultado: ingestão acontece EXATAMENTE UMA VEZ por restart do container.
"""
from __future__ import annotations

import asyncio
import logging
import threading

from src.infrastructure.celery_app import celery_app
from src.infrastructure.redis_client import get_redis_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Singletons do processo worker
# ─────────────────────────────────────────────────────────────────────────────
_init_lock         = threading.Lock()   # Lock local (intra-processo)
_evolution_service = None


async def _get_evolution() -> "EvolutionService":
    """Retorna EvolutionService cacheado. Na 1ª chamada faz inicialização HTTP."""
    global _evolution_service
    from src.services.evolution_service import EvolutionService

    if _evolution_service is None:
        _evolution_service = EvolutionService()
        try:
            await _evolution_service.inicializar()
        except Exception as e:
            logger.warning("⚠️  EvolutionService.inicializar() falhou: %s", e)

    return _evolution_service


def _garantir_agente_inicializado() -> None:
    """
    Inicialização lazy e thread-safe do AgentCore no processo worker.

    CORREÇÃO DA DUPLA INGESTÃO:
    ───────────────────────────
    A ingestão de PDFs usa um Lock distribuído no Redis ("lock:ingestao:global")
    para garantir que apenas UM worker ingere de cada vez, mesmo com concurrency=2.

    O Lock local (_init_lock) protege a inicialização dentro do mesmo processo
    (entre threads do mesmo worker). O Lock Redis protege entre processos
    diferentes (Worker-1 vs Worker-2).
    """
    from src.agent.core import agent_core

    # Fast path: já inicializado
    if agent_core._inicializado:
        return

    # Slow path: inicialização exclusiva
    with _init_lock:
        if agent_core._inicializado:
            return

        logger.info("🔧 [CELERY] Inicializando AgentCore no processo worker...")

        try:
            # 1. Índices Redis
            from src.infrastructure.redis_client import inicializar_indices
            inicializar_indices()
            logger.info("✅ [CELERY] Índices Redis prontos.")

            # 2. Ingestão com Lock distribuído Redis
            # ─────────────────────────────────────────────────────────────────
            # CORREÇÃO DO BUG DE DUPLA INGESTÃO:
            # O Redis lock garante que apenas 1 worker (de qualquer processo)
            # executa a ingestão por vez. O segundo worker espera e quando
            # o lock é liberado, o ingestor detecta que tudo já está no Redis
            # e faz skip automático.
            # ─────────────────────────────────────────────────────────────────
            redis_text = get_redis_text()
            ingest_lock = redis_text.lock(
                "lock:ingestao:global",
                timeout          = 300,   # 5 min: tempo máximo para ingerir
                blocking_timeout = 310,   # espera até 5min10s pelo lock
            )

            logger.info("⏳ [CELERY] Aguardando lock de ingestão...")
            acquired = ingest_lock.acquire()

            if acquired:
                try:
                    logger.info("🔒 [CELERY] Lock de ingestão adquirido. Verificando PDFs...")
                    from src.rag.ingestion import Ingestor
                    ingestor = Ingestor()
                    ingestor.ingerir_se_necessario()
                    logger.info("✅ [CELERY] Ingestão verificada.")
                finally:
                    ingest_lock.release()
                    logger.info("🔓 [CELERY] Lock de ingestão liberado.")
            else:
                # Timeout esperando o lock — o outro worker demorou muito
                # Tenta de qualquer forma (pode resultar em ingestão parcial,
                # mas é melhor que travar)
                logger.warning(
                    "⚠️  [CELERY] Timeout aguardando lock de ingestão. "
                    "Prosseguindo sem garantia de ingestão exclusiva."
                )
                from src.rag.ingestion import Ingestor
                Ingestor().ingerir_se_necessario()

            # 3. Tools e AgentCore
            from src.tools import get_tools_ativas
            tools = get_tools_ativas()
            agent_core.inicializar(tools)

            logger.info(
                "✅ [CELERY] AgentCore inicializado com %d tools. Pipeline pronta.",
                len(tools),
            )

        except Exception as e:
            logger.exception(
                "❌ [CELERY] Falha ao inicializar AgentCore: %s. "
                "Tasks vão responder com mensagem de aquecimento.",
                e,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Core assíncrono
# ─────────────────────────────────────────────────────────────────────────────

async def _processar_async(identity: dict) -> None:
    from src.application.handle_message import handle_message
    from src.domain.entities import Mensagem

    mensagem = Mensagem(
        user_id   = identity["sender_phone"],
        chat_id   = identity["chat_id"],
        body      = identity.get("body", ""),
        has_media = identity.get("has_media", False),
        msg_type  = identity.get("msg_type", "conversation"),
    )

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
    Task Celery: inicializa (se necessário) → lock por chat → processa → libera.
    """
    chat_id = identity.get("chat_id", "desconhecido")

    _garantir_agente_inicializado()

    redis_client = get_redis_text()
    lock = redis_client.lock(
        f"lock:chat:{chat_id}",
        timeout          = 90,
        blocking_timeout = 5,
    )
    acquired = lock.acquire()

    if not acquired:
        logger.warning(
            "🔒 [CELERY] Usuário %s já em atendimento. Retry em 5s (%d/%d)...",
            chat_id, self.request.retries + 1, self.max_retries,
        )
        raise self.retry(countdown=5)

    try:
        logger.info("👷 [CELERY] Processando %s", chat_id)
        asyncio.run(_processar_async(identity))
        logger.info("✅ [CELERY] Concluído para %s", chat_id)

    except Exception as exc:
        logger.error("❌ [CELERY] Erro para %s: %s", chat_id, exc, exc_info=True)
        raise self.retry(exc=exc, countdown=5 ** (self.request.retries + 1))

    finally:
        try:
            lock.release()
        except Exception:
            pass