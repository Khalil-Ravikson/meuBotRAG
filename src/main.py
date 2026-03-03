"""
main.py — Bootstrap FastAPI (v4 — Arquitetura Assíncrona com Filas)
===================================================================

O QUE MUDOU vs v3 (V4 Update):
──────────────────────────────
  REMOVIDO:
    - O processamento síncrono no endpoint /webhook.
    - A chamada direta para `handle_webhook` que aguardava o RAG/Gemini.

  ADICIONADO:
    - Importação de `processar_mensagem_task` do Celery.
    - O endpoint /webhook agora despacha a tarefa em background (.delay()) 
      e responde instantaneamente HTTP 200 OK à Evolution API.
    - Fim dos erros ECONNREFUSED e Timeouts!
"""
from __future__ import annotations
import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.infrastructure.settings import settings
from src.infrastructure.redis_client import (
    get_redis_text,
    inicializar_indices,
    redis_ok,
)
from src.infrastructure.observability import obs
from src.agent.core import agent_core
from src.middleware.dev_guard import DevGuard
from src.services.evolution_service import EvolutionService
from src.rag.ingestion import Ingestor
from src.tools import get_tools_ativas

# Importe a task do Celery (V4)
from src.application.tasks import processar_mensagem_task

# =============================================================================
# Logging
# =============================================================================

_NIVEL = logging.DEBUG if settings.DEV_MODE else logging.INFO
logging.basicConfig(
    level=_NIVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

for _nome in [
    "httpcore.http11", "httpcore.connection", "httpx",
    "urllib3.connectionpool",
    "google.auth",                  
    "google.generativeai",          
    "sentence_transformers",        
    "transformers",                 
]:
    logging.getLogger(_nome).setLevel(logging.WARNING)

class _WebhookFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/webhook" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(_WebhookFilter())
logger = logging.getLogger(__name__)

# =============================================================================
# App e singletons
# =============================================================================

app         = FastAPI(title="Bot UEMA", version="4.0")
api_service = EvolutionService()
guard       = DevGuard(get_redis_text())

# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info(
        "🚀 Iniciando Bot UEMA v4 (Assíncrono) | DEV=%s | modelo=%s",
        settings.DEV_MODE,
        settings.GEMINI_MODEL,  
    )

    await asyncio.to_thread(inicializar_indices)
    logger.info("✅ Índices Redis Stack prontos")

    ingestor = Ingestor()
    await asyncio.to_thread(ingestor.ingerir_se_necessario)

    if settings.DEV_MODE:
        await asyncio.to_thread(ingestor.diagnosticar)

    tools = get_tools_ativas()
    await asyncio.to_thread(agent_core.inicializar, tools)

    for tentativa in range(1, 4):
        try:
            await api_service.inicializar()
            break
        except Exception as e:
            if tentativa < 3:
                logger.warning(
                    "⚠️  Evolution API ainda não responde (tentativa %d/3). "
                    "Aguardando 5s... (%s)", tentativa, e,
                )
                await asyncio.sleep(5)
            else:
                logger.error("❌ Evolution API inacessível após 3 tentativas.")

    obs.info("SYSTEM", "Startup", f"DEV={settings.DEV_MODE} | tools={len(tools)} | modelo={settings.GEMINI_MODEL}")
    logger.info("✅ Bot UEMA v4 pronto para receber webhooks!")

# =============================================================================
# Routes — Webhook principal (V4 - Filas)
# =============================================================================

@app.post("/webhook")
async def webhook(request: Request):
    """
    Rececionista V4: Apenas valida e atira para a fila. 
    Não espera pelo Gemini nem pelo RAG.
    """
    payload = await request.json()
    
    # 1. Validação super rápida (spam, dedup)
    is_valid, identity_or_reason = await guard.validar(payload)
    
    if is_valid:
        identity = identity_or_reason
        
        # 2. Despacha a tarefa para o Celery em background (.delay)
        processar_mensagem_task.delay(identity)
        logger.debug("📥 Tarefa enviada para a fila do Celery: %s", identity.get("chat_id"))
    else:
        logger.debug("🛑 DevGuard bloqueou: %s", identity_or_reason)

    # 3. Responde imediatamente à Evolution API
    return JSONResponse(content={"status": "ok", "message": "Recebido"}, status_code=200)

# =============================================================================
# Routes — Health & Observabilidade
# =============================================================================

@app.get("/health")
async def health():
    redis_status = redis_ok()
    agente_ok    = agent_core._inicializado

    return {
        "status":   "ok" if (redis_status and agente_ok) else "starting",
        "version":  "4.0",
        "redis":    redis_status,
        "agente":   agente_ok,
        "modelo":   settings.GEMINI_MODEL,
        "dev_mode": settings.DEV_MODE,
    }

@app.get("/logs")
async def get_logs(limit: int = 20):
    return {"errors": obs.get_recent_errors(limit)}

@app.get("/metrics")
async def get_metrics(limit: int = 50):
    return {"metrics": obs.get_recent_metrics(limit)}

@app.get("/banco/sources")
async def banco_sources():
    ingestor = Ingestor()
    sources  = await asyncio.to_thread(ingestor.diagnosticar)
    return {"sources": list(sources)}

# =============================================================================
# Routes — Debug de Memória
# =============================================================================

@app.get("/fatos/{user_id}")
async def get_fatos(user_id: str):
    from src.memory.long_term_memory import listar_todos_fatos
    fatos = await asyncio.to_thread(listar_todos_fatos, user_id)
    return {"user_id": user_id, "total": len(fatos), "fatos": fatos}

@app.get("/memoria/{session_id}")
async def get_memoria(session_id: str):
    from src.memory.working_memory import get_historico_compactado, get_sinais
    historico = await asyncio.to_thread(get_historico_compactado, session_id)
    sinais    = await asyncio.to_thread(get_sinais, session_id)
    return {
        "session_id":    session_id,
        "turns":         historico.turns_incluidos,
        "total_chars":   historico.total_chars,
        "sinais":        sinais,
        "historico_txt": historico.texto_formatado[:500] + "…" if historico.texto_formatado else "",
    }

@app.delete("/memoria/{session_id}")
async def limpar_memoria(session_id: str):
    from src.memory.working_memory import limpar_sessao
    await asyncio.to_thread(limpar_sessao, session_id)
    return {"status": "ok", "session_id": session_id}