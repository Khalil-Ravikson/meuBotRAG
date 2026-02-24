"""
main.py — Bootstrap FastAPI (arquitetura nova)
==============================================
Substitui o main.py antigo.

MUDANÇAS vs versão anterior:
  - Imports da nova arquitetura (infrastructure/ em vez de services/)
  - Startup usa Ingestor() + agent_core.inicializar(tools)
  - WahaService.inicializar() chamado no startup
  - Endpoint /metrics adicionado
  - Sem instanciação de Redis direto aqui (usa get_redis() de infrastructure/)
"""
from __future__ import annotations
import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.infrastructure.settings import settings
from src.infrastructure.redis_client import get_redis, redis_ok
from src.infrastructure.observability import obs
from src.agent.core import agent_core
from src.middleware.dev_guard import DevGuard
from src.services.waha_service import WahaService
from src.application.handle_webhook import handle_webhook
from src.rag.ingestor import Ingestor
from src.tools import get_tools_ativas

# =============================================================================
# Logging
# =============================================================================

_NIVEL = logging.DEBUG if settings.DEV_MODE else logging.INFO
logging.basicConfig(
    level=_NIVEL,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
for nome_logger in [
    "httpcore.http11", "httpcore.connection", "httpx",
    "urllib3.connectionpool", "groq._base_client",
]:
    logging.getLogger(nome_logger).setLevel(logging.WARNING)


class _WebhookFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/webhook" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_WebhookFilter())
logger = logging.getLogger(__name__)

# =============================================================================
# App e singletons
# =============================================================================

app   = FastAPI(title="Bot UEMA", version="2.0")
waha  = WahaService()
guard = DevGuard(get_redis())

# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info("🚀 Iniciando Bot UEMA | DEV=%s | modelo=%s",
                settings.DEV_MODE, settings.GROQ_MODEL)

    # 1. Ingestão de PDFs (em thread separada — não bloqueia o event loop)
    ingestor = Ingestor()
    await asyncio.to_thread(ingestor.ingerir_se_necessario)

    # 2. Diagnóstico em DEV (confirma sources no banco)
    if settings.DEV_MODE:
        await asyncio.to_thread(ingestor.diagnosticar)

    # 3. Inicializa o agente com as tools ativas
    tools = get_tools_ativas()
    await asyncio.to_thread(agent_core.inicializar, tools)

    # 4. Conecta e configura o webhook do WAHA
    await waha.inicializar()

    obs.info("SYSTEM", "Startup", f"DEV={settings.DEV_MODE} | tools={len(tools)}")
    logger.info("✅ Bot pronto!")

# =============================================================================
# Routes
# =============================================================================

@app.post("/webhook")
async def webhook(request: Request):
    payload   = await request.json()
    resultado = await handle_webhook(payload, guard, waha)
    return JSONResponse(content=resultado)


@app.get("/health")
async def health():
    from src.api.schemas import HealthResponse
    return HealthResponse(
        status   = "ok",
        redis    = redis_ok(),
        agente   = agent_core._agent_with_history is not None,
        dev_mode = settings.DEV_MODE,
    )


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