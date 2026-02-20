"""
================================================================================
main.py — v3: Logs limpos, endpoint /logs, LogService integrado
================================================================================

MELHORIAS v3:
  1. Logging formatado de forma limpa: só o essencial no terminal
  2. Suprime loggers barulhentos (httpcore, httpx, urllib3, groq debug)
  3. LogService usado no startup para registrar inicialização
  4. Endpoint /logs para visualizar erros recentes sem abrir o terminal
  5. diagnose_banco() chamado automaticamente no startup (só em DEV_MODE)
================================================================================
"""

import asyncio
import logging
import redis

from fastapi import FastAPI, Request
from src.config import settings
from src.services.rag_service import RagService
from src.services.waha_service import WahaService
from src.services.menu_service import MenuService
from src.services.router_service import RouterService
from src.handlers.webhook_handler import WebhookHandler
from src.middleware.dev_guard import DevGuard
from src.services.logger_service import LogService

# =============================================================================
# Configuração de logging limpo
# =============================================================================

# Nível base: DEBUG em dev, INFO em prod
_NIVEL_BASE = logging.DEBUG if getattr(settings, "DEV_MODE", True) else logging.INFO

logging.basicConfig(
    level=_NIVEL_BASE,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── Suprime loggers muito verbosos que poluem o terminal ────────────────────
# httpcore e httpx: mostram cada header HTTP — úteis só para debug de rede
# urllib3: conexões internas do LangChain Smith
# groq._base_client: request options gigantes com todo o payload
_LOGGERS_SILENCIOSOS = [
    "httpcore.http11",
    "httpcore.connection",
    "httpx",
    "urllib3.connectionpool",
    "groq._base_client",
]
for nome in _LOGGERS_SILENCIOSOS:
    logging.getLogger(nome).setLevel(logging.WARNING)

# Suprime log de acesso do Uvicorn para /webhook (muito repetitivo)
class _EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/webhook" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(_EndpointFilter())

logger = logging.getLogger(__name__)

# =============================================================================
# Bootstrap dos serviços
# =============================================================================

app    = FastAPI(title="Bot UEMA", version="5.0")
rag    = RagService()
waha   = WahaService()
menu   = MenuService()
router = RouterService()
log    = LogService()

# Redis
try:
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    r.ping()
    print("✅ Redis conectado!")
except redis.ConnectionError as e:
    raise RuntimeError(f"❌ Redis offline: {e}")

guard   = DevGuard(r)
handler = WebhookHandler(
    rag_service    = rag,
    waha_service   = waha,
    menu_service   = menu,
    router_service = router,
)

# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup():
    print(f"🚀 Bot iniciado | DEV: {guard.dev_mode} | Whitelist: {guard.dev_whitelist}")

    # Inicializa o RAG em thread separada para não bloquear o event loop
    await asyncio.to_thread(rag.inicializar)

    # Em modo dev, executa diagnóstico do banco para verificar sources
    if guard.dev_mode:
        await asyncio.to_thread(rag.diagnose_banco)

    await waha.inicializar()
    log.log_info("SYSTEM", "Startup completo", f"dev_mode={guard.dev_mode}")

# =============================================================================
# Endpoints
# =============================================================================

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    aprovado, resultado = await guard.validar(data)
    if not aprovado:
        logger.debug("🚫 Bloqueado: %s", resultado)
        return {"status": resultado}

    await handler.processar(resultado)
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Verifica saúde dos serviços principais."""
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status":   "ok" if redis_ok else "degraded",
        "redis":    "online" if redis_ok else "offline",
        "agente":   "pronto" if rag.agent_with_history else "inicializando",
        "dev_mode": guard.dev_mode,
    }


@app.get("/logs")
async def get_logs(limit: int = 20):
    """
    Retorna os últimos erros registrados pelo LogService.
    Útil para debug sem precisar abrir o terminal do Docker.

    Acesse: http://localhost:8000/logs
    """
    erros = log.get_recent_errors(limit)
    return {
        "total":  len(erros),
        "errors": erros,
    }


@app.get("/banco/sources")
async def banco_sources():
    """
    Mostra quais 'source' estão presentes no banco vetorial.
    Use para verificar se os nomes dos PDFs batem com PDF_CONFIG.

    Acesse: http://localhost:8000/banco/sources
    """
    sources = await asyncio.to_thread(rag.diagnose_banco)
    return {
        "sources_no_banco": list(sources),
        "sources_esperados": list(rag.PDF_CONFIG.keys()) if hasattr(rag, "PDF_CONFIG") else [],
    }