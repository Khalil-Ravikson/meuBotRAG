"""
main.py — Bootstrap FastAPI (v5 — Modularizado)
================================================

MUDANÇAS v5 vs v4:
  MODULARIZADO:
    - Monitor movido para src/api/monitor.py (APIRouter dedicado)
    - Static files: Jinja2 + CSS/JS isolados em static/ e templates/
    - main.py agora é puro bootstrap: startup + webhook + health

  MANTIDO:
    - DevGuard + SecurityGuard + Celery task queue
    - Todos os endpoints de memória/fatos/banco

ESTRUTURA:
  main.py                         ← bootstrap (este ficheiro)
  src/api/monitor.py              ← router do dashboard
  static/css/monitor.css          ← estilos isolados
  static/js/monitor.js            ← JS isolado
  templates/monitor/dashboard.html ← Jinja2 template
"""
from __future__ import annotations
import asyncio
import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.infrastructure.settings import settings
from src.infrastructure.redis_client import (
    get_redis_text,
    inicializar_indices,
    redis_ok,
)
from src.infrastructure.observability import obs
from src.infrastructure.semantic_cache import init_cache_index
from src.agent.core import agent_core
from src.middleware.dev_guard import DevGuard
from src.middleware.security_guard import SecurityGuard
from src.services.evolution_service import EvolutionService
from src.rag.ingestion import Ingestor
from src.tools import get_tools_ativas
from src.application.tasks import processar_mensagem_task

# ── Router do Monitor ─────────────────────────────────────────────────────────
from src.api.monitor import router as monitor_router

# =============================================================================
# Logging
# =============================================================================

_NIVEL = logging.DEBUG if settings.DEV_MODE else logging.INFO
logging.basicConfig(
    level  = _NIVEL,
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt= "%H:%M:%S",
)
for _nome in ["httpcore", "httpx", "urllib3",
              "google.auth", "google.generativeai",
              "sentence_transformers", "transformers"]:
    logging.getLogger(_nome).setLevel(logging.WARNING)

class _WebhookFilter(logging.Filter):
    def filter(self, r): return "/webhook" not in r.getMessage()

logging.getLogger("uvicorn.access").addFilter(_WebhookFilter())
logger = logging.getLogger(__name__)

# =============================================================================
# App
# =============================================================================

app = FastAPI(
    title       = "Bot UEMA",
    version     = "5.0",
    description = "Assistente Académico UEMA — WhatsApp + RAG + Redis Stack",
)

# Static files e templates (para o monitor)
app.mount("/static", StaticFiles(directory="/app/static"), name="static")
templates = Jinja2Templates(directory="/app/templates")

# CORS — só permite localhost em dev
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:8001", "http://127.0.0.1:8001"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Inclui o router do monitor com prefix /monitor
app.include_router(monitor_router, prefix="/monitor", tags=["Monitor"])

# Singletons
api_service = EvolutionService()
guard       = DevGuard(get_redis_text())
sec_guard   = SecurityGuard(get_redis_text(), settings)

# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info("🚀 Bot UEMA v5 | DEV=%s | modelo=%s", settings.DEV_MODE, settings.GEMINI_MODEL)

    await asyncio.to_thread(inicializar_indices)
    await asyncio.to_thread(init_cache_index)
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
                logger.warning("⚠️  Evolution API tentativa %d/3: %s", tentativa, e)
                await asyncio.sleep(5)
            else:
                logger.error("❌ Evolution API inacessível após 3 tentativas.")

    obs.info("SYSTEM", "Startup", f"v5 | DEV={settings.DEV_MODE} | tools={len(tools)}")
    logger.info("✅ Bot UEMA v5 pronto!")

# =============================================================================
# Webhook principal
# =============================================================================

@app.post("/webhook")
async def webhook(request: Request):
    """Valida payload Evolution API e despacha para Celery."""
    x_api_key = request.headers.get("apikey", "")
    if settings.WEBHOOK_SECRET and x_api_key != settings.WEBHOOK_SECRET:
        return JSONResponse({"status": "unauthorized"}, status_code=401)

    payload = await request.json()
    is_valid, identity_or_reason = await guard.validar(payload)

    if is_valid:
        processar_mensagem_task.delay(identity_or_reason)
        logger.debug("📥 Tarefa enfileirada: %s", identity_or_reason.get("chat_id"))
    else:
        logger.debug("🛑 DevGuard bloqueou: %s", identity_or_reason)

    return JSONResponse({"status": "ok"}, status_code=200)

# =============================================================================
# Health & Observabilidade (mantidos em main.py — simples e críticos)
# =============================================================================

@app.get("/health", tags=["Sistema"])
async def health():
    return {
        "status":   "ok" if (redis_ok() and agent_core._inicializado) else "starting",
        "version":  "5.0",
        "redis":    redis_ok(),
        "agente":   agent_core._inicializado,
        "modelo":   settings.GEMINI_MODEL,
        "dev_mode": settings.DEV_MODE,
    }

@app.get("/logs", tags=["Sistema"])
async def get_logs(limit: int = 20):
    return {"errors": obs.get_recent_errors(limit)}

@app.get("/metrics", tags=["Sistema"])
async def get_metrics(limit: int = 50):
    return {"metrics": obs.get_recent_metrics(limit)}

# =============================================================================
# Memória & Debug
# =============================================================================

@app.get("/banco/sources", tags=["Debug"])
async def banco_sources():
    ingestor = Ingestor()
    sources  = await asyncio.to_thread(ingestor.diagnosticar)
    return {"sources": list(sources)}

@app.get("/fatos/{user_id}", tags=["Debug"])
async def get_fatos(user_id: str):
    from src.memory.long_term_memory import listar_todos_fatos
    fatos = await asyncio.to_thread(listar_todos_fatos, user_id)
    return {"user_id": user_id, "total": len(fatos), "fatos": fatos}

@app.get("/memoria/{session_id}", tags=["Debug"])
async def get_memoria(session_id: str):
    from src.memory.working_memory import get_historico_compactado, get_sinais
    historico = await asyncio.to_thread(get_historico_compactado, session_id)
    sinais    = await asyncio.to_thread(get_sinais, session_id)
    return {
        "session_id":    session_id,
        "turns":         historico.turns_incluidos,
        "sinais":        sinais,
        "historico_txt": historico.texto_formatado[:500],
    }

@app.delete("/memoria/{session_id}", tags=["Debug"])
async def limpar_memoria(session_id: str):
    from src.memory.working_memory import limpar_sessao
    await asyncio.to_thread(limpar_sessao, session_id)
    return {"status": "ok"}

# =============================================================================
# Admin REST (alternativa ao WhatsApp)
# =============================================================================

@app.post("/admin/ingerir", tags=["Admin"])
async def admin_ingerir(
    request: Request,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    if not settings.ADMIN_API_KEY or x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Acesso negado.")
    body          = await request.json()
    nome_ficheiro = body.get("ficheiro", "")
    if not nome_ficheiro:
        raise HTTPException(status_code=400, detail="Campo 'ficheiro' obrigatório.")
    import os
    caminho = os.path.join(settings.DATA_DIR, nome_ficheiro)
    if not os.path.exists(caminho):
        raise HTTPException(status_code=404, detail=f"'{nome_ficheiro}' não encontrado.")
    ingestor = Ingestor()
    chunks   = await asyncio.to_thread(ingestor._ingerir_ficheiro, caminho)
    return {"status": "ok", "ficheiro": nome_ficheiro, "chunks": chunks}

@app.delete("/cache/{rota}", tags=["Admin"])
async def limpar_cache_rota(
    rota: str,
    x_admin_key: str = Header(None, alias="X-Admin-Key"),
):
    if not settings.ADMIN_API_KEY or x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Acesso negado.")
    from src.infrastructure.semantic_cache import invalidar_cache_rota
    n = invalidar_cache_rota(rota.upper())
    return {"status": "ok", "rota": rota, "removidas": n}