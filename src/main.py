"""
main.py — Bootstrap FastAPI (v6 — Registo Conversacional Integrado)
====================================================================

MUDANÇAS v6 vs v5:
  ADICIONADO:
    - handle_registration() no fluxo do webhook
    - Lógica de "porteiro inteligente": não recusa número novo, guia registo
    - Import de RegistrationService singleton

  FLUXO DO WEBHOOK (actualizado):
    ┌── Evolution API envia mensagem ──┐
    │                                  │
    │  1. DevGuard valida payload      │
    │  2. Busca phone na DB (Pessoas)  │
    │         │                        │
    │    ┌────┴─────┐                  │
    │    │          │                  │
    │  FOUND    NOT FOUND              │
    │    │          │                  │
    │    │    handle_registration()    │
    │    │      ├─ em registo:         │
    │    │      │   → processa passo   │
    │    │      ├─ novo número:        │
    │    │      │   → inicia registo   │
    │    │      └─ registo completo:   │
    │    │          → cria no DB       │
    │    │          → cai no FOUND ↓  │
    │    │                             │
    │  Celery task → bot normal        │
    └──────────────────────────────────┘

NOTA SOBRE DB vs REDIS:
  O porteiro usa PostgreSQL para verificar o cadastro.
  A máquina de estados usa Redis para guardar o progresso temporário.
  Assim, o Redis é apenas estado efémero (TTL 15min) e o Postgres é a fonte
  de verdade permanente.
"""
from __future__ import annotations
import asyncio
import logging

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from src.api.hub import _HUB_HTML
from fastapi.responses import HTMLResponse
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
from src.infrastructure.database import AsyncSessionLocal

# ── Routers ───────────────────────────────────────────────────────────────────
from src.api.monitor     import router as monitor_router
from src.api.router_pessoa import router as pessoa_router
from src.api.eval_dashboard import router as eval_router      # Phase 3
from src.api.admin_portal   import router as admin_router     # Phase 4
from src.api.hub import router as hub_router
# ── Registo Conversacional (Phase 5) ─────────────────────────────────────────
from src.application.handle_registration import handle_registration

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
    title       = "Oráculo UEMA",
    version     = "6.0",
    description = "Assistente Académico UEMA — WhatsApp + RAG + Redis Stack",
)

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:8001", "http://127.0.0.1:8001"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers montados ──────────────────────────────────────────────────────────
app.include_router(monitor_router, prefix="/monitor", tags=["Monitor"])
app.include_router(pessoa_router)
app.include_router(eval_router,   prefix="/eval",    tags=["RAG Eval"])
app.include_router(admin_router,  prefix="/admin",   tags=["Admin Portal"])
app.include_router(hub_router)        # serve GET /
# ── Singletons de serviços ────────────────────────────────────────────────────
api_service = EvolutionService()
guard       = DevGuard(get_redis_text())
sec_guard   = SecurityGuard(get_redis_text(), settings)

# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info(
        "🚀 Oráculo UEMA v6 | DEV=%s | modelo=%s",
        settings.DEV_MODE, settings.GEMINI_MODEL,
    )

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
                logger.warning(
                    "⚠️  Evolution API tentativa %d/3: %s", tentativa, e,
                )
                await asyncio.sleep(5)
            else:
                logger.error("❌ Evolution API inacessível após 3 tentativas.")

    obs.info(
        "SYSTEM", "Startup",
        f"v6 | DEV={settings.DEV_MODE} | tools={len(tools)}",
    )
    logger.info("✅ Oráculo UEMA v6 pronto!")


# =============================================================================
# Webhook principal — com registo conversacional integrado
# =============================================================================

@app.post("/webhook")
async def webhook(request: Request):
    """
    Valida payload Evolution API e despacha para Celery.
    
    NOVO v6: antes de despachar para Celery, verifica se o número está
    cadastrado. Se não está, gere o fluxo de registo directamente (sem Celery)
    para manter a latência baixa durante o processo de registo.
    """
    payload = await request.json()
    is_valid, identity_or_reason = await guard.validar(payload)

    if not is_valid:
        logger.debug("🛑 DevGuard bloqueou: %s", identity_or_reason)
        return JSONResponse({"status": "ok"}, status_code=200)

    identity = identity_or_reason
    chat_id  = identity.get("chat_id", "")
    body     = identity.get("body", "")

    # Normaliza o telefone (remove @s.whatsapp.net, +, espaços)
    raw_phone = chat_id.split("@")[0] if "@" in chat_id else chat_id
    import re as _re
    telefone_limpo = _re.sub(r"\D", "", raw_phone)

    # ── 1. Verifica cadastro no PostgreSQL ────────────────────────────────────
    async with AsyncSessionLocal() as session:
        from src.application.crud_pessoa import buscar_pessoa_por_telefone
        pessoa = await buscar_pessoa_por_telefone(session, telefone=telefone_limpo)

    # ── 2. Número não cadastrado → fluxo de registo ───────────────────────────
    if not pessoa:
        logger.info("❓ Número não cadastrado: %s — redirecionando para registo", telefone_limpo)
        # handle_registration retorna True se tratou a mensagem (registo em curso)
        # retorna False se o registo foi concluído (deve continuar para bot)
        tratado_pelo_registo = await handle_registration(
            chat_id   = chat_id,
            phone     = telefone_limpo,
            body      = body,
            evolution = api_service,
        )
        if tratado_pelo_registo:
            return JSONResponse({"status": "ok"}, status_code=200)
        
        # Registo concluído — re-busca a pessoa recém-criada
        async with AsyncSessionLocal() as session:
            from src.application.crud_pessoa import buscar_pessoa_por_telefone
            pessoa = await buscar_pessoa_por_telefone(session, telefone=telefone_limpo)
        
        if not pessoa:
            # Segurança extra: se ainda não existe (ex: falha de DB), abort
            logger.error("❌ Registo concluído mas Pessoa não encontrada: %s", telefone_limpo)
            return JSONResponse({"status": "ok"}, status_code=200)

    # ── 3. Número cadastrado → enriquece identity e despacha para Celery ──────
    identity["nome_usuario"]  = pessoa.nome
    identity["role_usuario"]  = pessoa.role.value
    identity["email_usuario"] = pessoa.email

    processar_mensagem_task.delay(identity)
    logger.debug("📥 Tarefa enfileirada para %s: %s", pessoa.nome, chat_id)

    return JSONResponse({"status": "ok"}, status_code=200)


# =============================================================================
# Health & Observabilidade
# =============================================================================

@app.get("/health", tags=["Sistema"])
async def health():
    """Estado de saúde agregado de todos os subsistemas."""
    redis_status = redis_ok()

    # Verifica Postgres
    try:
        async with AsyncSessionLocal() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        postgres_status = True
    except Exception:
        postgres_status = False

    tudo_ok = redis_status and postgres_status and agent_core._inicializado

    return {
        "status":   "ok" if tudo_ok else "degraded",
        "version":  "6.0",
        "redis":    redis_status,
        "postgres": postgres_status,
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
# Admin REST
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