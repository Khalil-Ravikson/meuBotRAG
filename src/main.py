"""
main.py — Bootstrap FastAPI (v3 — Clean Architecture)
======================================================

O QUE MUDOU vs v2:
───────────────────
  REMOVIDO:
    - Referências ao Groq (settings.GROQ_MODEL)
    - Referências ao AgentState e agent_core._agent_with_history
    - Import de redis_ok() que agora vem do redis_client novo
    - DATABASE_URL / pgvector (eliminado da stack)

  ADICIONADO:
    - inicializar_indices() do novo redis_client (cria índices híbridos)
    - GEMINI_MODEL no log de startup
    - /health verifica agent_core._inicializado (atributo do novo core)
    - /fatos/:user_id endpoint para debug de long-term memory
    - /memoria/:session_id endpoint para debug de working memory

  MANTIDO INTACTO:
    - EvolutionService (WhatsApp — não muda)
    - DevGuard (middleware de validação — não muda)
    - handle_webhook (entrada do webhook — não muda)
    - Ingestor (ingestão de PDFs — não muda)
    - get_tools_ativas() (definição de tools — não muda)
    - /logs, /metrics, /banco/sources
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
from src.application.handle_webhook import handle_webhook
from src.rag.ingestion import Ingestor
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

# Silencia loggers muito verbosos
for _nome in [
    "httpcore.http11", "httpcore.connection", "httpx",
    "urllib3.connectionpool",
    "google.auth",                  # SDK Gemini
    "google.generativeai",          # SDK Gemini
    "sentence_transformers",        # BAAI/bge-m3
    "transformers",                 # HuggingFace
]:
    logging.getLogger(_nome).setLevel(logging.WARNING)


class _WebhookFilter(logging.Filter):
    """Suprime logs de acesso ao /webhook no uvicorn (muito verboso)."""
    def filter(self, record: logging.LogRecord) -> bool:
        return "/webhook" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_WebhookFilter())
logger = logging.getLogger(__name__)

# =============================================================================
# App e singletons
# =============================================================================

app         = FastAPI(title="Bot UEMA", version="3.0")
api_service = EvolutionService()
guard       = DevGuard(get_redis_text())  # DevGuard usa cliente texto (sem bytes)

# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info(
        "🚀 Iniciando Bot UEMA v3 | DEV=%s | modelo=%s",
        settings.DEV_MODE,
        settings.GEMINI_MODEL,  # Gemini em vez de Groq
    )

    # ── 1. Inicializa índices Redis Stack ─────────────────────────────────────
    # Cria idx:rag:chunks (BM25 + Vector) e idx:tools (routing semântico)
    # Idempotente: se já existem, não faz nada
    await asyncio.to_thread(inicializar_indices)
    logger.info("✅ Índices Redis Stack prontos")

    # ── 2. Ingestão de PDFs ───────────────────────────────────────────────────
    # Guarda chunks no Redis (substitui o pgvector)
    # ingerir_se_necessario() verifica se já existem chunks antes de re-ingerir
    ingestor = Ingestor()
    await asyncio.to_thread(ingestor.ingerir_se_necessario)

    # ── 3. Diagnóstico em DEV ─────────────────────────────────────────────────
    if settings.DEV_MODE:
        await asyncio.to_thread(ingestor.diagnosticar)

    # ── 4. Inicializa AgentCore com tools ─────────────────────────────────────
    # Regista tools no Redis para roteamento semântico (sem LLM)
    tools = get_tools_ativas()
    await asyncio.to_thread(agent_core.inicializar, tools)

    # ── 5. Configura webhook Evolution API (com retry) ────────────────────────
    # A Evolution API pode demorar a arrancar — tentamos 3× com pausa de 5s
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
                logger.error(
                    "❌ Evolution API inacessível após 3 tentativas. "
                    "O bot vai funcionar mas não consegue enviar mensagens. "
                    "Verifique se o container 'evolution-api' está a correr."
                )

    obs.info("SYSTEM", "Startup", f"DEV={settings.DEV_MODE} | tools={len(tools)} | modelo={settings.GEMINI_MODEL}")
    logger.info("✅ Bot UEMA v3 pronto!")


# =============================================================================
# Routes — Webhook principal
# =============================================================================

@app.post("/webhook")
async def webhook(request: Request):
    payload   = await request.json()
    resultado = await handle_webhook(payload, guard, api_service)
    return JSONResponse(content=resultado)


# =============================================================================
# Routes — Health & Observabilidade
# =============================================================================

@app.get("/health")
async def health():
    """
    Verifica estado de todos os componentes críticos.
    Usado pelo healthcheck do Docker e pelo Chainlit.
    """
    redis_status = redis_ok()
    agente_ok    = agent_core._inicializado  # atributo do novo AgentCore

    return {
        "status":      "ok" if (redis_status and agente_ok) else "degraded",
        "version":     "3.0",
        "redis":       redis_status,
        "agente":      agente_ok,
        "modelo":      settings.GEMINI_MODEL,
        "dev_mode":    settings.DEV_MODE,
        "pgvector":    False,   # Eliminado na v3 — documentado explicitamente
    }


@app.get("/logs")
async def get_logs(limit: int = 20):
    return {"errors": obs.get_recent_errors(limit)}


@app.get("/metrics")
async def get_metrics(limit: int = 50):
    return {"metrics": obs.get_recent_metrics(limit)}


@app.get("/banco/sources")
async def banco_sources():
    """Sources presentes no Redis (substitui o endpoint de pgvector)."""
    ingestor = Ingestor()
    sources  = await asyncio.to_thread(ingestor.diagnosticar)
    return {"sources": list(sources)}


# =============================================================================
# Routes — Debug de Memória (úteis no Chainlit e em testes)
# =============================================================================

@app.get("/fatos/{user_id}")
async def get_fatos(user_id: str):
    """
    Lista todos os fatos long-term de um utilizador.
    Útil para confirmar que a extração de fatos está a funcionar.
    """
    from src.memory.long_term_memory import listar_todos_fatos
    fatos = await asyncio.to_thread(listar_todos_fatos, user_id)
    return {"user_id": user_id, "total": len(fatos), "fatos": fatos}


@app.get("/memoria/{session_id}")
async def get_memoria(session_id: str):
    """
    Mostra o estado actual da working memory de uma sessão.
    Inclui histórico compactado e sinais de sessão.
    """
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
    """Limpa a working memory de uma sessão (para testes e suporte)."""
    from src.memory.working_memory import limpar_sessao
    await asyncio.to_thread(limpar_sessao, session_id)
    return {"status": "ok", "session_id": session_id}