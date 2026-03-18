"""
api/monitor.py — Router exclusivo do Dashboard de Monitoramento
================================================================

Montado em main.py com prefix="/monitor":
  GET  /monitor            → dashboard HTML (Jinja2)
  GET  /monitor/data       → JSON dos logs (para o JS fazer polling)
  GET  /monitor/{user_id}  → dados de um utilizador específico
  POST /monitor/reset      → limpa logs de monitoramento (admin)

Por que APIRouter separado?
  - main.py fica limpo: só bootstrap, webhook e health
  - O router pode ser versionado, testado e desactivado independentemente
  - O template HTML e o CSS/JS são servidos por StaticFiles — sem string gigante no Python
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.infrastructure.redis_client import get_redis_text
from src.infrastructure.settings import settings

logger    = logging.getLogger(__name__)
router    = APIRouter()
templates = Jinja2Templates(directory="templates")


# =============================================================================
# Dashboard HTML
# =============================================================================

@router.get("", response_class=HTMLResponse)
async def dashboard(request: Request, limit: int = 100):
    """
    Dashboard principal — renderiza templates/monitor/dashboard.html via Jinja2.
    Os dados reais vêm do endpoint /monitor/data via fetch() no JS.
    """
    contexto = {
        "request":    request,
        "modelo":     settings.GEMINI_MODEL,
        "redis_url":  settings.REDIS_URL,
        "dev_mode":   settings.DEV_MODE,
        "version":    "5.0",
        "updated_at": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    }
    return templates.TemplateResponse("monitor/dashboard.html", contexto)


# =============================================================================
# JSON API — para o JS fazer polling a cada 30s
# =============================================================================

@router.get("/data")
async def monitor_data(limit: int = 100):
    """
    Retorna os dados do monitor em JSON estruturado.
    O dashboard.html faz fetch("/monitor/data") a cada 30 segundos.
    """
    r = get_redis_text()

    try:
        raw_logs = r.lrange("monitor:logs", 0, limit - 1)
        logs     = [json.loads(l) for l in raw_logs]
    except Exception:
        logs = []

    # Métricas agregadas
    total_msgs   = len(logs)
    total_tokens = sum(l.get("tokens_total", 0) for l in logs)
    avg_lat      = int(sum(l.get("latencia_ms", 0) for l in logs) / max(total_msgs, 1))

    # Distribuição por nível
    niveis: dict[str, int] = {}
    for l in logs:
        n = l.get("nivel", "GUEST")
        niveis[n] = niveis.get(n, 0) + 1

    # Distribuição por rota
    rotas: dict[str, int] = {}
    for l in logs:
        r_ = l.get("rota", "GERAL")
        rotas[r_] = rotas.get(r_, 0) + 1

    # Erros recentes
    try:
        raw_errs = r.lrange("system_logs:error", 0, 9)
        erros    = [json.loads(e) for e in raw_errs]
    except Exception:
        erros = []

    return {
        "updated_at":   datetime.now().isoformat(),
        "total_msgs":   total_msgs,
        "total_tokens": total_tokens,
        "avg_latencia": avg_lat,
        "niveis":       niveis,
        "rotas":        rotas,
        "logs":         logs[:50],    # últimas 50 entradas para a tabela
        "erros":        erros,
    }


@router.get("/{user_id}")
async def monitor_usuario(user_id: str):
    """Dados de monitoramento de um utilizador específico."""
    r    = get_redis_text()
    try:
        dados = r.hgetall(f"monitor:user:{user_id}")
    except Exception:
        dados = {}

    return {
        "user_id":       user_id,
        "total_msgs":    int(dados.get("total_msgs", 0)),
        "total_tokens":  int(dados.get("total_tokens", 0)),
        "avg_latencia":  (
            int(dados.get("total_latencia", 0)) //
            max(int(dados.get("total_msgs", 1)), 1)
        ),
        "ultima_msg":    dados.get("ultima_msg", ""),
        "nivel":         dados.get("nivel", "GUEST"),
    }


@router.post("/reset")
async def reset_logs(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    """Limpa todos os logs de monitoramento (apenas admin)."""
    if not settings.ADMIN_API_KEY or x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Acesso negado.")
    r = get_redis_text()
    r.delete("monitor:logs")
    return {"status": "ok", "mensagem": "Logs de monitoramento limpos."}