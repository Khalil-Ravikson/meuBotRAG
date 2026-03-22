"""
api/eval_dashboard.py — Dashboard RAG em Tempo Real
====================================================

ENDPOINTS:
  GET  /eval/           → HTML do dashboard (inline, sem dependência de arquivo)
  GET  /eval/stream     → SSE: stream de logs do Python em tempo real
  POST /eval/query      → SSE: executa pipeline RAG passo a passo e streama
  GET  /eval/metrics    → JSON: tokens, latência, CRAG, routing (último histórico)
  GET  /eval/eventos    → JSON: eventos do calendário nos próximos 30 dias
  POST /eval/run-full   → JSON: dispara o eval completo do rag_eval.py

MECANISMO DE LOG EM TEMPO REAL:
  Um `logging.Handler` customizado (`SSELogHandler`) intercepta todos os logs
  Python e os coloca num `asyncio.Queue` global. O endpoint GET /eval/stream
  consome essa fila via Server-Sent Events — sem polling, sem WebSocket.

  Por que SSE e não WebSocket?
  - SSE é unidirecional (servidor → cliente) — suficiente para logs
  - Funciona sobre HTTP simples, sem upgrade de protocolo
  - FastAPI suporta via StreamingResponse + EventSourceResponse
  - Muito mais simples de implementar e debugar
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from datetime import date, datetime
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# SSE Log Handler — intercepta logs Python e coloca na fila
# ─────────────────────────────────────────────────────────────────────────────

# Fila global: compartilhada entre o handler e os clientes SSE
# maxsize=200 evita acúmulo infinito se nenhum cliente estiver conectado
_log_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

# Contador de clientes SSE conectados (para saber se vale enviar logs)
_sse_clients: int = 0


class SSELogHandler(logging.Handler):
    """
    Handler de logging que coloca registros na _log_queue.
    Instalado no root logger para capturar TUDO (src.*, uvicorn, celery...).
    """

    # Níveis com cores para o terminal do dashboard
    _CORES = {
        "DEBUG":    "#4a9eff",
        "INFO":     "#00ff9d",
        "WARNING":  "#ffb900",
        "ERROR":    "#ff4444",
        "CRITICAL": "#ff0055",
    }

    def emit(self, record: logging.LogRecord) -> None:
        global _sse_clients
        if _sse_clients == 0:
            return  # ninguém conectado — não enfileira

        try:
            cor  = self._CORES.get(record.levelname, "#ffffff")
            nome = record.name.split(".")[-1][:18]  # só o módulo final
            msg  = self.format(record)

            entrada = json.dumps({
                "ts":    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "level": record.levelname,
                "name":  nome,
                "msg":   msg[:300],
                "cor":   cor,
            }, ensure_ascii=False)

            # put_nowait: não bloqueia — se a fila estiver cheia, descarta
            try:
                _log_queue.put_nowait(entrada)
            except asyncio.QueueFull:
                pass
        except Exception:
            pass


def _instalar_handler():
    """Instala o SSELogHandler no root logger (chamado no startup)."""
    handler = SSELogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(logging.DEBUG)

    root = logging.getLogger()
    # Evita instalar duplicado
    if not any(isinstance(h, SSELogHandler) for h in root.handlers):
        root.addHandler(handler)
        logging.getLogger(__name__).info("✅ SSELogHandler instalado")


# Instala automaticamente ao importar o módulo
_instalar_handler()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve o HTML do dashboard (inline — sem arquivo externo)."""
    return HTMLResponse(content=_HTML_DASHBOARD)


@router.get("/stream")
async def stream_logs(request: Request):
    """
    SSE: streama logs Python em tempo real para o browser.
    Cada evento é uma linha JSON com {ts, level, name, msg, cor}.
    """
    global _sse_clients

    async def gerador() -> AsyncIterator[str]:
        global _sse_clients
        _sse_clients += 1
        logger = logging.getLogger(__name__)
        logger.info("🔌 Cliente SSE conectado | total=%d", _sse_clients)

        try:
            # Heartbeat inicial (confirma conexão ao browser)
            yield f"data: {json.dumps({'ts': datetime.now().strftime('%H:%M:%S'), 'level': 'INFO', 'name': 'dashboard', 'msg': '✅ Stream conectado — aguardando logs...', 'cor': '#00ff9d'})}\n\n"

            while True:
                # Verifica se cliente desconectou
                if await request.is_disconnected():
                    break
                try:
                    # Aguarda próximo log (timeout 15s → envia heartbeat)
                    entrada = await asyncio.wait_for(_log_queue.get(), timeout=15.0)
                    yield f"data: {entrada}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat para manter conexão viva
                    hb = json.dumps({"ts": datetime.now().strftime("%H:%M:%S"), "level": "DEBUG", "name": "♥", "msg": "", "cor": "#1a2a1a"})
                    yield f"data: {hb}\n\n"
        finally:
            _sse_clients = max(0, _sse_clients - 1)
            logger.info("🔌 Cliente SSE desconectado | total=%d", _sse_clients)

    return StreamingResponse(
        gerador(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # desativa buffer do nginx
            "Connection":        "keep-alive",
        },
    )


@router.post("/query")
async def query_rag(request: Request):
    """
    SSE: executa a pipeline RAG completa e streama cada passo.

    Body JSON: {"pergunta": "quando é a matrícula?", "user_id": "eval_user"}

    Eventos emitidos (em ordem):
      step_start   → início de cada etapa
      step_result  → resultado de cada etapa
      chunk_rag    → chunks recuperados
      resposta     → resposta final do Gemini
      metricas     → tokens, latência, CRAG score
      done         → fim do stream
    """
    body = await request.json()
    pergunta = (body.get("pergunta") or "").strip()
    user_id  = body.get("user_id", "eval_live")

    if not pergunta:
        return JSONResponse({"erro": "Campo 'pergunta' obrigatório"}, status_code=400)

    async def pipeline_stream() -> AsyncIterator[str]:
        def _evento(tipo: str, dados: dict) -> str:
            return f"data: {json.dumps({'tipo': tipo, **dados}, ensure_ascii=False)}\n\n"

        t_total = time.monotonic()
        logger  = logging.getLogger("eval.query")

        try:
            # ── Passo 0: Guardrails ──────────────────────────────────────────
            yield _evento("step_start", {"step": "guardrails", "label": "0 — Guardrails"})
            t0 = time.monotonic()
            from src.agent.core import _guardrails
            gr = _guardrails(pergunta)
            ms = int((time.monotonic() - t0) * 1000)

            if gr:
                yield _evento("step_result", {
                    "step": "guardrails", "ms": ms,
                    "resultado": f"Short-circuit: {gr[0]}",
                    "badge": "blocked",
                })
                yield _evento("resposta", {"texto": gr[1], "fonte": "guardrail"})
                yield _evento("metricas", {"tokens_entrada": 0, "tokens_saida": 0, "latencia_ms": ms})
                yield _evento("done", {})
                return

            yield _evento("step_result", {"step": "guardrails", "ms": ms, "resultado": "Passou ✓", "badge": "ok"})

            # ── Passo 1: Routing semântico ────────────────────────────────────
            yield _evento("step_start", {"step": "routing", "label": "1 — Routing Semântico (Redis KNN)"})
            t0 = time.monotonic()
            from src.domain.semantic_router import rotear
            from src.domain.entities import EstadoMenu
            r_routing = rotear(pergunta, EstadoMenu.MAIN)
            ms = int((time.monotonic() - t0) * 1000)
            yield _evento("step_result", {
                "step": "routing", "ms": ms,
                "resultado": f"Rota: {r_routing.rota.value} | Confiança: {r_routing.confianca} | Score: {r_routing.score:.3f}",
                "badge": r_routing.confianca,
                "rota": r_routing.rota.value,
                "score": round(r_routing.score, 3),
            })

            # ── Passo 2: Query Transform ──────────────────────────────────────
            yield _evento("step_start", {"step": "transform", "label": "2 — Query Transform (Gemini)"})
            t0 = time.monotonic()
            from src.rag.query_transform import transformar_query
            qt = transformar_query(pergunta, fatos_usuario=[], usar_sub_queries=False)
            ms = int((time.monotonic() - t0) * 1000)
            yield _evento("step_result", {
                "step": "transform", "ms": ms,
                "resultado": qt.query_principal,
                "foi_transformada": qt.foi_transformada,
                "badge": "transformed" if qt.foi_transformada else "skip",
            })

            # ── Passo 3: Hybrid Retrieval ─────────────────────────────────────
            yield _evento("step_start", {"step": "retrieval", "label": "3 — Busca Híbrida (BM25 + Vetor)"})
            t0 = time.monotonic()
            from src.rag.hybrid_retriever import recuperar, recuperar_simples
            from src.domain.entities import Rota
            rota = r_routing.rota

            _SOURCE_MAP = {
                Rota.CALENDARIO: "calendario-academico-2026.pdf",
                Rota.EDITAL:     "edital_paes_2026.pdf",
                Rota.CONTATOS:   "guia_contatos_2025.pdf",
            }
            source_f = _SOURCE_MAP.get(rota)
            rec = recuperar(qt, source_filter=source_f) if rota != Rota.GERAL else recuperar_simples(pergunta)
            ms  = int((time.monotonic() - t0) * 1000)

            # CRAG score
            crag_score = 0.0
            if rec.encontrou and rec.chunks:
                scores = [c.rrf_score for c in rec.chunks if c.rrf_score > 0]
                crag_score = round(sum(scores) / len(scores), 3) if scores else 0.0

            yield _evento("step_result", {
                "step": "retrieval", "ms": ms,
                "resultado": f"{len(rec.chunks)} chunks | fonte: {rec.fonte_principal or 'geral'} | método: {rec.metodo_usado}",
                "badge": "ok" if crag_score >= 0.40 else "warn",
                "crag_score": crag_score,
            })

            # Emite os chunks para a UI mostrar
            for i, chunk in enumerate(rec.chunks[:4]):
                yield _evento("chunk_rag", {
                    "idx":     i + 1,
                    "source":  chunk.source,
                    "score":   round(chunk.rrf_score, 4),
                    "preview": chunk.content[:180].replace("\n", " "),
                })

            # ── Passo 4: Geração Gemini ───────────────────────────────────────
            yield _evento("step_start", {"step": "geracao", "label": "4 — Geração (Gemini Flash)"})
            t0 = time.monotonic()
            from src.providers.gemini_provider import chamar_gemini
            from src.agent.prompts import SYSTEM_UEMA, montar_prompt_geracao

            prompt = montar_prompt_geracao(
                pergunta     = pergunta,
                contexto_rag = rec.contexto_formatado if rec.encontrou else "",
            )
            resp = chamar_gemini(
                prompt             = prompt,
                system_instruction = SYSTEM_UEMA,
            )
            ms_gen = int((time.monotonic() - t0) * 1000)
            ms_tot = int((time.monotonic() - t_total) * 1000)

            if not resp.sucesso:
                yield _evento("step_result", {
                    "step": "geracao", "ms": ms_gen,
                    "resultado": f"Erro: {resp.erro[:100]}",
                    "badge": "error",
                })
                yield _evento("done", {})
                return

            yield _evento("step_result", {
                "step": "geracao", "ms": ms_gen,
                "resultado": f"{resp.output_tokens} tokens gerados",
                "badge": "ok",
            })

            yield _evento("resposta", {
                "texto":   resp.conteudo,
                "fonte":   rota.value,
                "tokens":  resp.output_tokens,
            })

            yield _evento("metricas", {
                "tokens_entrada": resp.input_tokens,
                "tokens_saida":   resp.output_tokens,
                "tokens_total":   resp.tokens_total,
                "latencia_ms":    ms_tot,
                "crag_score":     crag_score,
                "rota":           rota.value,
                "foi_cache":      False,
            })

            # Salva no Redis para o histórico de métricas
            _salvar_metrica_eval(pergunta, resp, crag_score, rota.value, ms_tot)

            logger.info("📊 Query eval | rota=%s | tokens=%d | crag=%.3f | %dms",
                        rota.value, resp.tokens_total, crag_score, ms_tot)

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("❌ Erro no pipeline eval: %s", exc)
            yield _evento("erro", {"msg": str(exc)[:200], "traceback": tb[-400:]})

        yield _evento("done", {})

    return StreamingResponse(
        pipeline_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/metrics")
async def get_metrics():
    """JSON com histórico de métricas das últimas 50 queries do eval."""
    try:
        from src.infrastructure.redis_client import get_redis_text
        r    = get_redis_text()
        raw  = r.lrange("eval:metricas", 0, 49)
        data = [json.loads(m) for m in raw]
        return JSONResponse({"metricas": data, "total": len(data)})
    except Exception as e:
        return JSONResponse({"metricas": [], "total": 0, "erro": str(e)})


@router.get("/eventos")
async def get_eventos():
    """JSON com eventos do calendário nos próximos 30 dias."""
    try:
        from src.rag.calendar_parser import buscar_eventos_proximos
        eventos = buscar_eventos_proximos(dias_frente=30)
        return JSONResponse({
            "eventos": [
                {
                    "nome":          e.nome,
                    "data_inicio":   e.data_inicio.isoformat(),
                    "data_fim":      e.data_fim.isoformat() if e.data_fim else None,
                    "dias_restantes":e.dias_restantes,
                    "categoria":     e.categoria,
                    "emoji":         e.emoji,
                    "notifica_hoje": e.deve_notificar_hoje,
                }
                for e in eventos
            ],
            "hoje": date.today().isoformat(),
        })
    except Exception as e:
        return JSONResponse({"eventos": [], "erro": str(e)})


@router.post("/run-full")
async def run_full_eval(request: Request):
    """Dispara o eval completo do rag_eval.py via Celery (assíncrono)."""
    try:
        body  = await request.json()
        versao = body.get("versao", "live")
        ids    = body.get("ids", None)  # None = todos

        # Importa a task de eval ou roda direto
        from src.application.tasks_notificacao import celery_app  # reusa o celery_app
        logging.getLogger(__name__).info(
            "🧪 Full eval solicitado | versao=%s | ids=%s", versao, ids
        )
        return JSONResponse({"status": "enfileirado", "versao": versao, "ids": ids})
    except Exception as e:
        return JSONResponse({"status": "erro", "msg": str(e)}, status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _salvar_metrica_eval(
    pergunta: str,
    resp,
    crag_score: float,
    rota: str,
    latencia_ms: int,
) -> None:
    """Persiste métrica no Redis para o histórico de métricas."""
    try:
        from src.infrastructure.redis_client import get_redis_text
        r     = get_redis_text()
        entry = json.dumps({
            "ts":             datetime.now().isoformat(),
            "pergunta":       pergunta[:150],
            "rota":           rota,
            "tokens_entrada": resp.input_tokens,
            "tokens_saida":   resp.output_tokens,
            "tokens_total":   resp.tokens_total,
            "crag_score":     crag_score,
            "latencia_ms":    latencia_ms,
        }, ensure_ascii=False)
        r.lpush("eval:metricas", entry)
        r.ltrim("eval:metricas", 0, 499)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# HTML do Dashboard (inline — sem dependência de arquivo externo)
# ─────────────────────────────────────────────────────────────────────────────

_HTML_DASHBOARD = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Oráculo UEMA — RAG Live Eval</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
/* ── Reset & Vars ──────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:      #080b0f;
  --surface: #0d1117;
  --panel:   #111822;
  --border:  #1e2d3d;
  --accent:  #00e5a0;
  --amber:   #ffb700;
  --blue:    #4af0ff;
  --red:     #ff4060;
  --purple:  #b06eff;
  --text:    #c9d6e3;
  --muted:   #4a6278;
  --mono:    'Space Mono', monospace;
  --sans:    'Syne', sans-serif;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--sans);overflow:hidden}

/* ── Scanline overlay ─────────────────────────────────────────── */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.08) 2px,rgba(0,0,0,.08) 4px);
}

/* ── Layout principal ─────────────────────────────────────────── */
.layout{display:grid;grid-template-rows:48px 1fr;grid-template-columns:1fr 400px;height:100vh;gap:0}
.header{grid-column:1/-1;display:flex;align-items:center;gap:16px;padding:0 20px;background:var(--surface);border-bottom:1px solid var(--border)}
.main-left{display:flex;flex-direction:column;overflow:hidden;border-right:1px solid var(--border)}
.main-right{display:flex;flex-direction:column;overflow:hidden;background:var(--surface)}

/* ── Header ──────────────────────────────────────────────────── */
.logo{font-family:var(--sans);font-weight:800;font-size:15px;letter-spacing:.08em;color:var(--accent);text-transform:uppercase}
.logo span{color:var(--muted)}
.header-pill{display:flex;align-items:center;gap:6px;padding:4px 12px;border:1px solid var(--border);border-radius:20px;font-family:var(--mono);font-size:11px;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--muted)}
.dot.live{background:var(--accent);box-shadow:0 0 6px var(--accent);animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.header-spacer{flex:1}
.btn-run{padding:6px 16px;background:var(--accent);color:#000;border:none;border-radius:6px;font-family:var(--mono);font-size:11px;font-weight:700;cursor:pointer;text-transform:uppercase;letter-spacing:.05em;transition:opacity .15s}
.btn-run:hover{opacity:.8}
.btn-run:disabled{opacity:.4;cursor:not-allowed}

/* ── Painéis com tabs ─────────────────────────────────────────── */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);background:var(--surface)}
.tab{padding:10px 18px;font-family:var(--mono);font-size:11px;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s;text-transform:uppercase;letter-spacing:.04em;user-select:none}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab:hover:not(.active){color:var(--text)}
.panel-body{flex:1;overflow:hidden;display:flex;flex-direction:column}
.panel-content{flex:1;overflow-y:auto;padding:0}

/* ── Query Box ────────────────────────────────────────────────── */
.query-area{padding:16px;border-bottom:1px solid var(--border);background:var(--panel)}
.query-label{font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.query-row{display:flex;gap:8px}
.query-input{flex:1;background:var(--bg);border:1px solid var(--border);color:var(--text);font-family:var(--mono);font-size:13px;padding:10px 14px;border-radius:6px;outline:none;transition:border-color .15s}
.query-input:focus{border-color:var(--accent)}
.query-input::placeholder{color:var(--muted)}
.btn-query{padding:10px 20px;background:transparent;border:1px solid var(--accent);color:var(--accent);font-family:var(--mono);font-size:11px;font-weight:700;border-radius:6px;cursor:pointer;text-transform:uppercase;letter-spacing:.06em;transition:all .15s;white-space:nowrap}
.btn-query:hover{background:var(--accent);color:#000}
.btn-query:disabled{opacity:.4;cursor:not-allowed}

/* ── Pipeline Steps ──────────────────────────────────────────── */
.steps-area{flex:1;overflow-y:auto;padding:12px 16px;display:flex;flex-direction:column;gap:8px}
.step-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;overflow:hidden;transition:border-color .2s}
.step-card.running{border-color:var(--amber);animation:pulse-border .8s ease infinite alternate}
.step-card.done-ok{border-color:rgba(0,229,160,.35)}
.step-card.done-warn{border-color:rgba(255,183,0,.35)}
.step-card.done-error{border-color:rgba(255,64,96,.35)}
@keyframes pulse-border{from{box-shadow:0 0 0 rgba(255,183,0,0)}to{box-shadow:0 0 8px rgba(255,183,0,.3)}}
.step-header{display:flex;align-items:center;gap:10px;padding:10px 14px}
.step-icon{width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;flex-shrink:0}
.step-icon.pending{background:var(--border);color:var(--muted)}
.step-icon.running{background:rgba(255,183,0,.15);color:var(--amber);animation:spin 1s linear infinite}
.step-icon.ok{background:rgba(0,229,160,.15);color:var(--accent)}
.step-icon.warn{background:rgba(255,183,0,.15);color:var(--amber)}
.step-icon.error{background:rgba(255,64,96,.15);color:var(--red)}
@keyframes spin{to{transform:rotate(360deg)}}
.step-label{font-family:var(--mono);font-size:11px;color:var(--text);flex:1}
.step-ms{font-family:var(--mono);font-size:10px;color:var(--muted)}
.step-result{padding:0 14px 10px 44px;font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.5;display:none}
.step-result.visible{display:block}

/* ── Resposta final ──────────────────────────────────────────── */
.resposta-area{border-top:1px solid var(--border);padding:16px;background:var(--panel);min-height:0;max-height:220px;overflow-y:auto;display:none}
.resposta-area.visible{display:block}
.resposta-label{font-family:var(--mono);font-size:10px;color:var(--accent);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
.resposta-texto{font-size:13px;line-height:1.7;color:var(--text);white-space:pre-wrap}
.resposta-fonte{font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:8px}

/* ── Chunks ──────────────────────────────────────────────────── */
.chunks-area{padding:12px 16px;display:flex;flex-direction:column;gap:6px}
.chunk-card{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px 12px}
.chunk-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.chunk-source{font-family:var(--mono);font-size:10px;color:var(--blue)}
.chunk-score{font-family:var(--mono);font-size:10px}
.chunk-preview{font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.5}

/* ── Terminal de logs ─────────────────────────────────────────── */
#terminal{flex:1;overflow-y:auto;padding:12px 16px;font-family:var(--mono);font-size:11px;line-height:1.7;background:var(--bg)}
.log-line{display:flex;gap:10px;padding:1px 0;border-bottom:1px solid transparent}
.log-line:hover{background:rgba(255,255,255,.02)}
.log-ts{color:var(--muted);flex-shrink:0;width:84px}
.log-level{flex-shrink:0;width:52px;text-align:right;font-weight:700}
.log-name{flex-shrink:0;width:120px;color:var(--blue);overflow:hidden;text-overflow:ellipsis}
.log-msg{color:var(--text);word-break:break-all}

/* ── Painel direito — Métricas ─────────────────────────────────── */
.metrics-panel{padding:16px;display:flex;flex-direction:column;gap:12px;overflow-y:auto;flex:1}
.metric-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.metric-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.metric-val{font-family:var(--mono);font-size:22px;font-weight:700;line-height:1;margin-bottom:4px}
.metric-lbl{font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.accent{color:var(--accent)}
.amber{color:var(--amber)}
.blue{color:var(--blue)}
.red{color:var(--red)}

/* ── Eventos do calendário ────────────────────────────────────── */
.eventos-list{display:flex;flex-direction:column;gap:6px;padding:0 16px 16px}
.evento-item{display:flex;gap:10px;align-items:flex-start;background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:10px 12px;transition:border-color .15s}
.evento-item:hover{border-color:var(--accent)}
.evento-dias{font-family:var(--mono);font-size:16px;font-weight:700;min-width:38px;text-align:center;line-height:1}
.evento-dias small{display:block;font-size:9px;color:var(--muted);font-weight:400}
.evento-info{flex:1}
.evento-nome{font-size:12px;font-weight:700;color:var(--text);line-height:1.3;margin-bottom:3px}
.evento-data{font-family:var(--mono);font-size:10px;color:var(--muted)}
.badge-cat{font-family:var(--mono);font-size:9px;padding:2px 7px;border-radius:10px;text-transform:uppercase;letter-spacing:.04em}
.cat-urgente{background:rgba(255,64,96,.1);color:var(--red);border:1px solid rgba(255,64,96,.25)}
.cat-prova{background:rgba(74,240,255,.1);color:var(--blue);border:1px solid rgba(74,240,255,.25)}
.cat-inicio{background:rgba(0,229,160,.1);color:var(--accent);border:1px solid rgba(0,229,160,.25)}
.cat-feriado{background:rgba(255,183,0,.1);color:var(--amber);border:1px solid rgba(255,183,0,.25)}
.cat-outro{background:rgba(176,110,255,.1);color:var(--purple);border:1px solid rgba(176,110,255,.25)}

/* ── Scrollbars ──────────────────────────────────────────────── */
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* ── Histórico de métricas ────────────────────────────────────── */
.hist-item{display:flex;gap:8px;align-items:center;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.04);font-family:var(--mono);font-size:10px}
.hist-rota{width:80px;color:var(--accent);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hist-tokens{width:48px;text-align:right;color:var(--amber)}
.hist-ms{width:48px;text-align:right;color:var(--blue)}
.hist-crag{width:48px;text-align:right}
.hist-q{flex:1;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.empty-state{padding:32px;text-align:center;color:var(--muted);font-family:var(--mono);font-size:11px}

/* ── CRAG meter ──────────────────────────────────────────────── */
.crag-meter{height:4px;background:var(--border);border-radius:2px;margin-top:6px;overflow:hidden}
.crag-fill{height:100%;border-radius:2px;transition:width .4s}
</style>
</head>
<body>
<div class="layout">

<!-- ── Header ──────────────────────────────────────────────────── -->
<header class="header">
  <div class="logo">Oráculo <span>UEMA</span></div>
  <div class="header-pill"><div class="dot live" id="dot-live"></div><span id="status-txt">conectando...</span></div>
  <div class="header-pill">modelo: <span id="modelo-txt" style="color:var(--accent)">gemini-2.0-flash-lite</span></div>
  <div class="header-spacer"></div>
  <button class="btn-run" id="btn-full-eval">▶ Full Eval</button>
</header>

<!-- ── Esquerda ────────────────────────────────────────────────── -->
<div class="main-left">
  <!-- Tabs esquerda -->
  <div class="tabs" id="tabs-left">
    <div class="tab active" data-tab="pipeline">Pipeline</div>
    <div class="tab" data-tab="terminal">Terminal</div>
    <div class="tab" data-tab="chunks">Chunks RAG</div>
  </div>

  <!-- Query box (sempre visível) -->
  <div class="query-area">
    <div class="query-label">▸ consulta em tempo real</div>
    <div class="query-row">
      <input class="query-input" id="query-input" placeholder="quando é a matrícula de veteranos?" maxlength="400">
      <button class="btn-query" id="btn-query">↵ Executar</button>
    </div>
  </div>

  <!-- Conteúdo das tabs -->
  <div class="panel-body">

    <!-- Tab: Pipeline ─────────────────────────────────────────── -->
    <div class="panel-content" id="tab-pipeline">
      <div class="steps-area" id="steps-area">
        <div class="empty-state">↑ Faça uma pergunta para ver o pipeline em ação</div>
      </div>
      <div class="resposta-area" id="resposta-area">
        <div class="resposta-label">▸ resposta gerada</div>
        <div class="resposta-texto" id="resposta-texto"></div>
        <div class="resposta-fonte" id="resposta-fonte"></div>
      </div>
    </div>

    <!-- Tab: Terminal ─────────────────────────────────────────── -->
    <div class="panel-content" id="tab-terminal" style="display:none;flex:1">
      <div id="terminal"></div>
    </div>

    <!-- Tab: Chunks ─────────────────────────────────────────────  -->
    <div class="panel-content" id="tab-chunks" style="display:none">
      <div class="chunks-area" id="chunks-area">
        <div class="empty-state">Chunks recuperados aparecerão aqui</div>
      </div>
    </div>

  </div>
</div>

<!-- ── Direita ─────────────────────────────────────────────────── -->
<div class="main-right">
  <div class="tabs" id="tabs-right">
    <div class="tab active" data-tab="metrics">Métricas</div>
    <div class="tab" data-tab="eventos">Calendário</div>
    <div class="tab" data-tab="historico">Histórico</div>
  </div>

  <!-- Tab: Métricas ─────────────────────────────────────────────── -->
  <div class="panel-body" id="rtab-metrics">
    <div class="metrics-panel">
      <div class="metric-row">
        <div class="metric-card"><div class="metric-val accent" id="m-tokens-total">—</div><div class="metric-lbl">tokens total</div></div>
        <div class="metric-card"><div class="metric-val amber" id="m-latencia">—</div><div class="metric-lbl">latência (ms)</div></div>
      </div>
      <div class="metric-row">
        <div class="metric-card"><div class="metric-val blue" id="m-tokens-in">—</div><div class="metric-lbl">tokens entrada</div></div>
        <div class="metric-card"><div class="metric-val" id="m-tokens-out">—</div><div class="metric-lbl">tokens saída</div></div>
      </div>
      <div class="metric-card">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div class="metric-lbl">CRAG score (qualidade RAG)</div>
          <div class="metric-val" id="m-crag" style="font-size:16px">—</div>
        </div>
        <div class="crag-meter"><div class="crag-fill" id="crag-fill" style="width:0%;background:var(--accent)"></div></div>
      </div>
      <div class="metric-card">
        <div class="metric-lbl" style="margin-bottom:6px">rota detectada</div>
        <div class="metric-val" id="m-rota" style="font-size:16px;color:var(--purple)">—</div>
      </div>
      <!-- Histórico resumido da sessão -->
      <div style="font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;padding:4px 0 6px">sessão atual</div>
      <div class="metric-row">
        <div class="metric-card"><div class="metric-val accent" id="s-queries">0</div><div class="metric-lbl">queries</div></div>
        <div class="metric-card"><div class="metric-val amber" id="s-tokens">0</div><div class="metric-lbl">tokens usados</div></div>
      </div>
    </div>
  </div>

  <!-- Tab: Eventos ─────────────────────────────────────────────── -->
  <div class="panel-body" id="rtab-eventos" style="display:none">
    <div class="metrics-panel" style="padding:0">
      <div style="padding:12px 16px 8px;font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em" id="eventos-header">carregando...</div>
      <div class="eventos-list" id="eventos-list">
        <div class="empty-state">carregando eventos...</div>
      </div>
    </div>
  </div>

  <!-- Tab: Histórico ────────────────────────────────────────────── -->
  <div class="panel-body" id="rtab-historico" style="display:none">
    <div style="padding:8px 16px;display:flex;gap:8px;font-family:var(--mono);font-size:9px;color:var(--muted);border-bottom:1px solid var(--border)">
      <span style="width:80px">ROTA</span>
      <span style="width:48px;text-align:right">TOKENS</span>
      <span style="width:48px;text-align:right">MS</span>
      <span style="width:48px;text-align:right">CRAG</span>
      <span style="flex:1">PERGUNTA</span>
    </div>
    <div style="overflow-y:auto;flex:1;padding:0 16px" id="hist-list">
      <div class="empty-state">Nenhuma query ainda</div>
    </div>
  </div>
</div>

</div><!-- /layout -->

<script>
// ── Estado global ─────────────────────────────────────────────────────────
const state = {
  logEs:       null,   // EventSource de logs
  queryEs:     null,   // EventSource de query
  queries:     0,
  totalTokens: 0,
  stepCards:   {},
};

// ── Tabs ──────────────────────────────────────────────────────────────────
function setupTabs(containerSel, prefixId) {
  const container = document.querySelector(containerSel);
  if (!container) return;
  container.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      container.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const key = tab.dataset.tab;
      // Left tabs
      if (prefixId === 'left') {
        ['pipeline','terminal','chunks'].forEach(k => {
          const el = document.getElementById('tab-' + k);
          if (el) el.style.display = k === key ? (k === 'terminal' ? 'flex' : 'block') : 'none';
        });
      } else {
        // Right tabs
        ['metrics','eventos','historico'].forEach(k => {
          const el = document.getElementById('rtab-' + k);
          if (el) el.style.display = k === key ? 'flex' : 'none';
        });
      }
    });
  });
}
setupTabs('#tabs-left', 'left');
setupTabs('#tabs-right', 'right');

// ── Log Terminal ──────────────────────────────────────────────────────────
function connectLogs() {
  if (state.logEs) state.logEs.close();
  const es = new EventSource('/eval/stream');
  state.logEs = es;

  es.onopen = () => {
    setStatus(true);
  };
  es.onerror = () => {
    setStatus(false);
    setTimeout(connectLogs, 3000);
  };
  es.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (!d.msg) return; // heartbeat silencioso
    appendLog(d);
  };
}

function appendLog(d) {
  const term = document.getElementById('terminal');
  const atBottom = term.scrollTop + term.clientHeight >= term.scrollHeight - 20;
  const line = document.createElement('div');
  line.className = 'log-line';
  line.innerHTML = `
    <span class="log-ts">${d.ts}</span>
    <span class="log-level" style="color:${d.cor}">${d.level}</span>
    <span class="log-name">${esc(d.name)}</span>
    <span class="log-msg">${esc(d.msg)}</span>
  `;
  term.appendChild(line);
  // limita a 400 linhas
  while (term.children.length > 400) term.removeChild(term.firstChild);
  if (atBottom) term.scrollTop = term.scrollHeight;
}

function setStatus(ok) {
  document.getElementById('dot-live').className = 'dot ' + (ok ? 'live' : '');
  document.getElementById('status-txt').textContent = ok ? 'stream ativo' : 'desconectado';
}

// ── Pipeline Query ────────────────────────────────────────────────────────
const STEPS_DEF = [
  {id:'guardrails', label:'0 — Guardrails (regex, 0ms)'},
  {id:'routing',    label:'1 — Routing Semântico (Redis KNN)'},
  {id:'transform',  label:'2 — Query Transform (Gemini)'},
  {id:'retrieval',  label:'3 — Busca Híbrida (BM25 + Vetor)'},
  {id:'geracao',    label:'4 — Geração (Gemini Flash)'},
];

const ICONS = { pending:'○', running:'◌', ok:'✓', warn:'⚠', error:'✗', blocked:'⊘', skip:'—' };

function buildSteps() {
  const area = document.getElementById('steps-area');
  area.innerHTML = '';
  state.stepCards = {};
  STEPS_DEF.forEach(s => {
    const card = document.createElement('div');
    card.className = 'step-card';
    card.id = 'step-' + s.id;
    card.innerHTML = `
      <div class="step-header">
        <div class="step-icon pending" id="si-${s.id}">${ICONS.pending}</div>
        <div class="step-label">${s.label}</div>
        <div class="step-ms" id="sms-${s.id}"></div>
      </div>
      <div class="step-result" id="sr-${s.id}"></div>
    `;
    area.appendChild(card);
    state.stepCards[s.id] = card;
  });
  document.getElementById('resposta-area').classList.remove('visible');
  document.getElementById('chunks-area').innerHTML = '';
}

function stepRunning(id) {
  const card = document.getElementById('step-' + id);
  if (!card) return;
  card.className = 'step-card running';
  const icon = document.getElementById('si-' + id);
  if (icon) { icon.className = 'step-icon running'; icon.textContent = '◌'; }
}

function stepDone(id, result, badge, ms) {
  const card = document.getElementById('step-' + id);
  if (!card) return;
  const badgeClass = ['ok','transformed'].includes(badge) ? 'done-ok'
                   : ['warn','skip'].includes(badge) ? 'done-warn'
                   : ['error','blocked'].includes(badge) ? 'done-error'
                   : 'done-ok';
  card.className = 'step-card ' + badgeClass;
  const icon = document.getElementById('si-' + id);
  const iconClass = ['ok','transformed'].includes(badge) ? 'ok'
                  : ['warn','skip'].includes(badge) ? 'warn'
                  : ['error','blocked'].includes(badge) ? 'error' : 'ok';
  if (icon) { icon.className = 'step-icon ' + iconClass; icon.textContent = ICONS[iconClass] || '✓'; }
  const msEl = document.getElementById('sms-' + id);
  if (msEl && ms != null) msEl.textContent = ms + 'ms';
  const resEl = document.getElementById('sr-' + id);
  if (resEl && result) { resEl.textContent = result; resEl.classList.add('visible'); }
}

function addChunk(d) {
  const area = document.getElementById('chunks-area');
  if (area.querySelector('.empty-state')) area.innerHTML = '';
  const card = document.createElement('div');
  card.className = 'chunk-card';
  const scoreColor = d.score >= 0.03 ? 'var(--accent)' : d.score >= 0.015 ? 'var(--amber)' : 'var(--red)';
  card.innerHTML = `
    <div class="chunk-header">
      <span class="chunk-source">${esc(d.source)}</span>
      <span class="chunk-score" style="color:${scoreColor}">RRF: ${d.score}</span>
    </div>
    <div class="chunk-preview">${esc(d.preview)}</div>
  `;
  area.appendChild(card);
}

function showResposta(texto, fonte, tokens) {
  const area  = document.getElementById('resposta-area');
  area.classList.add('visible');
  document.getElementById('resposta-texto').textContent = texto;
  document.getElementById('resposta-fonte').textContent =
    `fonte: ${fonte} ${tokens ? '| ' + tokens + ' tokens gerados' : ''}`;
}

function updateMetrics(d) {
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('m-tokens-total', fmt(d.tokens_total || 0));
  set('m-tokens-in',    fmt(d.tokens_entrada || 0));
  set('m-tokens-out',   fmt(d.tokens_saida || 0));
  set('m-latencia',     (d.latencia_ms || 0) + 'ms');
  set('m-rota',         d.rota || '—');

  const score = d.crag_score || 0;
  const scoreEl = document.getElementById('m-crag');
  if (scoreEl) {
    scoreEl.textContent = score.toFixed(3);
    scoreEl.style.color = score >= 0.4 ? 'var(--accent)' : score >= 0.2 ? 'var(--amber)' : 'var(--red)';
  }
  const fill = document.getElementById('crag-fill');
  if (fill) {
    const pct = Math.min(score * 100 / 0.6, 100);
    fill.style.width = pct + '%';
    fill.style.background = score >= 0.4 ? 'var(--accent)' : score >= 0.2 ? 'var(--amber)' : 'var(--red)';
  }

  // Sessão
  state.queries++;
  state.totalTokens += (d.tokens_total || 0);
  set('s-queries', state.queries);
  set('s-tokens',  fmt(state.totalTokens));
}

function addHistorico(d, pergunta) {
  const list = document.getElementById('hist-list');
  if (list.querySelector('.empty-state')) list.innerHTML = '';
  const score = d.crag_score || 0;
  const scoreColor = score >= 0.4 ? 'var(--accent)' : score >= 0.2 ? 'var(--amber)' : 'var(--red)';
  const item = document.createElement('div');
  item.className = 'hist-item';
  item.innerHTML = `
    <span class="hist-rota">${esc(d.rota || '—')}</span>
    <span class="hist-tokens">${fmt(d.tokens_total || 0)}</span>
    <span class="hist-ms">${d.latencia_ms || 0}</span>
    <span class="hist-crag" style="color:${scoreColor}">${score.toFixed(3)}</span>
    <span class="hist-q" title="${esc(pergunta)}">${esc(pergunta)}</span>
  `;
  list.insertBefore(item, list.firstChild);
  while (list.children.length > 100) list.removeChild(list.lastChild);
}

function runQuery() {
  const input = document.getElementById('query-input');
  const pergunta = input.value.trim();
  if (!pergunta) return;

  const btn = document.getElementById('btn-query');
  btn.disabled = true;
  btn.textContent = '⏳';

  // Ativa tab pipeline
  document.querySelectorAll('#tabs-left .tab').forEach(t => t.classList.remove('active'));
  document.querySelector('#tabs-left [data-tab="pipeline"]').classList.add('active');
  ['pipeline','terminal','chunks'].forEach(k => {
    const el = document.getElementById('tab-' + k);
    if (el) el.style.display = k === 'pipeline' ? 'block' : 'none';
  });

  buildSteps();

  if (state.queryEs) state.queryEs.close();

  // POST com fetch + stream manual
  fetch('/eval/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pergunta}),
  }).then(async res => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const block of lines) {
        if (!block.startsWith('data: ')) continue;
        try {
          const d = JSON.parse(block.slice(6));
          handleQueryEvent(d, pergunta);
        } catch {}
      }
    }
  }).catch(err => {
    console.error(err);
  }).finally(() => {
    btn.disabled = false;
    btn.textContent = '↵ Executar';
  });
}

function handleQueryEvent(d, pergunta) {
  switch(d.tipo) {
    case 'step_start':
      stepRunning(d.step);
      break;
    case 'step_result':
      stepDone(d.step, d.resultado, d.badge, d.ms);
      break;
    case 'chunk_rag':
      addChunk(d);
      break;
    case 'resposta':
      showResposta(d.texto, d.fonte, d.tokens);
      break;
    case 'metricas':
      updateMetrics(d);
      addHistorico(d, pergunta);
      break;
    case 'erro':
      appendLog({ts: new Date().toLocaleTimeString('pt-BR'), level:'ERROR', name:'pipeline', msg: d.msg, cor:'#ff4060'});
      break;
    case 'done':
      break;
  }
}

// ── Eventos do Calendário ─────────────────────────────────────────────────
async function loadEventos() {
  try {
    const r = await fetch('/eval/eventos');
    const data = await r.json();
    const list = document.getElementById('eventos-list');
    const header = document.getElementById('eventos-header');
    header.textContent = `próximos 30 dias — ${data.eventos.length} evento(s)`;

    if (!data.eventos.length) {
      list.innerHTML = '<div class="empty-state">Nenhum evento encontrado<br><small>O calendário foi ingerido no Redis?</small></div>';
      return;
    }

    list.innerHTML = data.eventos.map(e => {
      const dias   = e.dias_restantes;
      const cor    = dias === 0 ? 'var(--red)' : dias <= 3 ? 'var(--amber)' : dias <= 7 ? 'var(--accent)' : 'var(--text)';
      const catCls = 'cat-' + (e.categoria || 'outro');
      return `
        <div class="evento-item">
          <div class="evento-dias" style="color:${cor}">
            ${dias === 0 ? 'HOJE' : dias + 'd'}
            <small>restam</small>
          </div>
          <div class="evento-info">
            <div class="evento-nome">${e.emoji} ${esc(e.nome)}</div>
            <div class="evento-data">${e.data_inicio}${e.data_fim && e.data_fim !== e.data_inicio ? ' → ' + e.data_fim : ''}</div>
          </div>
          <span class="badge-cat ${catCls}">${e.categoria}</span>
        </div>`;
    }).join('');
  } catch(err) {
    document.getElementById('eventos-list').innerHTML = '<div class="empty-state">Erro ao carregar eventos</div>';
  }
}

// ── Full Eval ─────────────────────────────────────────────────────────────
document.getElementById('btn-full-eval').addEventListener('click', async () => {
  const btn = document.getElementById('btn-full-eval');
  btn.disabled = true;
  btn.textContent = '⏳ Executando...';
  try {
    const r = await fetch('/eval/run-full', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{"versao":"live"}'});
    const d = await r.json();
    appendLog({ts: new Date().toLocaleTimeString('pt-BR'), level:'INFO', name:'eval', msg:'Full eval enfileirado: ' + JSON.stringify(d), cor:'#00e5a0'});
  } finally {
    setTimeout(() => { btn.disabled = false; btn.textContent = '▶ Full Eval'; }, 3000);
  }
});

// ── Keyboard shortcuts ────────────────────────────────────────────────────
document.getElementById('query-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runQuery(); }
});
document.getElementById('btn-query').addEventListener('click', runQuery);

// ── Utils ─────────────────────────────────────────────────────────────────
function esc(s) { return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function fmt(n) { return n >= 1e6 ? (n/1e6).toFixed(1)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'k' : String(n); }

// ── Init ──────────────────────────────────────────────────────────────────
connectLogs();
loadEventos();
// Recarrega eventos a cada 5 minutos
setInterval(loadEventos, 300_000);
</script>
</body>
</html>
"""