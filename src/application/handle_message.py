"""
application/handle_message.py — A Ponte (v5 — SecurityGuard + RBAC + Admin)
=============================================================================

MUDANÇAS v5 vs v3/v4:
──────────────────────
  ADICIONADO:
    - SecurityGuard antes do AgentCore
    - RBAC: tools filtradas por nível (GUEST/STUDENT/ADMIN)
    - Rate Limiting: Fixed Window via Redis
    - Comandos admin via WhatsApp (!ingerir, !status, !tools, etc.)
    - Ingestão de documentos pelo ADMIN via ZapZap
    - Registo detalhado no monitor (tiktoken + latência + nível)

  MANTIDO:
    - domain/menu.py (lógica de menu não muda)
    - AgentCore.responder() (interface idêntica)
    - EvolutionService (não muda)
    - Working memory / long-term memory (não muda)

FLUXO v5:
──────────
  1. DevGuard valida payload Evolution API
  2. SecurityGuard verifica:
       a. RBAC: resolve nível (GUEST/STUDENT/ADMIN)
       b. Rate limit: Fixed Window Redis
       c. Comandos admin (regex, 0 tokens)
       d. Ingestão de media (ADMIN com ficheiro)
  3. domain/menu.py: navegação de menu (regex, 0 tokens)
  4. AgentCore com tools filtradas por nível
  5. Monitor: regista tokens + latência + nível
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from src.agent.core import agent_core
from src.domain.entities import EstadoMenu, Mensagem, Rota
from src.domain.menu import processar_mensagem
from src.infrastructure.redis_client import get_redis_text
from src.infrastructure.settings import settings
from src.memory.redis_memory import (
    clear_estado_menu,
    get_estado_menu,
    set_contexto,
    set_estado_menu,
)
from src.middleware.security_guard import NivelAcesso, SecurityGuard
from src.services.evolution_service import EvolutionService

logger = logging.getLogger(__name__)

# Singleton do SecurityGuard (inicializado uma vez)
_security_guard: SecurityGuard | None = None

_MSG_MEDIA_SEM_TEXTO = (
    "Recebi a tua mensagem! Por enquanto só consigo processar texto. 📝\n"
    "Digita a tua dúvida que te respondo rapidinho."
)


def get_security_guard() -> SecurityGuard:
    """Retorna o singleton do SecurityGuard (lazy init)."""
    global _security_guard
    if _security_guard is None:
        _security_guard = SecurityGuard(get_redis_text(), settings)
    return _security_guard


async def handle_message(mensagem: Mensagem, evolution: EvolutionService) -> None:
    """
    Orquestra o fluxo completo com RBAC + Rate Limit + Comandos Admin.
    """
    user_id    = mensagem.user_id
    chat_id    = mensagem.chat_id
    body       = mensagem.body
    has_media  = mensagem.has_media
    msg_type   = getattr(mensagem, "msg_type", "conversation")
    msg_key_id = getattr(mensagem, "msg_key_id", "")

    t0 = time.monotonic()

    # ── 1. Mensagem vazia ─────────────────────────────────────────────────────
    if not body.strip():
        if has_media:
            await evolution.enviar_mensagem(chat_id, _MSG_MEDIA_SEM_TEXTO)
        else:
            logger.debug("🔇 Mensagem vazia ignorada [%s]", user_id)
        return

    logger.info("📨 [%s] '%s'", user_id, body[:80])

    # ── 2. SecurityGuard: RBAC + Rate Limit + Comandos Admin ─────────────────
    guard  = get_security_guard()
    result = guard.verificar(
        user_id    = user_id,
        body       = body,
        has_media  = has_media,
        msg_type   = msg_type,
        msg_key_id = msg_key_id,
    )

    # ── 3. Bloqueado (rate limit) ─────────────────────────────────────────────
    if result.bloqueado:
        await evolution.enviar_mensagem(chat_id, result.resposta)
        return

    # ── 4. Resposta rápida do SecurityGuard (sem LLM) ─────────────────────────
    if result.resposta_rapida:
        await evolution.enviar_mensagem(chat_id, result.resposta_rapida)

    # ── 5. Ações especiais ────────────────────────────────────────────────────

    # 5a. Ingestão de documento via WhatsApp
    if result.acao == "INGERIR_DOC":
        identity_ingest = {
            "chat_id":      chat_id,
            "sender_phone": user_id,
            "msg_key_id":   msg_key_id,
            "body":         body,
        }
        try:
            from src.application.tasks_admin import ingerir_documento_task
            ingerir_documento_task.delay(identity_ingest)
        except Exception as e:
            logger.exception("❌ Falha ao disparar ingestão: %s", e)
            await evolution.enviar_mensagem(chat_id, "❌ Erro ao enfileirar ingestão.")
        return

    # 5b. Ingestão de ficheiro por nome (sem media)
    if result.acao == "INGERIR_FICHEIRO":
        identity_ingest = {
            "chat_id":      chat_id,
            "sender_phone": user_id,
            "msg_key_id":   "",
            "body":         result.parametro,
        }
        try:
            from src.application.tasks_admin import ingerir_documento_task
            ingerir_documento_task.delay(identity_ingest)
        except Exception as e:
            await evolution.enviar_mensagem(chat_id, f"❌ Erro: {str(e)[:80]}")
        return

    # 5c. Comandos admin (pesados — via Celery)
    if result.acao == "CMD_ADMIN" and result.precisa_celery:
        try:
            from src.application.tasks_admin import executar_comando_admin_task
            executar_comando_admin_task.delay(chat_id, result.parametro, user_id)
        except Exception as e:
            await evolution.enviar_mensagem(chat_id, f"❌ Erro: {str(e)[:80]}")
        return

    # 5d. Comandos admin síncronos (rápidos — responde na hora)
    if result.acao == "CMD_ADMIN" and not result.precisa_celery:
        resposta_cmd = await _executar_cmd_sincrono(result.parametro, user_id)
        await evolution.enviar_mensagem(chat_id, resposta_cmd)
        return

    # 5e. Erro (sem media, comando mal formado)
    if result.acao == "ERRO":
        return  # resposta_rapida já foi enviada acima

    # ── 6. Fluxo LLM normal ───────────────────────────────────────────────────
    estado_atual: EstadoMenu = get_estado_menu(user_id)
    resultado_menu = processar_mensagem(body, estado_atual)

    # 6a. Resposta directa de menu (sem LLM)
    if resultado_menu["type"] in ("menu_principal", "submenu"):
        novo_estado: EstadoMenu = resultado_menu["novo_estado"]
        set_estado_menu(user_id, novo_estado)
        await evolution.enviar_mensagem(chat_id, resultado_menu["content"])
        return

    # 6b. Actualiza estado do menu
    novo_estado = resultado_menu["novo_estado"]
    if novo_estado != estado_atual:
        if novo_estado == EstadoMenu.MAIN:
            clear_estado_menu(user_id)
        else:
            set_estado_menu(user_id, novo_estado)

    texto_para_agente = resultado_menu.get("prompt") or body

    # ── 7. AgentCore com tools filtradas por nível ────────────────────────────
    logger.info(
        "🤖 AgentCore [%s] nivel=%s tools=%d texto='%s'",
        user_id, result.nivel.value, len(result.tools_disponiveis),
        texto_para_agente[:60],
    )

    resposta_obj = await asyncio.to_thread(
        agent_core.responder,
        user_id     = user_id,
        session_id  = user_id,
        mensagem    = texto_para_agente,
        estado_menu = estado_atual,
    )

    set_contexto(user_id, {"ultima_intencao": resposta_obj.rota.value})

    # ── 8. Registo no Monitor (tokens + latência + nível) ─────────────────────
    latencia_ms = int((time.monotonic() - t0) * 1000)
    _registar_monitor(
        user_id     = user_id,
        nivel       = result.nivel,
        tokens_in   = resposta_obj.tokens_entrada,
        tokens_out  = resposta_obj.tokens_saida,
        latencia_ms = latencia_ms,
        rota        = resposta_obj.rota.value,
        pergunta    = texto_para_agente,
        resposta    = resposta_obj.conteudo,
    )

    # ── 9. Envia resposta ─────────────────────────────────────────────────────
    conteudo = resposta_obj.conteudo or _MSG_MEDIA_SEM_TEXTO
    await evolution.enviar_mensagem(chat_id, conteudo)

    logger.info(
        "✅ [%s] nivel=%s | rota=%s | tokens=%d | latência=%dms",
        user_id, result.nivel.value, resposta_obj.rota.value,
        resposta_obj.tokens_total, latencia_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Comandos síncronos admin (rápidos, sem Celery)
# ─────────────────────────────────────────────────────────────────────────────

async def _executar_cmd_sincrono(parametro: str, user_id: str) -> str:
    """Executa comandos admin que retornam resposta imediatamente."""

    if parametro == "STATUS":
        from src.infrastructure.redis_client import redis_ok
        from src.infrastructure.semantic_cache import cache_stats
        redis_status = redis_ok()
        stats        = cache_stats()
        return (
            f"⚙️  *Status do Sistema*\n\n"
            f"🟢 Redis: {'OK' if redis_status else '❌ Offline'}\n"
            f"🤖 AgentCore: {'✅ Pronto' if agent_core._inicializado else '⏳ Iniciando'}\n"
            f"🧠 Cache: {stats.get('total_entradas', 0)} entradas\n"
            f"📦 Modelo: `{settings.GEMINI_MODEL}`\n"
            f"👤 Teu nível: ADMIN"
        )

    if parametro == "TOOLS":
        from src.domain.semantic_router import listar_tools_registadas
        tools = listar_tools_registadas()
        if not tools:
            return "⚠️  Sem tools registadas no Redis."
        linhas = [f"🔧 *{len(tools)} Tools:*\n"]
        for t in tools:
            linhas.append(f"• `{t['name']}`")
        return "\n".join(linhas)

    return f"ℹ️  Comando `{parametro}` recebido."


# ─────────────────────────────────────────────────────────────────────────────
# Monitor — registo rico para /monitor endpoint
# ─────────────────────────────────────────────────────────────────────────────

def _registar_monitor(
    user_id:    str,
    nivel:      NivelAcesso,
    tokens_in:  int,
    tokens_out: int,
    latencia_ms:int,
    rota:       str,
    pergunta:   str,
    resposta:   str,
) -> None:
    """
    Guarda log rico no Redis para o dashboard /monitor.

    Chave: monitor:logs (lista LIFO, máx 500)
    Cada entrada:
      ts, user_id, nivel, tokens_in, tokens_out, tokens_total,
      latencia_ms, rota, pergunta[:200], resposta[:300]
    """
    try:
        r = get_redis_text()
        entrada = json.dumps({
            "ts":           __import__("datetime").datetime.now().isoformat(),
            "user_id":      user_id,
            "nivel":        nivel.value,
            "tokens_in":    tokens_in,
            "tokens_out":   tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "latencia_ms":  latencia_ms,
            "rota":         rota,
            "pergunta":     pergunta[:200],
            "resposta":     resposta[:300],
        }, ensure_ascii=False)
        r.lpush("monitor:logs", entrada)
        r.ltrim("monitor:logs", 0, 499)

        # Contador por utilizador (para /monitor/{user_id})
        r.hincrby(f"monitor:user:{user_id}", "total_msgs",   1)
        r.hincrby(f"monitor:user:{user_id}", "total_tokens", tokens_in + tokens_out)
        r.hincrby(f"monitor:user:{user_id}", "total_latencia", latencia_ms)
        r.hset(f"monitor:user:{user_id}", "nivel",     nivel.value)
        r.hset(f"monitor:user:{user_id}", "ultima_msg", pergunta[:80])
        r.expire(f"monitor:user:{user_id}", 86400 * 7)  # 7 dias

    except Exception as e:
        logger.debug("⚠️  Monitor log falhou: %s", e)