"""
application/handle_message.py — A Ponte (v4 — Clean Architecture Pura)
======================================================================

RESPONSABILIDADE DESTE FICHEIRO:
──────────────────────────────────
  Este ficheiro é a "cola" entre a comunicação WhatsApp (Evolution API)
  e o AgentCore. 

  Fluxo resumido:
    1. Recebe Mensagem (domain entity) vinda do handle_webhook.py
    2. Valida se tem texto ou apenas media
    3. Aciona o AgentCore.responder() numa thread separada
    4. Envia resposta via EvolutionService (ou WahaService)

O QUE MUDOU NA v4 (Remoção do Menu/Estado):
───────────────────────────────────────────
  - Todo o sistema de menus engessados (domain/menu.py) foi ELIMINADO.
  - O controlo de EstadoMenu no Redis foi ELIMINADO.
  - As saudações e bloqueios (Guardrails) agora são geridos de forma
    inteligente dentro do próprio `agent_core`.
  - O código ficou extremamente limpo, focado apenas em I/O (Entrada/Saída).
"""
from __future__ import annotations

import asyncio
import logging

from src.agent.core import agent_core
from src.domain.entities import Mensagem
from src.memory.redis_memory import set_contexto
from src.services.evolution_service import EvolutionService

logger = logging.getLogger(__name__)

# Resposta para quando o utilizador envia media sem texto (áudio, sticker, etc.)
_MSG_MEDIA_SEM_TEXTO = (
    "Recebi a tua mensagem! Por enquanto só consigo processar texto. 📝\n"
    "Digita a tua dúvida que te respondo rapidinho."
)

async def handle_message(mensagem: Mensagem, evolution: EvolutionService) -> None:
    """
    Orquestra o fluxo completo de uma mensagem recebida.

    ESTE MÉTODO É ASSÍNCRONO para integrar com o FastAPI sem bloquear
    o event loop. O AgentCore.responder() é executado
    via asyncio.to_thread() para não bloquear.
    """
    user_id = mensagem.user_id
    chat_id = mensagem.chat_id
    body    = mensagem.body

    # ── 1. Ignora mensagens sem texto ─────────────────────────────────────────
    if not body.strip():
        if mensagem.has_media:
            # Media sem legenda → responde com mensagem informativa
            logger.debug("📎 Media sem texto [%s] → resposta padrão", user_id)
            await evolution.enviar_mensagem(chat_id, _MSG_MEDIA_SEM_TEXTO)
        else:
            logger.debug("🔇 Mensagem vazia ignorada [%s]", user_id)
        return

    logger.info("📨 [%s] '%s'", user_id, body[:80])

    # ── 2. Aciona o AgentCore (pipeline Gemini + Redis + Guardrails) ─────────
    # asyncio.to_thread() executa o código síncrono do AgentCore sem bloquear
    # o event loop do FastAPI — essencial para alta concorrência
    logger.info("🤖 AgentCore [%s] a processar texto='%s'", user_id, body[:60])

    resposta_obj = await asyncio.to_thread(
        agent_core.responder,
        user_id=user_id,
        session_id=user_id,          # session_id = user_id para bots WhatsApp 1:1
        mensagem=body,
    )

    # ── 3. Persiste contexto (última intenção) ────────────────────────────────
    # Mantemos set_contexto() para não quebrar o endpoint /logs e diagnósticos
    set_contexto(user_id, {"ultima_intencao": resposta_obj.rota.value})

    # ── 4. Envia resposta via Evolution API ───────────────────────────────────
    conteudo = resposta_obj.conteudo or _MSG_MEDIA_SEM_TEXTO
    await evolution.enviar_mensagem(chat_id, conteudo)

    logger.info(
        "✅ Resposta enviada [%s] | rota=%s | latência=%dms | sucesso=%s",
        user_id,
        resposta_obj.rota.value,
        getattr(resposta_obj, 'latencia_ms', 0),
        resposta_obj.sucesso,
    )