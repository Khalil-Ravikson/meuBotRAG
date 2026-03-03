"""
application/handle_message.py — A Ponte (v3 — Clean Architecture)
==================================================================

RESPONSABILIDADE DESTE FICHEIRO:
──────────────────────────────────
  Este ficheiro é a "cola" entre a comunicação WhatsApp (Evolution API)
  e o novo AgentCore. É deliberadamente FINO — toda a lógica de negócio
  vive nos módulos especializados.

  Fluxo resumido:
    1. Recebe Mensagem (domain entity) vinda do handle_webhook.py
    2. Carrega estado do menu do Redis
    3. Passa pelo domain/menu.py → navegação directa OU aciona AgentCore
    4. Envia resposta via EvolutionService (ou WahaService — interface idêntica)

O QUE MUDOU vs v2:
─────────────────────
  ANTES:
    handle_message.py → domain/menu.py → domain/router.py (regex)
                      → agent/prompts.py → AgentState → AgentCore (LangChain)
                      → evolution_service.enviar_mensagem()

  AGORA:
    handle_message.py → domain/menu.py (mantido intacto)
                      → AgentCore.responder() [novo — pipeline limpa]
                      → evolution_service.enviar_mensagem() (mantido intacto)

  Removido:
    - Criação manual de AgentState (substituído por parâmetros simples)
    - Importação de domain/router.py (feita dentro do semantic_router)
    - Importação de agent/prompts.py (prompts vivem no gemini_provider)
    - Importação de memory/redis_memory.py (parcialmente — estado menu mantém)

  Mantido intacto:
    - domain/menu.py (lógica de menu não muda)
    - memory/redis_memory.py (get_estado_menu, set_estado_menu, set_contexto)
    - EvolutionService (comunicação WhatsApp não muda)

COMPATIBILIDADE COM WAHA:
──────────────────────────
  O ficheiro importa EvolutionService por padrão.
  Se ainda usas WahaService, basta trocar a importação — a interface
  .enviar_mensagem(chat_id, texto) é idêntica nos dois serviços.

TRATAMENTO DE MENSAGENS VAZIAS E MEDIA:
────────────────────────────────────────
  - Mensagem vazia → ignora silenciosamente (log DEBUG)
  - Mensagem com media mas sem texto → resposta padrão informativa
  - Mensagem com media + legenda → processa a legenda normalmente
"""
from __future__ import annotations

import asyncio
import logging

from src.agent.core import agent_core
from src.domain.entities import EstadoMenu, Mensagem
from src.domain.menu import processar_mensagem
from src.memory.redis_memory import (
    clear_estado_menu,
    get_estado_menu,
    set_contexto,
    set_estado_menu,
)
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
    o event loop. O AgentCore.responder() é síncrono mas é executado
    via asyncio.to_thread() para não bloquear.

    Parâmetros:
      mensagem:  Mensagem normalizada (vinda do handle_webhook.py)
      evolution: Serviço de envio WhatsApp (injetado)
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

    # ── 2. Carrega estado do menu do Redis ────────────────────────────────────
    estado_atual: EstadoMenu = get_estado_menu(user_id)

    # ── 3. domain/menu.py — decisão de navegação (stateless, regex puro) ─────
    resultado_menu = processar_mensagem(body, estado_atual)

    # ── 4a. Resposta directa de menu (sem LLM) ────────────────────────────────
    if resultado_menu["type"] in ("menu_principal", "submenu"):
        novo_estado: EstadoMenu = resultado_menu["novo_estado"]
        set_estado_menu(user_id, novo_estado)

        logger.debug("📋 Menu [%s]: %s → %s", user_id, estado_atual.value, novo_estado.value)
        await evolution.enviar_mensagem(chat_id, resultado_menu["content"])
        return

    # ── 4b. Actualiza estado do menu ──────────────────────────────────────────
    novo_estado = resultado_menu["novo_estado"]
    if novo_estado != estado_atual:
        if novo_estado == EstadoMenu.MAIN:
            clear_estado_menu(user_id)
        else:
            set_estado_menu(user_id, novo_estado)

    # ── 5. Determina o texto a processar pelo AgentCore ───────────────────────
    # Se o menu expandiu a pergunta (ex: opção "1" do submenu calendário),
    # usa o prompt expandido. Caso contrário usa o body original.
    texto_para_agente = resultado_menu.get("prompt") or body

    # ── 6. Aciona o novo AgentCore (pipeline Gemini + Redis) ─────────────────
    # asyncio.to_thread() executa o código síncrono do AgentCore sem bloquear
    # o event loop do FastAPI — essencial para alta concorrência
    logger.info("🤖 AgentCore [%s] estado=%s texto='%s'",
                user_id, estado_atual.value, texto_para_agente[:60])

    resposta_obj = await asyncio.to_thread(
        agent_core.responder,
        user_id=user_id,
        session_id=user_id,          # session_id = user_id para bots WhatsApp 1:1
        mensagem=texto_para_agente,
        estado_menu=estado_atual,
    )

    # ── 7. Persiste contexto (última intenção) — compatibilidade redis_memory ─
    # Mantemos set_contexto() para não quebrar o endpoint /logs e diagnósticos
    set_contexto(user_id, {"ultima_intencao": resposta_obj.rota.value})

    # ── 8. Envia resposta via Evolution API ───────────────────────────────────
    conteudo = resposta_obj.conteudo or _MSG_MEDIA_SEM_TEXTO
    await evolution.enviar_mensagem(chat_id, conteudo)

    logger.info(
        "✅ Resposta enviada [%s] | rota=%s | tokens=%d | sucesso=%s",
        user_id,
        resposta_obj.rota.value,
        resposta_obj.tokens_total,
        resposta_obj.sucesso,
    )