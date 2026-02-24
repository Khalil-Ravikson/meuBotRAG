"""
application/handle_message.py — Orquestrador principal
=======================================================
Decide: menu direto OU agente.
Substitui o webhook_handler.py + parte do menu_service.

Fluxo:
  Mensagem
    → domain/menu.py (stateless: é navegação?)
       ├─ SIM → waha_service.enviar(texto do menu)
       └─ NÃO → domain/router.py → Rota
                → memory/ → carrega contexto do usuário
                → agent/prompts.py → monta prompt enriquecido
                → AgentState → agent/core.py
                → memory/ → salva novo estado
                → waha_service.enviar(resposta)
"""
from __future__ import annotations
import logging

from src.domain.entities import Mensagem, EstadoMenu
from src.domain.menu     import processar_mensagem
from src.domain.router   import analisar
from src.agent.core      import agent_core
from src.agent.state     import AgentState
from src.agent.prompts   import montar_prompt_enriquecido
from src.memory.redis_memory import (
    get_estado_menu, set_estado_menu, clear_estado_menu,
    get_contexto, set_contexto,
)
from src.services.waha_service import WahaService
from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)


async def handle_message(mensagem: Mensagem, waha: WahaService) -> None:
    """
    Processa uma mensagem recebida e envia a resposta via WAHA.

    Parâmetros:
      mensagem : Mensagem (domain entity)
      waha     : WahaService (injetado)
    """
    user_id = mensagem.user_id
    body    = mensagem.body

    if not body.strip():
        logger.debug("🔇 Mensagem vazia ignorada [%s].", user_id)
        return

    logger.info("📨 [%s] '%s'", user_id, body[:80])

    # ── 1. Carrega estado do menu do Redis ────────────────────────────────────
    estado_atual = get_estado_menu(user_id)

    # ── 2. domain/menu.py (stateless): decide o tipo de resposta ─────────────
    resultado = processar_mensagem(body, estado_atual)

    # ── 3. Resposta direta do menu (sem LLM) ──────────────────────────────────
    if resultado["type"] in ("menu_principal", "submenu"):
        novo_estado = resultado["novo_estado"]
        set_estado_menu(user_id, novo_estado)
        await waha.enviar_mensagem(mensagem.chat_id, resultado["content"])
        return

    # ── 4. Atualiza estado do menu (limpa para MAIN após ação do submenu) ─────
    novo_estado = resultado["novo_estado"]
    if novo_estado != estado_atual:
        if novo_estado == EstadoMenu.MAIN:
            clear_estado_menu(user_id)
        else:
            set_estado_menu(user_id, novo_estado)

    # ── 5. Determina rota e monta prompt enriquecido ──────────────────────────
    prompt_base = resultado["prompt"] or body
    rota        = analisar(prompt_base, estado_atual)
    ctx_usuario = get_contexto(user_id)

    prompt_final = montar_prompt_enriquecido(
        texto_usuario    = prompt_base,
        rota             = rota,
        contexto_usuario = ctx_usuario,
    )

    # ── 6. Cria AgentState ────────────────────────────────────────────────────
    state = AgentState(
        user_id            = user_id,
        session_id         = user_id,  # 1 sessão por usuário
        mensagem_original  = body,
        chat_id            = mensagem.chat_id,
        rota               = rota,
        modo_menu          = estado_atual,
        prompt_enriquecido = prompt_final,
        contexto_usuario   = ctx_usuario,
        max_iteracoes      = settings.AGENT_MAX_ITERATIONS,
    )

    # ── 7. Agente gera a resposta ─────────────────────────────────────────────
    logger.info("🤖 [%s] rota=%s → AgentCore", user_id, rota.value)
    resposta_obj = agent_core.responder(state)

    # ── 8. Persiste contexto (última intenção) ────────────────────────────────
    set_contexto(user_id, {"ultima_intencao": rota.value})

    # ── 9. Envia resposta ─────────────────────────────────────────────────────
    conteudo = resposta_obj.conteudo or "Desculpe, não consegui processar sua solicitação."
    await waha.enviar_mensagem(mensagem.chat_id, conteudo)