"""
application/handle_webhook.py — Extração e validação do payload (v2 — Evolution API)
=====================================================================================
Recebe o payload bruto do FastAPI, valida com DevGuard,
converte para Mensagem (domain entity) e chama handle_message.

MIGRAÇÃO WAHA → EVOLUTION:
  - WahaService substituído por EvolutionService
  - identity agora inclui "push_name" (nome do contato no WhatsApp)
"""
from __future__ import annotations
import logging

from src.domain.entities import Mensagem
from src.application.handle_message import handle_message
from src.middleware.dev_guard import DevGuard
from src.services.evolution_service import EvolutionService

logger = logging.getLogger(__name__)


async def handle_webhook(
    payload: dict,
    guard: DevGuard,
    evolution: EvolutionService,
) -> dict:
    """
    Ponto de entrada de toda mensagem recebida.

    Retorna:
      {"status": "ok"} sempre (Evolution não precisa de resposta específica)
    """
    ok, resultado = await guard.validar(payload)

    if not ok:
        logger.debug("🛑 DevGuard bloqueou: %s", resultado)
        return {"status": "blocked", "reason": resultado}

    identity: dict = resultado

    mensagem = Mensagem(
        user_id   = identity["sender_phone"],
        chat_id   = identity["chat_id"],
        body      = identity.get("body", ""),
        has_media = identity.get("has_media", False),
        msg_type  = identity.get("msg_type", "conversation"),
    )

    await handle_message(mensagem, evolution)
    return {"status": "ok"}