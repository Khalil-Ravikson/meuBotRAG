"""
application/handle_webhook.py — Extração e validação do payload WAHA
=====================================================================
Recebe o payload bruto do FastAPI, valida com DevGuard,
converte para Mensagem (domain entity) e chama handle_message.
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
    api_service: EvolutionService,  # Modificado aqui
) -> dict:
    
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
        msg_type  = identity.get("msg_type", "text"),
    )

    await handle_message(mensagem, api_service)
    return {"status": "ok"}