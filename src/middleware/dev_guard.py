"""
middleware/dev_guard.py — Porteiro do webhook (v5 — Evolution API completo)
============================================================================

PROBLEMA CORRIGIDO v5:
  A Evolution API manda MUITOS eventos além de mensagens. Alguns têm
  estrutura completamente diferente (lista em vez de objeto):

  ✉️  messages.upsert     → data: { key: {...}, message: {...} }  ← PROCESSA
  👁️  messages.update     → data: [{ key: {...}, update: {...} }]  ← IGNORA (ACK/lido)
  👤  contacts.upsert     → data: [ {remoteJid, pushName, ...} ]   ← IGNORA (lista)
  👤  contacts.update     → data: [ {remoteJid, pushName, ...} ]   ← IGNORA (lista)
  🔗  connection.update   → data: { state: 'open' }               ← IGNORA
  📱  qrcode.updated      → data: { qrcode: '...' }               ← IGNORA
  📤  send.message        → data: { key: {fromMe: true} }         ← IGNORA (fromMe)
  🏷️  groups.upsert       → data: [ {id, subject, ...} ]          ← IGNORA (lista)
  🔔  presence.update     → data: { id, presences: {...} }        ← IGNORA

  FILTRO DUPLO:
    1. Aceita apenas eventos na lista _EVENTOS_MENSAGEM
    2. Rejeita data que seja lista (não é mensagem individual)
    3. Rejeita fromMe=true (mensagem enviada pelo bot)
    4. Rejeita grupos (@g.us), broadcasts, newsletters

ESTRUTURA DO messages.upsert (Evolution API v2):
  {
    "event": "messages.upsert",
    "instance": "bot_uema",
    "data": {
      "key": {
        "remoteJid": "5598...@s.whatsapp.net",
        "fromMe": false,
        "id": "ABC123DEF456"
      },
      "message": {
        "conversation": "Olá",                         ← texto simples
        "extendedTextMessage": { "text": "com link" }, ← texto com preview
        "imageMessage": { "caption": "legenda" },      ← imagem
        "videoMessage": { "caption": "legenda" },      ← vídeo
        "audioMessage": {},                            ← áudio (sem texto)
        "documentMessage": { "caption": "legenda" }   ← documento
      },
      "messageType": "conversation",
      "pushName": "João Silva",
      "instanceId": "ac41b6fd-..."
    }
  }
"""
from __future__ import annotations
import json
import uuid
import logging

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

# Únicos eventos que representam mensagens recebidas na Evolution API v2
# Todos os outros (contacts.*, groups.*, connection.*, presence.*) são ignorados
_EVENTOS_MENSAGEM = {"messages.upsert"}

# Tipos de mídia (sem texto para processar pelo LLM)
_TIPOS_MIDIA = {
    "audioMessage", "stickerMessage", "reactionMessage",
    "protocolMessage", "pollCreationMessage",
}


class DevGuard:
    def __init__(self, redis_client):
        self.r = redis_client
        self.dev_mode = getattr(settings, "DEV_MODE", False)

        whitelist_raw = getattr(settings, "DEV_WHITELIST", "")
        if isinstance(whitelist_raw, str):
            self.dev_whitelist = {n.strip() for n in whitelist_raw.split(",") if n.strip()}
        else:
            self.dev_whitelist = set(whitelist_raw)

        logger.info(
            "🛡️  DevGuard v5 (Evolution API) | dev_mode=%s | whitelist=%s",
            self.dev_mode,
            self.dev_whitelist or "(vazia — todos passam)",
        )

    async def validar(self, data: dict) -> tuple[bool, dict | str]:
        """
        Valida payload da Evolution API.

        Retorno:
          (True,  identity: dict) → aprovado
          (False, motivo: str)    → bloqueado
        """

        # ── Log DEBUG do payload bruto ─────────────────────────────────────────
        logger.debug(
            "📦 Payload bruto:\n%s",
            json.dumps(data, ensure_ascii=False, indent=2)[:800],
        )

        # ── 1. Filtro de evento ────────────────────────────────────────────────
        # Rejeita contacts.upsert, contacts.update, connection.update,
        # groups.upsert, presence.update, qrcode.updated, send.message, etc.
        evento = data.get("event", "")
        if evento not in _EVENTOS_MENSAGEM:
            logger.debug("⏭️  Evento ignorado: '%s'", evento)
            return False, "ignored_event"

        # ── 2. Extrai o bloco "data" ───────────────────────────────────────────
        msg_data = data.get("data", {})

        # GUARD CRÍTICO: contacts.upsert e groups.upsert mandam data como LISTA
        # Se chegou aqui (evento passou), mas data é lista → estrutura inválida
        if isinstance(msg_data, list):
            logger.debug("⏭️  data é lista (não é mensagem individual), ignorando.")
            return False, "ignored_event"

        if not msg_data:
            logger.warning("⚠️  Evento '%s' sem campo 'data'", evento)
            return False, "empty_payload"

        # ── 3. Extrai key ──────────────────────────────────────────────────────
        key = msg_data.get("key", {})

        # ── 4. Ignora mensagens próprias ───────────────────────────────────────
        if key.get("fromMe", False):
            logger.debug("⏭️  Mensagem própria ignorada (fromMe=true).")
            return False, "ignored_self"

        # ── 5. Extração do chat_id ─────────────────────────────────────────────
        chat_id = key.get("remoteJid", "")
        if not chat_id:
            logger.warning("⚠️  Sem remoteJid. Chaves em 'key': %s", list(key.keys()))
            return False, "invalid_payload"

        sender_phone = chat_id.split("@")[0]

        # ── 6. Filtro de grupos ────────────────────────────────────────────────
        if "@g.us" in chat_id:
            logger.debug("⏭️  Grupo ignorado: %s", chat_id)
            return False, "ignored_group"

        # ── 7. Filtro de broadcast e status ───────────────────────────────────
        if "status@broadcast" in chat_id or "broadcast" in chat_id:
            logger.debug("⏭️  Broadcast ignorado: %s", chat_id)
            return False, "ignored_broadcast"

        # ── 8. Filtro de newsletter ────────────────────────────────────────────
        if "@newsletter" in chat_id:
            logger.debug("⏭️  Newsletter ignorada: %s", chat_id)
            return False, "ignored_newsletter"

        # ── 9. Modo DEV: whitelist ─────────────────────────────────────────────
        if self.dev_mode and self.dev_whitelist and sender_phone not in self.dev_whitelist:
            logger.info("🚧 DevGuard bloqueou: %s (não está na DEV_WHITELIST)", sender_phone)
            return False, "not_in_whitelist"

        # ── 10. Deduplicação via Redis ─────────────────────────────────────────
        event_id = key.get("id") or data.get("id") or str(uuid.uuid4())
        if self.r:
            chave = f"evt:{event_id}"
            if self.r.get(chave):
                logger.debug("🔁 Duplicado ignorado: %s", event_id)
                return False, "duplicate"
            self.r.setex(chave, 300, "1")

        # ── 11. Extração do body ───────────────────────────────────────────────
        message = msg_data.get("message", {})
        body = (
            message.get("conversation")
            or message.get("extendedTextMessage", {}).get("text")
            or message.get("imageMessage", {}).get("caption")
            or message.get("videoMessage", {}).get("caption")
            or message.get("documentMessage", {}).get("caption")
            or ""
        ).strip()

        # ── 12. Tipo da mensagem ───────────────────────────────────────────────
        msg_type = msg_data.get("messageType", "unknown")
        has_media = msg_type in _TIPOS_MIDIA or (
            msg_type in ("imageMessage", "videoMessage", "audioMessage", "documentMessage")
        )

        # ── 13. Ignora mídia sem legenda (áudio, sticker, reação) ─────────────
        if msg_type in _TIPOS_MIDIA and not body:
            logger.debug("⏭️  Mídia sem texto ignorada: %s [%s]", msg_type, sender_phone)
            return False, "ignored_media_no_text"

        push_name = msg_data.get("pushName", "")

        identity = {
            "chat_id":      chat_id,
            "sender_phone": sender_phone,
            "body":         body,
            "has_media":    has_media,
            "msg_type":     msg_type,
            "push_name":    push_name,
        }

        logger.info(
            "✅ [%s / %s] tipo=%s | '%s'",
            push_name or sender_phone, sender_phone, msg_type, body[:80],
        )
        return True, identity