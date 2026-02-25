from __future__ import annotations
import json
import uuid
import logging

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

# A Evolution API usa esse evento para novas mensagens
_EVENTOS_MENSAGEM = {"messages.upsert", "MESSAGES_UPSERT"}

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
            "🛡️  DevGuard (Evolution) | dev_mode=%s | whitelist=%s",
            self.dev_mode,
            self.dev_whitelist or "(vazia — todos passam)",
        )

    async def validar(self, req_data: dict) -> tuple[bool, dict | str]:
        logger.debug(
            "📦 Payload bruto recebido:\n%s",
            json.dumps(req_data, ensure_ascii=False, indent=2)[:1000],
        )

        # ── 1. Filtro de evento ────────────────────────────────────────────────
        evento = req_data.get("event", "")
        if evento not in _EVENTOS_MENSAGEM:
            logger.debug("⏭️  Evento ignorado: '%s'", evento)
            return False, "ignored_event"

        # ── 2. Extrai dados (Evolution usa 'data' em vez de 'payload') ────────
        data = req_data.get("data", {})
        if not data:
            logger.warning("⚠️  Evento '%s' sem chave 'data'.", evento)
            return False, "empty_payload"

        # ── 3. Ignora mensagens próprias ──────────────────────────────────────
        key = data.get("key", {})
        if key.get("fromMe"):
            logger.debug("⏭️  Mensagem própria ignorada.")
            return False, "ignored_self"

        # ── 4. Extração do chat_id ────────────────────────────────────────────
        chat_id = key.get("remoteJid", "")
        if not chat_id:
            logger.warning("⚠️  Payload sem chat_id (remoteJid).")
            return False, "invalid_payload"

        sender_phone = chat_id.split("@")[0]

        # ── 5. Filtro de grupos e status broadcast ────────────────────────────
        if "@g.us" in chat_id or "status@broadcast" in chat_id:
            logger.debug("⏭️  Grupo/broadcast ignorado: %s", chat_id)
            return False, "ignored_group_status"

        # ── 6. Modo DEV: whitelist ────────────────────────────────────────────
        if self.dev_mode and self.dev_whitelist and sender_phone not in self.dev_whitelist:
            logger.info("🚧 DevGuard bloqueou: %s (não está na DEV_WHITELIST)", sender_phone)
            return False, "not_in_whitelist"

        # ── 7. Extração do event_id para deduplicação ─────────────────────────
        event_id = key.get("id", str(uuid.uuid4()))

        # ── 8. Deduplicação via Redis (TTL 5 min) ─────────────────────────────
        if self.r:
            chave = f"evt:{event_id}"
            if self.r.get(chave):
                logger.debug("🔁 Evento duplicado ignorado: %s", event_id)
                return False, "duplicate"
            self.r.setex(chave, 300, "1")

        # ── 9. Extração do corpo da mensagem ──────────────────────────────────
        msg_obj = data.get("message", {})
        if not msg_obj:
            logger.debug("⏭️  Payload sem objeto de mensagem válido.")
            return False, "empty_message_object"

        # O Baileys envia o texto em diferentes chaves dependendo se tem citação, mídia, etc.
        body = (
            msg_obj.get("conversation") or
            msg_obj.get("extendedTextMessage", {}).get("text") or
            msg_obj.get("imageMessage", {}).get("caption") or
            msg_obj.get("videoMessage", {}).get("caption") or
            msg_obj.get("documentMessage", {}).get("caption") or
            ""
        ).strip()

        # ── 10. Monta identity ────────────────────────────────────────────────
        msg_type = data.get("messageType", "unknown")
        has_media = msg_type in ["imageMessage", "videoMessage", "audioMessage", "documentMessage"]

        identity = {
            "chat_id":      chat_id,
            "sender_phone": sender_phone,
            "body":         body,
            "has_media":    has_media,
            "msg_type":     msg_type,
        }

        logger.info("✅ [%s] '%s'", sender_phone, body[:80])
        return True, identity