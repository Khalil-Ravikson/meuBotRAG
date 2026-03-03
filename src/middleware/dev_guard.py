"""
middleware/dev_guard.py — v9 (Evolution API v2.3.7+)
=====================================================

CONTEXTO @LID (história completa):
  O @lid é um ID interno do WhatsApp usado quando o remetente usa dispositivo
  vinculado (WhatsApp Web/Desktop). O número real não é divulgado pela API.

  Histórico de versões Evolution API:
    ≤ v2.2.3  → @lid no remoteJid, sem resolução → erro 400 garantido
    v2.3.0–4  → senderPn intermitente, inconsistente
    v2.3.5+   → resolve @lid via call rejection silenciosa → chega @s.whatsapp.net

  Com v2.3.7 (versão actual) o remoteJid deve chegar resolvido na maioria dos
  casos. Mantemos senderPn como fallback para os casos edge ainda existentes.

CAMPOS DO PAYLOAD (Evolution v2.3.7):
  "event": "messages.upsert"
  "sender": "559887400509@s.whatsapp.net"   ← instância do BOT (nunca usar para reply)
  "data": {
    "key": {
      "remoteJid": "559812345678@s.whatsapp.net",  ← resolvido pela v2.3.5+
       OU (caso edge ainda possível)
      "remoteJid": "191555725959219@lid",           ← fallback: usa senderPn
    },
    "senderPn": "559812345678@s.whatsapp.net",      ← presente quando remoteJid = @lid
    "pushName": "Nome do Utilizador",
    "message": { "conversation": "Olá" },
    "messageType": "conversation"
  }
"""
from __future__ import annotations
import json
import uuid
import logging

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)

_EVENTOS_MENSAGEM = {"messages.upsert"}

_TIPOS_MIDIA_SEM_TEXTO = {
    "audioMessage", "stickerMessage", "reactionMessage",
    "protocolMessage", "pollCreationMessage",
}


def _normalizar_numero(jid: str) -> str:
    """'559887400509@s.whatsapp.net' → '559887400509'"""
    return jid.split("@")[0].replace("+", "").replace(" ", "").strip()


def _resolver_chat_id(key: dict, msg_data: dict) -> str | None:
    """
    Resolve o chat_id para enviar a resposta.

    Prioridade (v2.3.7+):
      1. remoteJid @s.whatsapp.net → caso normal e caso resolvido pela v2.3.5+
      2. senderPn @s.whatsapp.net  → fallback para @lid ainda não resolvido
      3. None                       → não resolvível, rejeita a mensagem

    NUNCA usa data.sender (= número da instância do bot).
    """
    remote_jid = key.get("remoteJid", "")

    # Caso 1: número real já presente (normal, ou resolvido pela v2.3.5+)
    if "@s.whatsapp.net" in remote_jid:
        return remote_jid

    # Caso 2: ainda @lid (edge case) — tenta senderPn da v2.3.0+
    if "@lid" in remote_jid:
        sender_pn = msg_data.get("senderPn", "")
        if sender_pn and "@s.whatsapp.net" in sender_pn:
            logger.info("📱 @lid resolvido via senderPn: %s → %s", remote_jid, sender_pn)
            return sender_pn

        # Sem senderPn: não resolvível nesta versão
        logger.warning(
            "⚠️  @lid sem senderPn. remoteJid=%s\n"
            "   A Evolution API v2.3.7 devia ter resolvido isto.\n"
            "   Verifica se a imagem está actualizada: docker pull atendai/evolution-api:v2.3.7",
            remote_jid,
        )
        return None

    # Formato inesperado (broadcast/newsletter já filtrados antes)
    logger.warning("⚠️  remoteJid formato desconhecido: %s", remote_jid)
    return None


class DevGuard:
    def __init__(self, redis_client):
        self.r        = redis_client
        self.dev_mode = getattr(settings, "DEV_MODE", False)

        whitelist_raw = getattr(settings, "DEV_WHITELIST", "")
        if isinstance(whitelist_raw, str):
            self.dev_whitelist = {
                _normalizar_numero(n)
                for n in whitelist_raw.split(",")
                if n.strip()
            }
        else:
            self.dev_whitelist = {_normalizar_numero(n) for n in whitelist_raw}

        if self.dev_mode and not self.dev_whitelist:
            logger.warning(
                "⚠️  DEV_MODE=True mas DEV_WHITELIST vazia — "
                "todas as mensagens passam. Define DEV_WHITELIST=<teu_numero> no .env."
            )
        elif self.dev_mode:
            logger.info(
                "🛡️  DevGuard v9 (Evolution v2.3.7+) | DEV_MODE=True | whitelist=%s",
                self.dev_whitelist,
            )
        else:
            logger.info("🛡️  DevGuard v9 (Evolution v2.3.7+) | DEV_MODE=False | todas as msgs passam")

    async def validar(self, data: dict) -> tuple[bool, dict | str]:
        """
        Valida e filtra payload da Evolution API v2.3.7.
        Retorno: (True, identity) → aprovado | (False, motivo) → bloqueado
        """
        logger.debug("📦 Payload: %s", json.dumps(data, ensure_ascii=False)[:400])

        # ── 1. Filtro de evento ────────────────────────────────────────────────
        evento = data.get("event", "")
        if evento not in _EVENTOS_MENSAGEM:
            logger.debug("⏭️  Evento ignorado: '%s'", evento)
            return False, "ignored_event"

        # ── 2. Extrai bloco data ───────────────────────────────────────────────
        msg_data = data.get("data", {})
        if isinstance(msg_data, list):
            logger.debug("⏭️  data é lista — ignorado.")
            return False, "ignored_event"
        if not msg_data:
            logger.warning("⚠️  messages.upsert sem campo 'data'")
            return False, "empty_payload"

        key        = msg_data.get("key", {})
        remote_jid = key.get("remoteJid", "")

        # ── 3. Ignora mensagens próprias (enviadas pelo bot) ───────────────────
        if key.get("fromMe", False):
            logger.debug("⏭️  fromMe=true — ignorado.")
            return False, "ignored_self"

        # ── 4. Filtros de origem ───────────────────────────────────────────────
        if "@g.us" in remote_jid:
            logger.debug("⏭️  Grupo ignorado: %s", remote_jid)
            return False, "ignored_group"
        if "broadcast" in remote_jid or "@newsletter" in remote_jid:
            logger.debug("⏭️  Broadcast/newsletter ignorado.")
            return False, "ignored_broadcast"
        if not remote_jid:
            logger.warning("⚠️  remoteJid vazio no payload.")
            return False, "invalid_payload"

        # ── 5. Resolve chat_id para resposta ───────────────────────────────────
        chat_id = _resolver_chat_id(key, msg_data)
        if chat_id is None:
            return False, "unresolvable_lid"

        sender_phone = _normalizar_numero(remote_jid)
        push_name    = msg_data.get("pushName", "")

        # ── 6. DEV_MODE: whitelist ─────────────────────────────────────────────
        if self.dev_mode and self.dev_whitelist:
            numero_check = _normalizar_numero(chat_id)
            if numero_check not in self.dev_whitelist:
                logger.info(
                    "🚧 DEV bloqueou %s ('%s')\n"
                    "   → Adiciona '%s' ao DEV_WHITELIST no .env",
                    numero_check, push_name, numero_check,
                )
                return False, "not_in_whitelist"

        # ── 7. Extrai corpo da mensagem ────────────────────────────────────────
        message  = msg_data.get("message", {})
        msg_type = msg_data.get("messageType", "unknown")
        body = (
            message.get("conversation")
            or message.get("extendedTextMessage", {}).get("text")
            or message.get("imageMessage", {}).get("caption")
            or message.get("videoMessage", {}).get("caption")
            or message.get("documentMessage", {}).get("caption")
            or ""
        ).strip()

        # ── 8. Ignora mídia sem texto ──────────────────────────────────────────
        if msg_type in _TIPOS_MIDIA_SEM_TEXTO and not body:
            logger.debug("⏭️  Mídia sem texto ignorada [%s]", sender_phone)
            return False, "ignored_media_no_text"

        # ── 9. Deduplicação via Redis ──────────────────────────────────────────
        event_id = key.get("id") or data.get("id") or str(uuid.uuid4())
        if self.r:
            chave = f"evt:{event_id}"
            try:
                if self.r.get(chave):
                    logger.debug("🔁 Duplicado ignorado: %s", event_id)
                    return False, "duplicate"
                self.r.setex(chave, 300, "1")
            except Exception as e:
                logger.warning("⚠️  Redis indisponível para dedup: %s", e)

        # ── 10. Aprovado ───────────────────────────────────────────────────────
        has_media = msg_type in {
            "imageMessage", "videoMessage", "audioMessage",
            "documentMessage", *_TIPOS_MIDIA_SEM_TEXTO,
        }

        identity = {
            "chat_id":      chat_id,       # destino resolvido — para enviar resposta
            "sender_phone": sender_phone,  # ID original — para logs e Redis
            "body":         body,
            "has_media":    has_media,
            "msg_type":     msg_type,
            "push_name":    push_name,
        }

        logger.info(
            "✅ Aprovada | destino=%s | user=%s ('%s') | tipo=%s | '%s'",
            chat_id, sender_phone, push_name, msg_type, body[:60],
        )
        return True, identity