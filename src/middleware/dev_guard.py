"""
================================================================================
dev_guard.py — Middleware de Validação e Segurança (v2)
================================================================================

RESUMO:
  Porteiro do sistema. Toda mensagem recebida pelo /webhook passa aqui primeiro.
  Só libera para o handler o que for válido, deduplicado e autorizado.

CORREÇÕES v2:
  1. __init__ simplificado: não precisa receber settings como parâmetro,
     lê diretamente do módulo (evita o bug de chamada DevGuard(r) sem settings)
  2. Método renomeado para validar() (era validar_requisicao) — consistência com main.py
  3. dev_whitelist e dev_mode lidos do settings (com fallback para valores padrão)
  4. identity retorna 'chat_id' e 'body' prontos para o WebhookHandler usar direto

FLUXO DE VALIDAÇÃO (em ordem):
  1. Evento deve ser "message"
  2. Não pode ser mensagem própria (fromMe)
  3. chat_id deve existir e ser válido
  4. Não pode ser grupo (@g.us) ou status broadcast
  5. Se dev_mode ativo: sender_phone deve estar na whitelist
  6. Deduplicação via Redis (TTL 5 min): mesmo event_id não passa duas vezes
  7. Retorna identity pronta para o handler
================================================================================
"""

import uuid
import logging
from src.config import settings

logger = logging.getLogger(__name__)


class DevGuard:
    def __init__(self, redis_client):
        """
        Parâmetros:
          redis_client : instância já conectada do Redis (vinda da main.py)

        Lê dev_mode e dev_whitelist do settings (com fallback seguro).
        Não recebe settings como parâmetro para simplificar a instanciação.
        """
        self.r = redis_client

        # Lê do settings com fallback — não quebra se a variável não existir
        self.dev_mode = getattr(settings, "DEV_MODE", True)

        # Whitelist: pode ser definida no settings como lista ou set
        # Fallback para conjunto vazio (ninguém passa em dev_mode sem whitelist)
        whitelist_raw = getattr(settings, "DEV_WHITELIST", "559887680098,175174737518829")
        if isinstance(whitelist_raw, str):
            # Suporte a formato "55999...,55988..." no .env
            self.dev_whitelist = set(n.strip() for n in whitelist_raw.split(",") if n.strip())
        else:
            self.dev_whitelist = set(whitelist_raw)

        logger.info(
            "🛡️  DevGuard iniciado | dev_mode=%s | whitelist=%s",
            self.dev_mode,
            self.dev_whitelist,
        )

    async def validar(self, data: dict) -> tuple[bool, dict | str]:
        """
        Valida a requisição recebida no /webhook.

        Parâmetro:
          data : dict bruto do JSON recebido pelo FastAPI

        Retorno:
          (True,  identity: dict) → aprovado, segue para o handler
          (False, motivo: str)    → bloqueado, retorna status para o WAHA

        identity contém:
          chat_id      : JID completo (ex: "5598...@s.whatsapp.net")
          sender_phone : só o número (ex: "5598...")
          body         : texto da mensagem
          has_media    : bool
          msg_type     : tipo da mensagem ("chat", "image", etc.)
        """

        # ── 1. Filtro de evento ────────────────────────────────────────────────
        # Só processa eventos do tipo "message". Ignora session.status, etc.
        if data.get("event") != "message":
            logger.debug("⏭️  Evento ignorado: %s", data.get("event"))
            return False, "ignored_event"

        payload = data.get("payload", {})

        # ── 2. Ignora mensagens próprias ──────────────────────────────────────
        if not payload or payload.get("fromMe"):
            return False, "ignored_self"

        # ── 3. Extração do chat_id ────────────────────────────────────────────
        # O WAHA pode enviar o remetente em 'from' ou em 'key.remoteJid'
        # dependendo da versão. Tentamos os dois.
        chat_id = payload.get("from") or payload.get("key", {}).get("remoteJid", "")
        if not chat_id:
            logger.warning("⚠️  Payload sem chat_id: %s", str(payload)[:200])
            return False, "invalid_payload"

        sender_phone = chat_id.split("@")[0]

        # ── 4. Filtro de grupos e status broadcast ────────────────────────────
        if "@g.us" in chat_id or "status@broadcast" in chat_id:
            logger.debug("⏭️  Grupo/broadcast ignorado: %s", chat_id)
            return False, "ignored_group_status"

        # ── 5. Modo DEV: whitelist ────────────────────────────────────────────
        if self.dev_mode and sender_phone not in self.dev_whitelist:
            logger.info("🚧 DevGuard bloqueou: %s (fora da whitelist)", sender_phone)
            return False, "not_in_whitelist"

        # ── 6. Extração do event_id para deduplicação ─────────────────────────
        # Tenta pegar ID do evento ou da mensagem; gera UUID como fallback.
        event_id = (
            data.get("id")
            or payload.get("id")
            or payload.get("key", {}).get("id")
            or str(uuid.uuid4())
        )

        # ── 7. Deduplicação via Redis ─────────────────────────────────────────
        # Garante que a mesma mensagem não seja processada duas vezes
        # (pode acontecer se o WAHA reenviar o webhook por timeout).
        if self.r:
            chave_evt = f"evt:{event_id}"
            if self.r.get(chave_evt):
                logger.debug("🔁 Evento duplicado ignorado: %s", event_id)
                return False, "duplicate"
            # Marca como processado por 5 minutos
            self.r.setex(chave_evt, 300, "1")

        # ── 8. Monta identity aprovada ────────────────────────────────────────
        body = (payload.get("body") or "").strip()

        identity = {
            "chat_id":      chat_id,
            "sender_phone": sender_phone,
            "body":         body,
            "has_media":    payload.get("hasMedia", False),
            "msg_type":     (
                payload.get("_data", {}).get("type")
                or ("chat" if body else None)
            ),
        }

        logger.debug("✅ DevGuard aprovado: %s | body: '%s'", sender_phone, body[:60])
        return True, identity