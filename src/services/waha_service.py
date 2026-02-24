"""
services/waha_service.py — Integração com WAHA (WhatsApp HTTP API)
===================================================================
Sem mudanças na lógica — só atualização do import de settings.

MIGRAÇÃO:
  Antes: from src.config import settings
  Agora: from src.infrastructure.settings import settings
"""
from __future__ import annotations
import logging
import httpx

from src.infrastructure.settings import settings

logger = logging.getLogger(__name__)


class WahaService:
    def __init__(self):
        self.base_url   = settings.WAHA_BASE_URL.rstrip("/")
        self.api_key    = settings.WAHA_API_KEY
        self.session    = settings.WAHA_SESSION
        self.headers    = {
            "Content-Type": "application/json",
            "X-Api-Key":    self.api_key,
        }
        self.webhook_url = settings.WHATSAPP_HOOK_URL
        self.events      = ["message", "session.status"]

    # ------------------------------------------------------------------
    # WEBHOOK
    # ------------------------------------------------------------------

    async def configurar_webhook(self) -> None:
        """Registra/atualiza o Webhook via PUT /api/sessions/{session}."""
        url = f"{self.base_url}/api/sessions/{self.session}"
        payload = {
            "name": self.session,
            "config": {
                "webhooks": [{
                    "url":           self.webhook_url,
                    "events":        self.events,
                    "hmac":          None,
                    "retries":       None,
                    "customHeaders": None,
                }]
            },
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.put(url, json=payload, headers=self.headers)
                if r.status_code == 200:
                    logger.info("✅ Webhook configurado → %s", self.webhook_url)
                else:
                    logger.warning("⚠️  Webhook status %s | %s", r.status_code, r.text)
            except httpx.ConnectError:
                logger.error("❌ Não foi possível conectar ao WAHA: %s", self.base_url)
            except httpx.TimeoutException:
                logger.error("❌ Timeout ao configurar webhook.")
            except Exception as e:
                logger.exception("❌ Erro ao configurar webhook: %s", e)

    # ------------------------------------------------------------------
    # ENVIO DE MENSAGEM
    # ------------------------------------------------------------------

    async def enviar_mensagem(self, chat_id: str, texto: str) -> None:
        """Envia mensagem de texto via POST /api/sendText."""
        if not chat_id or not texto:
            logger.warning("⚠️  enviar_mensagem: chat_id ou texto vazio.")
            return

        url = f"{self.base_url}/api/sendText"
        payload = {
            "session": self.session,
            "chatId":  chat_id,
            "text":    texto,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                r = await client.post(url, json=payload, headers=self.headers)
                if r.status_code in (200, 201):
                    logger.info("✅ Mensagem enviada para %s", chat_id)
                else:
                    logger.warning("⚠️  Falha ao enviar. Status %s | %s", r.status_code, r.text)
            except httpx.ConnectError:
                logger.error("❌ Não foi possível conectar ao WAHA: %s", self.base_url)
            except httpx.TimeoutException:
                logger.error("❌ Timeout ao enviar para %s", chat_id)
            except Exception as e:
                logger.exception("❌ Erro inesperado ao enviar mensagem: %s", e)

    # ------------------------------------------------------------------
    # STATUS DA SESSÃO
    # ------------------------------------------------------------------

    async def verificar_sessao(self) -> str | None:
        """Consulta o status atual da sessão."""
        url = f"{self.base_url}/api/sessions/{self.session}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.get(url, headers=self.headers)
                if r.status_code == 200:
                    status = r.json().get("status", "UNKNOWN")
                    logger.info("ℹ️  Sessão '%s': %s", self.session, status)
                    return status
                logger.warning("⚠️  Status sessão: %s | %s", r.status_code, r.text)
                return None
            except httpx.ConnectError:
                logger.error("❌ WAHA inacessível: %s", self.base_url)
            except httpx.TimeoutException:
                logger.error("❌ Timeout ao verificar sessão.")
            except Exception as e:
                logger.exception("❌ Erro ao verificar sessão: %s", e)
            return None

    # ------------------------------------------------------------------
    # INICIALIZAÇÃO (chamada no startup)
    # ------------------------------------------------------------------

    async def inicializar(self) -> None:
        """Verifica sessão e configura o webhook."""
        logger.info("🚀 Inicializando WahaService...")
        status = await self.verificar_sessao()
        if status is None:
            logger.error("❌ WAHA inacessível. Verifique o container.")
            return
        if status not in ("WORKING", "SCAN_QR_CODE", "STARTING"):
            logger.warning("⚠️  Sessão com status '%s'. Webhook será registrado mesmo assim.", status)
        await self.configurar_webhook()