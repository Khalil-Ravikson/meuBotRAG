import httpx
import logging
import asyncio
from src.config import settings

logger = logging.getLogger(__name__)


class WahaService:
    def __init__(self):
        self.base_url = settings.WAHA_BASE_URL.rstrip("/")
        self.api_key = settings.WAHA_API_KEY
        self.session = "default"
        self.headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
        }
        self.webhook_url = settings.WHATSAPP_HOOK_URL or "http://bot-rag:8000/webhook"
        self.events = ["message", "session.status"]

    # ------------------------------------------------------------------
    # WEBHOOK
    # ------------------------------------------------------------------

    async def configurar_webhook(self):
        """
        Registra/atualiza o Webhook via PUT /api/sessions/{session}.
        Chamado uma vez na inicialização do bot.
        """
        url = f"{self.base_url}/api/sessions/{self.session}"
        payload = {
            "name": self.session,
            "config": {
                "webhooks": [
                    {
                        "url": self.webhook_url,
                        "events": self.events,
                        "hmac": None,
                        "retries": None,
                        "customHeaders": None,
                    }
                ]
            },
        }

        logger.debug("📡 PUT %s", url)
        logger.debug("📦 Payload webhook: %s", payload)

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.put(url, json=payload, headers=self.headers)
                self._log_response("configurar_webhook", r)

                if r.status_code == 200:
                    logger.info("✅ Webhook configurado com sucesso! → %s", self.webhook_url)
                else:
                    logger.warning(
                        "⚠️  Webhook não configurado. Status %s | Body: %s",
                        r.status_code,
                        r.text,
                    )

            except httpx.ConnectError:
                logger.error("❌ Não foi possível conectar ao WAHA em: %s", self.base_url)
            except httpx.TimeoutException:
                logger.error("❌ Timeout ao tentar configurar webhook em: %s", url)
            except Exception as e:
                logger.exception("❌ Erro inesperado ao configurar webhook: %s", e)

    # ------------------------------------------------------------------
    # ENVIO DE MENSAGEM
    # ------------------------------------------------------------------

    async def enviar_mensagem(self, chat_id: str, texto: str):
        """
        Envia mensagem de texto via POST /api/sendText.
        """
        if not chat_id or not texto:
            logger.warning("⚠️  enviar_mensagem chamado com chat_id ou texto vazio.")
            return

        url = f"{self.base_url}/api/sendText"
        payload = {
            "session": self.session,
            "chatId": chat_id,
            "text": texto,
        }

        logger.debug("📡 POST %s", url)
        logger.debug("📦 Payload mensagem: %s", payload)

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                r = await client.post(url, json=payload, headers=self.headers)
                self._log_response("enviar_mensagem", r)

                if r.status_code in (200, 201):
                    logger.info("✅ Mensagem enviada para %s", chat_id)
                else:
                    logger.warning(
                        "⚠️  Falha ao enviar mensagem. Status %s | Body: %s",
                        r.status_code,
                        r.text,
                    )

            except httpx.ConnectError:
                logger.error("❌ Não foi possível conectar ao WAHA em: %s", self.base_url)
            except httpx.TimeoutException:
                logger.error("❌ Timeout ao enviar mensagem para %s", chat_id)
            except Exception as e:
                logger.exception("❌ Erro inesperado ao enviar mensagem: %s", e)

    # ------------------------------------------------------------------
    # STATUS DA SESSÃO
    # ------------------------------------------------------------------

    async def verificar_sessao(self) -> str | None:
        """
        Consulta o status atual da sessão via GET /api/sessions/{session}.
        Retorna o status (ex: 'WORKING', 'STOPPED') ou None em caso de erro.
        """
        url = f"{self.base_url}/api/sessions/{self.session}"
        logger.debug("📡 GET %s", url)

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                r = await client.get(url, headers=self.headers)
                self._log_response("verificar_sessao", r)

                if r.status_code == 200:
                    data = r.json()
                    status = data.get("status", "UNKNOWN")
                    logger.info("ℹ️  Status da sessão '%s': %s", self.session, status)
                    return status
                else:
                    logger.warning(
                        "⚠️  Não foi possível obter status da sessão. Status %s | Body: %s",
                        r.status_code,
                        r.text,
                    )
                    return None

            except httpx.ConnectError:
                logger.error("❌ Não foi possível conectar ao WAHA em: %s", self.base_url)
            except httpx.TimeoutException:
                logger.error("❌ Timeout ao verificar sessão.")
            except Exception as e:
                logger.exception("❌ Erro inesperado ao verificar sessão: %s", e)
            return None

    # ------------------------------------------------------------------
    # INICIALIZAÇÃO COMPLETA (chamada no startup do bot)
    # ------------------------------------------------------------------

    async def inicializar(self):
        """
        Sequência de inicialização:
        1. Verifica se a sessão está ativa
        2. Configura o webhook
        """
        logger.info("🚀 Inicializando WahaService...")
        logger.debug("   base_url   : %s", self.base_url)
        logger.debug("   session    : %s", self.session)
        logger.debug("   webhook_url: %s", self.webhook_url)
        logger.debug("   events     : %s", self.events)

        status = await self.verificar_sessao()
        if status is None:
            logger.error("❌ WAHA inacessível. Verifique se o container está rodando.")
            return

        if status not in ("WORKING", "SCAN_QR_CODE", "STARTING"):
            logger.warning(
                "⚠️  Sessão está com status '%s'. O webhook será registrado mas mensagens podem não funcionar.",
                status,
            )

        await self.configurar_webhook()

    # ------------------------------------------------------------------
    # HELPER
    # ------------------------------------------------------------------

    @staticmethod
    def _log_response(metodo: str, r: httpx.Response):
        """Loga detalhes da resposta HTTP no nível DEBUG."""
        logger.debug(
            "[%s] ← %s %s | Status: %s | Body: %s",
            metodo,
            r.request.method,
            r.request.url,
            r.status_code,
            r.text[:500],  # limita pra não poluir o log
        )